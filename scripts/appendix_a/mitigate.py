"""预先固定 cap=5 的真实模型重加权对照；同子集、初始化种子、轮数与测试集。

作为本组扩展单独保存，不纳入原四实验的 132 个模型。配对基线为实验2/q=.1
的三个 ERM 模型；不通过测试结果挑选权重上限。
"""
from pathlib import Path
import sys,json,time,hashlib,random,os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES,GENDERS,sample_biased,group_recall,sha256_file
from appendix_a.reweight import clipped_cell_weights
from appendix_a.trainer import FaceDataset,create_model,seed_all,worker_seed,_atomic_save


def run(config,repetition,cap=5.):
    experiment=2;step=1;seed=config['seed']+experiment*10000+step*100+repetition
    manifest=pd.read_csv(config['manifest']);train=manifest[manifest.split=='train'];test=manifest[manifest.split=='test']
    seed_all(seed);selected,counts=sample_biased(train,experiment,step,seed,config['class_totals'])
    baseline=f'exp2_q01_rep{repetition}';base=Path(config['output_dir'])/baseline
    base_result=json.loads((base/'complete.json').read_text())
    saved_selection=pd.read_csv(Path(config['checkpoint_dir'])/baseline/'train_selection.csv')
    assert selected.sample_id.tolist()==saved_selection.sample_id.tolist()
    n=counts.to_numpy(float);w,ess_fraction=clipped_cell_weights(n,cap)
    assert np.allclose((n*w).sum(0),n.sum(0))
    ess=ess_fraction*n.sum()
    name=f'weighted_exp2_q01_rep{repetition}';out=ROOT/'results/appendixA_mitigation'/name;out.mkdir(parents=True,exist_ok=True)
    ck=ROOT/'checkpoints/appendixA_mitigation'/name;ck.mkdir(parents=True,exist_ok=True)
    meta={'method':'class-marginal-preserving clipped reweighting','cap_before_class_normalization':cap,
          'max_final_weight':float(w.max()),'ESS_fraction':float(ess/n.sum()),'baseline':baseline,
          'baseline_config_hash':base_result['config_hash'],'seed':seed,'epochs':20,'repetition':repetition,
          'manifest_sha256':sha256_file(config['manifest']),'source_sha256':sha256_file(__file__),
          'reweight_implementation_sha256':sha256_file(ROOT/'src/appendix_a/reweight.py'),
          'test_used_for_cap_or_epoch_selection':False,'original_paper_data_match':False,
          'config':config,'cell_weights':w.tolist()}
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    if (out/'complete.json').exists():
        prior=json.loads((out/'complete.json').read_text());assert prior['config_hash']==fingerprint;return
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    model,_=create_model(config['pretrained_path']);model.to('cuda',memory_format=torch.channels_last)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(FaceDataset(selected,config['image_root'],224,True,config['horizontal_flip']),batch_size=config['batch_size'],
                      shuffle=True,num_workers=config['workers'],pin_memory=True,generator=gen,worker_init_fn=worker_seed)
    testloader=DataLoader(FaceDataset(test,config['image_root'],224),batch_size=config['batch_size'],shuffle=False,num_workers=config['workers'],pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['max_lr'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config['max_lr'],epochs=20,steps_per_epoch=len(loader),
        pct_start=config['onecycle_pct_start'],div_factor=config['onecycle_div_factor'],final_div_factor=config['onecycle_final_div_factor'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp']);weight_tensor=torch.tensor(w,dtype=torch.float32,device='cuda')
    start_epoch=0;history=[];checkpoint=ck/'latest.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['config_hash']==fingerprint
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch']+1;history=saved['history'];gen.set_state(saved['loader_rng'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for epoch in range(start_epoch,20):
        model.train();loss_sum=0.;correct=0;seen=0;started=time.perf_counter()
        for images,y,g in loader:
            images=images.cuda(non_blocking=True,memory_format=torch.channels_last);y=y.cuda(non_blocking=True)
            gi=torch.tensor([GENDERS.index(k) for k in g],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=config['amp']):
                logits=model(images);loss=(nn.functional.cross_entropy(logits,y,reduction='none')*weight_tensor[gi,y]).mean()
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();schedule.step()
            loss_sum+=loss.detach().item()*len(y);correct+=(logits.argmax(1)==y).sum().item();seen+=len(y)
        row={'epoch':epoch+1,'train_weighted_loss':loss_sum/seen,'train_accuracy':correct/seen,'rows':seen,'seconds':time.perf_counter()-started}
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        _atomic_save({'epoch':epoch,'config_hash':fingerprint,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':schedule.state_dict(),'scaler':scaler.state_dict(),'history':history,'loader_rng':gen.get_state(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint)
        print(json.dumps({'run':name,**row}),flush=True)
    model.eval();ys=[];ps=[];gs=[]
    with torch.inference_mode():
        for images,y,g in testloader:
            with torch.autocast('cuda',enabled=config['amp']):pred=model(images.cuda(non_blocking=True,memory_format=torch.channels_last)).argmax(1).cpu().tolist()
            ys.extend(y.tolist());ps.extend(pred);gs.extend(g)
    recalls=group_recall(ys,ps,gs);recalls.to_csv(out/'recall.csv',index=False)
    confusion=np.zeros((2,7,7),dtype=np.int64)
    for y,p,g in zip(ys,ps,gs):confusion[GENDERS.index(g),y,p]+=1
    np.savez_compressed(out/'confusion.npz',counts=confusion,classes=np.array(CLASSES),genders=np.array(GENDERS))
    result={'run':name,'config_hash':fingerprint,'repetition':repetition,'epochs':20,'test_accuracy':float(np.mean(np.array(ys)==ps)),
            'baseline_accuracy':base_result['test_accuracy'],'ESS_fraction':float(ess/n.sum()),'max_weight':float(w.max()),
            'test_rows':len(test),'original_paper_data_match':False,
            'artifact_sha256':{k:sha256_file(out/k) for k in ['history.csv','recall.csv','confusion.npz']}}
    (out/'complete.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json');a=p.parse_args()
    config=json.loads(a.config.read_text());torch.set_num_threads(8)
    for repetition in range(3):run(config,repetition,cap=5.)
    records=[]
    for repetition in range(3):
        weighted=ROOT/f'results/appendixA_mitigation/weighted_exp2_q01_rep{repetition}'
        wr=pd.read_csv(weighted/'recall.csv');br=pd.read_csv(Path(config['output_dir'])/f'exp2_q01_rep{repetition}'/'recall.csv')
        for method,df in [('ERM',br),('Reweighted',wr)]:
            for row in df.to_dict(orient='records'):records.append({'repetition':repetition,'method':method,**row})
    allrows=pd.DataFrame(records);allrows.to_csv(ROOT/'results/appendixA_mitigation/paired_recalls.csv',index=False)
    summary=allrows.groupby(['method','gender','label']).recall.agg(['mean','std','count']).reset_index()
    summary.to_csv(ROOT/'results/appendixA_mitigation/summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,label in zip(axes,['happy','mean']):
        x=np.arange(2)
        for j,method in enumerate(['ERM','Reweighted']):
            sub=summary[(summary.method==method)&(summary.label==label)].set_index('gender').loc[list(GENDERS)]
            ax.bar(x+(-.18 if j==0 else .18),sub['mean'],width=.36,yerr=sub['std'],capsize=4,label=method,color=['#AAB9C6','#137C8B'][j])
        ax.set_xticks(x,GENDERS);ax.set_ylim(0,1);ax.set_title('Happy recall' if label=='happy' else '7-class macro recall');ax.legend()
    fig.suptitle('Real-model mitigation | same exp2 q=.1 subsets, initialization and test set | 3 seeds')
    fig.tight_layout();fig.savefig(ROOT/'results/figures/appendixA_real_mitigation.png',dpi=170)
    (ROOT/'results/appendixA_mitigation/FINISHED.json').write_text(json.dumps({'complete':True,'real_models':3,'paired_baselines':3,'cap':5.,'test_used_for_cap_selection':False},indent=2))
    print('PAIRED REAL-MODEL MITIGATION FINISHED',flush=True)
