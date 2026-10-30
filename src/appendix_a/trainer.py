"""真实 ResNet50 训练：独立 CUDA 环境、断点恢复、最终 epoch 固定测试评估。

所有未被论文明确指定的实现选项都写入 run.json；不根据测试集挑 epoch。
"""
from pathlib import Path
import os,json,time,random,hashlib,copy
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from PIL import Image
from .data import CLASSES,GENDERS,sha256_file,group_recall,sample_biased

class FaceDataset(Dataset):
    def __init__(self,frame,root,size=224,training=False,flip=True):
        self.frame=frame.reset_index(drop=True);self.root=Path(root)
        ops=[transforms.Resize((size,size))]
        if training and flip:ops.append(transforms.RandomHorizontalFlip())
        ops.extend([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.transform=transforms.Compose(ops)
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(self.root/row.path) as im:image=self.transform(im.convert('RGB'))
        return image,CLASSES.index(row.label),row.gender


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    # 确定性优先；CUDA 非确定性算子显式报错，不悄悄宣称逐位复现。
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)


def create_model(pretrained_path,pretrained_sha256=None):
    path=Path(pretrained_path)
    if not path.is_file():raise FileNotFoundError(f'Pretrained ImageNet ResNet50 weights missing: {path}')
    digest=sha256_file(path)
    if pretrained_sha256 and digest!=pretrained_sha256:raise ValueError('Pretrained model SHA256 mismatch')
    if path.name=='resnet50-0676ba61.pth' and not digest.startswith('0676ba61'):
        raise ValueError('Official ResNet50 V1 weight filename/hash mismatch')
    model=models.resnet50(weights=None)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    return model,digest


def _atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def run_training(config,train,test,experiment,step,repetition,manifest_hash,resume=True,smoke=False):
    seed=int(config['seed'])+experiment*10000+step*100+repetition
    seed_all(seed)
    selected,counts=sample_biased(train,experiment,step,seed,config['class_totals'])
    name=f'exp{experiment}_q{step:02d}_rep{repetition}'
    root=Path(config['output_dir'])/name;root.mkdir(parents=True,exist_ok=True)
    ckroot=Path(config['checkpoint_dir'])/name;ckroot.mkdir(parents=True,exist_ok=True)
    model,weight_hash=create_model(config['pretrained_path'],config.get('pretrained_sha256'))
    resolved={**config,'experiment':experiment,'step':step,'fraction':step/10,'repetition':repetition,
              'run_seed':seed,'manifest_sha256':manifest_hash,'pretrained_sha256':weight_hash,
              'training_rows':len(selected),'test_rows':len(test),'smoke_test_only':smoke,
              'training_schedule_complete':not smoke and config['epochs']==20,
              'original_paper_data_match':bool(config.get('original_paper_data_match',False)),
              'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
              'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
              'implementation_sha256':{p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('data.py')]},
              'test_used_for_checkpoint_selection':False}
    key=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
    if (root/'complete.json').exists():
        prior=json.loads((root/'complete.json').read_text())
        if prior['config_hash']!=key:raise ValueError(f'Completed run config changed: {name}; choose a new output directory')
        for file,digest in prior.get('artifact_sha256',{}).items():
            if sha256_file(root/file)!=digest:raise ValueError(f'Completed run artifact changed: {name}/{file}')
        return prior
    (root/'run.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2))
    counts.to_csv(root/'train_counts.csv');selected.to_csv(ckroot/'train_selection.csv',index=False)
    device=torch.device('cuda');model=model.to(device,memory_format=torch.channels_last)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(FaceDataset(selected,config['image_root'],config['image_size'],True,config['horizontal_flip']),
                      batch_size=config['batch_size'],shuffle=True,num_workers=config['workers'],
                      pin_memory=True,worker_init_fn=worker_seed,generator=generator,
                      persistent_workers=False,drop_last=False)
    # BatchNorm 必须有至少两个训练图像；允许采样总数的最后一个批次合并。
    if len(selected)%config['batch_size']==1:
        raise ValueError('Final train batch would contain one image; choose a different batch size')
    testloader=DataLoader(FaceDataset(test,config['image_root'],config['image_size']),
                          batch_size=config['batch_size'],shuffle=False,num_workers=config['workers'],pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['max_lr'],weight_decay=config['weight_decay'])
    schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config['max_lr'],epochs=config['epochs'],
              steps_per_epoch=len(loader),pct_start=config['onecycle_pct_start'],div_factor=config['onecycle_div_factor'],
              final_div_factor=config['onecycle_final_div_factor'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['amp'])
    start_epoch=0;history=[]
    checkpoint=ckroot/'latest.pt'
    if resume and checkpoint.exists():
        # 仅加载本项目自己产生且 config_hash 一致的检查点；包含 RNG，故非 weights-only。
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config_hash']!=key:raise ValueError('Checkpoint config/manifest mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch']+1;history=saved['history']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        generator.set_state(saved['loader_rng'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
    started=time.perf_counter()
    for epoch in range(start_epoch,config['epochs']):
        model.train();loss_sum=0.;correct=0;n=0;t=time.perf_counter()
        for images,y,_ in loader:
            images=images.to(device,non_blocking=True,memory_format=torch.channels_last);y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=config['amp']):
                logits=model(images);loss=nn.functional.cross_entropy(logits,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();schedule.step()
            loss_sum+=loss.detach().item()*len(y);correct+=(logits.argmax(1)==y).sum().item();n+=len(y)
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'train_loss':loss_sum/n,'train_accuracy':correct/n,
             'lr':schedule.get_last_lr()[0],'seconds':time.perf_counter()-t,'rows':n}
        history.append(row);pd.DataFrame(history).to_csv(root/'history.csv',index=False)
        _atomic_save({'epoch':epoch,'config_hash':key,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                      'scheduler':schedule.state_dict(),'scaler':scaler.state_dict(),'history':history,
                      'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                      'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()},checkpoint)
        print(json.dumps({'run':name,**row}),flush=True)
    model.eval();truth=[];pred=[];genders=[]
    with torch.inference_mode():
        for images,y,g in testloader:
            with torch.autocast('cuda',enabled=config['amp']):logits=model(images.to(device,non_blocking=True,memory_format=torch.channels_last))
            truth.extend(y.tolist());pred.extend(logits.argmax(1).cpu().tolist());genders.extend(g)
    recall=group_recall(truth,pred,genders)
    recall.to_csv(root/'recall.csv',index=False)
    confusion=np.zeros((2,len(CLASSES),len(CLASSES)),dtype=np.int64)
    for y,p,g in zip(truth,pred,genders):confusion[GENDERS.index(g),y,p]+=1
    np.savez_compressed(root/'confusion.npz',counts=confusion,classes=np.array(CLASSES),genders=np.array(GENDERS))
    result={'run':name,'config_hash':key,'epochs':config['epochs'],'experiment':experiment,'step':step,
            'fraction':step/10,'repetition':repetition,'smoke_test_only':smoke,'training_rows':len(selected),
            'test_rows':len(test),'seconds':time.perf_counter()-started,'test_accuracy':float(np.mean(np.array(truth)==pred)),
            'manifest_sha256':manifest_hash,'pretrained_sha256':weight_hash}
    result['artifact_sha256']={p:sha256_file(root/p) for p in ['recall.csv','confusion.npz','history.csv','train_counts.csv']}
    (root/'complete.json').write_text(json.dumps(result,indent=2))
    return result
