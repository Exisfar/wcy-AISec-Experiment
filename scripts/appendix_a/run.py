#!/usr/bin/env python3
"""默认生成 132 项正式实验计划；--run 才执行训练。"""
from pathlib import Path
import sys,json,argparse,fcntl,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from appendix_a.data import validate_manifest,required_counts,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a.json')
    p.add_argument('--run',action='store_true',help='Require real data and execute all unfinished training runs')
    p.add_argument('--max-runs',type=int,help='Explicitly limit new runs, e.g. for timing; report remains partial')
    p.add_argument('--output-dir',type=Path,help='Use a new output directory to truly retrain instead of skipping completed models')
    p.add_argument('--checkpoint-dir',type=Path,help='Checkpoint directory paired with a new output directory')
    args=p.parse_args();config=json.loads(args.config.read_text())
    if args.output_dir:
        config['output_dir']=str(args.output_dir)
        config['checkpoint_dir']=str(args.checkpoint_dir or args.output_dir/'checkpoints')
    elif args.checkpoint_dir:config['checkpoint_dir']=str(args.checkpoint_dir)
    steps=[int(round(float(v)*10)) for v in config['fractions']]
    if any(abs(v-s/10)>1e-10 for v,s in zip(config['fractions'],steps)):p.error('Fractions must use the paper 0.1 grid')
    jobs=[{'experiment':e,'step':s,'fraction':s/10,'repetition':r}
          for e in config['experiments'] for s in steps for r in range(config['repetitions'])]
    for job in jobs:
        counts=required_counts(job['experiment'],job['step'],config['class_totals'])
        job['n_train']=int(counts.to_numpy().sum())
    root=Path(config['output_dir']);root.mkdir(parents=True,exist_ok=True)
    training_lock=None
    if args.run:
        training_lock=(root/'.training.lock').open('a+')
        try:fcntl.flock(training_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another training process already owns this output directory')
        training_lock.seek(0);training_lock.truncate();training_lock.write(str(os.getpid()));training_lock.flush()
    pd.DataFrame(jobs).to_csv(root/'plan.csv',index=False)
    manifest=Path(config['manifest'])
    status={'status':'awaiting_affectnet_images' if not manifest.is_file() else 'manifest_found_not_validated',
            'planned_runs':len(jobs),'epochs_per_run':config['epochs'],'gpu_training_completed':False,
            'manifest_path':str(manifest),'required_data':'Real face images, image-linked FairFace gender predictions and declared train/test partitions',
            'original_paper_data_match':bool(config.get('original_paper_data_match',False)),
            'protocol_variant':config.get('protocol_variant'),
            'public_anonymous_csv_is_not_an_image_manifest':True}
    if args.run or not (root/'status.json').exists():
        (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not args.run:return
    if not manifest.is_file():
        p.error('Missing real AffectNet manifest/images. Obtain official data, preprocess and run again; no paper model results can be generated from anonymous CSVs.')
    # 导入训练依赖前先审计输入，阻止匿名 CSV 错配。
    df=validate_manifest(pd.read_csv(manifest,dtype={'sample_id':str}),config['image_root'])
    if 'image_sha256' not in df: p.error('Manifest requires image_sha256 for cross-split image leakage audit')
    provenance_path=manifest.with_suffix('.provenance.json')
    if not provenance_path.is_file():p.error('Require manifest.provenance.json with data/cropping/FairFace provenance')
    provenance=json.loads(provenance_path.read_text())
    if provenance.get('smoke_test_only'):p.error('Smoke fixtures are not allowed as paper training data')
    train=df[df.split=='train'];test=df[df.split=='test'];digest=sha256_file(manifest)
    (root/'data_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    pd.crosstab(train.gender,train.label).to_csv(root/'source_training_counts.csv')
    pd.crosstab(test.gender,test.label).to_csv(root/'test_counts.csv')
    available=pd.crosstab(train.gender,train.label)
    # 所有配置先检查支持，再开始耗时训练，避免半程才发现不能构造某端点。
    for job in jobs:
        target=required_counts(job['experiment'],job['step'],config['class_totals'])
        actual=available.reindex(index=target.index,columns=target.columns,fill_value=0)
        if (actual<target).any().any():p.error(f'Insufficient source samples for exp{job["experiment"]}, q={job["fraction"]}: cannot silently rescale paper counts')
        if int(target.to_numpy().sum())%config['batch_size']==1:
            p.error('A planned subset ends with a singleton BatchNorm batch; choose another batch size before starting any runs')
    import torch
    if not torch.cuda.is_available():p.error('CUDA device unavailable; use GPU container')
    from appendix_a.trainer import run_training
    status.update(status='running',manifest_sha256=digest,gpu=torch.cuda.get_device_name(0),data_provenance=provenance)
    (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    count=0
    try:
        for job in jobs:
            complete=root/f'exp{job["experiment"]}_q{job["step"]:02d}_rep{job["repetition"]}'/'complete.json'
            existing=complete.exists()
            run_training(config,train,test,job['experiment'],job['step'],job['repetition'],digest)
            if not existing:count+=1
            if args.max_runs and count>=args.max_runs:break
    except BaseException:
        status['status']='interrupted_or_failed';(root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2));raise
    finished=sum((root/f'exp{x["experiment"]}_q{x["step"]:02d}_rep{x["repetition"]}'/'complete.json').exists() for x in jobs)
    status.update(status='complete' if finished==len(jobs) else 'partial',completed_runs=finished,
                  gpu_training_completed=finished==len(jobs))
    (root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
