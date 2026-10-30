"""只读训练进度，不影响 GPU 作业；预计时间按已完成配置外推。"""
from pathlib import Path
import argparse,json,fcntl
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json');a=p.parse_args()
c=json.loads(a.config.read_text());root=Path(c['output_dir'])
expected=len(c['experiments'])*len(c['fractions'])*c['repetitions']
complete=[]
for path in root.glob('exp*_q*_rep*/complete.json'):
    run=json.loads(path.read_text())
    if not run.get('smoke_test_only'):complete.append(run)
active=[]
for path in root.glob('exp*_q*_rep*/history.csv'):
    if not (path.parent/'complete.json').exists():
        h=pd.read_csv(path)
        if len(h):active.append({'run':path.parent.name,'epoch':int(h.epoch.iloc[-1]),'epoch_seconds':float(h.seconds.tail(3).mean())})
durations=[]
for x in complete:
    # 完成记录 seconds 可能只覆盖恢复后的进程；历史保留了全部 20 epochs。
    history=pd.read_csv(root/x['run']/'history.csv')
    durations.append(float(history.seconds.sum())*1.05)
mean_seconds=sum(durations)/len(durations) if durations else None
eta=mean_seconds*(expected-len(complete)) if mean_seconds else None
out={'completed_runs':len(complete),'expected_runs':expected,'active':active,
     'mean_run_seconds':mean_seconds,'estimated_remaining_hours':eta/3600 if eta is not None else None,
     'formal_grid_complete':len(complete)==expected,'scope':'modified Kaggle real-image experiment; not original AffectNet replication'}
status_path=root/'status.json'
out['training_status']=json.loads(status_path.read_text()).get('status') if status_path.exists() else 'unknown'
out['training_lock_held']=False
if (root/'.training.lock').exists():
    with (root/'.training.lock').open('r') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        except BlockingIOError:out['training_lock_held']=True
print(json.dumps(out,ensure_ascii=False,indent=2));(root/'progress.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
