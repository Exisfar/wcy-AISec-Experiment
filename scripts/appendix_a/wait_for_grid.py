"""容器内只读完成监视：完整网格结束后自动核验并汇总，不重启被停止的训练。"""
from pathlib import Path
import sys,json,time,subprocess,datetime
ROOT=Path(__file__).resolve().parents[2]
config=ROOT/'configs/appendix_a_kaggle.json';c=json.loads(config.read_text());out=Path(c['output_dir'])
expected=len(c['experiments'])*len(c['fractions'])*c['repetitions'];last=-1
while True:
    count=len(list(out.glob('exp*_q*_rep*/complete.json')))
    state={'completed_runs':count,'expected_runs':expected,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'watch.json').write_text(json.dumps(state,indent=2))
    if count!=last:print(json.dumps(state),flush=True);last=count
    if count==expected:break
    time.sleep(30)
for name,extra in [('verify_results.py',['--require-complete']),('summarize.py',[]),('analyze.py',[])]:
    subprocess.run([sys.executable,str(Path(__file__).with_name(name)),'--config',str(config),*extra],cwd=ROOT,check=True)
(out/'GRID_FINISHED.json').write_text(json.dumps({'verified_complete':True,'real_runs':expected,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('FULL REAL-IMAGE GRID FINISHED AND VERIFIED',flush=True)
