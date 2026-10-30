"""离线执行所有 Notebook，任何出错立即终止；结果保存在 Notebook 输出中。"""
from pathlib import Path
import sys,time,json
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
t0=time.time(); logs=[]
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    print(f'RUN {p.name}',flush=True)
    start=time.time(); nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    logs.append({'notebook':p.name,'seconds':round(time.time()-start,2),'status':'passed'})
    print(f'PASS {p.name} ({logs[-1]["seconds"]}s)',flush=True)
(ROOT/'results/execution.json').write_text(json.dumps({'notebooks':logs,'total_seconds':round(time.time()-t0,2)},ensure_ascii=False,indent=2))
from result_files import cleanup_results
cleanup_results()
print('ALL NOTEBOOKS PASSED',flush=True)
