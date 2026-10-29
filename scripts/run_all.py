"""离线执行所有 Notebook，任何出错立即终止；执行后导出便于快速检查的 HTML。"""
from pathlib import Path
import sys,time,json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
t0=time.time(); logs=[]
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    print(f'RUN {p.name}',flush=True)
    start=time.time(); nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    exporter=HTMLExporter(); exporter.exclude_input=False
    body,_=exporter.from_notebook_node(nb)
    (ROOT/'results'/f'{p.stem}.html').write_text(body,encoding='utf-8')
    logs.append({'notebook':p.name,'seconds':round(time.time()-start,2),'status':'passed'})
    print(f'PASS {p.name} ({logs[-1]["seconds"]}s)',flush=True)
(ROOT/'results/execution.json').write_text(json.dumps({'notebooks':logs,'total_seconds':round(time.time()-t0,2)},ensure_ascii=False,indent=2))
print('ALL NOTEBOOKS PASSED',flush=True)
