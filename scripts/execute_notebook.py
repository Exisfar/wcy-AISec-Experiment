"""执行指定Notebook并保存输出，其他笔记保持不变。"""
from pathlib import Path
import argparse,time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('notebook',type=Path);a=p.parse_args()
path=a.notebook if a.notebook.is_absolute() else ROOT/a.notebook
nb=nbformat.read(path,as_version=4);start=time.perf_counter()
NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.write(nb,path)
import sys
sys.path.insert(0,str(ROOT/'src'))
from result_files import cleanup_results
cleanup_results()
print(f'PASS {path.name} ({time.perf_counter()-start:.2f}s)')
