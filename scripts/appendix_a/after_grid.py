"""容器内等待正式网格完成，再串行运行三次真实模型重加权对照。"""
from pathlib import Path
import time,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
while not (ROOT/'results/appendixA_kaggle/GRID_FINISHED.json').exists():time.sleep(30)
subprocess.run([sys.executable,str(Path(__file__).with_name('mitigate.py'))],cwd=ROOT,check=True)
