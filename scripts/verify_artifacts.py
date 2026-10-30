"""无需重算的交付完整性验证；失败时非零退出。"""
from pathlib import Path
import hashlib,json
import nbformat
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'results/provenance.json').read_text())
for name,digest in p['csv_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
notebooks=list((ROOT/'notebooks').rglob('*.ipynb'))
expected={'01_二十个真实数据集完整指标复现.ipynb','03_不足分析与改进探索.ipynb','02_真实人脸表情识别模型训练验证.ipynb'}
assert {f.name for f in notebooks}==expected,'主目录应为01、02、03三份实验笔记'
for f in notebooks:
    nb=nbformat.read(f,as_version=4)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':
            assert c.execution_count is not None,(f.name,i,'not executed')
            assert all(o.output_type!='error' for o in c.outputs),(f.name,i,'error')
print(f'PASS: 20 data hashes; {len(notebooks)} executed notebooks without errors')
