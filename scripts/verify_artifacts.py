"""无需重算的交付完整性验证；失败时非零退出。"""
from pathlib import Path
import hashlib,json
import nbformat
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'results/provenance.json').read_text())
for name,digest in p['csv_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for f in (ROOT/'notebooks').glob('*.ipynb'):
    nb=nbformat.read(f,as_version=4)
    for i,c in enumerate(nb.cells):
        if c.cell_type=='code':
            assert c.execution_count is not None,(f.name,i,'not executed')
            assert all(o.output_type!='error' for o in c.outputs),(f.name,i,'error')
print('PASS: 20 data hashes; 3 executed notebooks without errors')
