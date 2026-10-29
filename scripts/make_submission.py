from pathlib import Path
import tarfile,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
dist=ROOT/'dist'
# 打包的是现有完整项目与镜像，排除冗余的 source.tar.gz 二次副本和开发缓存。
target=dist/'课程论文复现_交付包.tar.gz'
with tarfile.open(target,'w:gz',compresslevel=1) as tar:
    for p in sorted(ROOT.iterdir()):
        if p.name in {'.venv','.git','.pytest_cache','dist','teacher-output'}:continue
        def keep(info):
            if any(x in Path(info.name).parts for x in ['.git','__pycache__','.ipynb_checkpoints']):return None
            return info
        tar.add(p,arcname='fer-bias-reproduction/'+p.name,filter=keep)
    for name in ['fer-bias-repro-image.tar.gz','SHA256SUMS','clean-container.log','build.log','image-inspect.json','DELIVERY.txt']:
        p=dist/name
        if p.exists():tar.add(p,arcname='fer-bias-reproduction/dist/'+name)
sha=hashlib.sha256()
with target.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
(dist/'课程论文复现_交付包.sha256').write_text(f'{sha.hexdigest()}  课程论文复现_交付包.tar.gz\n')
print('Submission:',target,'bytes:',target.stat().st_size)
