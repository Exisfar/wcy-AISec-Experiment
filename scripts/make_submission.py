"""Package one complete offline image and reviewable source; local slides remain on host."""
from pathlib import Path
import tarfile
ROOT=Path(__file__).resolve().parents[1]
dist=ROOT/'dist'; target=dist/'课程论文复现_交付包.tar.gz'
excluded={'make_slides.py','export_speech.py','speech_content.json','check_slides.py','presentation_local.sh'}
def keep(info):
    parts=Path(info.name).parts
    if any(p in {'.git','.venv','__pycache__','.ipynb_checkpoints','.pytest_cache'} for p in parts):return None
    if Path(info.name).name in excluded or info.name.endswith('.training.lock'):return None
    return info
with tarfile.open(target,'w:gz',compresslevel=1) as tar:
    for p in sorted(ROOT.iterdir()):
        if p.name in {'.venv','.git','.pytest_cache','dist','teacher-output','data','checkpoints','runtime','presentation','image.png','paper.txt','paper_raw.txt'}:continue
        tar.add(p,arcname='fer-bias-reproduction/'+p.name,filter=keep)
    for name in ['fer-bias-repro-image.tar.gz','clean-container.log','build.log','image-inspect.json','DELIVERY.txt','gpu-clean-inference.log','final-execution-summary.json']:
        p=dist/name
        if p.exists():tar.add(p,arcname='fer-bias-reproduction/dist/'+name)
with tarfile.open(target,'r:gz') as check:
    names=check.getnames()
    assert not any('presentation' in Path(n).parts or n.endswith(('.pptx','.docx')) for n in names)
    assert any(n.endswith('/results/SOURCES.md') for n in names)
    assert any(n.endswith('/启动Notebook.sh') for n in names)
    assert not any(Path(n).name in {'SHA256SUMS','课程论文复现_交付包.sha256'} for n in names)
    assert not any(n.endswith('.html') or Path(n).name in {'复现报告.md','真实人脸训练结果.md','附录A真实训练进度.md'} for n in names)
    assert not any('reports' in Path(n).parts for n in names)
    assert any(n.endswith('/results/appendixA_demo/test_manifest.csv') for n in names)
    assert not any('data' in Path(n).parts for n in names), '数据统一由完整镜像提供，外层不重复打包'
    assert not any('vendor' in Path(n).parts or Path(n).name in {'LICENSE','LICENSE.txt'} for n in names)
    assert any(n.endswith('/src/dataset_bias_metrics/representational.py') for n in names)
    assert sum('/notebooks/' in n and n.endswith('.ipynb') for n in names)==3
    assert any(n.endswith('/notebooks/02_真实人脸表情识别模型训练验证.ipynb') for n in names)
    assert any(n.endswith('/dist/fer-bias-repro-image.tar.gz') for n in names)
print('PASS: single offline image and experiment source packaged:',target,target.stat().st_size)
