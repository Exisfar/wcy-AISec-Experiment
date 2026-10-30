"""下载固定版本并核对真实训练图片；复用交付中保存的逐图人口属性清单。"""
from pathlib import Path
import argparse,csv,json,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import sha256_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    meta=json.loads((ROOT/'results/appendixA_demo/model_provenance.json').read_text())
    archive=ROOT/'data/kaggle-affectnet-yolo/dataset-v2.zip'
    folder=archive.parent/'extracted'
    manifest=ROOT/'results/appendixA_kaggle/input_manifest.csv'
    if not archive.exists():
        if not a.download:p.error('缺少 dataset-v2.zip；请提供文件或使用 --download')
        subprocess.run([sys.executable,str(Path(__file__).with_name('download_parallel.py')),meta['dataset_download_url'],str(archive)],check=True)
    assert sha256_file(archive)==meta['source_zip_sha256'],'ZIP version/hash mismatch'
    folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert (folder/name).resolve().is_relative_to(folder.resolve()),name
        for name in z.namelist():
            # 已存在的文件留给下方哈希核验，避免触碰正在读取的数据。
            if not (folder/name).exists():z.extract(name,folder)
    rows=list(csv.DictReader(manifest.open()))
    for row in rows:
        path=(folder/row['path']).resolve()
        assert path.is_relative_to(folder.resolve())
        assert sha256_file(path)==row['image_sha256'],row['path']
    (folder/'manifest.csv').write_bytes(manifest.read_bytes())
    (folder/'manifest.provenance.json').write_bytes(manifest.with_name('input_manifest.provenance.json').read_bytes())
    print(json.dumps({'verified_images':len(rows),'manifest_sha256':sha256_file(manifest),'source_zip_sha256':meta['source_zip_sha256'],'original_paper_data_match':False},indent=2))

if __name__=='__main__':main()
