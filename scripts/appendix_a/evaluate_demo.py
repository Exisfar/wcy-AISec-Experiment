"""快速重新执行真实人脸推理；缺少测试图片时可从用户指定的公开 Kaggle 源下载。"""
from pathlib import Path
import sys,json,argparse,zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from appendix_a.data import sha256_file,group_recall,CLASSES,GENDERS
from appendix_a.trainer import FaceDataset,seed_all


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--image-root',type=Path,default=ROOT/'data/kaggle-affectnet-yolo/extracted')
    p.add_argument('--download-images',action='store_true');p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args();demo=ROOT/'results/appendixA_demo';meta=json.loads((demo/'model_provenance.json').read_text())
    weights=ROOT/'data/models/resnet50_balanced_kaggle.pth'
    for path,key in [(weights,'model_sha256'),(demo/'test_manifest.csv','test_manifest_sha256'),(demo/'expected_confusion.npz','expected_confusion_sha256')]:
        assert sha256_file(path)==meta[key],str(path)
    frame=pd.read_csv(demo/'test_manifest.csv');image_root=a.image_root.resolve()
    missing=[p for p in frame.path if not (image_root/p).is_file()]
    if missing and a.download_images:
        import subprocess
        archive=ROOT/'data/kaggle-affectnet-yolo/dataset-v2.zip'
        if not archive.exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('download_parallel.py')),meta['dataset_download_url'],str(archive)],check=True)
        assert sha256_file(archive)==meta['source_zip_sha256'],'Dataset version/hash mismatch'
        with zipfile.ZipFile(archive) as z:
            for rel in missing:
                assert (image_root/rel).resolve().is_relative_to(image_root)
                z.extract(rel,image_root)
    elif missing:
        p.error(f'{len(missing)} real test images missing. Add --download-images or provide --image-root; no synthetic substitutes are used.')
    for r in frame.itertuples():assert sha256_file(image_root/r.path)==r.image_sha256,r.path
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device)
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    seed_all(20241028);torch.set_num_threads(8)
    model=models.resnet50(weights=None);model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.to(device,memory_format=torch.channels_last).eval()
    loader=DataLoader(FaceDataset(frame,image_root,224),batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=device=='cuda')
    truth=[];prediction=[];genders=[]
    with torch.inference_mode():
        for images,y,g in loader:
            images=images.to(device,memory_format=torch.channels_last,non_blocking=True)
            with torch.autocast('cuda',enabled=device=='cuda'):logits=model(images)
            truth.extend(y.tolist());prediction.extend(logits.argmax(1).cpu().tolist());genders.extend(g)
    counts=np.zeros((2,7,7),dtype=np.int64)
    for y,pred,g in zip(truth,prediction,genders):counts[GENDERS.index(g),y,pred]+=1
    expected=np.load(demo/'expected_confusion.npz',allow_pickle=False)['counts']
    out=ROOT/'results/appendixA_demo';out.mkdir(parents=True,exist_ok=True)
    table=group_recall(truth,prediction,genders);table.to_csv(out/'recall.csv',index=False)
    result={'device':device,'real_test_images':len(truth),'accuracy':float(np.mean(np.array(truth)==prediction)),
            'saved_training_run_accuracy':meta['expected_accuracy'],'confusion_matches_saved':bool(np.array_equal(counts,expected)),
            'max_confusion_cell_change':int(np.abs(counts-expected).max()),'original_paper_data_match':False,
            'note':'GPU AMP matches the training evaluation mode. CPU FP32 may change near-tied predictions; differences are reported, not hidden.'}
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
