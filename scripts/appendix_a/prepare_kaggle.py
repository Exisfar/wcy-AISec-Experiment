#!/usr/bin/env python3
"""审核并转换用户指定的 Kaggle v2；以 FairFace 给每张原图重新推断人口属性。

保留发布者划分，独立留存 validation；按像素文件哈希去重和检查冲突。
正式输出为“加工版改动实验”，不能冒充论文原始 AffectNet 划分。
"""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from appendix_a.data import CLASSES,GENDERS,sha256_file,validate_manifest

NAMES=('angry','contempt','disgust','fear','happy','neutral','sad','surprise')
RACES=('White','Black','Latino_Hispanic','East Asian','Southeast Asian','Indian','Middle Eastern')
AGES=('0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','70+')

class Images(Dataset):
    def __init__(self,frame,root):
        self.frame=frame;self.root=root
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                           transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        with Image.open(self.root/self.frame.iloc[i].path) as im:x=self.transform(im.convert('RGB'))
        return x,i


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=ROOT/'data/kaggle-affectnet-yolo/extracted')
    p.add_argument('--fairface-weights',type=Path,default=ROOT/'data/models/fairface_resnet34_7race_20190809.pt')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();folder=a.dataset_root.resolve();yolo=folder/'YOLO_format'
    torch.set_num_threads(8)
    rows=[]
    for split in ['train','valid','test']:
        paths=sorted((yolo/split/'images').glob('*'))
        for path in paths:
            if path.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
            labelpath=yolo/split/'labels'/(path.stem+'.txt')
            lines=[line.split() for line in labelpath.read_text().splitlines() if line.strip()]
            if len(lines)!=1 or len(lines[0])!=5:raise ValueError(f'Expected one labeled face: {labelpath}')
            label=NAMES[int(float(lines[0][0]))]
            if label=='contempt':continue
            with Image.open(path) as im:w,h=im.size
            rows.append({'sample_id':f'kaggle-v2:{split}:{path.name}','path':str(path.relative_to(folder)),
                         'label':label,'source_split':split,'image_sha256':sha256_file(path),'width':w,'height':h})
    df=pd.DataFrame(rows)
    if df.empty:raise FileNotFoundError('No YOLO images found; extract dataset-v2.zip first')
    # 保留 test 优先，防止评估图像泄漏到训练；标签冲突的重复图像全部剔除。
    conflicts=set(df.groupby('image_sha256').label.nunique().loc[lambda s:s>1].index)
    conflict_removed=int(df.image_sha256.isin(conflicts).sum())
    clean=df[~df.image_sha256.isin(conflicts)].copy()
    clean['priority']=clean.source_split.map({'test':0,'valid':1,'train':2})
    clean=clean.sort_values(['priority','path']).drop_duplicates('image_sha256',keep='first').drop(columns='priority').reset_index(drop=True)
    duplicate_removed=len(df)-conflict_removed-len(clean)
    weights=a.fairface_weights.resolve()
    model=models.resnet34(weights=None);model.fc=nn.Linear(model.fc.in_features,18)
    state=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);model.cuda().eval()
    loader=DataLoader(Images(clean,folder),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True)
    gender=[];gender_conf=[];races=[];ages=[]
    with torch.inference_mode():
        for x,idx in loader:
            logits=model(x.cuda(non_blocking=True)).float().cpu()
            gender_logits=logits[:,7:9];gender.extend(gender_logits.argmax(1).tolist())
            gender_conf.extend(gender_logits.softmax(1).max(1).values.tolist())
            races.extend(logits[:,:7].argmax(1).tolist());ages.extend(logits[:,9:18].argmax(1).tolist())
            print(f'FairFace {len(gender)}/{len(clean)}',flush=True)
    clean['gender']=[['Male','Female'][i] for i in gender];clean['gender_confidence']=gender_conf
    clean['race']=[RACES[i] for i in races];clean['age']=[AGES[i] for i in ages]
    clean['split']=clean.source_split.map({'train':'train','test':'test','valid':'validation'})
    clean.to_csv(folder/'all_demographics.csv',index=False)
    manifest=clean[clean.split.isin(['train','test'])].copy().reset_index(drop=True)
    validate_manifest(manifest,folder)
    manifest_path=folder/'manifest.csv';manifest.to_csv(manifest_path,index=False)
    train=manifest[manifest.split=='train'];counts=pd.crosstab(train.gender,train.label).reindex(index=GENDERS,columns=CLASSES,fill_value=0)
    totals=counts.min(axis=0).astype(int).to_dict()
    if min(totals.values())<2:raise ValueError(f'Not enough training support for 7-class sweeps: {totals}')
    base=json.loads((ROOT/'configs/appendix_a.json').read_text())
    base.update(image_root=str(folder),manifest=str(manifest_path),class_totals=totals,
                output_dir='/workspace/results/appendixA_kaggle',checkpoint_dir='/workspace/checkpoints/appendixA_kaggle',
                protocol_variant='Kaggle 96x96 re-split subset; upsampled to 224; newly inferred FairFace demographics; target per-class size=min(Female,Male) in source training data',
                original_paper_data_match=False)
    base['protocol_assumptions'].extend([
        'Explicitly authorized modified-data experiment, NOT a strict reproduction of paper AffectNet results.',
        'Kaggle published train/test partitions retained after exact-file-hash deduplication; validation is unused and never merged into training.',
        'Kaggle images were already cropped/downscaled to 96; use provided crop, no MMOD redetection, then resize to 224 (cannot restore lost details).',
        'FairFace official public 20190809 7-race ResNet34 weights are used; these may differ from the original paper demographic checkpoint.'
    ])
    out=ROOT/'results/appendixA_kaggle';out.mkdir(parents=True,exist_ok=True)
    provenance={'dataset':'fatihkgg/affectnet-yolo-format','version':2,
                'url':'https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format',
                'archive_sha256':sha256_file(ROOT/'data/kaggle-affectnet-yolo/dataset-v2.zip'),
                'fairface_model_sha256':sha256_file(weights),'fairface_model_file':weights.name,
                'demographics':'FairFace predictions, not self-reported identity',
                'crop_method':'Kaggle provided 96px crops; no MMOD recropping',
                'original_paper_data_match':False,'smoke_test_only':False,
                'raw_7class_rows':len(df),'label_conflict_rows_removed':conflict_removed,
                'duplicate_rows_removed':duplicate_removed,'rows_after_dedup':len(clean),
                'splits_after_dedup':clean.split.value_counts().to_dict(),
                'subject_identity_available':False,'exact_hash_dedup_is_not_subject_disjointness':True,
                'manifest_sha256':sha256_file(manifest_path)}
    manifest_path.with_suffix('.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    (out/'data_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    counts.to_csv(out/'source_training_counts.csv')
    pd.crosstab(manifest[manifest.split=='test'].gender,manifest[manifest.split=='test'].label).to_csv(out/'test_counts.csv')
    (ROOT/'configs/appendix_a_kaggle.json').write_text(json.dumps(base,ensure_ascii=False,indent=2))
    print(json.dumps(provenance,ensure_ascii=False,indent=2));print('Class totals:',totals,flush=True)

if __name__=='__main__':main()
