"""Manifest 合同、严格的偏差构造与数据泄漏检查。该模块不依赖 PyTorch。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

CLASSES=('angry','disgust','fear','happy','neutral','sad','surprise')
GENDERS=('Female','Male')
PAPER_TOTALS=dict(zip(CLASSES,[6962,1733,3117,54241,33708,11175,6439]))


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_manifest(frame, image_root=None, check_files=True):
    """split=train/test；test 必须包含所有 2×7 组，禁止静默丢弃样本。

    cropped face path 为相对 image_root 的路径。可选 image_sha256/subject_id
    在可用时进行跨分割泄漏检查；缺失身份信息时不伪称已排除身份泄漏。
    """
    required={'sample_id','path','label','gender','split'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Manifest missing columns: {sorted(required-set(frame.columns))}')
    df=frame.copy()
    if df[list(required)].isna().any().any():raise ValueError('Required manifest fields contain missing values')
    if df.sample_id.astype(str).duplicated().any():raise ValueError('Duplicate sample_id')
    if df.path.astype(str).duplicated().any():raise ValueError('Duplicate image path (including cross-split duplicates)')
    if not set(df.label).issubset(CLASSES):raise ValueError('Use only the 7 paper emotion classes; exclude contempt/invalid labels')
    if set(df.gender)-set(GENDERS):raise ValueError('gender must be Female/Male FairFace proxy predictions')
    if set(df.split)!={'train','test'}:raise ValueError('Provide distinct train and test partitions')
    if 'image_sha256' in df:
        train=set(df.loc[df.split=='train','image_sha256'].dropna())
        test=set(df.loc[df.split=='test','image_sha256'].dropna())
        if train & test:raise ValueError('Image SHA256 leakage between train/test')
    if 'subject_id' in df:
        train=set(df.loc[df.split=='train','subject_id'].dropna())
        test=set(df.loc[df.split=='test','subject_id'].dropna())
        if train & test:raise ValueError('Subject leakage between train/test')
    counts=pd.crosstab(df.loc[df.split=='test','gender'],df.loc[df.split=='test','label']).reindex(index=GENDERS,columns=CLASSES,fill_value=0)
    if (counts==0).any().any():raise ValueError('Held-out test set must contain every gender x class cell')
    if check_files:
        root=Path(image_root).resolve()
        for row in df.itertuples():
            rel=Path(row.path)
            if rel.is_absolute():raise ValueError('Manifest path must be relative to image_root')
            path=(root/rel).resolve()
            if not path.is_relative_to(root):raise ValueError('Image path escapes image_root')
            if not path.is_file():raise FileNotFoundError(path)
    return df


def required_counts(experiment, step, totals=None):
    """step 为 0..10 整数，整数除法精确实现 floor(n*q)，避免浮点取整偏差。"""
    if experiment not in [1,2,3,4] or step not in range(11):raise ValueError('Invalid experiment or fraction step')
    totals=totals or PAPER_TOTALS
    out=pd.DataFrame(0,index=GENDERS,columns=CLASSES,dtype=int)
    for label in CLASSES:
        n=int(totals[label])
        if experiment==1:
            a,b=n*step//10,n*(10-step)//10
        else:
            a=b=n//2
            if label=='happy':a,b=n*step//10,n*(10-step)//10
            if experiment>=3 and label=='angry':
                k=10-step if experiment==3 else step
                a,b=n*k//10,n*(10-k)//10
        out.loc['Female',label]=a;out.loc['Male',label]=b
    return out


def sample_biased(train, experiment, step, seed, totals=None):
    """不重复采样，不借用测试样本；不足时抛错，不静默缩小论文样本量。"""
    if set(train.split)!={'train'}:raise ValueError('Only training rows may be sampled')
    targets=required_counts(experiment,step,totals)
    rng=np.random.default_rng(seed)
    frames=[]
    for gender in GENDERS:
        for label in CLASSES:
            n=int(targets.loc[gender,label])
            pool=train[(train.gender==gender)&(train.label==label)].sort_values('sample_id')
            if n>len(pool):raise ValueError(f'Insufficient train support for {gender}/{label}: need {n}, have {len(pool)}')
            if n:frames.append(pool.iloc[rng.choice(len(pool),size=n,replace=False)])
    result=pd.concat(frames).sample(frac=1,random_state=seed).reset_index(drop=True)
    assert not result.sample_id.duplicated().any()
    actual=pd.crosstab(result.gender,result.label).reindex(index=GENDERS,columns=CLASSES,fill_value=0)
    assert actual.equals(targets)
    return result,targets


def group_recall(y_true,y_pred,genders):
    """返回每个性别、每个类别的 TP/support；分母为零时 NaN，禁止伪造为 0。"""
    y=np.asarray(y_true);p=np.asarray(y_pred);g=np.asarray(genders)
    rows=[]
    for gender in GENDERS:
        values=[]
        for i,label in enumerate(CLASSES):
            mask=(g==gender)&(y==i);n=int(mask.sum());tp=int(((p==i)&mask).sum())
            recall=tp/n if n else np.nan
            rows.append({'gender':gender,'label':label,'support':n,'tp':tp,'recall':recall})
            values.append(recall)
        rows.append({'gender':gender,'label':'mean','support':int((g==gender).sum()),'tp':None,
                     'recall':float(np.mean(values)) if np.isfinite(values).all() else np.nan})
    return pd.DataFrame(rows)
