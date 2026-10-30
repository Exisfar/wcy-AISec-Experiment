"""附录 A 的科学协议测试：构造量、独立性、支持条件、召回分母和泄漏。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from appendix_a.data import CLASSES,GENDERS,required_counts,sample_biased,group_recall,validate_manifest
from metrics import association


def fixture_frame(n=20):
    return pd.DataFrame([{'sample_id':f'{split}-{g}-{c}-{i}','path':f'{split}/{g}/{c}/{i}.jpg',
                          'gender':g,'label':c,'split':split}
                         for split in ['train','test'] for g in GENDERS for c in CLASSES for i in range(n)])


def test_paper_table_endpoints_and_midpoint():
    assert required_counts(1,0).loc['Female'].sum()==0
    assert required_counts(1,10).loc['Male'].sum()==0
    assert required_counts(1,5).loc['Female'].sum()==58685
    assert required_counts(1,5).loc['Female'].equals(required_counts(1,5).loc['Male'])
    assert required_counts(3,0).loc['Female','angry']==6962
    assert required_counts(3,0).loc['Female','happy']==0
    assert required_counts(4,10).loc['Female','angry']==6962
    assert required_counts(4,10).loc['Male','happy']==0


def test_sampling_without_replacement_has_exact_counts():
    f=fixture_frame();train=f[f.split=='train'];totals={c:20 for c in CLASSES}
    for e in range(1,5):
        for step in range(11):
            selected,counts=sample_biased(train,e,step,42,totals)
            assert len(selected)==counts.to_numpy().sum()
            assert not selected.sample_id.duplicated().any()
            assert set(selected.split)=={'train'}


def test_sampling_seed_reproducible():
    f=fixture_frame();train=f[f.split=='train'];totals={c:20 for c in CLASSES}
    a,_=sample_biased(train,2,3,10,totals);b,_=sample_biased(train,2,3,10,totals)
    c,_=sample_biased(train,2,3,11,totals)
    assert a.sample_id.tolist()==b.sample_id.tolist()
    assert a.sample_id.tolist()!=c.sample_id.tolist()


def test_insufficient_support_is_not_silently_resampled():
    f=fixture_frame(n=2)
    with pytest.raises(ValueError,match='Insufficient train support'):
        sample_biased(f[f.split=='train'],1,0,42)


def test_pure_representational_change_does_not_create_dependence():
    for step in range(1,10):
        t=required_counts(1,step,{c:100 for c in CLASSES})
        assert np.isclose(association(t)[0]['V'],0)
    assert association(required_counts(2,0))[0]['V']>.5


def test_opposite_and_aligned_constructs_are_distinct():
    a=required_counts(3,0);b=required_counts(4,0)
    assert a.loc['Female','happy']==b.loc['Female','happy']==0
    assert a.loc['Female','angry']>b.loc['Female','angry']
    assert a.loc['Female'].sum()!=b.loc['Female'].sum()


def test_macro_recall_is_not_pooled_accuracy():
    # 女性 class0 9/10，class1 0/1；macro 为 .45，不能用 9/11 替代。
    y=[0]*10+[1];p=[0]*9+[1]+[0];g=['Female']*11
    df=group_recall(y,p,g)
    assert np.isclose(df.query("gender=='Female' and label=='angry'").recall.iloc[0],.9)
    assert df.query("gender=='Female' and label=='disgust'").recall.iloc[0]==0
    # 缺少其余类别，正式7类macro不应冒充为两类macro或0。
    assert np.isnan(df.query("gender=='Female' and label=='mean'").recall.iloc[0])
    assert df.query("gender=='Male'").recall.isna().all()


def test_manifest_requires_linked_images_not_anonymous_csv():
    with pytest.raises(ValueError,match='missing columns'):
        validate_manifest(pd.DataFrame({'gender':['Female'],'label':['happy']}),check_files=False)
    assert len(validate_manifest(fixture_frame(),check_files=False))>0


def test_test_partition_missing_group_is_rejected():
    f=fixture_frame();f=f[~((f.split=='test')&(f.gender=='Female')&(f.label=='happy'))]
    with pytest.raises(ValueError,match='test set'):validate_manifest(f,check_files=False)


def test_identity_and_image_leakage_is_rejected():
    f=fixture_frame(n=1);f['subject_id']='same-subject'
    with pytest.raises(ValueError,match='Subject leakage'):validate_manifest(f,check_files=False)
    f=f.drop(columns='subject_id');f['image_sha256']='same-image'
    with pytest.raises(ValueError,match='SHA256 leakage'):validate_manifest(f,check_files=False)


def test_path_escape_is_rejected(tmp_path):
    f=fixture_frame(n=1);f.loc[0,'path']='../not-allowed.jpg'
    with pytest.raises(ValueError,match='escapes'):validate_manifest(f,tmp_path)


def test_reweighting_preserves_class_mass_and_reports_ess():
    from appendix_a.reweight import clipped_cell_weights
    counts=np.array([[10,50],[90,50]])
    weights,ess_fraction=clipped_cell_weights(counts,cap=5)
    adjusted=counts*weights
    assert np.allclose(adjusted.sum(0),counts.sum(0))
    assert np.allclose(adjusted,[[50,50],[50,50]])
    assert 0<ess_fraction<1
    # 再归一之后实际最大权重可能超过预先截断值，不能把 cap 当作最终上界。
    weights,_=clipped_cell_weights([[1,50],[99,50]],cap=2)
    assert weights.max()>2


def test_reweighting_cannot_create_missing_group_class_cells():
    from appendix_a.reweight import clipped_cell_weights
    with pytest.raises(ValueError,match='positive support'):
        clipped_cell_weights([[0,50],[100,50]])
