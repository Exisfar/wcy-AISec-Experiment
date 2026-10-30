"""CPU 即可独立检查真实训练日志、配置和混淆矩阵，重算每条 recall。"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES,GENDERS,required_counts,sha256_file,validate_manifest


def verify(config_path,require_complete=False):
    config=json.loads(Path(config_path).read_text());root=Path(config['output_dir'])
    expected={(e,int(round(q*10)),r) for e in config['experiments'] for q in config['fractions'] for r in range(config['repetitions'])}
    provenance=json.loads((root/'data_provenance.json').read_text())
    supports=pd.read_csv(root/'test_counts.csv',index_col=0).reindex(index=GENDERS,columns=CLASSES).to_numpy()
    manifest=root/'input_manifest.csv'
    assert sha256_file(manifest)==provenance['manifest_sha256']
    frame=validate_manifest(pd.read_csv(manifest),check_files=False)
    actual_supports=pd.crosstab(frame.loc[frame.split=='test','gender'],frame.loc[frame.split=='test','label']).reindex(index=GENDERS,columns=CLASSES).to_numpy()
    assert np.array_equal(actual_supports,supports)
    seen=set();mean_recalls=[]
    for path in sorted(root.glob('exp*_q*_rep*/complete.json')):
        c=json.loads(path.read_text());run=json.loads((path.parent/'run.json').read_text())
        key=(c['experiment'],c['step'],c['repetition'])
        assert key in expected and key not in seen,key
        seen.add(key)
        assert not c['smoke_test_only'] and c['epochs']==20
        assert c['manifest_sha256']==provenance['manifest_sha256']
        digest=hashlib.sha256(json.dumps(run,sort_keys=True).encode()).hexdigest()
        assert digest==c['config_hash'],path
        assert run['test_used_for_checkpoint_selection'] is False
        assert run['original_paper_data_match'] is False
        assert run['batch_size']==config['batch_size'] and run['max_lr']==1e-4
        for name,digest in run['implementation_sha256'].items():
            assert sha256_file(ROOT/'src/appendix_a'/name)==digest,name
        for name,digest in c['artifact_sha256'].items():assert sha256_file(path.parent/name)==digest,(path,name)
        hist=pd.read_csv(path.parent/'history.csv')
        assert hist.epoch.tolist()==list(range(1,21))
        assert np.isfinite(hist.train_loss).all() and (hist.train_loss>=0).all()
        counts=pd.read_csv(path.parent/'train_counts.csv',index_col=0).reindex(index=GENDERS,columns=CLASSES)
        target=required_counts(key[0],key[1],config['class_totals'])
        assert np.array_equal(counts.to_numpy(),target.to_numpy())
        assert (hist.rows==c['training_rows']).all()
        matrices=np.load(path.parent/'confusion.npz',allow_pickle=False)
        assert matrices['classes'].tolist()==list(CLASSES) and matrices['genders'].tolist()==list(GENDERS)
        m=matrices['counts'];assert m.shape==(2,7,7) and (m>=0).all()
        assert np.array_equal(m.sum(2),supports)
        assert m.sum()==c['test_rows']
        accuracy=(np.trace(m[0])+np.trace(m[1]))/m.sum()
        assert np.isclose(accuracy,c['test_accuracy'],atol=1e-12)
        table=pd.read_csv(path.parent/'recall.csv')
        for i,g in enumerate(GENDERS):
            values=np.diag(m[i])/supports[i]
            for j,label in enumerate(CLASSES):
                row=table[(table.gender==g)&(table.label==label)].iloc[0]
                assert row.support==supports[i,j] and row.tp==m[i,j,j]
                assert np.isclose(row.recall,values[j],atol=1e-12)
            assert np.isclose(table[(table.gender==g)&(table.label=='mean')].recall.iloc[0],values.mean(),atol=1e-12)
        mean_recalls.append(float(table[table.label=='mean'].recall.mean()))
    if require_complete:assert seen==expected,f'Only {len(seen)}/{len(expected)} real runs complete'
    result={'verified_runs':len(seen),'expected_runs':len(expected),'complete':seen==expected,
            'epochs_per_run':20,'recall_entries_recomputed':len(seen)*16,
            'scope':'Kaggle modified-data real-image experiment, not original paper AffectNet',
            'all_hashes_counts_histories_and_recalls_match':True}
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json');p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.config,a.require_complete),ensure_ascii=False,indent=2))
