"""独立核验三个真实重加权对照的配对基线、权重约束和分组召回率。"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES,GENDERS,sha256_file
from appendix_a.reweight import clipped_cell_weights


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    out=ROOT/'results/appendixA_mitigation';verified=0
    for rep in range(3):
        folder=out/f'weighted_exp2_q01_rep{rep}'
        if not (folder/'complete.json').exists():continue
        result=json.loads((folder/'complete.json').read_text());meta=json.loads((folder/'run.json').read_text())
        assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==result['config_hash']
        assert sha256_file(Path(__file__).with_name('mitigate.py'))==meta['source_sha256']
        assert sha256_file(ROOT/'src/appendix_a/reweight.py')==meta['reweight_implementation_sha256']
        assert sha256_file(ROOT/'results/appendixA_kaggle/input_manifest.csv')==meta['manifest_sha256']
        baseline=ROOT/'results/appendixA_kaggle'/meta['baseline'];erm=json.loads((baseline/'complete.json').read_text())
        assert erm['config_hash']==meta['baseline_config_hash']
        assert meta['test_used_for_cap_or_epoch_selection'] is False
        assert meta['cap_before_class_normalization']==5 and meta['epochs']==20
        assert meta['seed']==json.loads((baseline/'run.json').read_text())['run_seed']
        for name,digest in result['artifact_sha256'].items():assert sha256_file(folder/name)==digest
        count=pd.read_csv(baseline/'train_counts.csv',index_col=0).reindex(index=GENDERS,columns=CLASSES).to_numpy()
        w,ess=clipped_cell_weights(count,5)
        assert np.allclose(w,meta['cell_weights']) and np.isclose(ess,meta['ESS_fraction'])
        hist=pd.read_csv(folder/'history.csv');assert hist.epoch.tolist()==list(range(1,21))
        assert np.isfinite(hist.train_weighted_loss).all()
        confusion=np.load(folder/'confusion.npz',allow_pickle=False)['counts']
        bc=np.load(baseline/'confusion.npz',allow_pickle=False)['counts']
        assert np.array_equal(confusion.sum(2),bc.sum(2))
        table=pd.read_csv(folder/'recall.csv')
        for i,g in enumerate(GENDERS):
            recalls=np.diag(confusion[i])/confusion[i].sum(1)
            for j,label in enumerate(CLASSES):
                row=table[(table.gender==g)&(table.label==label)].iloc[0]
                assert row.tp==confusion[i,j,j] and row.support==confusion[i,j].sum()
                assert np.isclose(row.recall,recalls[j],atol=1e-12)
            assert np.isclose(table[(table.gender==g)&(table.label=='mean')].recall.iloc[0],recalls.mean(),atol=1e-12)
        assert np.isclose((np.trace(confusion[0])+np.trace(confusion[1]))/confusion.sum(),result['test_accuracy'])
        verified+=1
    if a.require_complete:assert verified==3,f'Only {verified}/3 mitigation models complete'
    state={'verified_real_mitigation_models':verified,'expected':3,'complete':verified==3,
           'same_subsets_initializations_and_test_sets':True,'cap_predeclared':5,'test_used_for_cap_selection':False}
    out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
if __name__=='__main__':main()
