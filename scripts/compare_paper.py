"""对论文 Fig.3/5 的印刷相关矩阵逐项比对，容差对应其显示精度。"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
def compare(fig,filename,cols,tri):
    actual=pd.read_csv(ROOT/'results/tables'/filename,index_col=0)
    for i,vals in enumerate(tri):
        for j,v in enumerate(vals,i):
            target=float(v);tol=.0005 if len(v.split('.')[-1])==3 else .005
            got=actual.loc[cols[i],cols[j]]
            rows.append({'source':fig,'metric_1':cols[i],'metric_2':cols[j],
                         'paper_printed':target,'reproduction':got,'abs_error':abs(target-got),
                         'rounding_tolerance':tol,'match':abs(target-got)<=tol+1e-12})
compare('Fig.3','agreement_representation.csv',['R','1/IR','1-NSD','SEI','ENS','H','1-BP','1/D','1-D'],[
['1','-.5','.033','-.072','.57','.57','.36','.46','.46'],
['1','.62','.75','.17','.17','.24','.25','.25'],
['1','.96','.79','.79','.94','.93','.93'],
['1','.71','.71','.84','.84','.84'],
['1','1','.89','.95','.95'],
['1','.89','.95','.95'],
['1','.97','.97'],['1','1'],['1']])
compare('Fig.5','agreement_global.csv',['V','Ux','T','NMI','C','Uy'],[
['1','.98','.98','.94','.94','.93'],
['1','.97','.94','.92','.91'],
['1','.98','.98','.97'],
['1','.98','.99'],['1','.99'],['1']])
loc=pd.read_csv(ROOT/'results/tables/local_agreement.csv').rho.dropna()
for name,actual,target,tol in [('local_mean',loc.mean(),.96,.005),('local_std',loc.std(ddof=0),.02,.005)]:
    rows.append({'source':'Sec.V-C','metric_1':name,'metric_2':'NPMI vs Z','paper_printed':target,
                 'reproduction':actual,'abs_error':abs(target-actual),'rounding_tolerance':tol,'match':abs(target-actual)<=tol})
out=pd.DataFrame(rows)
out.to_csv(ROOT/'results/tables/paper_comparison.csv',index=False)
print(out.groupby('source').match.agg(['sum','count']))
assert out[out.source.isin(['Fig.3','Fig.5'])].match.all(), 'Unexpected mismatch in paper matrices'
print('Mismatches (documented local aggregation/tie sensitivity):')
print(out[~out.match].to_string(index=False))
