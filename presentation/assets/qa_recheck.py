"""Recheck existing saved results for classroom Q&A; run inside bias-repro-gpu."""
from pathlib import Path
import json,sys,math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path('/workspace');sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES,GENDERS,required_counts
from appendix_a.reweight import clipped_cell_weights
from metrics import association,representation
root=ROOT/'results/appendixA_kaggle'
summary=pd.read_csv(root/'recall_summary.csv')
metrics=pd.read_csv(root/'metric_all_runs.csv')

def row(exp,q,g,label):
 s=summary[(summary.experiment==exp)&(summary.fraction==q)&(summary.gender==g)&(summary.label==label)].iloc[0]
 return {'mean':float(s['mean']),'std':float(s['std'])}

def wilson(tp,n,z=1.959963984540054):
 p=tp/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [center-half,center+half]

def matrix(exp,step,rep):return np.load(root/f'exp{exp}_q{step:02d}_rep{rep}'/'confusion.npz')['counts']

out={'scope':'Post-hoc reading of existing saved experiment outputs; no retraining; original paper curves are not available as numeric arrays.'}
plan=pd.read_csv(root/'plan.csv');out['train_size_ranges']=plan.groupby('experiment').n_train.agg(['min','max']).to_dict(orient='index')
out['test_support']=pd.read_csv(root/'test_counts.csv',index_col=0).to_dict(orient='index')
out['female_happy_seed_intervals']=[]
for step in [5,0]:
 for rep in range(3):
  m=matrix(2,step,rep)[0];tp=int(m[3,3]);n=int(m[3].sum())
  out['female_happy_seed_intervals'].append({'q':step/10,'rep':rep,'tp':tp,'support':n,'recall':tp/n,'wilson95_iid_images':wilson(tp,n)})
out['endpoint_comparisons']={}
for exp,g,label,missing_q in [(2,'Female','happy',0),(3,'Female','angry',1)]:
 out['endpoint_comparisons'][f'exp{exp}_{g}_{label}']={'balanced':row(exp,.5,g,label),'missing':row(exp,missing_q,g,label)}
out['exp2_female_class_changes']={label:{'q05':row(2,.5,'Female',label),'q00':row(2,0,'Female',label)} for label in [*CLASSES,'mean']}
out['exp2_female_happy_error_flow']={}
for step in [5,0]:
 counts=np.stack([matrix(2,step,r)[0,3] for r in range(3)])
 out['exp2_female_happy_error_flow'][f'q{step:02d}']={'predicted_class_mean_counts':dict(zip(CLASSES,counts.mean(0).tolist())),'predicted_class_seed_counts':dict(zip(CLASSES,counts.T.tolist()))}
# Middle-range wiggles: report actual q and recall values rather than assuming monotonicity.
out['exp2_middle_happy_curves']={g:[{'q':q,'mean':row(2,q,g,'happy')['mean'],'std':row(2,q,g,'happy')['std']} for q in [.1,.2,.3,.4,.5,.6,.7,.8,.9]] for g in GENDERS}
# Descriptive correlation only: three-seed means at 11 q values, not comparison with original-paper curves.
out['posthoc_V_gap_correlations']=[]
for exp in [1,2,3,4]:
 m=metrics[metrics.experiment==exp].groupby('fraction').V.mean()
 sub=summary[(summary.experiment==exp)&(summary.label=='mean')].pivot(index='fraction',columns='gender',values='mean')
 gap=(sub.Female-sub.Male).abs()
 paired=pd.concat([m.rename('V'),gap.rename('gap')],axis=1).dropna()
 if exp==1:
  rho=None;note='V is essentially zero for pure representational bias; exclude endpoint NaNs; correlation is not interpreted.'
 else:
  rho=float(spearmanr(paired.V.round(12),paired.gap).statistic);note='11 q values; V rounded to 12 decimals for ties; no significance or causality claim.'
 out['posthoc_V_gap_correlations'].append({'experiment':exp,'rho':rho,'n_q':len(paired),'V_range':[float(paired.V.min()),float(paired.V.max())],'definition':'abs(Female macro recall mean - Male macro recall mean)','note':note})
# One-vs-rest FPR and precision for the happy label; argmax classifier, no threshold tuning.
out['posthoc_weighting_happy_errors']=[]
for gidx,g in enumerate(GENDERS):
 for method in ['ERM','Reweighted']:
  values=[]
  for rep in range(3):
   if method=='ERM':m=matrix(2,1,rep)[gidx]
   else:m=np.load(ROOT/f'results/appendixA_mitigation/weighted_exp2_q01_rep{rep}/confusion.npz')['counts'][gidx]
   tp=int(m[3,3]);positive=int(m[3].sum());fp=int(m[:,3].sum())-tp;negative=int(m.sum())-positive
   values.append({'rep':rep,'tp':tp,'positive':positive,'fp':fp,'negative':negative,'FPR':fp/negative,'precision':tp/(tp+fp),'recall':tp/positive})
  out['posthoc_weighting_happy_errors'].append({'gender':g,'method':method,'per_seed':values,'mean_FPR':float(np.mean([v['FPR'] for v in values])),'mean_precision':float(np.mean([v['precision'] for v in values])),'mean_recall':float(np.mean([v['recall'] for v in values]))})
# Boundary examples for Z, IR and extreme clipping.
out['zero_joint_Z']=association([[50,0],[0,50]])[2].tolist()
out['one_group_Z']=association([[50,50]])[2].tolist()
out['one_group_IR']=float(representation([100])['IR'])
w,ess=clipped_cell_weights([[1],[99]],cap=5)
out['one_percent_toy']={'counts':[[1],[99]],'weights_after_class_normalization':w.tolist(),'effective_mass_fraction_minority':float(w[0,0]/100),'ESS_fraction':ess,'scope':'algebraic toy only; not a trained 1% experiment'}
priors=json.loads((ROOT/'configs/appendix_a_kaggle.json').read_text())['class_totals']
out['class_prior']={'target_train_total':sum(priors.values()),'happy_fraction':priors['happy']/sum(priors.values())}
p=ROOT/'presentation/assets/qa_evidence.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
