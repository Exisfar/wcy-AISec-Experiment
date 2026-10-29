"""可复用实验与绘图。所有随机过程显式给定种子；原始 CSV 只读。"""
from pathlib import Path
import sys, json, hashlib, warnings, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import representation, association, corrected_v, fixed_universe

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/dataset_bias_metrics'))
import dataset_bias_metrics as dbm
COMPONENTS=['age','race','gender']
UNIVERSE={'age':9,'race':7,'gender':2}
LAB={'jaffe','kdef','ck','ck+','oulu-casia','gemep','mug','wsefep','adfes','liris-cse','isafe'}
MOTION={'sfew','caer-s'}
GLOBAL_MAP={'ϕ_C':'V','T':'T','C':'C','U→':'Ux','U←':'Uy','NMI':'NMI'}
for p in ['results/tables','results/figures','results/local']:
    (ROOT/p).mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
sns.set_palette(['#137C8B','#E88C30','#4467B0','#B94C64','#749D5C'])


def load_data():
    data={p.stem:pd.read_csv(p) for p in sorted((ROOT/'vendor/dataset_bias_metrics/example_data').glob('*.csv'))}
    assert len(data)==20
    for name, df in data.items():
        assert list(df)==['age','race','gender','label'] and not df.isna().any().any(), name
    return data


def savefig(name):
    plt.savefig(ROOT/f'results/figures/{name}.png',bbox_inches='tight',facecolor='white')
    plt.close()


def heatmap(df,name,title,center=None,fmt='.2f',size=(11,6)):
    plt.figure(figsize=size)
    sns.heatmap(df,annot=df.shape[1]<=10,fmt=fmt,cmap='vlag' if center is not None else 'YlGnBu',
                center=center,cbar_kws={'shrink':.7},linewidths=.25)
    plt.title(title,pad=15); plt.tight_layout(); savefig(name)


def main_metrics(data):
    reps=[]; globals_=[]; local=[]; checks=[]; inventory=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        for name,df in data.items():
            inventory.append({'dataset':name,'n':len(df),'collection':'LAB' if name in LAB else 'ITW-M' if name in MOTION else 'ITW-I',
                              **{f'{c}_groups':df[c].nunique() for c in COMPONENTS+['label']}})
            for c in COMPONENTS+['label']:
                vals=representation(df[c].value_counts().to_numpy())
                reps.append({'dataset':name,'component':c,**vals})
                for k, fn in dbm.representational.metrics.items():
                    got=float(fn(df,c)); want=vals[k]
                    checks.append({'dataset':name,'component':c,'metric':k,'ours':want,'upstream':got,
                                   'match':bool(np.isclose(got,want,atol=1e-10,rtol=1e-9,equal_nan=True))})
                if c=='label': continue
                table=pd.crosstab(df[c],df.label)
                g,npmi,z=association(table)
                globals_.append({'dataset':name,'component':c,**g})
                for k,fn in dbm.stereotypical.metrics.items():
                    got=float(fn(df,c,'label')); want=g[GLOBAL_MAP[k]]
                    checks.append({'dataset':name,'component':c,'metric':k,'ours':want,'upstream':got,
                                   'match':bool(np.isclose(got,want,atol=1e-10,rtol=1e-9,equal_nan=True))})
                for key,arr,fn in [('NPMI',npmi,dbm.local_stereotypical.npmi),('Z',z,dbm.local_stereotypical.duchersz)]:
                    got=fn(df,c,'label').to_numpy(dtype=float)
                    checks.append({'dataset':name,'component':c,'metric':key,'ours':float(np.nanmean(arr)),
                                   'upstream':float(np.nanmean(got)),
                                   'match':bool(np.allclose(got,arr,atol=1e-10,rtol=1e-9,equal_nan=True))})
                    pd.DataFrame(arr,index=table.index,columns=table.columns).to_csv(ROOT/f'results/local/{name}_{c}_{key}.csv')
                table.to_csv(ROOT/f'results/local/{name}_{c}_counts.csv')
                if min(table.shape)>1:
                    rho=spearmanr(npmi.ravel(),z.ravel()).statistic
                else: rho=np.nan
                up_nm=dbm.local_stereotypical.npmi(df,c,'label').to_numpy(float).ravel()
                up_z=dbm.local_stereotypical.duchersz(df,c,'label').to_numpy(float).ravel()
                raw_rho=spearmanr(up_nm,up_z).statistic if min(table.shape)>1 else np.nan
                stable_rho=spearmanr(npmi.round(12).ravel(),z.round(12).ravel()).statistic if min(table.shape)>1 else np.nan
                local.append({'dataset':name,'component':c,'rho':rho,'upstream_raw_rho':raw_rho,'tie_stabilized_rho':stable_rho})
    rep=pd.DataFrame(reps); glob=pd.DataFrame(globals_); loc=pd.DataFrame(local); check=pd.DataFrame(checks)
    for name,obj in [('inventory',pd.DataFrame(inventory)),('representation',rep),('global',glob),('local_agreement',loc),('upstream_crosscheck',check)]:
        obj.to_csv(ROOT/f'results/tables/{name}.csv',index=False)
    assert check.match.all(), check.loc[~check.match].to_string()
    return rep,glob,loc,check


def agreement(rep,glob):
    # 与论文 Fig.3 一致：先各组件计算 Spearman，再逐单元平均。NaN 成对删除；复现论文图时将不可定义的组件相关记为 0 后平均，另在文档解释。
    r=rep.copy()
    r['1-NSD']=1-r.NSD; r['1/IR']=1/r.IR; r['1-BP']=1-r.BP
    rcols=['R','1/IR','1-NSD','SEI','ENS','H','1-BP','1/D','1-D']
    rcor=sum(r[r.component==c][rcols].corr(method='spearman').fillna(0) for c in COMPONENTS+['label'])/4
    # 按论文 Eq.13 的分母：UR = I/H(G) = Ux，U = I/H(Y) = Uy。
    gcols=['V','Ux','T','NMI','C','Uy']
    gcor=sum(glob[glob.component==c][gcols].corr(method='spearman') for c in COMPONENTS)/3
    rcor.to_csv(ROOT/'results/tables/agreement_representation.csv')
    gcor.to_csv(ROOT/'results/tables/agreement_global.csv')
    heatmap(rcor,'fig03_agreement','Fig.3 reproduction | Mean Spearman across 4 components',center=0)
    heatmap(gcor,'fig05_agreement','Fig.5 reproduction | Mean Spearman across 3 components',center=0,size=(9,6))
    for c in COMPONENTS+['label']:
        sub=r[r.component==c].set_index('dataset')[rcols]
        order=(sub/sub.max()).mean(axis=1).sort_values(ascending=False).index
        heatmap(sub.loc[order].T,f'fig02_{c}',f'Fig.2 reproduction | {c} | Raw values',size=(16,5))
    for c in COMPONENTS:
        sub=glob[glob.component==c].set_index('dataset')[gcols]
        order=(sub/sub.max()).mean(axis=1).sort_values().index
        heatmap(sub.loc[order].T,f'fig04_{c}',f'Fig.4 reproduction | {c} | Raw values',size=(16,4))
    return rcor,gcor


def local_plots(data):
    # 所有 20 数据集均导出三组件的 counts/NPMI/Z，对应附录 B。
    for name,df in data.items():
        fig,axes=plt.subplots(3,3,figsize=(17,12))
        for i,c in enumerate(COMPONENTS):
            t=pd.crosstab(df[c],df.label); _,nm,z=association(t)
            for j,(title,arr) in enumerate([('Support',t.to_numpy()),('NPMI',nm),("Ducher Z",z)]):
                sns.heatmap(pd.DataFrame(arr,index=t.index,columns=t.columns),ax=axes[i,j],
                            cmap='YlGnBu' if j==0 else 'vlag',vmin=None if j==0 else -1,
                            vmax=None if j==0 else 1,annot=True,fmt='.0f' if j==0 else '.2f',
                            annot_kws={'fontsize':7},cbar=False)
                axes[i,j].set_title(f'{c} | {title}');axes[i,j].set_xlabel('')
        fig.suptitle(f'{name.upper()} | observed FairFace proxy labels',fontsize=18)
        plt.tight_layout(); savefig('fig06_ferplus' if name=='ferplus' else f'appendixB_{name}')
    df=data['expw'];t=pd.crosstab(df.race,df.label);_,_,z=association(t)
    heatmap(pd.DataFrame(z,index=t.index,columns=t.columns),'fig08_expw','Fig.8 reproduction | ExpW race x expression | Ducher Z',center=0)


def summary_plot(rep,glob):
    inv=pd.read_csv(ROOT/'results/tables/inventory.csv').set_index('dataset')
    fig,axes=plt.subplots(1,3,figsize=(17,7),sharey=True)
    for ax,c in zip(axes,COMPONENTS):
        s=rep[rep.component==c].set_index('dataset');g=glob[glob.component==c].set_index('dataset')
        order=sorted(s.index,key=lambda n:({'LAB':0,'ITW-M':1,'ITW-I':2}[inv.loc[n,'collection']],n))
        out=pd.DataFrame({'K-ENS':UNIVERSE[c]-s.ENS,'1-SEI':1-s.SEI,'V':g.V}).loc[order]
        sns.heatmap(out,annot=True,fmt='.2f',ax=ax,cmap='YlOrBr',cbar=False)
        ax.set_title(c);ax.set_ylabel('')
    fig.suptitle('Fig.7 reproduction | fixed demographic universe K = 9 / 7 / 2',fontsize=16)
    plt.tight_layout();savefig('fig07_summary')


def appendix_curves():
    # 表 III–VI 的类计数；与论文一样将不变的非目标类保持为各性别 floor(n_y/2)。
    base=np.array([6962,1733,3117,54241,33708,11175,6439])
    rows=[];tables=[]
    for mode in range(1,5):
        for q in np.linspace(0,1,11):
            if mode==1:
                a=np.floor(base*q); b=np.floor(base*(1-q))
            else:
                a=np.floor(base/2);b=a.copy()
                a[3]=np.floor(base[3]*q);b[3]=np.floor(base[3]*(1-q))
                if mode>=3:
                    qa=1-q if mode==3 else q
                    a[0]=np.floor(base[0]*qa);b[0]=np.floor(base[0]*(1-qa))
            t=np.stack([a,b]);m=representation(t.sum(1));g,_,_=association(t)
            rows.append({'experiment':mode,'fraction':q,**{k:m[k] for k in ['ENS','SEI']},'V':g['V']})
            if q in [0,.5,1]:
                for sex,counts in zip(['Female','Male'],t):
                    tables.append({'experiment':mode,'fraction':q,'gender':sex,**dict(zip(['angry','disgust','fear','happy','neutral','sad','surprise'],counts))})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'results/tables/appendixA_metrics.csv',index=False)
    pd.DataFrame(tables).to_csv(ROOT/'results/tables/appendixA_reconstructed_counts.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,ax in enumerate(axes.ravel(),1):
        s=out[out.experiment==i]
        for k in ['ENS','SEI','V']:ax.plot(s.fraction,s[k],marker='o',markersize=3,label=k)
        ax.set_title(f'Experiment {i} | metric panel only');ax.set_xlabel('Female fraction');ax.set_ylim(0,2.1);ax.legend()
    plt.suptitle('Appendix A: count-based reproduction (no ResNet training)',fontsize=15)
    plt.tight_layout();savefig('appendixA_curves')
    return out


def reweighting(data):
    """改进探索：目标 q(g,y)=p(y)/K，截断权重后在每个 y 内归一，精确保留 p(y)。
    缺失 g×y 单元无法凭重加权创造样本，直接标记 infeasible，不报告虚假公平。
    """
    rows=[]
    for name,df in data.items():
        for c in COMPONENTS:
            t=pd.crosstab(df[c],df.label).to_numpy(float);k=UNIVERSE[c]
            feasible=t.shape[0]==k and (t>0).all()
            if not feasible:
                rows.append({'dataset':name,'component':c,'cap':0,'feasible':False,
                             'missing_cells':int((t==0).sum()+(k-t.shape[0])*t.shape[1])})
                continue
            target=np.broadcast_to(t.sum(0)[None,:]/k,t.shape)
            ratio=target/t
            for cap in [1,2,5,10,0]: # 0 表示不截断
                w=ratio if cap==0 else np.minimum(ratio,cap)
                weighted=t*w
                weighted*=t.sum(0)[None,:]/weighted.sum(0)[None,:]
                w=weighted/t
                m=representation(weighted.sum(1));g,_,_=association(weighted)
                ess=weighted.sum()**2/np.sum(t*w*w)
                assert np.allclose(weighted.sum(0),t.sum(0))
                rows.append({'dataset':name,'component':c,'cap':cap,'feasible':True,'missing_cells':0,
                             'V':g['V'],'ENS':m['ENS'],'SEI':m['SEI'],'ESS_fraction':ess/t.sum(),
                             'max_weight':w.max(),'class_marginal_error':abs(weighted.sum(0)-t.sum(0)).max()})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'results/tables/reweighting.csv',index=False)
    sub=out[(out.dataset=='ferplus')&(out.component=='gender')&out.feasible].copy()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(sub)); labels=['1','2','5','10','unclipped']
    axes[0].plot(x,sub.V,marker='o');axes[0].set_ylabel('Weighted Cramer V');axes[1].plot(x,sub.ESS_fraction,marker='o');axes[1].set_ylabel('Effective sample size / N')
    for ax in axes:ax.set_xticks(x,labels);ax.set_xlabel('Weight cap before class normalization')
    plt.suptitle('Extension | FER+ gender: descriptive balance vs sample efficiency')
    plt.tight_layout();savefig('extension_reweighting')
    return out


def uncertainty(data, reps=300):
    """对固定列联表 multinomial bootstrap；仅量化 iid 样本假设下的不确定性。"""
    rng=np.random.default_rng(20241028);rows=[]
    for name in ['jaffe','ferplus','expw']:
        df=data[name]
        for c in COMPONENTS:
            t=pd.crosstab(df[c],df.label).to_numpy();n=t.sum()
            v=association(t)[0]['V']
            samples=[association(rng.multinomial(n,(t/n).ravel()).reshape(t.shape))[0]['V'] for _ in range(reps)]
            lo,hi=np.nanquantile(samples,[.025,.975]) if np.isfinite(samples).any() else (np.nan,np.nan)
            # 固定边缘的随机列联表，检验独立性；不把非负 V 的 bootstrap 偏移误读为信号。
            from scipy.stats import random_table
            null_tables=random_table.rvs(t.sum(1),t.sum(0),size=reps,random_state=rng)
            null_v=np.array([association(nt)[0]['V'] for nt in null_tables])
            null_p=(1+np.sum(null_v>=v-1e-15))/(reps+1) if np.isfinite(v) else np.nan
            rows.append({'dataset':name,'component':c,'V':v,'V_corrected':corrected_v(t),'ci_low':lo,'ci_high':hi,'replicates':reps,
                         'null_V_mean':np.nanmean(null_v) if np.isfinite(null_v).any() else np.nan,'permutation_p':null_p})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'results/tables/uncertainty.csv',index=False)
    # 保持性别边缘的随机转移：对每一类把 alpha 的性别标签置换为从全局 p(g) 抽样，属于噪声敏感性分析。
    df=data['ferplus'];t=pd.crosstab(df.gender,df.label).to_numpy(float);p=t.sum(1)/t.sum();rows=[]
    for alpha in [0,.05,.1,.2,.5]:
        noisy=(1-alpha)*t+alpha*np.outer(p,t.sum(0))
        rows.append({'noise_fraction':alpha,'V':association(noisy)[0]['V']})
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/proxy_label_sensitivity.csv',index=False)
    return out


def counterexamples():
    # 全局边缘完全均匀，但交叉群体只有 2/4 出现。
    frame=pd.DataFrame({'gender':['F']*50+['M']*50,'race':['A']*50+['B']*50})
    joint=frame.groupby(['gender','race']).size()
    single=representation([100]);fixed=fixed_universe([100],2)
    result={'single_group':{**single,**fixed},'marginal_gender_ENS':representation([50,50])['ENS'],
            'marginal_race_ENS':representation([50,50])['ENS'],'joint_observed_groups':len(joint),
            'joint_fixed_groups':4,'upstream_joint_R':int(dbm.representational.richness(frame,['gender','race'])),
            'pearson_perfect_standard':association([[50,0],[0,50]])[0]['C'],
            'pearson_paper_eq12_denominator':0,
            'chisq_upstream_output':float(dbm.stereotypical.chisq(pd.DataFrame({'x':[0]*50+[1]*50,'y':[0]*50+[1]*50}),'x','y'))}
    (ROOT/'results/tables/counterexamples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def synthetic_downstream(seeds=20):
    """本组扩展，绝非 AffectNet 训练复现。
    SCM: U_Y~Bern(.5), X=2U_Y-1+U_X; A 在训练集的选择机制中与 Y 相关。
    干预 do(A) 固定 X、Y 与噪声，只替换敏感属性；此 SCM 下 X 不随 A 改变。
    """
    from sklearn.linear_model import LogisticRegression
    rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(seed)
        def sample(n,correlation):
            y=rng.integers(0,2,n);x=(2*y-1)+rng.normal(0,1.25,n)
            a=np.where(rng.random(n)<correlation,y,1-y)
            return np.column_stack([x,a]),y
        x,y=sample(4000,.95)
        counts=pd.crosstab(x[:,1],y).to_numpy()
        w=np.array([counts[:,yy].sum()/(2*counts[int(aa),yy]) for aa,yy in zip(x[:,1],y)])
        models={}
        for method in ['ERM','Reweighted','Drop A']:
            keep=method=='Drop A'
            model=LogisticRegression(C=1,max_iter=300,random_state=seed)
            model.fit(x[:,:1] if keep else x,y,sample_weight=w if method=='Reweighted' else None)
            models[method]=model
        for shift,p in [('same',.95),('independent',.5),('reversed',.05)]:
            xt,yt=sample(8000,p)
            for method,m in models.items():
                inp=xt[:,:1] if method=='Drop A' else xt
                pred=m.predict(inp);xcf=xt.copy();xcf[:,1]=1-xcf[:,1]
                cf=m.predict(xcf[:,:1] if method=='Drop A' else xcf)
                group_acc=[np.mean(pred[(xt[:,1]==a)&(yt==yy)]==yt[(xt[:,1]==a)&(yt==yy)]) for a in [0,1] for yy in [0,1]]
                rows.append({'seed':seed,'shift':shift,'method':method,'accuracy':np.mean(pred==yt),
                             'worst_group_accuracy':min(group_acc),'counterfactual_flip':np.mean(pred!=cf)})
    raw=pd.DataFrame(rows);raw.to_csv(ROOT/'results/tables/synthetic_raw.csv',index=False)
    summary=raw.groupby(['shift','method'])[['accuracy','worst_group_accuracy','counterfactual_flip']].agg(['mean','std'])
    summary.to_csv(ROOT/'results/tables/synthetic_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,col,title in zip(axes,['accuracy','worst_group_accuracy','counterfactual_flip'],['Accuracy','Worst (A,Y) group accuracy','Paired do(A) prediction flip']):
        sns.barplot(data=raw[raw['shift']=='reversed'],x='method',y=col,ax=ax,errorbar=('ci',95),seed=42)
        ax.set_ylim(0,1);ax.set_title(title);ax.set_xlabel('');ax.set_ylabel('')
    plt.suptitle(f'Extension only | Synthetic SCM | Reversed selection | {seeds} seeds')
    plt.tight_layout();savefig('extension_synthetic')
    return raw


def provenance(data):
    import subprocess,platform
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'vendor/dataset_bias_metrics/example_data').glob('*.csv'))}
    commit='a70bdc28cc96b338f4d5fdebce5064b56d80de57'
    out={'paper':'arXiv:2303.15889v2','paper_title':'Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition',
         'upstream_url':'https://github.com/irisdominguez/Dataset_Bias_Metrics','upstream_commit':commit,
         'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
         'csv_sha256':files,'total_rows':sum(map(len,data.values())),'seed':20241028}
    (ROOT/'results/provenance.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
