"""完整真实训练网格的描述性分析；所有结果取实际日志，不预设论文趋势。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from verify_results import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json');a=p.parse_args()
    verified=verify(a.config,require_complete=True)
    c=json.loads(a.config.read_text());root=Path(c['output_dir']);s=pd.read_csv(root/'recall_summary.csv')
    assert (s['count']==3).all()
    effects=[]
    def val(exp,q,gender,label):
        row=s[(s.experiment==exp)&np.isclose(s.fraction,q)&(s.gender==gender)&(s.label==label)].iloc[0]
        return float(row['mean']),float(row['std'])
    for exp in c['experiments']:
        for gender in ['Female','Male']:
            for label in ['happy','angry','mean']:
                b,bs=val(exp,.5,gender,label)
                for q in [0.,1.]:
                    v,sd=val(exp,q,gender,label)
                    effects.append({'experiment':exp,'gender':gender,'label':label,'fraction':q,
                                    'balanced_recall':b,'endpoint_recall':v,'endpoint_std':sd,'drop_from_balanced':b-v})
    effect=pd.DataFrame(effects);effect.to_csv(root/'endpoint_effects.csv',index=False)
    # 可投影的四实验总览：每个面板仅展示两条性别宏平均曲线。
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True,sharey=True)
    names={1:'Representational',2:'Happy only',3:'Happy / angry opposite',4:'Happy / angry aligned'}
    for exp,ax in zip(c['experiments'],axes.ravel()):
        for gender,color in [('Female','#B94C64'),('Male','#137C8B')]:
            sub=s[(s.experiment==exp)&(s.label=='mean')&(s.gender==gender)].sort_values('fraction')
            x=sub.fraction.to_numpy();y=sub['mean'].to_numpy();sd=sub['std'].to_numpy()
            ax.plot(x,y,label=gender,color=color,marker='o',markersize=4)
            ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color,alpha=.12)
        ax.set_title(f'{exp}. {names[exp]}',fontsize=14);ax.set_ylim(0,1);ax.set_xticks([0,.2,.4,.6,.8,1]);ax.legend()
        ax.set_xlabel('Induced female fraction');ax.set_ylabel('7-class macro recall')
    fig.suptitle('Kaggle modified-data experiment | 132 real ResNet50 runs | mean +/- seed SD',fontsize=15)
    fig.tight_layout();fig.savefig(ROOT/'results/figures/appendixA_real_macro.png',dpi=170);plt.close(fig)
    # 投影版使用整页宽度和更大字号；完整科研图另存，避免四面板缩进侧栏。
    fig,axes=plt.subplots(2,2,figsize=(14,5.2),sharex=True,sharey=True)
    for exp,ax in zip(c['experiments'],axes.ravel()):
        for gender,color in [('Female','#B94C64'),('Male','#137C8B')]:
            sub=s[(s.experiment==exp)&(s.label=='mean')&(s.gender==gender)].sort_values('fraction')
            x=sub.fraction.to_numpy();y=sub['mean'].to_numpy();sd=sub['std'].to_numpy()
            ax.plot(x,y,label=gender,color=color,marker='o',markersize=4,linewidth=2.5)
            ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color,alpha=.12)
        ax.set_title(f'{exp}. {names[exp]}',fontsize=18)
        ax.set_ylim(0,1);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.tick_params(labelsize=16)
    for ax in axes[:,0]:ax.set_ylabel('Macro recall',fontsize=17)
    for ax in axes[1]:ax.set_xlabel('Female fraction q',fontsize=17)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=17,frameon=False,bbox_to_anchor=(.5,1.05))
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(ROOT/'results/figures/appendixA_real_macro_slide.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for exp,ax in zip([1,2],axes):
        for gender,color in [('Female','#B94C64'),('Male','#137C8B')]:
            sub=s[(s.experiment==exp)&(s.label=='happy')&(s.gender==gender)].sort_values('fraction')
            x=sub.fraction.to_numpy();y=sub['mean'].to_numpy();sd=sub['std'].to_numpy()
            ax.plot(x,y,label=gender,color=color,marker='o',markersize=4)
            ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color,alpha=.12)
        ax.set_title(f'Experiment {exp}: {names[exp]}');ax.set_ylim(0,1);ax.legend();ax.set_xlabel('Induced female fraction');ax.set_ylabel('Happy-class recall')
    fig.suptitle('Real face images: same classifier, different bias construction | 3 seeds',fontsize=14)
    fig.tight_layout();fig.savefig(ROOT/'results/figures/appendixA_real_happy.png',dpi=170);plt.close(fig)
    runs=[json.loads(p.read_text()) for p in sorted(root.glob('exp*_q*_rep*/complete.json'))]
    train_history=pd.concat([pd.read_csv(root/r['run']/'history.csv').assign(run=r['run']) for r in runs],ignore_index=True)
    train_history.to_csv(root/'all_epoch_histories.csv',index=False)
    last=train_history[train_history.epoch==20]
    report={'complete':True,'real_model_runs':len(runs),'total_epochs':len(train_history),
            'test_images_per_run':runs[0]['test_rows'],'epoch_wall_seconds_sum':float(train_history.seconds.sum()),
            'mean_final_train_accuracy':float(last.train_accuracy.mean()),
            'mean_test_accuracy':float(np.mean([r['test_accuracy'] for r in runs])),
            'macro_recall_range':[float(s[s.label=='mean']['mean'].min()),float(s[s.label=='mean']['mean'].max())],
            'original_paper_data_match':False,'repetitions':3,
            'selected_effects':effect.to_dict(orient='records')}
    original=json.loads((ROOT/'configs/appendix_a.json').read_text())['class_totals']
    priors=pd.DataFrame({'original_paper_target':original,'kaggle_target':c['class_totals']})
    priors['original_class_share']=priors.original_paper_target/priors.original_paper_target.sum()
    priors['kaggle_class_share']=priors.kaggle_target/priors.kaggle_target.sum()
    priors.to_csv(root/'class_prior_comparison.csv')
    (root/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='selected_effects'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
