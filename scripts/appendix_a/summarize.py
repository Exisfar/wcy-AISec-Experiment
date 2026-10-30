#!/usr/bin/env python3
"""只绘制真实完成的实验；缺少重复/比例点时标记 PARTIAL，不插值补曲线。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES
from metrics import representation,association


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/appendix_a_kaggle.json')
    args=parser.parse_args();config=json.loads(args.config.read_text());root=Path(config['output_dir'])
    recall_rows=[];metric_rows=[];runs=[]
    for path in sorted(root.glob('exp*_q*_rep*/complete.json')):
        run=json.loads(path.read_text())
        if run.get('smoke_test_only'):continue
        frame=pd.read_csv(path.parent/'recall.csv')
        for k in ['experiment','step','fraction','repetition']:frame[k]=run[k]
        recall_rows.append(frame);runs.append(run)
        counts=pd.read_csv(path.parent/'train_counts.csv',index_col=0).to_numpy()
        rep=representation(counts.sum(1));g,_,_=association(counts)
        metric_rows.append({'experiment':run['experiment'],'fraction':run['fraction'],'repetition':run['repetition'],
                            'ENS':rep['ENS'],'SEI':rep['SEI'],'V':g['V']})
    expected=len(config['experiments'])*len(config['fractions'])*config['repetitions']
    if not runs:
        print('NO REAL MODEL RESULTS: AffectNet training not completed; no recall figures produced.')
        return
    recall=pd.concat(recall_rows,ignore_index=True)
    recall.to_csv(root/'recall_all_runs.csv',index=False)
    summary=recall.groupby(['experiment','fraction','gender','label']).recall.agg(['mean','std','count']).reset_index()
    summary.to_csv(root/'recall_summary.csv',index=False)
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(root/'metric_all_runs.csv',index=False)
    complete=len(runs)==expected
    for exp in config['experiments']:
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        m=metrics[metrics.experiment==exp].groupby('fraction')[['ENS','SEI','V']].mean()
        for k in m:axes[0].plot(m.index,m[k],marker='o',label=k)
        axes[0].set_title('(a) Dataset bias metrics');axes[0].legend();axes[0].set_ylim(0,2.1)
        for ax,gender in zip(axes[1:],['Female','Male']):
            for label in [*CLASSES,'mean']:
                sub=summary[(summary.experiment==exp)&(summary.gender==gender)&(summary.label==label)].sort_values('fraction')
                if sub.empty:continue
                x=sub.fraction.to_numpy();y=sub['mean'].to_numpy();sd=sub['std'].to_numpy()
                ax.plot(x,y,marker='o',label=label,linewidth=2.5 if label=='mean' else 1.3)
                # 单重复没有标准差，不画假置信带。阴影为重复间样本 std，非置信区间。
                if np.isfinite(sd).any():ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),alpha=.08)
            ax.set_title(f'({"b" if gender=="Female" else "c"}) Recall: {gender}')
            ax.set_ylim(0,1);ax.set_ylabel('Recall');ax.legend(fontsize=7)
        for ax in axes:ax.set_xlabel('Induced female fraction');ax.set_xticks(config['fractions'])
        scope='KAGGLE MODIFIED-DATA EXPERIMENT' if config.get('protocol_variant') else 'Appendix A reimplementation'
        fig.suptitle(f'{scope} / exp {exp} | {"FULL GRID" if complete else "PARTIAL: missing runs, no interpolation"}')
        fig.tight_layout();fig.savefig(root/f'figure_{8+exp:02d}_recall.png',dpi=160);plt.close(fig)
    (root/'aggregation.json').write_text(json.dumps({'completed_runs':len(runs),'expected_runs':expected,
        'complete':complete,'error_bands':'sample standard deviation across repetitions; not confidence intervals'},indent=2))
    print(f'{len(runs)}/{expected} real runs summarized; complete={complete}')
if __name__=='__main__':main()
