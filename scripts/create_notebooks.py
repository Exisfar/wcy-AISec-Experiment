"""Notebook 的可审阅生成源；修改叙述后可重新生成。"""
from pathlib import Path
import json
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
setup='''from pathlib import Path
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd() if (Path.cwd()/"pyproject.toml").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT/"src"))
from experiments import *
warnings.filterwarnings("ignore", category=FutureWarning)
print("项目目录:", ROOT)
print("Python:", sys.version.split()[0], "NumPy:", np.__version__, "Pandas:", pd.__version__)
'''
def md(s):return {'cell_type':'markdown','id':uuid4().hex[:8],'metadata':{},'source':s.splitlines(True)}
def code(s):return {'cell_type':'code','id':uuid4().hex[:8],'metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]}
def write(name,cells):
    doc={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3 (uv project)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
    (ROOT/'notebooks'/name).write_text(json.dumps(doc,ensure_ascii=False,indent=1),encoding='utf-8')

write('00_论文核对与公式审计.ipynb',[
md(r'''# 00 · 论文身份、复现边界与公式审计

**任务目标**：复现用户提供 PDF 的实际正文，而非根据文件名猜测内容。所有验证均在 CUDA Ubuntu 容器中、通过 uv 项目级 `.venv` 执行；核心实验是 CPU 统计计算，不需要 GPU。

| 项目 | 核对结果 |
|---|---|
| 当前正确文件名 | 2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf |
| 原始文件名（已更正） | 2024_CS_Metrics and Algorithms for Identifying and Mitigating Bias in AI Design- A Counterfactual Fairness Approach.pdf |
| PDF 正文标题 | Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition |
| 作者 | Iris Dominguez-Catena, Daniel Paternain, Mikel Galar |
| 版本 | arXiv:2303.15889v2，2024-06-25；正文及附录共 45 页 |
| 正确源码 | https://github.com/irisdominguez/Dataset_Bias_Metrics |
| 用户给的连字符仓库 | Dataset-Bias-Metrics，README 指向 2022 年 *Assessing Demographic Bias Transfer…*，不是本篇正文指标库 |

**数据真实性**：20 个 CSV 来自作者 `example_data`，包含打乱行序的 `age/race/gender/label`，不是我们生成的数据。人口属性由 FairFace 估计，不能解释为自报身份。

**范围**：正文 Sec.III–V、Fig.2–8 与附录 B 的统计分析可在公开元数据上重新计算。Sec.IV 的原图预处理/FairFace 标注不重跑；附录 A 只重建列联表和 Fig.9–12(a) 指标曲线，不声称重跑 AffectNet 图像 ResNet50 训练或复现其 recall。改进与合成下游实验均标记为“本组扩展”。

这篇论文研究描述性统计偏差，没有给出保证反事实公平的因果算法。后续的反事实实验是独立扩展，不能替代原论文。'''),code(setup),
code('''# 对原 PDF 做内容核对，记录 SHA256，避免文件名导致论文串题。
import fitz, hashlib
pdf = ROOT / '2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf'
doc = fitz.open(pdf)
assert len(doc) == 45
first = doc[0].get_text()
assert "Metrics for Dataset Demographic Bias" in first
print("PDF 页数:", len(doc))
print("PDF SHA256:", hashlib.sha256(pdf.read_bytes()).hexdigest())
print(first[:800])
data = load_data()
provenance_info = provenance(data)
print("公开真实数据集:", len(data), "总记录数:", provenance_info['total_rows'])
print("固定上游 commit:", provenance_info['upstream_commit'])
display(pd.DataFrame([{ 'dataset':n, 'rows':len(d), **{c:d[c].nunique() for c in COMPONENTS}} for n,d in data.items()]))'''),
md(r'''## 1. 指标与统一约定

令 $p_g=n_g/n$，$p_{gy}=n_{gy}/n$，$e_{gy}=p_gp_y$。`0 log 0=0`。Richness 为实际观测组数。先去掉全零边缘，缺失预设组另记覆盖率。

- 表征偏差：$R$、$H=-\sum p_g\log p_g$、$ENS=e^H$、$SEI=H/\log R$、NSD、IR、BP、$D=\sum p_g^2$、$1/D$、$1-D$。ENS/SEI 越大越均衡；D/NSD/IR/BP 越大越偏。
- 全局刻板偏差：Cramér V、Tschuprow T、Pearson C、Theil U（双方向）、NMI。论文 NMI 是 $I(G;Y)/H(G,Y)$，不能直接套 sklearn 默认归一化。
- 局部刻板偏差：NPMI、Ducher Z。正值表示超额出现，负值表示不足；不是分类错误率。

独立实现 `src/metrics.py` 使用 NumPy 概率表；作者原码保留不修改。NumPy 固定 1.26.4，兼容上游 `np.NaN`；Pandas 固定 2.1.4，兼容上游整数表写入小数的历史写法。'''),
code('''# 先用可手算的列联表验证：独立、不平衡但独立、完全关联。
from metrics import representation, association, fixed_universe
independent = np.array([[25,25],[25,25]])
imbalanced_independent = np.array([[45,45],[5,5]])
perfect = np.array([[50,0],[0,50]])
assert np.isclose(association(independent)[0]['V'], 0)
assert np.isclose(association(imbalanced_independent)[0]['V'], 0)
assert np.isclose(association(perfect)[0]['V'], 1)
assert np.isclose(representation([50,50])['ENS'], 2)
assert np.isclose(representation([100])['ENS'], 1)
assert np.isnan(representation([100])['SEI'])
# 单组 V 不可识别：NaN 不是“没有偏差”。
assert np.isnan(association([[50,50]])[0]['V'])
print("手算边界案例全部通过。")'''),
md(r'''## 2. 公式、源码与解释之间的差异

1. **Eq.(12)** 的 PDF 将 Pearson C 写成 $\sqrt{(\chi^2/n)/(1-\chi^2/n)}$。作者调用 SciPy 的定义为 $\sqrt{(\chi^2/n)/(1+\chi^2/n)}$。完全关联的 2×2 表中前者分母为 0；标准值为 $\sqrt{1/2}$。这是可直接证明的公式排版/定义差异，不是实验发现“算法必然失效”。
2. **Eq.(13)** 分子 $H(P_1)-H(P_1\mid P_2)=I$，但分母印为 $H(P_2)$；作者 `theilsu(x,y)` 用 $H(x)$。我们记录双方向并遵循作者实现进行主结果比较。
3. **Table I** 将 Shannon H 与 bias 关系列为 direct，与正文“熵越高越多样”相矛盾；应按正文解释。NSD 的表中区间也不包括其可取的 0。
4. **`chisq` 函数** 返回 p 值，不是 docstring 声称的统计量；不属于本文最终推荐指标。
5. **多组件 richness**：作者对每列 `nunique` 相乘，得到理论笛卡尔积大小，未必等于实际出现的交叉群体数。单组件主实验不受此问题影响。
6. **缺失群体**：只存在一个组时 IR=1；SEI 未定义。不能把“已出现组内部均衡”读成所有群体都得到覆盖。

以上与作者原码逐项对应，后面的 Notebook 使用固定定义，避免悄悄修复后宣称完全一致。'''),
code('''audit = counterexamples()
display(audit)
assert audit['joint_observed_groups'] == 2
assert audit['upstream_joint_R'] == 4
assert np.isclose(audit['pearson_perfect_standard'], np.sqrt(.5))
assert audit['single_group']['coverage'] == .5
print("审计反例已保存 results/tables/counterexamples.json")'''),
md(r'''## 3. 完整性与局限

- 20 数据集元数据全量计算，不抽样代替主结果；各 CSV 校验和存于 `results/provenance.json`。
- 观测偏差不等价于模型歧视或因果效应。未获得同一人身份、图片特征和模型输出，不能从这些 CSV 推断真实模型公平性。
- 原图、FairFace 自动标注的误差、同一主体重复帧、被剔除的人脸，均可能影响统计结论；公开元数据不含主体标识，无法修复聚类依赖。
- 原论文已讨论自动标注局限和“数据偏差不线性传递”。本组贡献是定量审计、边界反例和可复现的缓解探索，而非把这些已知局限包装成首次发现。

来源：本地 PDF Sec.III、IV、V、Appendix A/B；作者 README / 源码；https://arxiv.org/abs/2303.15889v2 。''')])

write('01_二十个真实数据集完整指标复现.ipynb',[
md(r'''# 01 · 20 个真实 FER 数据集：主要统计实验复现

本 Notebook 对正文四个研究问题逐项给出证据：表征指标是否冗余、全局指标是否一致、局部指标是否一致，以及不同数据源的偏差分布。

数据源是作者公开且固定 commit 的 20 个完整匿名统计 CSV。单一组件计算与作者逐值交叉验证；图形重绘使用独立实现。所有图和明细均导出到 `results/`，不依赖联网。'''),code(setup),code('''start = time.perf_counter()
data = load_data()
provenance_info = provenance(data)
rep, glob, loc, checks = main_metrics(data)
print(f"真实记录数: {sum(map(len,data.values())):,}; 数据集: {len(data)}")
print("与作者独立交叉验证项数:", len(checks), "全部通过:", checks.match.all())
assert checks.match.all()
display(pd.read_csv(ROOT/'results/tables/inventory.csv'))
display(rep.head())'''),
md(r'''## RQ1 / Fig.2–3：表征指标的排序一致性

论文先在 age、race、gender、label **分别**跨 20 个数据集计算 Spearman 相关矩阵，再逐单元平均，不能把 80 行混在一起做相关。D/NSD/BP 转为补数、IR 转为倒数，确保排序方向一致。组件内部先成对删除 NaN；复现图中将不可定义的相关填 0 再对四组件平均。该约定能复现印刷数值，但 0 不表示真实无相关。若改为仅平均可定义组件，R/1-NSD 为 0.0441 而非 0.0331，R/SEI 为 −0.0958 而非 −0.0718，展示聚合约定的敏感性。图中展示原始数值；配色尺度与论文可以不同。

待检验：ENS 与 H 完全单调等价；SEI 与 1-NSD 高相关；R 与 IR 的行为不同。'''),code('''rcor, gcor = agreement(rep, glob)
display(rcor.round(3))
display(Image(filename=str(ROOT/'results/figures/fig03_agreement.png')))
assert np.isclose(rcor.loc['ENS','H'], 1)
print("SEI 与 1-NSD:", rcor.loc['SEI','1-NSD'])
display(Image(filename=str(ROOT/'results/figures/fig02_race.png')))'''),
md(r'''## RQ2 / Fig.4–5：全局刻板偏差指标

每个指标衡量“人口属性与表情标签的统计依赖”，不是模型精度。由于标签分布和变量类别数不同，量纲或区间接近也不意味着可直接互换数值。

图中 Ux=I/H(G)，Uy=I/H(Y)。按论文 Eq.(13) 的分母，UR 对应 Ux、U 对应 Uy；作者函数 theilsu(x,y) 的命名方向与正文不一致。

论文建议 Cramér V：直观且不要求两个变量类别数相同。论文的经验强度阈值不是经过多重比较校正的显著性检验，演示中只称其为描述性参考。'''),code('''display(gcor.round(3))
display(Image(filename=str(ROOT/'results/figures/fig05_agreement.png')))
print("全局指标非对角相关最小值:", gcor.where(~np.eye(len(gcor),dtype=bool)).min().min())
display(Image(filename=str(ROOT/'results/figures/fig04_gender.png')))'''),
md(r'''## RQ3 / Fig.6 与附录 B：局部刻板偏差

对每个数据集、每个组件生成 count/NPMI/Z 的完整矩阵，展平成同一顺序计算 Spearman，随后平均。保留联合概率为零但边缘非零的单元：NPMI=-1，而不是删掉。只有一个组时关联无法识别，相关系数记为 NaN。

论文报告均值 ρ=0.96±0.02。我们同时导出有效组合数、均值和标准差；这不是对不同数据集“人类总体”的置信区间。作者浮点原始值在完全独立表中可能产生仅 1e-16 的伪排序，因此同时报告原始作者相关与 12 位舍入后的并列稳健相关。'''),code('''local_plots(data)
valid = loc.rho.dropna()
print("有效数据集×组件:", len(valid), "/", len(loc))
print("独立公式：局部相关均值:", valid.mean(), "样本标准差:", valid.std(ddof=1), "总体标准差:", valid.std(ddof=0))
print("原始作者浮点计算:", loc.upstream_raw_rho.mean(), loc.upstream_raw_rho.std(ddof=0))
print("将 1e-12 以下差异视为并列后:", loc.tie_stabilized_rho.mean(), loc.tie_stabilized_rho.std(ddof=0))
display(loc.pivot(index='dataset',columns='component',values='rho').round(4))
display(Image(filename=str(ROOT/'results/figures/fig06_ferplus.png')))
display(Image(filename=str(ROOT/'results/figures/fig08_expw.png')))'''),
md(r'''## RQ4 / Fig.7–8：按数据来源比较偏差

LAB=实验室；ITW-M=影视；ITW-I=互联网查询。固定年龄/种族/性别标签空间分别为 9/7/2。`K-ENS` 同时反映缺失与不均衡；`1-SEI` 仅反映已出现组的不均衡。

这里比较的是数据集集合的描述性结果。不能解释为“互联网收集必然更公平”的因果结论。'''),code('''summary_plot(rep,glob)
inv=pd.read_csv(ROOT/'results/tables/inventory.csv')
summary=rep[rep.component.isin(COMPONENTS)].merge(inv[['dataset','collection']],on='dataset')
summary['ENS_over_K']=summary.apply(lambda r:r.ENS/UNIVERSE[r.component],axis=1)
display(summary.groupby(['collection','component'])[['ENS_over_K','SEI']].mean().round(3))
display(Image(filename=str(ROOT/'results/figures/fig07_summary.png')))
# 上游 README 提供的未四舍五入锚点，比只看趋势更严格。
anchors=[('ADFES race ENS', rep.query("dataset=='adfes' and component=='race'").ENS.iloc[0],2.478206948646503),
         ('ExpW race V', glob.query("dataset=='expw' and component=='race'").V.iloc[0],0.04104288518527493)]
for name,actual,expected in anchors:
    assert np.isclose(actual,expected,atol=1e-12), (name,actual,expected)
    print(name, actual, 'PASS')'''),
md(r'''## 附录 A：四种受控偏差的列联表与指标曲线

重建 Table III–VI：全局性别比例、仅 happy 类、happy/angry 反向、happy/angry 同向偏差；比例 0–1、步长 0.1。遵循表格整数取整，balanced 时每类男女分别 floor(n_y/2)，因此样本总数与极端设置可相差几个样本。

**重要差异**：原文 Experiment 1 文字将单组端点的 V 描述为 0，但公式与 SciPy 在维度退化时 V 未定义。图中保留 NaN 空点，不假装测到了零。Table III–VI 的端点与中点计数另导出，便于逐格核对。

只复现 Fig.9–12(a)；(b)(c) recall 曲线依赖原图和预训练网络，本交付未执行。'''),code('''appendix = appendix_curves()
display(appendix.query('fraction==0.5'))
display(Image(filename=str(ROOT/'results/figures/appendixA_curves.png')))
print("本 Notebook 执行秒数:", round(time.perf_counter()-start,2))'''),
md(r'''## 结果判定

精确比对：所有单组件指标与固定上游作者实现一致，README 的两个未四舍五入数值锚点通过。论文图的数值/相关系数另在 `results/tables/paper_comparison.csv` 中逐项比对，保留任何不吻合项及可能原因，不能用“源码一致”替代“论文一致”。

本地输出包括：20 个数据集全量表征和全局指标、60 套局部矩阵、正文重绘、附录 B 的 20 幅矩阵总览、附录 A 的四种指标曲线。''')])

write('02_不足分析与改进探索.ipynb',[
md(r'''# 02 · 不足分析与改进探索（本组扩展）

本 Notebook 直接对应评议 (e)(f)(g)：找出不足、提出方案、实施探索。所有结果均标注真实元数据或合成实验，避免误导。

提出一个可落地流程：**固定群体空间审计 → 检查交叉支持 → 估计不确定性 → 在有支持时做限权重平衡 → 用保留测试集和适当因果假设验证模型**。这个流程不是新的理论公平保证；我们只验证其中可由现有数据支持的部分。'''),code(setup),code('''data=load_data()
audit=counterexamples()
display(pd.DataFrame([audit['single_group']]))
print("两个边缘均 ENS=2，但实际交叉群体只有 2/4；上游组合 richness 返回",audit['upstream_joint_R'])
assert audit['single_group']['IR']==1 and audit['single_group']['coverage']==.5'''),
md(r'''## 探索一：固定标签空间与交叉群体覆盖

原本只统计观测组，导致单组 IR=1；即使各人口属性的边缘均衡，也可能缺失交叉群体。补充 `coverage=R/K`、`H/log K`、`ENS/K`。这些数依赖预设标签空间，不能把数据集中不可适用的组任意列成“缺失”。这里 K 跟随论文 FairFace 类别，仅作该设定下的技术核验。

不要用平滑伪计数当作真实覆盖。没有样本的群体必须单独标记。'''),code('''coverage=[]
for name,df in data.items():
    for c in COMPONENTS:
        coverage.append({'dataset':name,'component':c,**fixed_universe(df[c].value_counts(),UNIVERSE[c])})
coverage=pd.DataFrame(coverage)
coverage.to_csv(ROOT/'results/tables/coverage.csv',index=False)
display(coverage.pivot(index='dataset',columns='component',values='coverage').round(3))'''),
md(r'''## 探索二：不确定性与自动标注误差

采用 300 次 multinomial bootstrap 给 V 加区间，并报告有限样本校正 V。此处将数据视作抽样用于**扩展总体推断**；原文将固定数据集视作总体、不做校正，两者问题设定不同，不应直接宣称原文“遗漏了必须的修正”。

限制：V 非负，在零附近有小样本上偏，普通百分位 bootstrap 不保证良好覆盖（JAFFE/race 的原始 V 甚至低于其百分位下界）；因此这里同时报告固定边缘的 300 次随机列联表零假设检验与校正 V，不把区间不含零直接解释为显著关联。置换 p 值仅在相应交换性假设下成立，且本探索未做多重比较校正。

无主体 ID，只能 iid 行级 bootstrap，重复帧可能导致区间过窄。对 FairFace 标签噪声，使用混合扰动展示敏感性，不假设已知真实错误率，也不把扰动当作真实人口属性纠正。'''),code('''ci=uncertainty(data,reps=300)
display(ci.round(5))
display(pd.read_csv(ROOT/'results/tables/proxy_label_sensitivity.csv'))'''),
md(r'''## 探索三：保持类别分布的截断重加权

设目标分布 $q(g,y)=p(y)/K$，未截断权重 $w_{gy}=q(g,y)/p(g,y)$。为限制方差，先将权重截断到 cap，再按每个 y 重新归一化，**保持各表情类的总权重不变**。截断后的最终最大权重可能高于 cap，所以同时报告实际最大权重。

观察指标：加权 V（越低越独立）、ENS/SEI、有效样本量 $ESS=(\sum w)^2/\sum w^2$。所有 20×3 组件均检查可行性；任一预设群体或群体×类单元缺失时标记 infeasible，不伪造权重。未截断可以按构造将 V 降为 0，这一代数结果不证明模型公平性，也不证明原始数据已改变。

要用于模型训练，必须仅在训练分割估计权重，并在 untouched 测试集评估；真实 CSV 无图像特征，本节不声称完成真实 FER 模型缓解。'''),code('''rw=reweighting(data)
print("检查的数据集×组件:",rw[['dataset','component']].drop_duplicates().shape[0])
print("不可行组合:",rw.loc[~rw.feasible,['dataset','component']].shape[0])
display(rw.query("dataset=='ferplus' and component=='gender'"))
display(rw.loc[~rw.feasible,['dataset','component','missing_cells']])
assert rw.loc[rw.feasible,'class_marginal_error'].max()<1e-8
unclipped=rw[rw.feasible & (rw.cap==0)]
assert unclipped.V.max()<1e-10
display(Image(filename=str(ROOT/'results/figures/extension_reweighting.png')))'''),
md(r'''## 探索四：偏差能否影响下游？明确 SCM 的合成压力测试

**这是本组独立扩展，绝非 AffectNet/ResNet50 结果。** 使用 20 个随机种子、4000 训练样本、每个测试环境 8000 样本。

数据机制：$Y\sim Bern(0.5)$，$X=2Y-1+\epsilon$，$\epsilon\sim N(0,1.25^2)$。敏感属性 A 通过选择机制与 Y 在训练集 95% 一致；测试包含同分布 95%、独立 50%、反转 5%。在这个 SCM 中 X 不受 A 因果影响，因此配对 `do(A=1-A)` 保持 X/Y/噪声不变。

比较：原始 ERM、仅根据训练表估计的重加权、删除 A 的对照；指标包括准确率、最差 (A,Y) 交叉群体准确率、配对反事实预测翻转率。所有方法共享同一测试样本。

删除 A 在现实中不能保证反事实公平：若 X 是 A 的后代或存在代理变量，必须根据真实因果图重新构造反事实。这里的 SCM 专门允许这个对照，结论仅限合成机制。'''),code('''raw=synthetic_downstream(seeds=20)
summary=raw.groupby(['shift','method'])[['accuracy','worst_group_accuracy','counterfactual_flip']].agg(['mean','std'])
display(summary.round(4))
display(Image(filename=str(ROOT/'results/figures/extension_synthetic.png')))
# 同种子配对差：仅以种子作为独立重复，而不把全部测试行视作 20 倍独立实验。
from scipy.stats import t as student_t
pivot=raw.query("shift=='reversed'").pivot(index='seed',columns='method',values='accuracy')
delta=pivot['Reweighted']-pivot['ERM']
margin=student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))
paired={'n_seeds':len(delta),'mean_accuracy_improvement':delta.mean(),'ci_low':delta.mean()-margin,'ci_high':delta.mean()+margin}
(ROOT/'results/tables/synthetic_paired_ci.json').write_text(json.dumps(paired,indent=2))
display(paired)
assert raw.query("method=='Drop A'").counterfactual_flip.max()==0'''),
md(r'''## 怎样解读改进结果

- **能够证明**：固定类别覆盖能揭示遗漏群体；交叉审计能发现边缘指标看不到的空缺；重加权可以在有支持的真实统计表上消除或降低描述性依赖；在明确设定的合成选择偏差环境中，可检验缓解后下游变化。
- **不能证明**：真实人脸分类器已达到反事实公平；统一人口比例必然适用于所有场景；匿名元数据可以替代原图/人工标注；合成收益能迁移到 AffectNet。
- **后续方案**：在合法获得 AffectNet 后，按主体划分训练/验证/测试，训练集估计截断权重，使用与原文相同的 ResNet50、1cycle、20 epochs、lr=1e-4、三次重复，报告 subgroup recall 与 ESS/不确定性；需使用真实因果假设才能讨论反事实公平。

报告中会同时展示改进的收益和代价：高权重导致有效样本减少，无支持单元无法仅靠加权解决，真实模型验证仍需图像与主体信息。''')])
print('Wrote 3 notebooks')
