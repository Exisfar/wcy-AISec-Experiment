from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'results/provenance.json').read_text());ex=json.loads((ROOT/'results/execution.json').read_text())
r=pd.read_csv(ROOT/'results/tables/agreement_representation.csv',index_col=0);g=pd.read_csv(ROOT/'results/tables/agreement_global.csv',index_col=0)
l=pd.read_csv(ROOT/'results/tables/local_agreement.csv');rw=pd.read_csv(ROOT/'results/tables/reweighting.csv');raw=pd.read_csv(ROOT/'results/tables/synthetic_raw.csv')
syn=raw.groupby(['shift','method'])[['accuracy','worst_group_accuracy','counterfactual_flip']].mean()
ci=json.loads((ROOT/'results/tables/synthetic_paired_ci.json').read_text())
sub=rw.query("dataset=='ferplus' and component=='gender' and cap==0").iloc[0]
comp=l.groupby('component').upstream_raw_rho.mean()
lines=f'''# 复现报告：数据集人口统计偏差

## 1. 论文身份与核验范围

实际正文：**Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition**，Iris Dominguez-Catena、Daniel Paternain、Mikel Galar；TPAMI 2024 / arXiv:2303.15889v2，2024-06-25，含附录 45 页。用户 PDF 的原始文件名误写 Counterfactual Fairness，现已更正为 `2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf`；正文不属于因果反事实公平算法论文。用户给的连字符 GitHub 仓库对应作者 2022 年另一篇文章；本项目使用正文指向的下划线仓库。

范围：Sec.III 指标、Sec.V 全部四个研究问题和 Fig.2–8、附录 B 的 20 数据集全部局部矩阵，均从公开元数据重算。附录 A 四种偏差构造重建 Table III–VI 与 Fig.9–12(a) 指标曲线。**没有重新执行 Sec.IV 原图预处理/FairFace 标注，没有训练 AffectNet 的 ResNet50，没有复现 Fig.9–12(b)(c) recall。** 20 种子合成下游实验是本组扩展，不是原文模型实验。

## 2. 数据与可运行环境

作者 commit `{p['upstream_commit']}`，20 个原始匿名 CSV，共 **{p['total_rows']:,}** 条记录。每条包含 age/race/gender/label；年龄、种族、性别为 FairFace 表观人口预测。行序由作者打乱，不含身份、图像或主体索引。每文件 SHA256 存于 `results/provenance.json`。

基镜像 `nvidia/cuda:12.5.1-base-ubuntu20.04`，固定摘要 `sha256:9608fa115ab6f9b5985f8a05019f707148454ed669791572d95dd9814e582149`。容器内使用 uv 创建 `/workspace/.venv`；Python 3.11.15、NumPy 1.26.4、Pandas 2.1.4、SciPy 1.12.0，其余依赖固定于 uv.lock。旧作者代码使用 `np.NaN` 与整数表写入浮点的旧行为，故明确固定兼容版本并保留原码不修改。

三个 Notebook 最近一次开发容器执行用时 **{ex['total_seconds']:.2f} 秒**（只是本机实测，不承诺其他机器相同时间）。最终镜像另以禁网、无 GPU、无项目挂载的新容器验证，日志位于 `dist/clean-container.log`。通过 `reproduce` 执行可重算全部输出；`verify` 只做秒级完整性检查，两者不能混淆。

## 3. 方法与独立核验

独立实现 `src/metrics.py` 直接从概率表计算，不调用作者计算函数。对照保留的作者原码：20×4×10 个表征标量、20×3×6 个全局标量、20×3×2 个局部整矩阵，共 **1,280 项全部一致**。局部比较逐单元验证 allclose，而不是只比较均值。8 项关键测试覆盖独立但不平衡、完全关联、维度退化、零单元、尺度不变性、方向对称及单行浮点误差。

论文表中共 17 类指标：10 类表征、5 类全局、2 类局部。Theil U 双方向产生 6 个全局读数。论文 NMI 为 I(G;Y)/H(G,Y)，没有使用 sklearn 默认归一化。观测组数量 R 与固定类别空间 K 明确区分。R=1 时 V/SEI 未定义，保留 NaN，不称为“公平”；非零边缘的零联合单元 NPMI=-1。

未修改任何原始 CSV。总体重复行不是无效记录：不同人或不同帧可以共享同一四字段值，不能按四字段去重。

## 4. 与论文数值对照

### 4.1 印刷结果与精确锚点

- **Fig.3 上三角 45/45 项** 在各自印刷精度的舍入容差内吻合。
- **Fig.5 上三角 21/21 项** 在印刷精度内吻合。
- 作者 README 的 ADFES/race ENS=2.478206948646503、ExpW/race V=0.04104288518527493 两个未四舍五入锚点，均在 1e-12 容差内通过。
- 明细：`results/tables/paper_comparison.csv`。目标来自用户 PDF 图中数字，录入源为 `scripts/compare_paper.py`。

重要约定：Fig.3 先在 age/race/gender/label 内分别计算相关再平均。性别 R 在成对删除单组缺失值后变成常数，相关不可定义；将该组件不可定义的相关填 0 再平均能复现印刷值。如果只平均可定义组件，R 对 1-NSD 为 0.044110（论文 0.033），R 对 SEI 为 −0.095752（论文 −0.072）。填 0 是图的复现约定，不意味着真实无相关，展示了聚合定义的影响。

Fig.5 中按正文 Eq.(13) 的分母明确映射：Ux=I/H(G)、Uy=I/H(Y)；正文的 UR 对应 Ux、U 对应 Uy。作者 `theilsu(x,y)` 默认用 H(x)，与正文分母方向不同。依靠箭头名容易错位，报告始终写出归一化分母。

### 4.2 局部指标相关的聚合差异

论文 Sec.V-C 报告 ρ=0.96±0.02，但该句未充分交代标准差的聚合层次和浮点并列处理。

| 计算约定 | 均值 | 标准差 | 解释 |
|---|---:|---:|---|
| 独立公式，逐数据集×组件，忽略退化 | {l.rho.mean():.6f} | {l.rho.std(ddof=0):.6f} | {l.rho.count()} 个有效组合；std 为组合层总体标准差 |
| 原始作者浮点值，逐数据集×组件 | {l.upstream_raw_rho.mean():.6f} | {l.upstream_raw_rho.std(ddof=0):.6f} | {l.upstream_raw_rho.count()} 个有效组合；完全独立表可出现约 1e-16 的伪排序 |
| 原始作者值，先按三组件平均，再跨组件 | {comp.mean():.6f} | {comp.std(ddof=1):.6f} | 三个均值的样本标准差；舍入为 0.96±0.02 |
| 独立公式，12 位舍入后视为并列 | {l.tie_stabilized_rho.mean():.6f} | {l.tie_stabilized_rho.std(ddof=0):.6f} | 去除数值近零造成的伪排序 |

第三种约定可得到与印刷报告相容的值，但在缺少作者原始分析脚本的情况下，不能断言这就是作者采用的唯一算法。默认逐组合稳健结果与论文 ± 部分不同，原始差异保留在对照 CSV。定性结论仍成立：NPMI/Z 排序高度相关；不同聚合与并列处理的具体数值并不相同。

### 4.3 主要结论

1. ENS 与 H 的平均 Spearman 为 **{r.loc['ENS','H']:.3f}**，符合严格单调变换；SEI 与 1-NSD 为 **{r.loc['SEI','1-NSD']:.3f}**，支持联合使用一个组合多样性指标与一个均匀性指标。
2. 六个全局读数的平均成对相关最小为 **{g.where(~__import__('numpy').eye(len(g),dtype=bool)).min().min():.3f}**，与原文“排序高度一致”相符。相同排序不等于相同量纲，也不保证区间宽度有同样含义。
3. LAB 常出现较低人口丰富度；互联网数据的表征多样性通常更高，但并不保证更低刻板偏差。属于数据集集合的描述性观察，不作因果归因。
4. Z/NPMI 能定位具体人口属性×表情的超额/不足，不能当作模型错误率。

所有 20 数据集主指标、60 套 count/NPMI/Z、正文重绘及 20 幅附录矩阵随包。附录 A 的 4×11 个构造可验证偏差类型，但单组端点的 V 在标准公式下不可定义，与原文文字称 0 的描述有差异；图中保留空点。

## 5. 可核验的不足与边界

### 5.1 公式/实现审计

- PDF Eq.(12) Pearson C 分母为 1−χ²/n；标准定义和作者 SciPy 实现为 1+χ²/n。完全关联 2×2 表的标准 C=√(1/2)，印刷式分母为 0。此处是表达差异，不证明作者实验用了错误公式。
- Eq.(13) 的 Theil U 分母与作者函数方向不同；保留双方向并按分母解释。
- Table I 将 H 与 bias 关系列为 direct，但正文解释为熵越高越多样；NSD 区间也不含其可达到的 0。
- 上游 `chisq` 文档称返回卡方统计量，实际返回 p 值；本组未把它混入推荐偏差指标。
- 上游多组件 richness 将各列 nunique 相乘，是笛卡尔积大小，不一定等于观测联合群体数量。例：只有 (F,A)、(M,B) 两组，却返回 2×2=4。单组件主结果不受影响。

### 5.2 方法限制

只看已出现组会掩盖缺失群体：单组 IR=1、ENS=1，SEI 未定义。两个边缘均匀也可能遗漏半数交叉群体。FairFace 属性是预测代理而非自报真值；年龄离散化损失序数信息。无主体 ID 导致主体重复和视频帧相关难以校正。元数据上的统计关联不能保证模型公平，更不能直接推断因果反事实公平。原论文已讨论其中部分限制，本组不把它们包装成首次发现。

## 6. 本组改进探索

### 6.1 固定空间、支持检查与不确定性

补充 coverage=R/K、ENS/K、H/log K，并检查交叉支持。没有样本的群体/单元标记缺失，不用伪计数冒充真实覆盖。

对 JAFFE、FER+、ExpW 三数据集的三人口组件进行 300 次 multinomial bootstrap，并报告有限样本校正 V，属于扩展总体推断；不替换原论文对固定数据集的描述量。V 非负且在零附近有小样本上偏，普通百分位 bootstrap 不保证良好覆盖；JAFFE/race 原始 V 小于百分位下界就是具体例子，不应把区间不含零直接视为显著关联。因此同时给出固定边缘随机列联表的 300 次零假设检验与有限样本校正 V。置换 p 值依赖交换性且未做多重比较校正。无主体 ID，只能采用 iid 行假设，重复帧可能使区间过窄。针对 FER+ 性别代理标签做混合噪声敏感性实验，测试结果随假设变化，不宣称已纠正真实标注错误。

### 6.2 真实统计数据的截断重加权

目标 q(g,y)=p(y)/K，未截断权重 w=q/p。先将权重限于 cap，再在每个 y 内归一化，严格保留表情类总权重。归一化后实际最大权重可能超过预截断 cap，因此一并报告实际最大值。

检查 20×3=60 个数据集×组件；其中 **{rw.loc[~rw.feasible,['dataset','component']].shape[0]} 个**因缺失群体或联合单元不满足支持条件而标记不可行。

FER+ gender：原始 V=0.170962；未截断加权 V≈0，ENS=2、SEI=1；**ESS/N={sub.ESS_fraction:.6f}**，实际最大权重 {sub.max_weight:.6f}。类别边缘保留误差小于 1e-8。这些结果表明目标权重在该统计表上可实现及其样本效率代价；V=0 是目标分布的代数性质，**不证明真实分类器公平**。

只按统计表加权不构成真实 FER 模型缓解验证。实际模型使用时，必须训练集估计权重、未动测试集评估，并根据场景决定目标人口比例，不把均匀比例视为普适最优。

### 6.3 合成 SCM 下游与反事实压力测试

20 随机种子，每种子 4,000 训练、每测试环境 8,000 样本。Y~Bern(.5)，X=2Y−1+N(0,1.25²)，A 在训练选择中与 Y 95% 一致；测试分别为同分布 95%、独立 50%、反转 5%。SCM 声明 X 不受 A 因果影响，因此 do(A) 配对固定 X/Y/噪声成立。比较 Logistic Regression 的 ERM、训练表重加权与 Drop A；共享测试样本。

| 反转环境，20 种子均值 | 准确率 | 最差 (A,Y) 组准确率 | 配对 do(A) 翻转率 |
|---|---:|---:|---:|
| ERM | {syn.loc[('reversed','ERM'),'accuracy']:.4f} | {syn.loc[('reversed','ERM'),'worst_group_accuracy']:.4f} | {syn.loc[('reversed','ERM'),'counterfactual_flip']:.4f} |
| 重加权 | {syn.loc[('reversed','Reweighted'),'accuracy']:.4f} | {syn.loc[('reversed','Reweighted'),'worst_group_accuracy']:.4f} | {syn.loc[('reversed','Reweighted'),'counterfactual_flip']:.4f} |
| Drop A | {syn.loc[('reversed','Drop A'),'accuracy']:.4f} | {syn.loc[('reversed','Drop A'),'worst_group_accuracy']:.4f} | {syn.loc[('reversed','Drop A'),'counterfactual_flip']:.4f} |

反转准确率重加权相对 ERM 的种子配对提升为 **{ci['mean_accuracy_improvement']:.4f}**，以 20 种子为独立重复的 t 区间 **[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]**。不是把全部测试行当作 20 倍独立重复。

**代价同时展示**：同分布 ERM 准确率 {syn.loc[('same','ERM'),'accuracy']:.4f}，重加权 {syn.loc[('same','Reweighted'),'accuracy']:.4f}。重加权牺牲了有利选择相关带来的同分布精度，提升对反转环境的稳健性。Drop A 在该特定 SCM 翻转率为 0，不代表现实删除敏感属性就实现反事实公平；若 X 是 A 的后代或代理，因果反事实必须重新建模。

## 7. 下一步真实图像实验协议

合法获得 AffectNet 图像及合适元信息后，按主体划分数据，复现原图预处理/FairFace 标注；构建四种偏差×11 比例，使用预训练 ResNet50、20 epochs、1cycle、最高学习率 1e-4，每配置 3 重复。训练分割估计受限权重，验证分割选择 cap，在 untouched 测试集报告每性别每类 recall、最差组指标、ESS 与不确定性。真实反事实公平还需明确因果结构和允许路径。

本交付不把这一后续协议当作已经执行，也不以合成结果填补真实图像训练缺口。

## 8. 课堂交付

`slides/论文复现汇报.pptx` 为可编辑 24 页中文 PPT（20 页主讲、4 页问题与答案），演讲者备注与 `docs/演示讲稿.md` 同步，主讲精确规划 25:00，另 5 分钟提问。`docs/答辩问答.md` 包含两个讨论问题及对应参考答案。老师可直接导入镜像执行 `reproduce`；无需 GPU 或联网。
'''
(ROOT/'docs/复现报告.md').write_text(lines,encoding='utf-8')
print('Wrote report')
