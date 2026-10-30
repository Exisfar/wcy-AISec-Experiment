"""生成02的代码与说明模板；执行脚本会替换该笔记，已有输出需重新执行。"""
from pathlib import Path
import nbformat as nbf
from notebook_text import normalize_notebook, MAIN_IMAGE_NAME
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md('''# 02 · 真实人脸表情识别模型训练验证

数据、训练配置、日志与结果。'''),code('''from pathlib import Path
import sys, json, subprocess
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path.cwd() if (Path.cwd()/'configs').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT/'src'))
from appendix_a.data import CLASSES, GENDERS, required_counts, validate_manifest, sha256_file
config_path=ROOT/'configs/appendix_a_kaggle.json'
config=json.loads(config_path.read_text())
# 结果目录相对于当前项目定位。
results=ROOT/'results/appendixA_kaggle'
provenance=json.loads((results/'data_provenance.json').read_text())
display(provenance)
assert provenance['original_paper_data_match'] is False
assert provenance['smoke_test_only'] is False
manifest_path=results/'input_manifest.csv'
assert sha256_file(manifest_path)==provenance['manifest_sha256']
manifest=validate_manifest(pd.read_csv(manifest_path),check_files=False)
display(manifest.groupby('split').size().rename('本次实际使用的图片数'))
'''),md('''## 1. 数据来源、去重和人口属性

原始下载包含 25,262 张 96×96 图片，8 种情绪。按论文七类任务剔除 contempt，对原始文件 SHA256 做冲突与重复审计，保留发布者分割；测试/验证优先，避免相同文件泄漏进训练。这个检查**不等于主体互斥**：源数据没有主体 ID，相同人的不同照片仍可能跨分割。

保留训练 14,903 张、验证 4,764 张、测试 2,421 张。验证集不并入训练，也不用于选 epoch。FairFace 20190809 7-race ResNet34 在 Docker GPU 环境中为每张图片重新推断表观人口属性。只按同一张图片的路径关联标签，绝不将作者打乱的匿名 CSV 与图片按行拼接。

性别类别是 FairFace 的二元预测代理，不是个体自报身份；96 像素输入可能影响人口属性推断质量，未获得独立人口真值验证。'''),code('''train_counts=pd.read_csv(results/'source_training_counts.csv',index_col=0).reindex(index=GENDERS,columns=CLASSES)
test_counts=pd.read_csv(results/'test_counts.csv',index_col=0).reindex(index=GENDERS,columns=CLASSES)
display(Markdown('**FairFace 预测后的训练支持量**'));display(train_counts)
display(Markdown('**保留测试集支持量**'));display(test_counts)
assert train_counts.to_numpy().sum()==14903
assert test_counts.to_numpy().sum()==2421
assert (test_counts>0).all().all()
assert train_counts.min(axis=0).astype(int).to_dict()==config['class_totals']
'''),md('''## 2. 四类偏差与精确采样

每类目标规模取训练集中该类别男女样本数的较小值，保证极端比例也可无放回采样。整数算术实现 `floor(n*q)`，避免浮点取整误差；所有比例点保持同一类规模规则，因取整导致少量总数差异与原文构造一致。

1. 各类同时改变性别比例：主要影响表征偏差。
2. 只改变 happy。
3. happy 与 angry 向相反方向改变。
4. happy 与 angry 向同一方向改变。

每个配置三次独立采样/初始化。无放回采样；缺少支持时报错，不用复制样本或测试集填充。'''),code('''plan=pd.read_csv(results/'plan.csv')
assert len(plan)==132
assert plan[['experiment','step','repetition']].drop_duplicates().shape[0]==132
assert sorted(plan['step'].unique())==list(range(11))
display(plan.groupby('experiment').agg(runs=('repetition','size'),min_train=('n_train','min'),max_train=('n_train','max')))
for exp in [1,2,3,4]:
    display(Markdown(f'**实验 {exp}：q=0 与 q=1 的计数**'))
    display(required_counts(exp,0,config['class_totals']))
    display(required_counts(exp,10,config['class_totals']))
'''),md('''## 3. 训练协议与可追溯性

论文明确提供：预训练 ResNet50、20 epochs、1cycle、最高学习率 1e-4、三次重复、按性别和类别统计 recall。本实现对论文未充分给出的细节做显式选择：ImageNet-1K V1 权重、7 类线性头、训练整个骨干、AdamW、batch=256、weight_decay=0.01、ImageNet 归一化、随机水平翻转、AMP 与 channels-last。

每次训练使用同一测试集，仅评估最终 epoch，不以测试分数选择检查点。每个 epoch 保存 RNG、优化器、调度器和采样器状态以恢复。已完成项校验数据/配置/实现/结果哈希后跳过。正式统一协议与早期 batch=128 吞吐试跑分开存放，不混合统计。

下面从 `run.json` 和逐 epoch 日志检查已实际完成的配置；不是只展示训练代码。'''),code('''completed=[]
for path in sorted(results.glob('exp*_q*_rep*/complete.json')):
    item=json.loads(path.read_text())
    assert item['epochs']==20 and item['smoke_test_only'] is False
    completed.append(item)
print(f'真实完成：{len(completed)} / 132')
print('状态：', '完整网格' if len(completed)==132 else 'PARTIAL，仍有配置未完成')
if completed:
    first=results/completed[0]['run']
    run=json.loads((first/'run.json').read_text())
    display({k:run[k] for k in ['torch','torchvision','cuda','gpu','epochs','batch_size','max_lr','manifest_sha256','pretrained_sha256','original_paper_data_match']})
    history=pd.read_csv(first/'history.csv')
    assert history.epoch.tolist()==list(range(1,21))
    display(history)
'''),md('''## 4. 用混淆矩阵独立重算 recall

对每个性别和表情，`recall=TP/该性别该类的测试样本数`。macro recall 是七类召回率的算术平均，不能被总体准确率或按样本数加权平均替代。保存的混淆矩阵按 `(gender, true_class, predicted_class)` 排列。

核验包括：132 个预期配置是否齐全；每项确有 20 个 epoch；训练列联表是否满足相应比例；所有配置的测试支持量是否相同；CSV 每个 recall 是否与混淆矩阵完全一致；结果文件是否通过 SHA256。

从保存的混淆矩阵重新计算召回率，并对照日志与结果文件。'''),code('''cmd=[sys.executable,str(ROOT/'scripts/appendix_a/verify_results.py'),'--config',str(config_path)]
# 完整网格按132项严格核验；缺项时验证脚本会报错。
if len(completed)==132:cmd.append('--require-complete')
result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=True)
print(result.stdout)
'''),md('''## 5. 真实图像召回率曲线

下面的 (b)(c) 来自上面真实训练模型在保留测试集上的预测，不是合成数据结果。曲线按同一配置三个种子求平均，阴影表示重复间样本标准差，**不是置信区间**。如果网格不齐，图上标记 PARTIAL；不补点、不把局部观察外推为完整趋势。'''),code('''result=subprocess.run([sys.executable,str(ROOT/'scripts/appendix_a/summarize.py'),'--config',str(config_path)],cwd=ROOT,text=True,capture_output=True,check=True)
print(result.stdout)
for exp in [1,2,3,4]:
    figure=results/f'figure_{8+exp:02d}_recall.png'
    if figure.exists():display(Image(filename=str(figure)))
if (results/'recall_summary.csv').exists():
    summary=pd.read_csv(results/'recall_summary.csv')
    display(summary[(summary.fraction.isin([0,.5,1]))&(summary.label.isin(['happy','angry','mean']))])
'''),md('''## 6. 本组扩展：真实分类器的配对重加权对照

在主网格结束后，固定实验 2 的 q=.1 子集，额外训练三个重加权模型。它们使用与各自 ERM 基线完全相同的样本、初始化种子、优化器、20 epochs 和测试集；只改变训练损失里的群体×类别权重。先限权重 cap=5，再按每个类别归一以保持类别总权重。cap 在测试前固定，不做测试集调参。

同时报告两性别的 happy recall、七类 macro recall、整体准确率和 ESS，不能只选改善的格子。3 个配对重复用于描述探索结果，不能据此作普适改善或因果公平保证。'''),code('''mitigation=ROOT/'results/appendixA_mitigation'
if (mitigation/'FINISHED.json').exists():
    checked=subprocess.run([sys.executable,str(ROOT/'scripts/appendix_a/verify_mitigation.py'),'--require-complete'],cwd=ROOT,text=True,capture_output=True,check=True)
    print(checked.stdout)
    paired=pd.read_csv(mitigation/'summary.csv')
    display(paired[paired.label.isin(['happy','mean'])])
    display(Image(filename=str(ROOT/'results/figures/appendixA_real_mitigation.png')))
else:
    print('真实模型重加权对照尚未全部完成；不显示假定收益。')
'''),md('''## 7. 快速重跑一个真实模型的预测

完整网格重训需要数小时。另导出一个已完成的均衡模型（实验 1，q=.5，种子 0），用于快速重跑 2,421 张真实测试图片的预测。模型权重、测试 manifest 与预期混淆矩阵均记录 SHA256；图片不随源码或 Git 分发，按同一 Kaggle 版本获取并校验。

下面为已保存的真实测试集推理记录。'''),code('''demo=ROOT/'results/appendixA_demo/evaluation.json'
if demo.exists():
    replay=json.loads(demo.read_text());display(replay)
    assert replay['real_test_images']==2421
    assert replay['original_paper_data_match'] is False
    if replay['device']=='cuda':assert replay['confusion_matches_saved'] is True
else:
    print('快速推理复验记录尚未生成。')
'''),md('''## 8. 从 Notebook 启动真正的训练（默认关闭）

下面是可执行的训练入口，不只是命令截图。训练开关默认关闭。需要真实图片和完整 GPU 环境；输出使用新目录，避免把“跳过已有完成项”误称为重新训练。

将 `RUN_REAL_TRAINING=True` 可实际执行；`MAX_NEW_RUNS=None` 表示完整 132 项，设为正整数可只做限时运行，但这类输出必须标为部分实验。'''),code('''RUN_REAL_TRAINING=False
MAX_NEW_RUNS=None
if RUN_REAL_TRAINING:
    gpu_python=ROOT/'.venv/bin/python'
    if not gpu_python.exists():raise RuntimeError('需要完整 GPU 容器与真实图片；请检查训练环境。')
    cmd=[str(gpu_python),str(ROOT/'scripts/appendix_a/run.py'),'--config',str(config_path),'--run',
         '--output-dir',str(ROOT/'results/retrain_from_notebook'),
         '--checkpoint-dir',str(ROOT/'checkpoints/retrain_from_notebook')]
    if MAX_NEW_RUNS is not None:cmd.extend(['--max-runs',str(MAX_NEW_RUNS)])
    subprocess.run(cmd,cwd=ROOT,check=True)
else:
    print('已完成的训练记录见上方。')
'''),md('''## 9. 恢复现有训练与从头重训

以下命令在宿主机调用 Docker，Python 实际在 GPU 容器内执行。该命令恢复现有作业，完成项会跳过；从头重训应使用上一节的新输出目录。完整网格使用独立的输出目录。交付 GPU 镜像的数据准备和从零运行方法见 项目 `README.md`。

```bash
docker exec -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=8 bias-repro-gpu \\
  .venv/bin/python scripts/appendix_a/run.py \\
  --config configs/appendix_a_kaggle.json --run
```

数据获取、转换、FairFace 推断分别见 `scripts/appendix_a/download_parallel.py`、`prepare_kaggle.py`；训练代码见 `src/appendix_a/trainer.py`，采样/召回率定义见 `src/appendix_a/data.py`。所有实验共用根目录的uv.lock。

**结果边界**：只对本加工版数据与其代理人口属性作结论。源图 96×96、非原始官方划分、缺少主体标识、FairFace 的预测误差，以及优化器/增广等未完全由原文指定的实现细节，均限制其与原文 recall 数值的可比性。不能把这些曲线标为“原始 AffectNet 精确复现”。''')]
next(c for c in reversed(cells) if c.cell_type=="code").source += "\n\nfrom result_files import cleanup_results\ncleanup_results()"
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
path=ROOT/'notebooks'/MAIN_IMAGE_NAME
nb=normalize_notebook(nb,path.name)
nbf.write(nb,path);print('Wrote',path.name)
