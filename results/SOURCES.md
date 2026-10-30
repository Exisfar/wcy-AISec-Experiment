# 来源与版本

1. 本地用户 PDF：正文标题 Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition，arXiv:2303.15889v2（2024-06-25），45 页含附录。https://arxiv.org/abs/2303.15889v2 。期刊 DOI：https://doi.org/10.1109/TPAMI.2024.3361979 。
2. 作者指标源码：https://github.com/irisdominguez/Dataset_Bias_Metrics ，固定 commit `a70bdc28cc96b338f4d5fdebce5064b56d80de57`。使用到的模块统一保存于 `src/dataset_bias_metrics/`，原许可文本集中记录如下，不另设许可证文件。
3. 20 个原始匿名统计 CSV：源自上述仓库的 `example_data/*.csv`，项目内统一存放于 `data/demographics/`（文件内容及SHA256不变）。README 说明：只提供无身份信息的人口数据，行序打乱；字段是 age/race/gender/label。没有原始人脸、主体 ID 或图像路径。
5. 统计参考：标准 Pearson C、Cramér V 与 Tschuprow T 使用作者代码中 SciPy `stats.contingency.association` 的定义；bootstrap 与有限样本校正属于本组扩展。有限样本校正采用 Bergsma (2013), A bias-correction for Cramér's V and Tschuprow's T, DOI https://doi.org/10.1016/j.jkss.2012.10.002 。

作者原始 README、实现文件及其注释随包可查。论文印刷数值的逐项目标记录在 scripts/compare_paper.py，输出在 results/tables/paper_comparison.csv。数据 SHA256、Python 版本和上游 commit 记录在 results/provenance.json；完整依赖以 uv.lock 为准。

原论文版权归其作者/出版方；随课程提交保留用户提供 PDF，仅供本次教学验证。不要将 PDF 的随包存在解释成重新授权公开分发。CSV 采用作者公开仓库提供的匿名元数据；原始人脸数据许可不由本项目授予。

## 新增：真实人脸加工版实验

- 用户明确指定并选择使用 Kaggle `fatihkgg/affectnet-yolo-format` 第 2 版：https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format 。原始数据 96×96，发布者重新划分 train/valid/test，不能与论文原始 AffectNet 视为同一实验条件。
- Kaggle 描述、版本及文件清单通过公开 Kaggle API 核对；ZIP SHA256 与实际图片/类别审计见 `results/appendixA/kaggle_dataset_audit.json`、`results/appendixA_kaggle/data_provenance.json`。
- FairFace 官方代码及公开模型：https://github.com/dchen236/FairFace 、https://github.com/joojs/fairface 。本轮使用公开的 20190809 7-race ResNet34，为每张图片重新预测表观人口属性；出处与模型 SHA256 另行记录。人口预测不是身份自报数据。
- ImageNet 预训练 ResNet50 V1：https://download.pytorch.org/models/resnet50-0676ba61.pth 。SHA256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`。
- 所有实验依赖统一在容器根目录 uv 项目：Python 3.12.13、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124；统计与 Notebook 同样使用该 Python 3.12 环境。CPU 可验证已保存的混淆矩阵；真实图像重训与重新推理需要输入图片，完整重训还需要 GPU。
- 官方原始 AffectNet 当前通过 AffectNet+ 申请渠道提供：https://www.mohammadmahoor.com/pages/databases/affectnetplus/ 。本轮没有将 从Kaggle下载的压缩数据标作原始官方数据精确复现。

## 作者指标源码原许可文本

```text
Copyright (c) 2023 Iris Dominguez-Catena

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
