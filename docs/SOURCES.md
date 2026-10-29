# 来源与版本

1. 本地用户 PDF：正文标题 Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition，arXiv:2303.15889v2（2024-06-25），45 页含附录。https://arxiv.org/abs/2303.15889v2 。期刊 DOI：https://doi.org/10.1109/TPAMI.2024.3361979 。原始文件名误含 Counterfactual Fairness，现已更正为 `2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf`；PDF 内容与 SHA256 未改变。
2. 正文作者源码：https://github.com/irisdominguez/Dataset_Bias_Metrics ，固定 commit `a70bdc28cc96b338f4d5fdebce5064b56d80de57`。`vendor/dataset_bias_metrics/` 为原样快照。作者 MIT 许可证 `vendor/dataset_bias_metrics/LICENSE.txt` 一并保留。
3. 20 个原始匿名统计 CSV：上述仓库的 `example_data/*.csv`。README 说明：只提供无身份信息的人口数据，行序打乱；字段是 age/race/gender/label。没有原始人脸、主体 ID 或图像路径。
4. 用户给的旧仓库：https://github.com/irisdominguez/Dataset-Bias-Metrics 。README 对应 Assessing Demographic Bias Transfer from Dataset to Model: A Case Study in Facial Expression Recognition（arXiv:2205.10049，2022）。本项目不运行该旧项目，也不将它误称本篇正确源码。
5. 统计参考：标准 Pearson C、Cramér V 与 Tschuprow T 使用作者代码中 SciPy `stats.contingency.association` 的定义；bootstrap 与有限样本校正属于本组扩展。有限样本校正采用 Bergsma (2013), A bias-correction for Cramér's V and Tschuprow's T, DOI https://doi.org/10.1016/j.jkss.2012.10.002 。

作者原始 README、实现文件及其注释随包可查。论文印刷数值的逐项目标记录在 scripts/compare_paper.py，输出在 results/tables/paper_comparison.csv。数据 SHA256、Python 版本和上游 commit 记录在 results/provenance.json；完整依赖以 uv.lock 为准。

原论文版权归其作者/出版方；随课程提交保留用户提供 PDF，仅供本次教学验证。不要将 PDF 的随包存在解释成重新授权公开分发。CSV 采用作者公开仓库提供的匿名元数据；原始人脸数据许可不由本项目授予。
