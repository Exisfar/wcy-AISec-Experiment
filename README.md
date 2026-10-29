# 人口统计偏差论文复现：课程交付包

**论文文件名已更正**：当前文件为 [2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf](<2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf>)。原始文件名误含 Counterfactual Fairness，已按正文标题更正；实际论文为 **Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition**（Dominguez-Catena 等，arXiv:2303.15889v2 / TPAMI 2024）。本项目以该 PDF 正文为复现对象。正确作者仓库是 [Dataset_Bias_Metrics](https://github.com/irisdominguez/Dataset_Bias_Metrics)（下划线）；用户提供的连字符仓库对应另一篇 2022 年论文。

交付包含：3 份已执行、带中文注释的 Jupyter Notebook；20 个真实匿名统计 CSV；完整指标和图表；24 页中文 PPT（20 页主讲 + 2 组问题/答案，共 4 页）；25 分钟逐页照读讲稿；源码、uv 锁文件及可离线运行镜像。**未重跑原图预处理/FairFace 标注和 AffectNet ResNet50 图像训练**；附录 A 只验证列联表与指标面板，合成下游结果属于本组扩展。

## 给老师：最快验证

需要 Docker，环境为 Linux amd64；Windows/macOS 可用 Docker Desktop。无需 GPU，无需 Python，无需联网。Apple Silicon 可添加 `--platform linux/amd64` 使用仿真，运行时间会变长。先进入解压后的项目目录。

```bash
# 检查交付文件（推荐）
sha256sum -c dist/SHA256SUMS

# 导入完整镜像：不需要重新拉基础镜像或下载依赖
docker load -i dist/fer-bias-repro-image.tar.gz

# 秒级检查：20 数据文件 SHA256、Notebook 已执行、关键公式测试
docker run --rm --network none fer-bias-repro:1.0 verify

# 完整离线重跑：3 个 Notebook + HTML + 论文数值对照 + 测试
docker run --rm --network none fer-bias-repro:1.0 reproduce
```

`reproduce` 完成时显示 `ALL NOTEBOOKS PASSED`；论文数值对照还会打印不吻合项。局部相关的稳健并列处理与原文聚合有差异，详见结果报告；这不被隐藏或当作运行失败。普通 CPU 上 Notebook 本体约 30–60 秒，导入大镜像的解压时间另算，以当前机器为准。

如需将新结果保存到宿主机，执行：

```bash
bash scripts/teacher_verify.sh
```

脚本自动导入缺失镜像，在**禁网、无 GPU、无项目挂载**的新容器完整执行，将结果与 Notebook 复制到 `teacher-output/`。执行失败会返回非零状态。

## 交互打开 Notebook

```bash
bash scripts/open_lab.sh
# 或：docker run --rm -it -p 127.0.0.1:8888:8888 fer-bias-repro:1.0 lab
```

打开终端日志中的 `http://127.0.0.1:8888/lab?token=...`。保留 Jupyter 自动 token；镜像里的数据、项目 `.venv`、Notebook 均已就绪。三份 Notebook 按 00→01→02 阅读，选择 Run All 可逐单元验证。网页中编辑是临时容器修改，退出即丢弃；保存结果使用上面的复制脚本。

没有 Docker 时，可直接阅读 `results/00_*.html`、`results/01_*.html`、`results/02_*.html` 与 `slides/论文复现汇报.pdf`。HTML 图片已内嵌，支持离线阅读；数学公式在部分浏览器中可能显示原始 LaTeX，结果图表不依赖外网。

## 文件索引

| 文件/目录 | 用途 |
|---|---|
| `notebooks/00_论文核对与公式审计.ipynb` | 身份、来源、定义、手算反例与公式审计 |
| `notebooks/01_二十个真实数据集完整指标复现.ipynb` | 正文主要统计、附录 B、附录 A 指标曲线 |
| `notebooks/02_不足分析与改进探索.ipynb` | 覆盖率、不确定性、重加权、合成反事实压力测试 |
| `slides/论文复现汇报.pptx` / `.pdf` | 可编辑中文汇报 / 无环境演示备份 |
| `docs/PPT演讲稿.docx` | 可直接照读和打印的 Word 演讲稿 |
| `docs/演示讲稿.md` | 20 页主汇报逐页计时，合计 25:00；4 页问题与答案，仅两个问题 |
| `docs/答辩问答.md` | 两个讨论问题，问题与答案各占一页 |
| `docs/复现报告.md` | 实际数值、论文对照、不足与改进结果、未验证项 |
| `results/tables/` / `results/local/` | 全部数值与 60 套局部矩阵 |
| `results/figures/` | 正文重绘、20 数据集附录图和改进实验图 |
| `results/provenance.json` | 数据哈希、版本、记录数、上游 commit |
| `dist/clean-container.log` | 禁网新容器完整复现证据 |
| `vendor/dataset_bias_metrics/` | 作者原样源码、匿名 CSV、许可证 |
| `src/` / `scripts/` / `tests/` | 独立实现、生成/执行/交付脚本与关键测试 |
| `runtime/` | 精确 Python、uv 与项目虚拟环境离线重建材料 |

## 从源码重新构建镜像

用户拉好的基础镜像为 `nvidia/cuda:12.5.1-base-ubuntu20.04`。本 Dockerfile 锁定其内容摘要；离线构建前需本地已有该镜像，或先 `docker load` 导入随包完整镜像。重建过程无需 apt/pip/uv 网络下载：

```bash
docker build --network none -t fer-bias-repro:1.0 .
```

`runtime/python.tar.gz` 为 CPython 3.11.15；`runtime/venv.tar.gz` 为**在容器里通过 uv 创建**的 `/workspace/.venv`，依赖以 `uv.lock` 为准。Dockerfile 将它们恢复到固定路径，不依赖宿主机虚拟环境。`Dockerfile.online` 供希望从 uv 锁文件重新安装依赖的人使用（需要网络；若自行更新锁文件，结果可能改变）。

```bash
# 在联网构建中使用可访问的代理地址（Linux 可用 --network host + 127.0.0.1）
docker build --network host -f Dockerfile.online \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
  -t fer-bias-repro:from-lock .
```

注意宿主 `.venv` 内解释器固定 `/opt/python`，请通过容器运行；直接在宿主激活不适用。导出运行环境为 Linux amd64，不应把其中二进制复制到其他架构直接使用。

## 复现定义与扩展

- 10 类表征指标、5 类全局关联指标（U 双方向）、2 类局部指标，共 17 类。20×4×10 + 20×3×6 + 20×3×2 = **1,280** 项作者交叉核验（局部每项是整矩阵）。
- 全局相关在 age/race/gender 内分别计算后平均；表征相关另包含 label。NMI 使用论文的 `I/H(G,Y)`，不套 sklearn 默认定义。
- 单组退化 V/SEI 保留 NaN；非零边缘中的零联合单元 NPMI=-1。统计独立不等于因果公平。
- 扩展使用真实元数据探索截断重加权，并明确固定群体空间、交叉支持和 ESS。20 随机种子合成模型实验声明 SCM；结果不冒充 AffectNet 真实训练结果。
- 全部公式、限制、数值对照、来源与许可见 `docs/`。原始 CSV 只读，数据 SHA256 随结果保存。
