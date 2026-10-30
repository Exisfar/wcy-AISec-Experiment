# 人口统计偏差论文复现

复现论文：**Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition**（TPAMI 2024）。

本项目包含三份已经运行完成的 Jupyter Notebook，保存了代码、中文说明、图表和实验输出。Python 环境、依赖、数据集和模型均已放入 Docker 镜像，不需要另外安装或下载。

## 怎么打开

需要一台已安装并启动 Docker 的 Linux x86_64 电脑。查看 Notebook 和运行统计验证不需要 GPU。

1. 解压 `课程论文复现_交付包.tar.gz`。
2. 在解压得到的 `fer-bias-reproduction` 文件夹中打开终端，运行：

   ```bash
   bash 启动Notebook.sh
   ```

3. 等待镜像导入。终端出现带有 `token` 的 `http://127.0.0.1:8888/lab?...` 链接后，在浏览器中打开。
4. 进入 `notebooks` 文件夹，按 01、02、03 的顺序查看。

脚本会自动完成镜像导入和 Jupyter 启动。Notebook 已有完整输出，可以直接阅读；如需重新执行，在 Jupyter 中选择 **Run → Run All Cells**。使用过程中保持终端窗口开启。

## 三份 Notebook 看什么

| 文件 | 内容 |
|---|---|
| `01_二十个真实数据集完整指标复现.ipynb` | 20个数据集的偏差指标、作者代码对照及论文主要统计结果 |
| `02_真实人脸表情识别模型训练验证.ipynb` | 四类偏差设置下的132次真实模型训练记录、分组召回率与3次配对重加权对照 |
| `03_不足分析与改进探索.ipynb` | 群体覆盖、标签敏感性、不确定性、重加权及合成实验 |

执行三份 Notebook 会重算统计结果并核对训练记录，不会自动重新训练135个模型。02中另有默认关闭的重训入口。

## 想一次性重新验证

在解压后的项目文件夹中运行：

```bash
bash scripts/teacher_verify.sh
```

脚本自动导入镜像、执行三份 Notebook 和测试，并将带有新输出的 Notebook 保存到 `teacher-output/notebooks/`。终端显示“验证成功”即完成。Notebook 执行在本机实测约半分钟，首次导入镜像另需时间。

## 数据和代码在哪里

解压后能直接查看 `notebooks/`、`src/`、`scripts/`、`configs/`、`tests/`、`results/` 和论文 PDF。

完整运行环境在 `dist/fer-bias-repro-image.tar.gz` 中，由启动脚本自动导入，不需要手动解压。容器中的 `/workspace/data/` 统一保存匿名CSV、真实人脸图片和模型权重；Python 环境位于 `/workspace/.venv/`。

`src/` 包含指标、采样、训练和重加权代码；`tests/` 检查关键公式与实验规则；`results/` 保留训练记录和验证所需的数据，主要结论直接在 Notebook 中展示。作者代码与数据来源见 `results/SOURCES.md`。

## 实验范围

统计部分使用作者公开的20份匿名CSV，共794,166条记录，完成1,280项指标实现对照。图像实验使用从Kaggle下载的压缩数据，完成132次ResNet50训练和3次重加权训练。

图像数据的分辨率、划分、样本规模及人口属性标注与原论文条件存在差异。主要现象支持论文的定性描述，具体数值和限制见 Notebook。真实模型重新训练或GPU推理需要相应的NVIDIA GPU与容器支持。
