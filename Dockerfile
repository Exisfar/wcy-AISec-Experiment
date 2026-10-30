# 单一离线镜像：Python、uv 项目环境、数据、模型、Notebook 与结果全部随包提供。
FROM nvidia/cuda:12.5.1-base-ubuntu20.04@sha256:9608fa115ab6f9b5985f8a05019f707148454ed669791572d95dd9814e582149
LABEL org.opencontainers.image.title="FER Dataset Bias Reproduction" \
      org.opencontainers.image.version="3.0"
WORKDIR /workspace
COPY runtime/uv /usr/local/bin/uv
ADD runtime/python.tar.gz /opt/python312/
ADD runtime/venv.tar.gz /workspace/.venv/
COPY pyproject.toml uv.lock README.md Dockerfile /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY notebooks /workspace/notebooks
COPY results /workspace/results
COPY data/demographics /workspace/data/demographics
COPY data/models /workspace/data/models
# 包含当前实验所用图像，推理无需联网或额外挂载。
COPY data/kaggle-affectnet-yolo/extracted /workspace/data/kaggle-affectnet-yolo/extracted
COPY ["2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf", "/workspace/"]
ENV PATH="/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 UV_LINK_MODE=copy NVIDIA_VISIBLE_DEVICES=all
RUN chmod +x scripts/entrypoint.sh \
    && python scripts/verify_artifacts.py && python -m pytest -q tests \
    && python scripts/appendix_a/verify_results.py --require-complete \
    && python scripts/appendix_a/verify_mitigation.py --require-complete
EXPOSE 8888
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["lab"]
