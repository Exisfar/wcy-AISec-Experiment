# 离线重建：基础镜像已在本地；运行时与 uv 创建的项目环境均随包提供。
FROM nvidia/cuda:12.5.1-base-ubuntu20.04@sha256:9608fa115ab6f9b5985f8a05019f707148454ed669791572d95dd9814e582149
LABEL org.opencontainers.image.title="FER Dataset Bias Reproduction" \
      org.opencontainers.image.version="1.0" \
      org.opencontainers.image.description="arXiv:2303.15889v2; 20 datasets; offline notebooks; uv project environment"
WORKDIR /workspace
COPY runtime/uv /usr/local/bin/uv
ADD runtime/python.tar.gz /opt/python/
ADD runtime/venv.tar.gz /workspace/.venv/
COPY pyproject.toml uv.lock README.md LICENSE Dockerfile Dockerfile.online /workspace/
COPY src /workspace/src
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY notebooks /workspace/notebooks
COPY results /workspace/results
COPY docs /workspace/docs
COPY slides /workspace/slides
COPY vendor/dataset_bias_metrics /workspace/vendor/dataset_bias_metrics
COPY vendor/UPSTREAM.json /workspace/vendor/UPSTREAM.json
COPY ["2024_Metrics for Dataset Demographic Bias - A Case Study on Facial Expression Recognition.pdf", "/workspace/"]
ENV PATH="/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin" \
    MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    NVIDIA_VISIBLE_DEVICES=void
RUN chmod +x /workspace/scripts/entrypoint.sh && python scripts/verify_artifacts.py && python -m pytest -q tests
EXPOSE 8888
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
CMD ["help"]
