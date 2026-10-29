#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
docker image inspect fer-bias-repro:1.0 > dist/image-inspect.json
docker save fer-bias-repro:1.0 | gzip -1 > dist/fer-bias-repro-image.tar.gz
# 保留完整离线重建材料，排除宿主虚拟环境、Git 内部文件和交付自身。
tar --exclude='./.venv' --exclude='./.pytest_cache' --exclude='./dist' \
  --exclude='./teacher-output' --exclude='*/.git' --exclude='*/__pycache__' \
  --exclude='*/.ipynb_checkpoints' \
  -czf dist/fer-bias-repro-source.tar.gz .
sha256sum dist/fer-bias-repro-image.tar.gz \
  runtime/python.tar.gz runtime/venv.tar.gz runtime/uv \
  'slides/论文复现汇报.pptx' 'slides/论文复现汇报.pdf' > dist/SHA256SUMS
sha256sum dist/fer-bias-repro-source.tar.gz > dist/source-SHA256SUMS
# 单文件课程交付包：镜像、全部源码与可直接打开的演示材料。
python3 scripts/make_submission.py
printf '%s\n' '交付包已生成：dist/课程论文复现_交付包.tar.gz'
