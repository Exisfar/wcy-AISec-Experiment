#!/usr/bin/env bash
# 仅在宿主机排版；读取既有实验结果，不运行统计实验、Notebook 或训练。
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /.dockerenv ]; then
  printf '%s\n' '请在宿主机运行本脚本；实验代码继续在 Docker 内运行。' >&2
  exit 2
fi
uv run --project presentation --frozen python presentation/make_slides.py
uv run --project presentation --frozen python presentation/check_slides.py
