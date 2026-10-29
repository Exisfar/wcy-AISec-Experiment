#!/usr/bin/env bash
set -euo pipefail
cd /workspace
case "${1:-help}" in
  verify)
    python scripts/verify_artifacts.py
    python -m pytest -q tests
    ;;
  reproduce)
    python scripts/run_all.py
    python scripts/compare_paper.py
    python scripts/verify_artifacts.py
    python -m pytest -q tests
    ;;
  lab)
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace "${@:2}"
    ;;
  slides)
    exec python scripts/make_slides.py
    ;;
  bash|sh)
    exec "$@"
    ;;
  help)
    printf '%s\n' 'FER Bias reproduction | arXiv:2303.15889v2' 'verify     检查数据哈希、Notebook 输出和关键公式测试（秒级）' 'reproduce  离线重跑三个 Notebook，导出 HTML 与论文对照' 'lab        启动 JupyterLab（8888 端口；token 见日志）' 'bash       进入容器'
    ;;
  *) exec "$@" ;;
esac
