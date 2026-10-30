#!/usr/bin/env bash
set -euo pipefail
cd /workspace
case "${1:-help}" in
  verify)
    python scripts/verify_artifacts.py
    python -m pytest -q tests
    python scripts/appendix_a/verify_results.py --require-complete
    python scripts/appendix_a/verify_mitigation.py --require-complete
    ;;
  reproduce)
    python scripts/run_all.py
    python scripts/compare_paper.py
    python scripts/verify_artifacts.py
    python -m pytest -q tests
    python scripts/appendix_a/verify_results.py --require-complete
    python scripts/appendix_a/verify_mitigation.py --require-complete
    ;;
  lab)
    exec jupyter lab --LabApp.extension_manager=readonly --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace "${@:2}"
    ;;
  appendix-verify)
    python scripts/appendix_a/verify_results.py --require-complete "${@:2}"
    python scripts/appendix_a/verify_mitigation.py --require-complete
    ;;
  appendix-evaluate)
    exec python scripts/appendix_a/evaluate_demo.py "${@:2}"
    ;;
  appendix-train)
    exec python scripts/appendix_a/run.py --config /workspace/configs/appendix_a_kaggle.json --run "${@:2}"
    ;;
  bash|sh)
    exec "$@"
    ;;
  help)
    printf '%s\n' 'FER Bias reproduction | arXiv:2303.15889v2' 'verify     检查数据哈希、Notebook 输出和关键测试' 'reproduce  运行01–03三份实验笔记，保留输出并完成论文对照' 'appendix-verify  核验真实模型网格及混淆矩阵' 'appendix-evaluate  运行真实人脸推理' 'appendix-train  训练/恢复真实模型（需要图片）' 'lab        启动 JupyterLab（8888端口；token见日志）' 'bash       进入容器'
    ;;
  *) exec "$@" ;;
esac
