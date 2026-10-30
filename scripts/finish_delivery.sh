#!/usr/bin/env bash
# 宿主机编排；项目 Python 与验证均在容器中执行。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
docker build --network none -t fer-bias-repro:3.0 . > dist/build.log 2>&1
container_name="fer-bias-check-$(date +%s)"
trap 'docker rm -f "$container_name" >/dev/null 2>&1 || true' EXIT
docker run --name "$container_name" --network none fer-bias-repro:3.0 reproduce > dist/clean-container.log 2>&1
docker cp "$container_name":/workspace/results/execution.json dist/final-execution-summary.json
docker rm "$container_name"
docker run --rm --gpus all --network none --shm-size=2g fer-bias-repro:3.0 appendix-evaluate --device cuda > dist/gpu-clean-inference.log 2>&1
docker exec "${EXPERIMENT_CONTAINER:-bias-repro-gpu}" /workspace/.venv/bin/python -c 'import json; from pathlib import Path; s=Path("dist/gpu-clean-inference.log").read_text(); d=json.loads(s[s.index("{"):]); assert d["confusion_matches_saved"] and d["real_test_images"] == 2421; print("PASS: isolated offline GPU inference")'
bash scripts/package.sh
date -u +'%Y-%m-%dT%H:%M:%SZ' > dist/DELIVERY_FINISHED
