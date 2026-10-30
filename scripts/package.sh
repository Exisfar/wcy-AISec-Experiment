#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
experiment_container="${EXPERIMENT_CONTAINER:-bias-repro-gpu}"
docker image inspect fer-bias-repro:3.0 > dist/image-inspect.json
docker save fer-bias-repro:3.0 | gzip -1 > dist/fer-bias-repro-image.tar.gz
docker exec "$experiment_container" /workspace/.venv/bin/python /workspace/scripts/make_submission.py
printf '%s\n' '交付包已生成：dist/课程论文复现_交付包.tar.gz'
