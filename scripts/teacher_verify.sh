#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image='fer-bias-repro:3.0'
if [ -f dist/fer-bias-repro-image.tar.gz ]; then
  docker load -i dist/fer-bias-repro-image.tar.gz
elif ! docker image inspect "$image" >/dev/null 2>&1; then
  printf '%s\n' '缺少交付镜像，请先解压完整交付包。' >&2
  exit 2
fi
mkdir -p teacher-output
container_name="fer-bias-check-$(date +%s)"
trap 'docker rm -f "$container_name" >/dev/null 2>&1 || true' EXIT
# 禁网、无 GPU、不挂载宿主项目：验证结果确实来自镜像内完整环境。
docker run --name "$container_name" --network none "$image" reproduce
docker cp "$container_name":/workspace/results teacher-output/
docker cp "$container_name":/workspace/notebooks teacher-output/
printf '%s\n' '验证成功：结果已保存至 teacher-output/'
