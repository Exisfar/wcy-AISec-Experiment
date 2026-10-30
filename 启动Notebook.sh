#!/usr/bin/env bash
# 在宿主机导入随包镜像并启动 Jupyter。
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v docker >/dev/null 2>&1; then
  printf '%s\n' '请先安装并启动 Docker，再运行本脚本。' >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  printf '%s\n' '无法连接 Docker。请确认 Docker 已启动，当前账号可以运行 docker 命令。' >&2
  exit 1
fi
if [ -f dist/fer-bias-repro-image.tar.gz ]; then
  printf '%s\n' '正在导入实验环境，请稍候……'
  docker load -i dist/fer-bias-repro-image.tar.gz
elif ! docker image inspect fer-bias-repro:3.0 >/dev/null 2>&1; then
  printf '%s\n' '没有找到随包镜像。请解压完整交付包后再运行。' >&2
  exit 1
fi
printf '\n%s\n' '即将启动 Jupyter。请打开下面日志中的 http://127.0.0.1:8888/lab?token=... 链接。'
printf '%s\n\n' '进入 notebooks 文件夹，按 01、02、03 查看。使用时保持本终端开启，结束时按 Ctrl+C。'
exec docker run --rm -it -p 127.0.0.1:8888:8888 fer-bias-repro:3.0 lab
