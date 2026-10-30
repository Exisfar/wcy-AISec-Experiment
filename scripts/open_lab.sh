#!/usr/bin/env bash
set -euo pipefail
docker run --rm -it -p 127.0.0.1:8888:8888 fer-bias-repro:3.0 lab
