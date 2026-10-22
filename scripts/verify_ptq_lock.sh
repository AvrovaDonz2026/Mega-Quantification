#!/usr/bin/env bash
set -euo pipefail
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}
report=${PIP_REPORT:-"$repo_root/ptq-resolver-report.json"}
exec "$python_bin" -m pip install \
  --dry-run --ignore-installed --report "$report" --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -c "$repo_root/docker/constraints-x86-cuda128-py312.txt" \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  'nvidia-modelopt[hf]==0.46.1' 'llmcompressor==0.8.1' \
  'compressed-tensors==0.10.2' 'transformers==5.14.1' \
  'accelerate==1.4.0' 'datasets==3.3.2'
