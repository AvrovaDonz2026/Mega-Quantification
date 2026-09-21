#!/usr/bin/env bash
# Serve a Mega-Quantification export with NVIDIA/SGLang cookbook flags.
# KV that does not fit GPU HBM is offloaded to host RAM (HiCache) so
# context-length can stay 262144 on a 32 GB card.
#
# Usage (repo root):
#   bash scripts/serve_sglang.sh outputs/Qwen3.8-27B-NVFP4-W4A8
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
MODEL="${1:-outputs/Qwen3.8-27B-NVFP4-W4A8}"
shift || true
PY="${MEGAQUANT_PYTHON:-python3}"
EVAL_RECIPE="${EVAL_RECIPE:-$ROOT/recipes/eval-gpqa-diamond.yaml}"
exec "${PY}" -m megaquant.cli serve -c "${EVAL_RECIPE}" --model "${MODEL}" "$@"
