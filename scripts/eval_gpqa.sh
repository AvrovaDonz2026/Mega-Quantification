#!/usr/bin/env bash
# GPQA Diamond against a served NVFP4 export (OpenAI-compatible SGLang).
# Does not truncate: remaining context + continue_on_length; KV → RAM via HiCache.
#
# Usage:
#   MEGAQUANT_SGLANG_BASE_URL=http://127.0.0.1:30000/v1 \
#     bash scripts/eval_gpqa.sh outputs/Qwen3.8-27B-NVFP4-W4A8
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-}"
shift || true
PY="${MEGAQUANT_PYTHON:-python3}"
EVAL_RECIPE="${EVAL_RECIPE:-$ROOT/recipes/eval-gpqa-diamond.yaml}"
ARGS=(-c "${EVAL_RECIPE}")
if [[ -n "${MODEL}" ]]; then
  ARGS+=(--model "${MODEL}")
fi
exec "${PY}" -m megaquant.cli eval "${ARGS[@]}" "$@"
