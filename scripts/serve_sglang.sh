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
# nvcc host preprocessor needs cc1plus on PATH (SGLang FP8 JIT / sm_120).
if command -v "${CXX:-g++}" >/dev/null 2>&1; then
  _cc1plus="$("${CXX:-g++}" -print-prog-name=cc1plus 2>/dev/null || true)"
  if [[ -n "${_cc1plus}" && "${_cc1plus}" != "cc1plus" && -x "${_cc1plus}" ]]; then
    export PATH="$(dirname "${_cc1plus}"):${PATH}"
  fi
  export CXX="${CXX:-$(command -v "${CXX:-g++}")}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
fi
cd "$ROOT"
MODEL="${1:-outputs/Qwen3.8-27B-NVFP4-W4A8}"
shift || true
PY="${MEGAQUANT_PYTHON:-python3}"
EVAL_RECIPE="${EVAL_RECIPE:-$ROOT/recipes/eval-gpqa-diamond.yaml}"
exec "${PY}" -m megaquant.cli serve -c "${EVAL_RECIPE}" --model "${MODEL}" "$@"
