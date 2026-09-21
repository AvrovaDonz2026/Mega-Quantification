#!/usr/bin/env bash
# Mega-Quantification container entrypoint.
# Usage: docker compose run --rm megaquant [plan|quantize|eval|serve|schemes|families|bash] ...
set -euo pipefail

log() { printf '[megaquant] %s\n' "$*"; }

print_gpu_report() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found. This container needs --gpus all / Compose device reservations."
    return 0
  fi
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader || true

  local total_mib=0
  local cap
  while IFS=',' read -r _idx _name cap mem; do
    cap="$(echo "${cap}" | xargs)"
    mem="$(echo "${mem}" | awk '{print $1}')"
    if [[ "${mem}" =~ ^[0-9]+$ ]]; then
      total_mib=$((total_mib + mem))
    fi
    if [[ "${cap}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      python3 - "${cap}" <<'PY' || true
import sys
cap = float(sys.argv[1])
if cap < 10.0:
    print("[megaquant] compute capability < 10.0: NVFP4 *kernels* need Blackwell (sm_100/sm_120). PTQ fake-quant can still run.")
elif cap < 12.0:
    print("[megaquant] Blackwell-class GPU detected (Hopper/B100 family). NVFP4 is supported.")
else:
    print("[megaquant] sm_120+ detected (RTX 5090 / Blackwell consumer). NVFP4 W4A8 PTQ is supported.")
PY
    fi
  done < <(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)

  if (( total_mib > 0 && total_mib < 45000 )); then
    log "visible VRAM is ${total_mib} MiB. Qwen3.8-27B BF16 is ~54 GB; use MEGAQUANT_LOW_MEMORY=1, device_map=auto, or 2+ GPUs."
  fi
}

cd "${MEGAQUANT_ROOT:-/opt/megaquant}"

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" || "${1:-}" == "/bin/bash" ]]; then
  exec "$@"
fi

if [[ "${1:-}" == "host-check" ]]; then
  exec /opt/megaquant/docker/host-check.sh
fi

# SGLang FP8 JIT on sm_120 needs host g++/cc1plus on PATH.
if command -v "${CXX:-g++}" >/dev/null 2>&1; then
  _cc1plus="$("${CXX:-g++}" -print-prog-name=cc1plus 2>/dev/null || true)"
  if [[ -n "${_cc1plus}" && "${_cc1plus}" != "cc1plus" && -x "${_cc1plus}" ]]; then
    export PATH="$(dirname "${_cc1plus}"):${PATH}"
  fi
  export CXX="${CXX:-$(command -v "${CXX:-g++}")}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
fi
if command -v "${CC:-gcc}" >/dev/null 2>&1; then
  export CC="${CC:-$(command -v "${CC:-gcc}")}"
fi

if [[ "${MEGAQUANT_SKIP_GPU_REPORT:-0}" != "1" ]]; then
  print_gpu_report
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ -n "${MEGAQUANT_DEVICE_MAP:-}" ]]; then
  export MEGAQUANT_DEVICE_MAP
fi
if [[ -n "${MEGAQUANT_LOW_MEMORY:-}" ]]; then
  export MEGAQUANT_LOW_MEMORY
fi
if [[ -n "${MEGAQUANT_OFFLOAD_DIR:-}" ]]; then
  export MEGAQUANT_OFFLOAD_DIR
fi
if [[ -n "${MEGAQUANT_MAX_MEMORY:-}" ]]; then
  export MEGAQUANT_MAX_MEMORY
fi

DEFAULT_RECIPE="${RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml}"

_has_config_flag() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      -c|--config|-c=*|--config=*) return 0 ;;
    esac
  done
  return 1
}

# Default: if the user passes nothing, run the recipe plan (dry-run).
# `docker compose run --rm megaquant plan` replaces CMD with just `plan`,
# and the CLI requires -c/--config — append the recipe in that case too.
if [[ $# -eq 0 ]]; then
  set -- plan -c "${DEFAULT_RECIPE}"
elif [[ "${1}" == "plan" || "${1}" == "quantize" ]]; then
  if ! _has_config_flag "$@"; then
    set -- "$@" -c "${DEFAULT_RECIPE}"
  fi
elif [[ "${1}" == "eval" || "${1}" == "serve" ]]; then
  if ! _has_config_flag "$@"; then
    set -- "$@" -c "${EVAL_RECIPE:-recipes/eval-gpqa-diamond.yaml}"
  fi
fi

# Allow `docker compose run megaquant --dry-run` style passthrough plus extras.
# shellcheck disable=SC2206
extra=( ${MEGAQUANT_EXTRA_ARGS:-} )

if python -c "import megaquant.cli" >/dev/null 2>&1; then
  exec python -m megaquant.cli "$@" "${extra[@]}"
fi

log "megaquant CLI is not importable yet. Falling back to help."
log "PYTHONPATH=${PYTHONPATH:-}"
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device0", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import failed:", exc)
try:
    import modelopt
    print("modelopt", getattr(modelopt, "__version__", "present"))
except Exception as exc:
    print("modelopt import failed:", exc)
PY
exec python -m megaquant.cli "$@" "${extra[@]}"
