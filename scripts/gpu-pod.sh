#!/usr/bin/env bash
# Run Mega-Quantification on a GPU cloud pod that is already a container
# (no nested Docker, often a small overlay plus a large read-only model mount).
#
# Typical 5090 box:
#   1x RTX 5090 32 GB (sm_120), 64 GB RAM, CUDA 13.2, conda py312 + torch cu132
#   Local BF16 snapshot via MEGAQUANT_MODEL or ./models / /workspace/models
#
# Usage (on the pod, from the repo root):
#   bash scripts/gpu-pod.sh plan|quantize|serve|eval|publish|rewrite-sglang [w4a8|w4a4|mixed]
#
# RECIPE env, if set, wins. Otherwise the committed 5090 packed recipe is used:
#   recipes/qwen3.8-27b-nvfp4-w4a8.5090.yaml
#   recipes/qwen3.8-27b-nvfp4-w4a4.5090.yaml
#   recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
# If that file is missing, recipes/qwen3.8-27b-nvfp4-<scheme>.pod.yaml is
# generated once (never overwritten). Scheme default is w4a8. Do not bake
# 27B weights into images. Local snapshot, if present, overrides model.source.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LC_ALL="${LC_ALL:-C}"
export LANG="${LANG:-C}"
export PYTHONUNBUFFERED=1
export MEGAQUANT_DEVICE_MAP="${MEGAQUANT_DEVICE_MAP:-auto}"
export MEGAQUANT_LOW_MEMORY="${MEGAQUANT_LOW_MEMORY:-1}"
export MEGAQUANT_OFFLOAD_DIR="${MEGAQUANT_OFFLOAD_DIR:-$ROOT/offload_folder}"
# Leave MEGAQUANT_MAX_MEMORY unset so Python packs from real VRAM/RAM
# (VRAM − 1 GiB, MemTotal − 6 GiB). Override explicitly if you need a cap.
export MEGAQUANT_GPU_HEADROOM_GIB="${MEGAQUANT_GPU_HEADROOM_GIB:-1}"
export MEGAQUANT_CPU_RESERVE_GIB="${MEGAQUANT_CPU_RESERVE_GIB:-6}"
export MEGAQUANT_BATCH_SIZE="${MEGAQUANT_BATCH_SIZE:-4}"
export MEGAQUANT_PIN_MEMORY="${MEGAQUANT_PIN_MEMORY:-1}"
NPROC="$(nproc)"
export MEGAQUANT_NUM_THREADS="${MEGAQUANT_NUM_THREADS:-$NPROC}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NPROC}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NPROC}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NPROC}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Idle 5090 reports PCIe gen1; CUDA warmup + extra copy connections train gen5 x16.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-16}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# SGLang FP8 JIT on sm_120 compiles with nvcc's host preprocessor. A missing
# g++/cc1plus fails as: gcc: fatal error: cannot execute 'cc1plus'.
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
# GDN fused kernels: without these, transformers prints
# "Falling back to torch implementation" and 5090 SM% stays ~5%.
export MEGAQUANT_INSTALL_FLA="${MEGAQUANT_INSTALL_FLA:-1}"
export MEGAQUANT_INSTALL_CAUSAL_CONV1D="${MEGAQUANT_INSTALL_CAUSAL_CONV1D:-0}"

PY="${MEGAQUANT_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  for candidate in \
      /usr/local/miniconda3/envs/py312/bin/python \
      /usr/local/miniconda3/bin/python \
      python3; do
    if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
      PY="${candidate}"
      break
    fi
  done
fi

DEFAULT_MODEL="${MEGAQUANT_MODEL:-}"
if [[ -z "${DEFAULT_MODEL}" ]]; then
  for candidate in \
      "$ROOT/models/Qwen3.8-27B" \
      "/workspace/models/Qwen3.8-27B" \
      "/models/Qwen3.8-27B"; do
    if [[ -f "${candidate}/config.json" ]]; then
      DEFAULT_MODEL="${candidate}"
      break
    fi
  done
fi
CMD="${1:-plan}"
shift || true

SCHEME="w4a8"
if [[ "${1:-}" == "w4a8" || "${1:-}" == "w4a4" || "${1:-}" == "mixed" ]]; then
  SCHEME="$1"
  shift
fi

PACKED_RECIPE="$ROOT/recipes/qwen3.8-27b-nvfp4-${SCHEME}.5090.yaml"
POD_RECIPE="$ROOT/recipes/qwen3.8-27b-nvfp4-${SCHEME}.pod.yaml"
if [[ -z "${RECIPE:-}" ]]; then
  if [[ -f "${PACKED_RECIPE}" ]]; then
    RECIPE="${PACKED_RECIPE}"
  else
    RECIPE="${POD_RECIPE}"
  fi
fi

MODEL_ARGS=()
if [[ -f "${DEFAULT_MODEL}/config.json" ]]; then
  MODEL_ARGS+=(--model "${DEFAULT_MODEL}")
fi

mkdir -p "$ROOT/outputs" "$MEGAQUANT_OFFLOAD_DIR" "$HF_HOME"

if [[ ! -f "${RECIPE}" ]]; then
  RECIPE="${POD_RECIPE}"
  case "${SCHEME}" in
    w4a4)
      NV_SCHEME="nvfp4_w4a4"
      OUT_NAME="Qwen3.8-27B-NVFP4-W4A4"
      ;;
    mixed)
      NV_SCHEME="nvfp4_mixed"
      OUT_NAME="Qwen3.8-27B-NVFP4-mixed"
      ;;
    *)
      SCHEME="w4a8"
      NV_SCHEME="nvfp4_w4a8"
      OUT_NAME="Qwen3.8-27B-NVFP4-W4A8"
      ;;
  esac
  cat > "${RECIPE}" <<YAML
name: qwen3.8-27b-nvfp4-${SCHEME}-pod
backend: modelopt
scheme: ${NV_SCHEME}
algorithm: max
kv_cache: fp8
family: qwen3_5
extra_ignore: []
model:
  source: ${DEFAULT_MODEL}
  dtype: bfloat16
  trust_remote_code: true
  device_map: auto
  quantize_vision: false
  quantize_mtp: false
calibration:
  # Nemotron is gated; ultrachat is public. Override RECIPE + HF_TOKEN for Nemotron.
  dataset: HuggingFaceH4/ultrachat_200k
  num_samples: 256
  max_seq_length: 1024
  batch_size: 4
  seed: 42
  with_images: false
export:
  output_dir: ${ROOT}/outputs/${OUT_NAME}
  format: hf
  pack: true
YAML
fi

log() { printf '[gpu-pod] %s\n' "$*"; }

log "python=${PY} threads=${MEGAQUANT_NUM_THREADS} gpu_headroom=${MEGAQUANT_GPU_HEADROOM_GIB}GiB cpu_reserve=${MEGAQUANT_CPU_RESERVE_GIB}GiB batch=${MEGAQUANT_BATCH_SIZE}"
log "recipe=${RECIPE} scheme=${SCHEME}"
case "${CMD}" in
  publish|rewrite-sglang|schemes|families)
    ;;
  *)
    "${PY}" - <<'PY'
import torch, sys, subprocess
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available in this Python")
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
try:
    raw = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
         "--format=csv,noheader"],
        text=True, timeout=8,
    ).strip()
    print("pcie_idle", raw, "(gen1 until CUDA warmup; PTQ trains gen5 x16)")
except Exception as exc:
    print("pcie query skipped", type(exc).__name__)
PY
    ;;
esac

if [[ "${CMD}" == "quantize" && "${MEGAQUANT_INSTALL_FLA}" == "1" ]]; then
  log "installing GDN Python kernels (kernels + flash-linear-attention; no CUDA compile)"
  "${PY}" -m pip install -q 'kernels>=0.9' 'flash-linear-attention' \
    || log "FLA wheel install skipped"
fi
if [[ "${CMD}" == "quantize" && "${MEGAQUANT_INSTALL_CAUSAL_CONV1D}" == "1" ]]; then
  log "installing causal-conv1d (CUDA extension; TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0})"
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    "${PY}" -m pip install -q causal-conv1d \
    || log "causal-conv1d skipped"
fi
case "${CMD}" in
  publish|rewrite-sglang|schemes|families)
    ;;
  *)
    "${PY}" - <<'PY'
mods = ["kernels", "fla", "causal_conv1d"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"[gpu-pod] {name} OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"[gpu-pod] {name} missing ({type(exc).__name__})")
PY
    ;;
esac

case "${CMD}" in
  plan)
    exec "${PY}" -m megaquant.cli plan -c "${RECIPE}" "${MODEL_ARGS[@]}" "$@"
    ;;
  quantize)
    exec "${PY}" -m megaquant.cli quantize -c "${RECIPE}" "${MODEL_ARGS[@]}" "$@"
    ;;
  serve|eval)
    EVAL_RECIPE="${EVAL_RECIPE:-$ROOT/recipes/eval-gpqa-diamond.yaml}"
    case "${SCHEME}" in
      w4a4) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-W4A4" ;;
      mixed) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-mixed" ;;
      *) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-W4A8" ;;
    esac
    EVAL_OUT="${EVAL_MODEL}/gpqa_diamond"
    if [[ "${CMD}" == "serve" ]]; then
      log "serve recipe=${EVAL_RECIPE} model=${EVAL_MODEL} (SGLang HiCache KV CPU offload = MemTotal − reserve)"
      exec "${PY}" -m megaquant.cli serve -c "${EVAL_RECIPE}" --model "${EVAL_MODEL}" "$@"
    fi
    export MEGAQUANT_SGLANG_BASE_URL="${MEGAQUANT_SGLANG_BASE_URL:-${MEGAQUANT_BASE_URL:-${MEGAQUANT_VLLM_BASE_URL:-http://127.0.0.1:30000/v1}}}"
    log "eval recipe=${EVAL_RECIPE} model=${EVAL_MODEL} out=${EVAL_OUT} base=${MEGAQUANT_SGLANG_BASE_URL}"
    exec "${PY}" -m megaquant.cli eval -c "${EVAL_RECIPE}" \
      --model "${EVAL_MODEL}" --output "${EVAL_OUT}" "$@"
    ;;
  publish|rewrite-sglang)
    case "${SCHEME}" in
      w4a4) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-W4A4" ;;
      mixed) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-mixed" ;;
      *) EVAL_MODEL="$ROOT/outputs/Qwen3.8-27B-NVFP4-W4A8" ;;
    esac
    if [[ "${CMD}" == "publish" ]]; then
      log "publish model=${EVAL_MODEL} scheme=${SCHEME}"
      exec "${PY}" "${ROOT}/scripts/oss_publish.py" "${EVAL_MODEL}" --scheme "${SCHEME}" "$@"
    fi
    log "rewrite-sglang model=${EVAL_MODEL}"
    exec "${PY}" -m megaquant.cli rewrite-sglang "${EVAL_MODEL}"
    ;;
  schemes|families)
    exec "${PY}" -m megaquant.cli "${CMD}" "$@"
    ;;
  *)
    exec "${PY}" -m megaquant.cli "${CMD}" "$@"
    ;;
esac
