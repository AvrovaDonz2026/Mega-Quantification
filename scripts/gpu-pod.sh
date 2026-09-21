#!/usr/bin/env bash
# Run Mega-Quantification on a GPU *cloud pod that is already a container*
# (Compshare / AutoDL / similar k8s GPU boxes). Those machines have no Docker,
# often a ~50G overlay, and a large read-only model mount.
#
# Typical 5090 pod we tested:
#   1x RTX 5090 32 GB (sm_120), 64 GB RAM, CUDA 13.2, conda py312 + torch cu132
#   BF16 weights already at /model/ModelScope/Qwen/Qwen3.8-27B (read-only)
#
# Usage (on the pod, from the repo root):
#   bash scripts/gpu-pod.sh plan
#   bash scripts/gpu-pod.sh quantize
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LC_ALL="${LC_ALL:-C}"
export LANG="${LANG:-C}"
export PYTHONUNBUFFERED=1
export MEGAQUANT_DEVICE_MAP="${MEGAQUANT_DEVICE_MAP:-auto}"
export MEGAQUANT_LOW_MEMORY="${MEGAQUANT_LOW_MEMORY:-1}"
export MEGAQUANT_OFFLOAD_DIR="${MEGAQUANT_OFFLOAD_DIR:-$ROOT/offload_folder}"
# Leave MEGAQUANT_MAX_MEMORY unset so Python packs from real VRAM/RAM
# (VRAM − 2 GiB, MemTotal − 6 GiB). Override explicitly if you need a cap.
export MEGAQUANT_GPU_HEADROOM_GIB="${MEGAQUANT_GPU_HEADROOM_GIB:-2}"
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
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
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

RECIPE="${RECIPE:-$ROOT/recipes/qwen3.8-27b-nvfp4-w4a8.pod.yaml}"
DEFAULT_MODEL="/workspace/models/Qwen3.8-27B"
if [[ ! -f "${DEFAULT_MODEL}/config.json" ]]; then
  DEFAULT_MODEL="/model/ModelScope/Qwen/Qwen3.8-27B"
fi
CMD="${1:-plan}"
shift || true

mkdir -p "$ROOT/outputs" "$MEGAQUANT_OFFLOAD_DIR" "$HF_HOME"

if [[ ! -f "${RECIPE}" ]]; then
  cat > "${RECIPE}" <<YAML
name: qwen3.8-27b-nvfp4-w4a8-pod
backend: modelopt
scheme: nvfp4_w4a8
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
  # Nemotron v2 is gated; ultrachat is public. Override RECIPE + HF_TOKEN for Nemotron.
  dataset: HuggingFaceH4/ultrachat_200k
  num_samples: 256
  max_seq_length: 1024
  batch_size: 4
  seed: 42
  with_images: false
export:
  output_dir: ${ROOT}/outputs/Qwen3.8-27B-NVFP4-W4A8
  format: hf
  pack: true
YAML
fi

log() { printf '[gpu-pod] %s\n' "$*"; }

log "python=${PY} threads=${MEGAQUANT_NUM_THREADS} gpu_headroom=${MEGAQUANT_GPU_HEADROOM_GIB}GiB cpu_reserve=${MEGAQUANT_CPU_RESERVE_GIB}GiB batch=${MEGAQUANT_BATCH_SIZE}"
"${PY}" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available in this Python")
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
PY

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
"${PY}" - <<'PY'
mods = ["kernels", "fla", "causal_conv1d"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"[gpu-pod] {name} OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"[gpu-pod] {name} missing ({type(exc).__name__})")
PY

case "${CMD}" in
  plan)
    exec "${PY}" -m megaquant.cli plan -c "${RECIPE}" "$@"
    ;;
  quantize)
    exec "${PY}" -m megaquant.cli quantize -c "${RECIPE}" "$@"
    ;;
  schemes|families)
    exec "${PY}" -m megaquant.cli "${CMD}" "$@"
    ;;
  *)
    exec "${PY}" -m megaquant.cli "${CMD}" "$@"
    ;;
esac
