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
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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
DEFAULT_MODEL="/model/ModelScope/Qwen/Qwen3.8-27B"
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
  dataset: nvidia/Nemotron-Post-Training-Dataset-v2
  num_samples: 512
  max_seq_length: 2048
  batch_size: 1
  seed: 42
  with_images: false
export:
  output_dir: ${ROOT}/outputs/Qwen3.8-27B-NVFP4-W4A8
  format: hf
  pack: true
YAML
fi

log() { printf '[gpu-pod] %s\n' "$*"; }

log "python=${PY}"
"${PY}" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available in this Python")
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
print("arch", torch.cuda.get_arch_list())
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
