#!/usr/bin/env bash
# Deprecated name: default inference is SGLang. Kept so older docs still work.
# KV that does not fit GPU HBM is offloaded to host RAM.
#
# Usage (repo root):
#   bash scripts/serve_vllm.sh outputs/Qwen3.8-27B-NVFP4-W4A8
# Optional: pass --engine vllm to use the NVIDIA GB300 vLLM flags instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/serve_sglang.sh" "$@"
