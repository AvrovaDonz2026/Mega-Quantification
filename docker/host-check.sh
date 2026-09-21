#!/usr/bin/env bash
# Run on the HOST (the 5090+ box) before the first compose build/run.
set -euo pipefail

ok() { printf '  [ok] %s\n' "$*"; }
bad() { printf '  [x]  %s\n' "$*"; FAIL=1; }

FAIL=0
echo "== Mega-Quantification host check (RTX 5090+ / Blackwell) =="

if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -n 1)"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | xargs)"
  python3 - "${driver}" <<'PY' || true
import sys
ver = sys.argv[1].split(".")[0]
try:
    major = int(ver)
except ValueError:
    sys.exit(0)
if major < 570:
    print("  [!] driver < 570: RTX 5090 / CUDA 12.8 typically need 570+")
PY
else
  bad "nvidia-smi missing — install NVIDIA driver 570+ (Blackwell / 5090)"
fi

if command -v docker >/dev/null 2>&1; then
  ok "docker: $(docker --version)"
else
  bad "docker missing"
fi

if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia\|nvidia'; then
  ok "NVIDIA container runtime is visible to docker"
else
  if command -v nvidia-container-cli >/dev/null 2>&1; then
    ok "nvidia-container-cli present ($(nvidia-container-cli --version 2>/dev/null | head -n 1))"
  else
    bad "NVIDIA Container Toolkit missing — https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  fi
fi

if docker compose version >/dev/null 2>&1; then
  ok "docker compose: $(docker compose version | head -n 1)"
else
  bad "docker compose v2 missing"
fi

free_gb="$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')"
if [[ "${free_gb}" =~ ^[0-9]+$ ]] && (( free_gb < 80 )); then
  bad "free disk ${free_gb}G — need ~20G for the image plus ~60G for Qwen3.8-27B BF16 + export"
else
  ok "free disk ~${free_gb}G"
fi

echo
if [[ "${FAIL}" -eq 0 ]]; then
  echo "Host looks ready. Next:"
  echo "  docker compose build"
  echo "  docker compose run --rm megaquant plan"
  echo "  docker compose run --rm quantize"
  exit 0
fi
echo "Fix the items marked [x], then re-run: bash docker/host-check.sh"
exit 1
