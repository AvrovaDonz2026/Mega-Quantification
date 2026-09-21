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
elif major < 580:
    print("  [!] driver < 580: megaquant:sglang (sglang 0.5.20 / torch cu130) needs 580+; PTQ cu128 is OK on 570")
PY
else
  bad "nvidia-smi missing — install NVIDIA driver 570+ (Blackwell / 5090)"
fi

# Suggestion only (no Docker/Python nvcc probe). FlashInfer SM 12.0 JIT is
# unblocked when the *serve image* `nvcc --version` is 12.9+ and the host
# driver is 580+. Default megaquant:sglang is CUDA 12.8.1-devel.
echo "  [i] FlashInfer SM 12.0 / DeepGEMM JIT: unblocked if serve-image nvcc --version is 12.9+ and host driver is 580+ (optional: SGLANG_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 docker compose build serve-sglang; confirm tag on NGC). Default image is CUDA 12.8.1 — use recipes/eval-gpqa-diamond.5090.yaml (triton) until rebuilt."

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

if [[ -f /etc/docker/daemon.json ]]; then
  python3 - <<'PY' || true
import json
from pathlib import Path
path = Path("/etc/docker/daemon.json")
try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(0)
dns = data.get("dns") or []
if dns:
    print(f"  [ok] docker daemon DNS: {len(dns)} resolver(s)")
else:
    print("  [!] /etc/docker/daemon.json has no dns — optional MEGAQUANT_DOCKER_DNS on install-host.sh")
PY
else
  echo "  [!] no /etc/docker/daemon.json — optional MEGAQUANT_DOCKER_DNS on install-host.sh"
fi

if grep -qF "# megaquant-docker-hub-pin" /etc/hosts 2>/dev/null; then
  ok "Docker Hub names pinned in /etc/hosts"
else
  echo "  [!] no Hub pin in /etc/hosts — install-host.sh --dns-only can add one when MEGAQUANT_DOCKER_DNS is set"
fi

free_gb="$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')"
if [[ "${free_gb}" =~ ^[0-9]+$ ]] && (( free_gb < 15 )); then
  bad "free disk ${free_gb}G on $(pwd) — need ~15G+ for the venv/export (BF16 weights can live on another mount)"
elif [[ "${free_gb}" =~ ^[0-9]+$ ]] && (( free_gb < 80 )); then
  echo "  [!] free disk ${free_gb}G on $(pwd) — OK if 27B BF16 is already on another mount (e.g. /model) and you are not baking weights into a Docker image"
else
  ok "free disk ~${free_gb}G"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "  [!] no Docker — on a VM run: bash docker/install-host.sh"
  echo "  [!] on a nested GPU pod without Docker: bash scripts/gpu-pod.sh plan"
fi

echo
if [[ "${FAIL}" -eq 0 ]]; then
  echo "Host looks ready. Next:"
  echo "  docker compose build"
  echo "  docker compose run --rm megaquant plan"
  echo "  docker compose --profile gpu run --rm quantize"
  exit 0
fi
echo "Fix the items marked [x], then re-run: bash docker/host-check.sh"
exit 1
