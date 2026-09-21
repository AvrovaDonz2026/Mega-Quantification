#!/usr/bin/env bash
# Idempotent host bootstrap for a fresh Ubuntu GPU VM:
#   Docker Engine + Compose v2 + NVIDIA Container Toolkit.
# Does not install the NVIDIA driver (needs 570+ already).
# Does not bake tenant DNS, SSH, or bucket names.
#
# Optional accelerator DNS (host resolver + Docker daemon only — never git):
#   MEGAQUANT_DOCKER_DNS="<ip> <ip>" bash docker/install-host.sh
# Prefer IPv4 for Hub / CUDA pulls (override with MEGAQUANT_PREFER_IPV4=0):
#   default on.
set -euo pipefail

log() { printf '[install-host] %s\n' "$*"; }

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

prefer_ipv4() {
  local gai=/etc/gai.conf
  local marker="# megaquant-ipv4-precedence"
  if [[ "${MEGAQUANT_PREFER_IPV4:-1}" != "1" ]]; then
    return 0
  fi
  "${SUDO[@]}" touch "${gai}"
  if "${SUDO[@]}" grep -qF "${marker}" "${gai}"; then
    log "IPv4 precedence already in ${gai}"
    return 0
  fi
  {
    echo "${marker}"
    echo "precedence :ffff:0:0/96  100"
  } | "${SUDO[@]}" tee -a "${gai}" >/dev/null
  log "prefer IPv4 via ${gai}"
}

apply_docker_dns() {
  local dns_csv="$1"
  python3 - "${dns_csv}" <<'PY'
import json, sys
from pathlib import Path
dns = [part for part in sys.argv[1].split() if part]
src = Path("/etc/docker/daemon.json")
existing = {}
if src.exists() and src.stat().st_size:
    try:
        existing = json.loads(src.read_text())
    except json.JSONDecodeError:
        existing = {}
existing["dns"] = dns
out = Path("/tmp/megaquant-daemon.json")
out.write_text(json.dumps(existing, indent=2) + "\n")
print("wrote", out, "dns", dns)
PY
  "${SUDO[@]}" mkdir -p /etc/docker
  "${SUDO[@]}" cp /tmp/megaquant-daemon.json /etc/docker/daemon.json
  log "Docker daemon DNS set from MEGAQUANT_DOCKER_DNS"
}

apply_host_dns() {
  local dns_csv="$1"
  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl missing; skip host resolver DNS"
    return 0
  fi
  if ! systemctl is-enabled systemd-resolved >/dev/null 2>&1 \
      && ! systemctl is-active systemd-resolved >/dev/null 2>&1; then
    log "systemd-resolved inactive; skip host resolver DNS"
    return 0
  fi
  "${SUDO[@]}" mkdir -p /etc/systemd/resolved.conf.d
  {
    echo "[Resolve]"
    echo "DNS=${dns_csv}"
    echo "Domains=~."
  } | "${SUDO[@]}" tee /etc/systemd/resolved.conf.d/99-megaquant-dns.conf >/dev/null
  "${SUDO[@]}" systemctl restart systemd-resolved
  log "host systemd-resolved DNS set from MEGAQUANT_DOCKER_DNS"
}

export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update -y
"${SUDO[@]}" apt-get install -y docker.io docker-compose-v2 curl ca-certificates gnupg python3
"${SUDO[@]}" usermod -aG docker "${SUDO_USER:-${USER}}" 2>/dev/null || true

prefer_ipv4

"${SUDO[@]}" mkdir -p /usr/share/keyrings /etc/apt/sources.list.d
curl -4 -fsSL --max-time 60 -o /tmp/nvidia-container.gpgkey \
  https://nvidia.github.io/libnvidia-container/gpgkey
"${SUDO[@]}" gpg --batch --yes --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  /tmp/nvidia-container.gpgkey
curl -4 -fsSL --max-time 60 -o /tmp/nvidia-container-toolkit.list \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  /tmp/nvidia-container-toolkit.list \
  | "${SUDO[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
"${SUDO[@]}" apt-get update -y
"${SUDO[@]}" apt-get install -y nvidia-container-toolkit
"${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker

if [[ -n "${MEGAQUANT_DOCKER_DNS:-}" ]]; then
  apply_docker_dns "${MEGAQUANT_DOCKER_DNS}"
  apply_host_dns "${MEGAQUANT_DOCKER_DNS}"
fi

"${SUDO[@]}" systemctl enable --now docker
"${SUDO[@]}" systemctl restart docker
log "docker $(docker --version 2>/dev/null || sudo docker --version)"
log "compose $(docker compose version 2>/dev/null || sudo docker compose version)"
log "next: bash docker/host-check.sh && docker compose --profile gpu build"
