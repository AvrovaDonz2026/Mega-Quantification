#!/usr/bin/env bash
# Idempotent host bootstrap for a fresh Ubuntu GPU VM:
#   Docker Engine + Compose v2 + NVIDIA Container Toolkit.
# Does not install the NVIDIA driver (needs 570+ already).
# Does not bake tenant DNS, SSH, bucket names, or accelerator IPs.
#
# Optional accelerator DNS (host resolver + Docker daemon only — never git):
#   MEGAQUANT_DOCKER_DNS="<ip> <ip>" bash docker/install-host.sh
# Re-apply DNS/Hub pins without apt:
#   MEGAQUANT_DOCKER_DNS="<ip> <ip>" bash docker/install-host.sh --dns-only
# Prefer IPv4 for Hub / CUDA pulls (override with MEGAQUANT_PREFER_IPV4=0).
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
print("wrote", out, "dns-count", len(dns))
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

apply_hub_pin() {
  local dns_csv="$1"
  local canary="${MEGAQUANT_DOCKER_PIN_CANARY:-pypi.org}"
  python3 - "${dns_csv}" "${canary}" <<'PY'
"""Pin Docker Hub names to a private/CGNAT accelerator IP from tenant DNS.

Some accelerator resolvers speed up pypi/GitHub/NGC but still return public
A records for registry-1.docker.io / CloudFront, which then RST. Discover the
accelerator by looking up a canary host and rewrite Hub names in /etc/hosts.
Does not hard-code tenant IPs.
"""
from __future__ import annotations

import ipaddress
import random
import socket
import struct
import sys
from pathlib import Path

dns_servers = [part for part in sys.argv[1].split() if part]
canary = sys.argv[2].strip() or "pypi.org"
hub_names = [
    "registry-1.docker.io",
    "docker.io",
    "index.docker.io",
    "auth.docker.io",
    "production.cloudflare.docker.com",
    "hub.docker.com",
    "registry.hub.docker.com",
    "production.cloudfront.docker.com",
    "d3ipk8a4rcghwr.cloudfront.net",
]
start_m = "# megaquant-docker-hub-pin"
end_m = "# megaquant-docker-hub-pin-end"


def _query_a(name: str, server: str, timeout: float = 2.5) -> str | None:
    labels = name.rstrip(".").split(".")
    query = struct.pack("!HHHHHH", random.randint(0, 65535), 0x0100, 1, 0, 0, 0)
    for label in labels:
        encoded = label.encode("idna")
        query += bytes([len(encoded)]) + encoded
    query += b"\x00" + struct.pack("!HH", 1, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(query, (server, 53))
        data, _ = sock.recvfrom(512)
    except OSError:
        return None
    finally:
        sock.close()
    if len(data) < 12:
        return None
    _, _, qcount, acount, _, _ = struct.unpack("!HHHHHH", data[:12])
    offset = 12
    for _ in range(qcount):
        while offset < len(data) and data[offset] != 0:
            if data[offset] & 0xC0:
                offset += 2
                break
            offset += 1 + data[offset]
        else:
            offset += 1
        offset += 4
    for _ in range(acount):
        if offset + 12 > len(data):
            break
        if data[offset] & 0xC0:
            offset += 2
        else:
            while offset < len(data) and data[offset] != 0:
                offset += 1 + data[offset]
            offset += 1
        rtype, _, _, rdlen = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlen]
        offset += rdlen
        if rtype == 1 and rdlen == 4:
            return socket.inet_ntoa(rdata)
    return None


def _is_accel(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return True
    return ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10")


accel = None
for server in dns_servers:
    addr = _query_a(canary, server)
    if addr and _is_accel(addr):
        accel = addr
        break
if not accel:
    print("no private/CGNAT A record for", canary, "- skip Hub pin", file=sys.stderr)
    raise SystemExit(0)

hosts = Path("/etc/hosts")
text = hosts.read_text(encoding="utf-8") if hosts.exists() else ""
block = start_m + "\n" + accel + " " + " ".join(hub_names) + "\n" + end_m + "\n"
if start_m in text and end_m in text:
    start = text.index(start_m)
    end = text.index(end_m) + len(end_m)
    if end < len(text) and text[end] == "\n":
        end += 1
    text = text[:start] + block + text[end:]
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += block
out = Path("/tmp/megaquant-hosts")
out.write_text(text, encoding="utf-8")
print("pinned", len(hub_names), "Hub names via canary", canary)
PY
  if [[ -f /tmp/megaquant-hosts ]]; then
    "${SUDO[@]}" cp /tmp/megaquant-hosts /etc/hosts
    log "Docker Hub names pinned to accelerator IP from ${canary}"
  fi
}

apply_network() {
  prefer_ipv4
  if [[ -z "${MEGAQUANT_DOCKER_DNS:-}" ]]; then
    return 0
  fi
  apply_docker_dns "${MEGAQUANT_DOCKER_DNS}"
  apply_host_dns "${MEGAQUANT_DOCKER_DNS}"
  apply_hub_pin "${MEGAQUANT_DOCKER_DNS}"
}

if [[ "${1:-}" == "--dns-only" ]]; then
  apply_network
  if command -v systemctl >/dev/null 2>&1; then
    "${SUDO[@]}" systemctl restart docker 2>/dev/null || true
  fi
  log "dns-only done"
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update -y
"${SUDO[@]}" apt-get install -y docker.io docker-compose-v2 curl ca-certificates gnupg python3
"${SUDO[@]}" usermod -aG docker "${SUDO_USER:-${USER}}" 2>/dev/null || true

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

apply_network

"${SUDO[@]}" systemctl enable --now docker
"${SUDO[@]}" systemctl restart docker
log "docker $(docker --version 2>/dev/null || sudo docker --version)"
log "compose $(docker compose version 2>/dev/null || sudo docker compose version)"
log "next: bash docker/host-check.sh && docker compose --profile gpu build"
