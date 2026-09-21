#!/usr/bin/env bash
# Save PTQ + SGLang images so another VM can `docker load` without rebuilding.
set -euo pipefail
OUT="${1:-megaquant-stack.tar}"
docker compose --profile gpu --profile dev build
IMAGES=(
  "${MEGAQUANT_IMAGE:-megaquant:nvfp4}"
  "${SGLANG_IMAGE:-megaquant:sglang}"
)
docker image save "${IMAGES[@]}" -o "${OUT}"
echo "Wrote ${OUT} ($(du -h "${OUT}" | awk '{print $1}'))"
echo "On the next GPU VM:"
echo "  bash docker/install-host.sh"
echo "  docker image load -i ${OUT}"
echo "  docker compose run --rm megaquant plan"
echo "  docker compose --profile gpu up serve-sglang"
