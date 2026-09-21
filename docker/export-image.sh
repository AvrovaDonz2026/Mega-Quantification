#!/usr/bin/env bash
# Save the built image so you can copy it to a 5090 box without rebuilding.
set -euo pipefail
IMAGE="${MEGAQUANT_IMAGE:-megaquant:nvfp4}"
OUT="${1:-megaquant-nvfp4.tar}"
docker compose build
docker image save "${IMAGE}" -o "${OUT}"
echo "Wrote ${OUT} ($(du -h "${OUT}" | awk '{print $1}'))"
echo "On the 5090 host:"
echo "  docker image load -i ${OUT}"
echo "  docker compose run --rm megaquant plan"
echo "  docker compose --profile gpu run --rm quantize"
