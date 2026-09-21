#!/usr/bin/env bash
# Fetch the official GPQA zip (password is published in idavidrein/gpqa README).
# Writes data/dataset/gpqa_diamond.csv for GPQA_CSV. Does not commit the CSV.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${GPQA_DIR:-$ROOT/data}"
mkdir -p "${DEST}"
ZIP="${DEST}/dataset.zip"
URL="${GPQA_ZIP_URL:-https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.zip}"
# Public password from https://github.com/idavidrein/gpqa README ("Dataset Download").
PASS="${GPQA_ZIP_PASSWORD:-deserted-untie-orchid}"
if [[ ! -f "${DEST}/dataset/gpqa_diamond.csv" ]]; then
  curl -fsSL -o "${ZIP}" "${URL}"
  python3 - "${ZIP}" "${DEST}" "${PASS}" <<'PY'
import sys, zipfile
from pathlib import Path
zpath, dest, pw = sys.argv[1], sys.argv[2], sys.argv[3]
with zipfile.ZipFile(zpath) as zf:
    zf.extractall(path=dest, pwd=pw.encode())
print("extracted", dest)
PY
fi
ls -l "${DEST}/dataset/gpqa_diamond.csv"
