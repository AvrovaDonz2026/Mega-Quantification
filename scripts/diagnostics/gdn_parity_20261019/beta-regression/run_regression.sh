#!/usr/bin/env bash
set -euo pipefail

artifact_root=$(cd "$(dirname "$0")" && pwd)
sglang_root=${SGLANG_ROOT:-"$artifact_root/../../sglang-gdn-20261019"}
export PYTHONPATH="$sglang_root/python${PYTHONPATH:+:$PYTHONPATH}"

variant=${1:?usage: $0 VARIANT [FAST_OPS] [REPORT] [SOURCE_REVISION]}
fast_ops=${2:-0}
report=${3:-"$artifact_root/report-${variant}-fast${fast_ops}.json"}
source_revision=${4:-${SOURCE_REVISION:-}}
revision_arg=()
if [[ -n "$source_revision" ]]; then
  revision_arg=(--source-revision "$source_revision")
fi
FLA_USE_FAST_OPS="$fast_ops" python3 "$artifact_root/probe.py" \
  --variant "$variant" --fla-use-fast-ops "$fast_ops" --report "$report" \
  "${revision_arg[@]}"
