# MLP BF16 precision recovery pilot

This is a remote reproduction procedure for the MSE/MLP BF16 diagnostic pilot.
Run these commands in a repository checkout on the GPU machine with Python
3.10 or later; replace the example checkout and model paths below. The default
experiment is a finite early-trajectory diagnostic; an explicit `long` phase
is provided for a separate full-budget trajectory. This document does not
contain or promise experiment results.

## 1. Build an independent MLP/control model pair

Start with the real scored MSE ModelOpt export and the matching original BF16
checkpoint. Both output paths must not exist. The ablation tool
copies selected BF16 tensor payloads and writes a derived `modelopt_mixed`
checkpoint; it is not a fresh PTQ run, a W4A8 model, or a replacement for the
scored MSE export.

```bash
MEGAQUANT_REPO=/path/to/Mega-Quantification
cd "$MEGAQUANT_REPO"

python3 scripts/prepare_precision_ablation.py \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --variant control \
  --output /path/to/new/mlp-pilot-control \
  --write

python3 scripts/prepare_precision_ablation.py \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --variant mlp \
  --output /path/to/new/mlp-pilot-mlp \
  --write
```

The original export's `provenance.json` is locked by `model_gate.py` to
SHA-256
`66edb2373a176d330c70a8445c52e33b9caa0c8b05378a9cafd57a97ce19b55c`.
The derived directories must retain identical provenance bytes and tokenizer
ancillary files. The writer verifies every output payload before publishing the
new directory; the controller then runs the full model gate again for both
arms. MLP recovery restores the selected language MLP projections while
preserving the quantized vision and MTP inventory. The control arm uses the
same loader and metadata path with no MLP modules restored.

## 2. Prepare the remote root

Use a fresh remote root on the GPU machine. Build the locked source package
from a local SGLang clone containing the commits in the parent
`source-lock.json`; `build_sources.py` does not fetch from a network.

```bash
cd "$MEGAQUANT_REPO/scripts/diagnostics/gdn_parity_20261019"

REMOTE_ROOT=/fresh/mlp-pilot-remote
EVIDENCE_ROOT=/fresh/mlp-pilot-evidence
mkdir -p "$REMOTE_ROOT"
# EVIDENCE_ROOT must not exist; the controller creates it.

python3 build_sources.py \
  --repo /path/to/sglang \
  --output "$REMOTE_ROOT/source-package" \
  --lock source-lock.json

mkdir -p "$REMOTE_ROOT/candidate-bundle-r1"
cp -a parity-v7 "$REMOTE_ROOT/candidate-bundle-r1/parity-v7"
cp source_lock.py source-lock.json "$REMOTE_ROOT/candidate-bundle-r1/"
mkdir -p "$REMOTE_ROOT/candidate-bundle-r1/mlp-pilot"
cp mlp-pilot/controller.py mlp-pilot/model_gate.py \
  "$REMOTE_ROOT/candidate-bundle-r1/mlp-pilot/"
```

The source package must include the `all-beta` directory, archive, and
manifest. The controller pins that variant to revision
`49d3acfd4d54b0cf1afffcfeb369c63af4f56499`; the parity-v7 runner, source lock,
and source-lock helper must come from the same bundle copy. Pass the exact
ablation tool used to create the models with `--ablation-tool`; the controller
and gate in the remote bundle must be the copied files above. The validated
serving image is pinned to
`sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf`.

## 3. Run the pilot

Keep the anonymous case-3 fixture private on the evaluation machine. It is
accepted only when its native private file SHA is
`8e151d76fdd25603216aeb79c59a39d637ad25c47efd87a3581f6b65b3dc7218`, its sole
case is `case=3`, and its prompt token digest is
`20e75efb3ff7b137eee5484d35b30e0bb3668d8cc810b36990c7d3cae662d436` (288
prompt tokens; 49152 historical output tokens). Do not copy its token IDs,
output IDs, prompts, or responses into the repository or evidence root.

```bash
cd "$MEGAQUANT_REPO"

python3 "$REMOTE_ROOT/candidate-bundle-r1/mlp-pilot/controller.py" \
  --root "$EVIDENCE_ROOT" \
  --remote-root "$REMOTE_ROOT" \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --control /path/to/new/mlp-pilot-control \
  --mlp /path/to/new/mlp-pilot-mlp \
  --ablation-tool "$MEGAQUANT_REPO/scripts/prepare_precision_ablation.py" \
  --native-private /private/path/native-case3.json \
  --budget 4096
```

The default `early` pilot is two cold-cache trials per arm with a 4096-token
budget. Its budget may be set explicitly from 512 through 8192. The fixed
protocol is context length 65536, temperature 0, TP1, no MTP or speculative
decoding, FP8 E4M3 KV cache, FP32 Mamba SSM, radix cache disabled, CUDA graphs
disabled, no auto-truncation, and the pinned serving image. Every payload,
tensor inventory, metadata transform, provenance byte, source archive, runtime
source hash, prompt echo, token count, and finite logprob is checked. Repeat
stability is reported separately; `completed` means both repeats agree, not
that natural stopping or answer correctness was established. Reports contain
the controller SHA, sanitized hashes, and scalar statistics only.
Both arms use the same startup random seed (`20261020`), checked against the
server identity. This does not make different precision paths deterministic;
the two cold-cache repetitions still check observed stability. Responses must
finish with `stop` or `length`; a length finish must reach the requested budget.

The optional `long` phase starts from the same 288-token original prompt and
uses exactly `49152` new tokens (`288 + 49152 < 65536`). It does not append a
previous output or use a historical prefix. Long requests use a 14400-second
HTTP timeout; early requests retain 3600 seconds. A short budget is rejected
when `--phase long` is selected.

Run the long phase in a new evidence root after the early run:

```bash
LONG_EVIDENCE_ROOT=/fresh/mlp-pilot-long-evidence
python3 "$REMOTE_ROOT/candidate-bundle-r1/mlp-pilot/controller.py" \
  --phase long \
  --root "$LONG_EVIDENCE_ROOT" \
  --remote-root "$REMOTE_ROOT" \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --control /path/to/new/mlp-pilot-control \
  --mlp /path/to/new/mlp-pilot-mlp \
  --ablation-tool "$MEGAQUANT_REPO/scripts/prepare_precision_ablation.py" \
  --native-private /private/path/native-case3.json
```

The earlier `r2` evidence remains an early run and is not automatically
extended into a long run. For each trial, the report records only scalar
stopping fields (`terminal_eos`, first thinking-end index, and a conservative
natural-termination boolean). It also checks the known lag-2032 period by
finding the longest contiguous run where `output[i] == output[i-2032]` and
reports its onset, comparison count, extent (`count + 2032`), complete-period
count, and confirmation flag. Confirmation requires at least two complete
periods. This detector checks only that known period and does not prove that
other repetition is absent. `stopping_outcome` and `known_cycle_outcome` are
independent summaries; a length-limited output is not treated as a failed
experiment, and a run without a detected cycle does not establish a repair.

## Scope

The derived weights and activation behavior are restored through the existing
loader path; this pilot does not claim a new quantization recipe. The early
budget observes only the early natural trajectory. The long budget observes a
single natural-start trajectory window and cannot establish loop repair,
stopping recovery, GPQA accuracy, throughput, or a production quality
improvement. A stop does not mean the answer is correct, and reaching the
length limit without a loop does not prove the repair worked. Neither phase
promises iterative or cyclic repair.

## 4. Remote supervisor deployment

The checked-in `supervisor.py`, `fallback_restore.py`, and
`supervisor-config.example.json` are deployment references copied from the
run snapshot. On the GPU host, place both Python files beside the real
`config.json` in `mlp-run-r3` and use the exact remote paths in that config. A systemd
transient unit can be started with:

```bash
systemd-run --unit=megaquant-mlp-r3-20261020 \
  --property=Type=exec \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=600s \
  --property=RuntimeMaxSec=25h \
  --property='ExecStopPost=/usr/bin/python3 /home/zhiman/megaquant-aitop-validation-20261019/mlp-run-r3/fallback_restore.py' \
  /usr/bin/python3 /home/zhiman/megaquant-aitop-validation-20261019/mlp-run-r3/supervisor.py
```

Stopping an existing service requires authorization; it was explicitly given
for this validation session. The supervisor stops and restores the GLM container
by its exact configured container ID; it never removes that container.
Experiment cleanup is limited to the exact owner, job, source, image, and
creation-time gates. `fallback_restore.py` is an `ExecStopPost` recovery path
and persists the restore result in the shared status JSON.

The early five-hour and long eighteen-hour values are watchdog ceilings, not
expected runtimes. Evidence roots and phase logs must be fresh. Status and
reports contain sanitized scalar fields and hashes; prompts, responses,
tokens, and raw Docker `Env`/inspect output must remain private.
