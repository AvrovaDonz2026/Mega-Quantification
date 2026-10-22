# MLP BF16 precision recovery pilot

This is a remote reproduction procedure for the MSE/MLP BF16 diagnostic pilot.
Run these commands in a repository checkout on the GPU machine with Python
3.10 or later; replace the example checkout and model paths below. The default
experiment is a finite early-trajectory diagnostic; an explicit `long` phase
is provided for a separate full-budget trajectory. This document does not
contain or promise experiment results.

The controller defaults to `--trajectory natural`, `--source-variant all-beta`,
and `--arms both`.
The single-host supervisor requires an auditable single-node launch before it
can pause a protected service. It rejects cross-node rank/master launches,
ambiguous topology, and opaque launch wrappers. A single-node `--tp 2` is
allowed when no cross-node flags are present. Services spanning multiple hosts
need a separate, authorized coordination procedure for both original containers.
Use `--source-variant original` to select the original source package, and
`--arms control` to run only the control arm. The selected source revision is
read from the complete parent `source-lock.json`; the source gate verifies the
selected directory, archive, and manifest before either arm starts. Reports
record `source_variant`, `source_revision`, and `selected_arms`. With one arm,
`comparison.both_repeat_stable` covers that arm's two repeats and the report
does not contain cross-arm `lcp` or `output_exact` fields.

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

For a diagnostic subset, pass the same selectors to the MLP build and to the
controller invocation. Projection names are canonicalized in the recorded
selection order; layer ranges are inclusive at `--layer-start` and exclusive
at `--layer-end`.

```bash
python3 scripts/prepare_precision_ablation.py \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --variant mlp \
  --mlp-projection down_proj \
  --layer-start 16 --layer-end 32 \
  --output /path/to/new/mlp-pilot-mlp-subset \
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
The gate reads the `selection` record from `precision_ablation.json` and
requires controller selectors to match it. Subset runs report the actual
restored module count and do not claim the full 192-module MLP recovery.

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

The source package must include all four locked variants (`original`,
`decode-beta`, `all-beta`, and `shared-decode`), each with its directory,
archive, and manifest. A candidate matrix containing only all-beta is not a
valid source lock. The selected revision is read from the parent lock and is
checked by the source gate; do not hard-code a revision copied from another
bundle. The parity-v7 runner, source lock, and source-lock helper must come
from the same bundle copy. Pass the exact ablation tool used to create the
models with `--ablation-tool`; the controller and gate in the remote bundle
must be the copied files above. The validated serving image is pinned to
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
  --native-private /path/to/private/native-case3.json \
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
New trial reports require the canonical `meta_info.num_retractions` field to be
an integer zero and retain that scalar. Missing, null, boolean or noninteger
values fail the gate. An omitted speculation counter is reported as unobserved;
the no-MTP condition is also checked against resolved server settings.
Frozen earlier trial reports did not retain the retraction scalar. Their
controller rejected positive canonical counts when present, but accepted an
absent field; do not relabel those reports as direct zero-count observations.

### Historical-prefix comparison

Pass `--trajectory historical-prefix` to run an independent prefix comparison.
This mode is valid only with `--phase early` and a budget from `512` through
`4096`; it performs two cold-cache trials for each selected arm. The private
case-3 fixture is checked by its native file and prompt hashes, then the server
receives the 288-token prompt followed by exactly `41472` native reference
tokens. The continuation reference is the next `--budget` native tokens.
Reports record `trajectory`, `input_tokens`, `native_prefix_tokens`, the hashed
reference window, and per-trial continuation LCP/exact fields. They also include
a scalar lag-2032 summary showing whether a confirmed run crosses the
input-prefix/continuation boundary. Token IDs, prompts, and response text are
never written.

This is a fresh-prefill comparison. It cannot reconstruct the historical
incremental KV or SSM state and cannot establish natural-generation repair. The
prefix execution must not be labeled as a natural trajectory. The existing
supervisor accepts only the default `natural` trajectory; invoke this controller
directly with a fresh evidence root for the prefix comparison.

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
  --native-private /path/to/private/native-case3.json
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

For the minimum original-source control-only check, use a fresh evidence root
and keep every other protocol value unchanged:

```bash
ORIGINAL_CONTROL_LONG_ROOT=/fresh/mlp-pilot-original-control-long
python3 "$REMOTE_ROOT/candidate-bundle-r1/mlp-pilot/controller.py" \
  --phase long \
  --source-variant original \
  --arms control \
  --root "$ORIGINAL_CONTROL_LONG_ROOT" \
  --remote-root "$REMOTE_ROOT" \
  --export /path/to/scored-mse-export \
  --bf16 /path/to/original-bf16-checkpoint \
  --control /path/to/new/mlp-pilot-control \
  --mlp /path/to/new/mlp-pilot-mlp \
  --ablation-tool "$MEGAQUANT_REPO/scripts/prepare_precision_ablation.py" \
  --native-private /path/to/private/native-case3.json
```

This is a procedure only; it does not state that a new experiment has started
or finished. A control-only report checks repeat stability and the selected
source and model/runtime gates, without claiming MSE repair. The next source
check must change only the source variant; keep context, seed, and radix
settings fixed. The historical protocol is not equivalent to this one. If
original still does not cycle, restore historical conditions one at a time
and record each change separately.

The deployed r3 snapshot has controller SHA
`679297e5eeef5e464dbf979e77b73af2c2f4aa2297583836220447c6ccc14673`.
Its EOS check uses only `config.json`, which omits generation/tokenizer EOS
IDs for these models. Its `natural_termination=false` therefore cannot prove
that an EOS stop did not occur. The original matched stop reason and token
sequence are not retained, so existing r3 records must keep their original
labels and this limitation. Keep a running snapshot unchanged; record a new
controller SHA and use fresh evidence directories for any corrected run.

The current helper combines config, generation-config, tokenizer EOS, and the
three additional stop tokens recognized by the pinned SGLang tokenizer hook.
For config EOS it prefers the root field, falling back to `text_config` when
the root field is absent, matching the pinned config propagation behavior.
It records only source counts/hashes and a sanitized `finish_reason_matched`
summary (`match_kind`, `matched_is_eos`, `matched_equals_terminal`). A natural
termination still requires a terminal EOS after thinking-end; when a matched
stop value is supplied, it must match that terminal token. A service `stop`
alone is insufficient. This correction has not been deployed into the running
r3 snapshot and cannot retrospectively relabel its results.
The EOS-only revision SHA is
`58abf92c21790043b5861c3be1cfb5399d0bc79a4831bfea993265317f3364cd`;
it is retained for the historical corrected helper and is not a claim that a
new run has been deployed.

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
`supervisor-config.example.json` are references for a fresh deployment. The
supervisor config accepts `source_variant` (`original` or `all-beta`), `arms`
(`both` or `control`), and optional `phases` (`early`, `long`); defaults remain
all-beta, both, and both phases. The example hashes identify source files at
the time they were written, not a future deployment. After the current r3
namespace has completed its cleanup and exact GLM restore, copy the files into
a new namespace and recalculate controller, model-gate, supervisor, and
fallback hashes at runtime with `sha256sum`; put those fresh values in the new
config. Do not modify the running r3 snapshot or reuse its namespace. GLM
stopping and restoration must retain the existing exact-ID/config mechanism.

Use placeholder paths and a fresh unit name for the new namespace:

```bash
systemd-run --unit=megaquant-mlp-new-YYYYMMDD \
  --property=Type=exec \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=600s \
  --property=RuntimeMaxSec=25h \
  --property='ExecStopPost=/usr/bin/python3 /path/to/new-namespace/fallback_restore.py' \
  /usr/bin/python3 /path/to/new-namespace/supervisor.py
```

Stopping an existing service requires authorization; it was explicitly given
for this validation session. The supervisor stops and restores the GLM container
by its exact configured container ID; it never removes that container.
Experiment cleanup is limited to the exact owner, job, source, image, and
creation-time gates. `fallback_restore.py` is an `ExecStopPost` recovery path
and persists the restore result in the shared status JSON.

Both restoration paths require the exact container identity and configuration
to remain unchanged throughout a continuous 120-second running observation.
If Docker exposes health metadata, it must stay `healthy` during that window.
The wait is bounded to 240 seconds; a container that exits after starting fails
the restore gate and keeps restoration pending. `observed_seconds` records the
window checked; it does not establish application readiness.

The early five-hour and long eighteen-hour values are watchdog ceilings, not
expected runtimes. Evidence roots and phase logs must be fresh. Status and
reports contain sanitized scalar fields and hashes; prompts, responses,
tokens, and raw Docker `Env`/inspect output must remain private.
