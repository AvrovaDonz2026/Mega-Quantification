# DGX Spark containers

This runbook builds the repository's ARM64 / CUDA 13 images and runs full-27B
W4A8 and W4A4 quantization, image inference and embedded MTP checks on one GB10.
The recipes use TP1. Cross-node TP2 requires a separate distributed validation.

The [v0.1.2 release rerun](validation/dgx-spark-release-20261001.md) records
the exact inputs, image digests and results at frozen revision `ac53833`.
The 2026-10-01 TP1 rerun passed two independent offline builds of each image,
fresh full W4A8/W4A4 PTQ, strict source/provenance/export audits, GPU regressions,
text/image/OCR and embedded quantized MTP checks, and isolated image save/load.
Both quantization and serving used that same revision. Each format matched
205 baseline/MTP token IDs across five fixed cases. This coverage does not
establish arbitrary-prompt determinism or TP2. The
[previous clean-container report](validation/dgx-spark-clean-container-20260930.md)
retains the earlier failures and repairs. The earlier
[quantization](validation/dgx-spark-pr46-20260929.md) and
[serving](validation/dgx-spark-serving-20260930.md) reports describe their actual
environments; they do not prove a clean build of these new images. No published
registry image is claimed here.

## What the images contain

| Image | Public base | Repository additions |
|---|---|---|
| `megaquant:gb10` | NGC PyTorch 26.08, pinned by SHA256 in `Dockerfile.spark` | Complete Spark PTQ dependency version list, installed MegaQuant wheel, recipes and scripts |
| `megaquant:sglang-spark` | SGLang 0.5.20 CUDA 13, pinned by SHA256 in `Dockerfile.sglang.spark` | Installed MegaQuant wheel, guarded visual/MTP loaders, GDN and SM121 projection repairs, recipes and scripts |

Both builds use the fixed wheel build backend in
`docker/requirements-build-spark.txt`, `--no-deps` and `--no-build-isolation`.
PTQ dependencies are listed in `docker/requirements-gpu-spark.txt`. The public
base supplies the ARM64 PyTorch/CUDA stack. These builds do not require a local
`quant-env` image or an option to skip dependency installation.

MegaQuant is installed as a wheel; the image does not depend on an editable
checkout or `PYTHONPATH`. Spark Compose services retain the baked code and
recipes. Their only host mounts are:

| Host variable | Container destination | Access |
|---|---|---|
| `MODELS_DIR` | `/models` | Read only |
| `DATA_DIR` | `/data` | Read only |
| `OUTPUTS_DIR` | `/opt/megaquant/outputs` | Read/write |
| `HF_CACHE` | `/cache/huggingface` | Read/write |
| `OFFLOAD_DIR` | `/opt/megaquant/offload` | Read/write |

Changing host `src/`, `recipes/` or `docs/` does not change a running Spark
image. Rebuild the image to change code or recipes.

## Build and verify provenance

Run from a clean checkout on an ARM64 Spark with Docker Compose and the NVIDIA
container runtime available. The host must be able to fetch the pinned public
images and Python packages.

```bash
git status --short
export GIT_REVISION="$(git rev-parse HEAD)"
export MEGAQUANT_SPARK_IMAGE="megaquant:gb10-${GIT_REVISION}"
export SGLANG_SPARK_IMAGE="megaquant:sglang-spark-${GIT_REVISION}"

docker compose --profile spark-ptq --profile spark build --pull --no-cache \
  quantize-spark-w4a8 serve-sglang-spark

docker run --rm --entrypoint python "$MEGAQUANT_SPARK_IMAGE" \
  scripts/container_manifest.py verify
docker run --rm --entrypoint python "$SGLANG_SPARK_IMAGE" \
  scripts/container_manifest.py verify
```

Each image contains `/opt/megaquant/container-manifest.json`, recording its base
reference, Git revision, architecture, baked file hashes, installed MegaQuant
Python file hashes and installed package versions. `verify` returns nonzero on
added, removed or changed recorded files, package version changes or a different
architecture. The normal CLI entrypoint verifies this manifest before running
the command. Commands that override the entrypoint must verify explicitly.

The manifest is a drift check against the build record. It does not replace an
external image digest or prove that every third-party package file is unchanged.
Record Docker's actual identifiers separately from the manifest checksum:

```bash
docker image inspect --format \
  'Id={{.Id}} RepoDigests={{json .RepoDigests}} Revision={{index .Config.Labels "org.opencontainers.image.revision"}}' \
  "$MEGAQUANT_SPARK_IMAGE" "$SGLANG_SPARK_IMAGE"
```

A local build can have empty `RepoDigests`. OCI image indexes, platform manifests
and configuration blobs have separate digests; preserve the field names when
recording them. A clean rebuild pins software inputs but does not promise an
identical image ID or byte-identical recalibrated weights.

## Prepare inputs

Provide the full BF16 model at `MODELS_DIR/Qwen3.8-27B`. `DATA_DIR` must contain
`calibration.jsonl` and its referenced image files. Both Spark PTQ recipes use
256 samples, maximum sequence length 1024, batch size 1 and seed 42, with vision
and MTP quantization enabled. KV quantizers remain unset in the export; serving
selects FP8 KV.

```bash
export MODELS_DIR=/path/to/models
export DATA_DIR=/path/to/calibration
export OUTPUTS_DIR="$PWD/outputs/spark-clean"
export HF_CACHE="$PWD/.cache/huggingface"
export OFFLOAD_DIR="$PWD/offload/spark-clean"
export MEGAQUANT_LOW_MEMORY=1
export MEGAQUANT_MAX_MEMORY=0:96GiB,cpu:24GiB
mkdir -p "$OUTPUTS_DIR" "$HF_CACHE" "$OFFLOAD_DIR"
```

GB10 shares CPU/GPU RAM. The memory limits are placement ceilings, not separate
physical pools; check available RAM before the full run. Quantization and
serving should run sequentially on the single GPU.

For a pinned calibration source and the image/text manifest generation command,
see [the Spark calibration instructions](../docker/README.md#dgx-spark--gb10).
Keep a SHA256 record of the BF16 shards, calibration JSONL and image manifest
alongside the resulting exports when reproducing a particular run.

## Quantize both formats

First exercise the installed CLI and baked recipe without loading weights:

```bash
docker compose --profile spark-ptq run --rm quantize-spark-w4a8 \
  plan -c recipes/qwen3.8-27b-nvfp4-w4a8.spark.yaml
docker compose --profile spark-ptq run --rm quantize-spark-w4a4 \
  plan -c recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml

docker compose --profile spark-ptq run --rm quantize-spark-w4a8
docker compose --profile spark-ptq run --rm quantize-spark-w4a4
```

Exports appear in `OUTPUTS_DIR/Qwen3.8-27B-NVFP4-W4A8-spark` and
`OUTPUTS_DIR/Qwen3.8-27B-NVFP4-W4A4-spark`. Retain the run logs, provenance and
`calibration_coverage.json`. Successful command exit alone does not prove that
all visual and MTP quantizers were exercised or that every exported tensor and
scale is valid; inspect those records and audit the checkpoint before serving.

Run the stdlib-only strict audit for each export, supplying the full Git SHA
used to build the PTQ image. The audit requires all source language, vision and
MTP tensors, quantized auxiliary projections with calibration coverage, valid
packed shapes and finite scales. It checks the recorded provenance revision
against the supplied SHA; it does not rewrite that record or prove inference
quality.

```bash
python scripts/audit_spark_export.py \
  "$OUTPUTS_DIR/Qwen3.8-27B-NVFP4-W4A8-spark" "$MODELS_DIR/Qwen3.8-27B" nvfp4_w4a8 \
  --expected-revision "$GIT_REVISION" --output-json w4a8-export-audit.json
python scripts/audit_spark_export.py \
  "$OUTPUTS_DIR/Qwen3.8-27B-NVFP4-W4A4-spark" "$MODELS_DIR/Qwen3.8-27B" nvfp4_w4a4 \
  --expected-revision "$GIT_REVISION" --output-json w4a4-export-audit.json
```

## Vision and MTP inference

Verify the serving image's manifest, seven-file patch guard, ARM64/SM121 GPU,
CUDA matrix multiplication, GDN decode/verify state parity and FP8/BF16
projection and attention decode/verify parity before loading the full checkpoint:

```bash
docker compose --profile spark run --rm --entrypoint python serve-sglang-spark \
  scripts/check_sglang_spark.py
```

For each format, start the baseline and MTP servers separately. Both use baked
FP32 SSM state recipes. Baseline uses `eval-gpqa-diamond.spark.yaml`; MTP uses
`eval-gpqa-diamond.spark-mtp.yaml` with embedded EAGLE, two speculative steps,
top-k 1 and three draft tokens. No external draft checkpoint is supplied.

The patch keeps the packed GDN decode sigmoid gate in FP32, matching the
speculative verification kernel. FP32 SSM cache alone does not remove the
upstream decode gate's BF16 rounding. The GPU preflight compares 32 sequential
decode steps with a speculative verification chain, including every cached
intermediate state, grouped heads and a strided state pool. An older patch
manifest or changed pinned source is rejected; rebuild from the official base.

On SM121, static FP8 projections retain the existing FlashInfer/cuBLAS path for
both single-row decode and multi-row verification. BF16 GDN BA projections use
SGLang's existing batch-invariant Triton matrix multiplication. GPU preflight
checks exact row equality and finite outputs; the BF16 check also uses a CPU
FP64 reference. These operator checks do not establish full-model token parity.
The clean-container report records the original W4A8 strict-comparison failure
and the separately validated repair.

For eager Qwen3.5 inference on SM121 with FP8 KV, page size 1, 24 query heads,
4 KV heads and head dimension 256, ordinary decode uses the existing FlashInfer
prefill wrapper with three copies of the current query. Each copy reads only
the real cached KV through an explicit mask; KV is written once and only the
first output is returned. This aligns the attention reduction shape with the
three-token MTP recipe. The GPU preflight covers short sequences and two
requests, including masked future tokens. The extra attention work has not
been benchmarked; this is a scoped compatibility repair, not a general
determinism or performance guarantee.

The following runs W4A8. For W4A4, change `SPARK_EVAL_MODEL` to the W4A4 export
and use distinct output JSON filenames. The HTTP clients do not attach a GPU.

```bash
export SPARK_EVAL_MODEL=outputs/Qwen3.8-27B-NVFP4-W4A8-spark
export SPARK_CHECK_OUTPUT=outputs/w4a8-baseline.json
export SPARK_MTP_CHECK_OUTPUT=outputs/w4a8-mtp.json

docker compose --profile spark up -d serve-sglang-spark
timeout 600 bash -c 'until curl -fsS http://127.0.0.1:30000/health >/dev/null; do sleep 5; done'
docker compose --profile spark-check run --rm check-sglang-spark
docker compose --profile spark stop serve-sglang-spark

docker compose --profile spark-mtp up -d serve-sglang-spark-mtp
timeout 600 bash -c 'until curl -fsS http://127.0.0.1:30001/health >/dev/null; do sleep 5; done'
docker compose --profile spark-check run --rm check-sglang-spark-mtp
docker compose --profile spark-mtp stop serve-sglang-spark-mtp
```

The default host ports are 30000 for baseline and 30001 for MTP. If changed via
`SGLANG_SPARK_PORT` or `SGLANG_SPARK_MTP_PORT`, change the health probes as well.
Inspect logs with `docker compose logs <service>` if readiness fails.

The public client sends arithmetic, Python generation, red/blue counterfactual
images and four-digit OCR requests. Passing MTP validation requires correct
functional answers, exact baseline output token IDs, actual MTP verification
steps and at least one accepted draft. Retain both JSON reports and server logs;
check for skipped visual/MTP weights or quantization scales during loading.

These checks establish functionality for the recorded requests. They do not
provide a GPQA score, a general vision quality result, controlled throughput,
video support, long-context coverage or TP2 validation. High draft acceptance
alone is not a measured speedup.

## Full GPQA Diamond at temperature 0

The historical **32k context trial** of Spark mixed W4A8 on 2026-10-02 scored
**157/198 (79.29%)**.
All 198 unique dataset-matching records and the summary passed the strict
checker. There were 30 truncated and 32 unparsed responses; those counts
overlap, and each affected response counts as wrong. See the
[verified evidence](validation/dgx-spark-gpqa-w4a8-20261002.json).
This measures the earlier 32768-token limit, not the requested 262144-token
rerun. The 32k uniform W4A4 trial was stopped and has no completed score.
The new **262144-context W4A8 and W4A4 results are both pending**; no completed
256k score or absence of truncation is established yet.

The new W4A8 run started from an empty journal at **2026-10-02 12:57:50 UTC**.
The baked client/profile revision is `79b5dfef54618d0e385c5974d0ea508cd2fbdd40`.
The live server reports 262144 context, a 16-request limit and a KV pool of
1,565,460 tokens, which accommodates one full 256k request. The two formats
share the same protocol and run sequentially; W4A4 starts after W4A8 completes
and passes verification. These startup checks establish the deployed settings;
the new accuracy results require the complete journals.

Evaluate both exports sequentially with
`recipes/eval-gpqa-diamond.spark-gpqa.yaml`. This is the GPQA baseline recipe: embedded
MTP is disabled during GPQA, although the exported vision and MTP tensors remain
in each checkpoint. The earlier text/image/MTP checks do not establish a
complete Spark GPQA result.

| Setting | Spark GPQA protocol |
|---|---|
| Dataset | Official GPQA Diamond CSV, all 198 rows, no `--limit` |
| Sampling | `temperature=0`, `top_p=0.95`, `top_k=20`, `min_p=0`, `presence_penalty=0`, `repetition_penalty=1` |
| Thinking | Enabled and preserved, `reasoning_effort=xhigh` |
| Seed and choices | Seed 0, shuffled choices |
| Context and output | 262144 context (256k); `max_new_tokens=0` fills its remaining window; `continue_on_length=true` |
| HTTP timeout | `generation.http_timeout_seconds=86400`: 24 hours per HTTP request, including each continuation; not a deadline for the full evaluation |
| Serving | TP1, 16 requests, 64 FP32 SSM slots, FP8 KV, 0.70 memory fraction, FlashInfer attention |
| Disabled | MTP, HiCache, host KV offload and CUDA graphs |

The full GPQA recipe now uses **262144** context, matching the default, 5090 and
6000D GPQA recipes. The separate four-request vision/MTP validation recipes
remain at 32768. Continuation is bounded by the remaining context and at most
eight follow-up requests; it does not create a larger context. A final length
finish counts as incorrect even if the text contains the gold letter. The
historical 32k journal stays separate from the new 256k runs.

The GPQA recipe increases concurrency from the four requests used by the
vision/MTP validation recipes to 16. It reserves 64 FP32 SSM slots and keeps the
same 0.70 memory fraction. Check available shared RAM and the server's actual
KV/Mamba allocations before starting; loading a recipe does not establish that
a full 198-item run succeeds. Use a new output directory when changing
concurrency or context instead of mixing rows from different protocols.

Use a serving image built from the current checkout for the Compose client
below. The original v0.1.2 image predates the GPQA client repair: SGLang extensions
must be top-level JSON fields, rather than a nested SDK `extra_body` object.
The repaired client also reserves chat-template tokens and counts truncated
answers as wrong. When retaining an older serving image and using a separate
updated CPU client, record both source revisions and the serving image digest.

Fetch the official CSV on the host into `DATA_DIR`; the Spark input mount is
read only inside the containers. `GPQA_CSV` below is the **container** path.
Keep the CSV unchanged, including any repeated answer options.

```bash
set -e
GPQA_DIR="$DATA_DIR" bash scripts/fetch_gpqa.sh
export GPQA_CSV=/data/dataset/gpqa_diamond.csv
export SPARK_EVAL_RECIPE=recipes/eval-gpqa-diamond.spark-gpqa.yaml
export SGLANG_SPARK_PORT="${SGLANG_SPARK_PORT:-30000}"
export GPQA_RUN="outputs/spark-gpqa-256k-$(date -u +%Y%m%dT%H%M%SZ)"

# The checker overrides the entrypoint, so verify its baked image explicitly.
docker compose --profile spark run --rm --entrypoint python eval-gpqa-spark \
  scripts/container_manifest.py verify

for format in W4A8 W4A4; do
  export SPARK_EVAL_MODEL="outputs/Qwen3.8-27B-NVFP4-${format}-spark"
  run_name="${format,,}-temp0-ctx262144-c16"
  docker compose --profile spark run --rm eval-gpqa-spark \
    eval -c "$SPARK_EVAL_RECIPE" --model "$SPARK_EVAL_MODEL" \
    --output "$GPQA_RUN/$run_name" --dry-run

  docker compose --profile spark up -d serve-sglang-spark
  timeout 600 bash -c 'until curl -fsS "http://127.0.0.1:${SGLANG_SPARK_PORT}/health" >/dev/null; do sleep 5; done'
  docker compose --profile spark run --rm eval-gpqa-spark \
    eval -c "$SPARK_EVAL_RECIPE" --model "$SPARK_EVAL_MODEL" \
    --output "$GPQA_RUN/$run_name"
  docker compose --profile spark run --rm --entrypoint python eval-gpqa-spark \
    scripts/check_gpqa_results.py --csv "$GPQA_CSV" \
    --run-dir "$GPQA_RUN/$run_name" \
    --output-json "$GPQA_RUN/$run_name-verification.json" --seed 0
  docker compose --profile spark stop serve-sglang-spark
done
```

Run these commands in Bash so a failed evaluation or checker stops the
sequence. Stop an existing MTP service before loading the baseline on the
single GPU. The HTTP client and checker attach no GPU. The two journals and
summaries appear under `OUTPUTS_DIR/spark-gpqa-256k-<timestamp>/` in
`w4a8-temp0-ctx262144-c16` and `w4a4-temp0-ctx262144-c16`.
Keep `GPQA_RUN` fixed when resuming an interrupted run: the client skips saved
`item_id` values. Start a new directory when changing weights, temperature,
context, seed, choice shuffle, server settings or client version.

For a CPU client outside the serving image, install the current repository in
a Python 3.10+ environment; a local CSV does not require Torch or Hub datasets.
Inside the same loop above, replace the Compose evaluation and checker commands
with the host commands below. This keeps the selected format, serving settings
and dated run directory the same:

```bash
python3 -m venv .venv-gpqa
.venv-gpqa/bin/python -m pip install .

# With the corresponding baseline server already ready inside the loop:
MEGAQUANT_SKIP_GPU_REPORT=1 GPQA_CSV="$DATA_DIR/dataset/gpqa_diamond.csv" \
  .venv-gpqa/bin/megaquant eval -c recipes/eval-gpqa-diamond.spark-gpqa.yaml \
  --model "$SPARK_EVAL_MODEL" \
  --base-url "http://127.0.0.1:${SGLANG_SPARK_PORT}/v1" \
  --output "$OUTPUTS_DIR/${GPQA_RUN#outputs/}/$run_name"
.venv-gpqa/bin/python scripts/check_gpqa_results.py \
  --csv "$DATA_DIR/dataset/gpqa_diamond.csv" \
  --run-dir "$OUTPUTS_DIR/${GPQA_RUN#outputs/}/$run_name" \
  --output-json "$OUTPUTS_DIR/${GPQA_RUN#outputs/}/$run_name-verification.json" --seed 0
```

Create the CPU environment before starting the loop. A score is publishable
only after `summary.json` exists and the
checker passes: it requires exactly 198 unique dataset-matching records,
rebuilds shuffled choices and gold letters from the CSV, reparses responses,
counts truncated/unparsed rows as wrong, and compares the recomputed scores
with the summary. Its JSON contains aggregate checks and file hashes, without
questions, answers, item IDs or local paths. Retain raw journals, the CSV,
server logs and run metadata locally; publish the sanitized verification
report and actual protocol with the final score. The historical W4A8 measurement
above used 32k context. Publish new W4A8 and W4A4 256k scores only after each
full run passes; reusing earlier 32k rows would invalidate the new result.

## Transfer a built image

The images contain the code and recipes, while weights and calibration remain
external. To move the exact local images without using a registry:

```bash
docker image save "$MEGAQUANT_SPARK_IMAGE" "$SGLANG_SPARK_IMAGE" \
  -o megaquant-spark-images.tar
sha256sum megaquant-spark-images.tar
```

On the destination, verify the archive checksum, use `docker image load`, set
the same image tags and external input paths, then repeat manifest and GPU
checks. Sharing an archive does not by itself establish that a second machine
completed quantization or inference.
