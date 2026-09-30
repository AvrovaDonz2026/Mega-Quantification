# DGX Spark containers

This runbook builds the repository's ARM64 / CUDA 13 images and runs full-27B
W4A8 and W4A4 quantization, image inference and embedded MTP checks on one GB10.
The recipes use TP1. Cross-node TP2 requires a separate distributed validation.

The frozen-container changes are in revision `03949ba`. Clean image builds and
the complete GPU/PTQ/vision/MTP rerun are still being verified. The earlier
[quantization](validation/dgx-spark-pr46-20260929.md) and
[serving](validation/dgx-spark-serving-20260930.md) reports describe their actual
environments; they do not prove a clean build of these new images. No published
registry image is claimed here.

## What the images contain

| Image | Public base | Repository additions |
|---|---|---|
| `megaquant:gb10` | NGC PyTorch 26.08, pinned by SHA256 in `Dockerfile.spark` | Complete Spark PTQ dependency version list, installed MegaQuant wheel, recipes and scripts |
| `megaquant:sglang-spark` | SGLang 0.5.20 CUDA 13, pinned by SHA256 in `Dockerfile.sglang.spark` | Installed MegaQuant wheel, guarded visual/MTP loader patch, recipes and scripts |

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

## Vision and MTP inference

Verify the serving image's manifest, loader patch, ARM64/SM121 GPU and CUDA
matrix multiplication before loading the full checkpoint:

```bash
docker compose --profile spark run --rm --entrypoint python serve-sglang-spark \
  scripts/check_sglang_spark.py
```

For each format, start the baseline and MTP servers separately. Both use baked
FP32 SSM state recipes. Baseline uses `eval-gpqa-diamond.spark.yaml`; MTP uses
`eval-gpqa-diamond.spark-mtp.yaml` with embedded EAGLE, two speculative steps,
top-k 1 and three draft tokens. No external draft checkpoint is supplied.

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
