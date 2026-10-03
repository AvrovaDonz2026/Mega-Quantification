---
name: megaquant
description: >
  Mega-Quantification PTQ pipeline for Qwen/Qwen3.8-27B BF16 to SGLang-serving
  NVFP4. Use when quantizing, exporting, serving, or scoring this repo; when
  the user mentions nvfp4_w4a8, nvfp4_w4a4, nvfp4_mixed, ModelOpt, SGLang GPQA,
  MTP, or OSS publish. Read this file before editing recipes, export code, or
  docs. Human narrative lives in README.md and docs/qwen3.8-27b.md.
---

# Mega-Quantification

Agent entry. Humans read `README.md` and `docs/qwen3.8-27b.md`.

## Hard rules

- `megaquant plan` and eval/serve `--dry-run` must not download the 27B checkpoint.
- A GPQA score is `correct/198` only when `summary.json` exists and the journal has exactly 198 unique dataset-matching rows. Truncated and unparsed rows count as wrong. Do not quote a partial journal. For temperature-0 runs, require `scripts/check_gpqa_results.py` to pass against the actual CSV, seed and choice-shuffle policy before publishing a score.
- Do not commit `.env`, `.oss.env`, `docker-compose.override.yml`, SSH material, bucket names, tenant DNS, or host paths.
- Do not publish the `*-draft` directory as scheme `w4a8`, `w4a4`, or `mixed`.
- Do not implement or document ModelOpt `W4A8_NVFP4_FP8` (NVFP4 block 32) as a supported scheme. SGLang rejects that tag.
- Re-running PTQ recomputes activation scales. It will not byte-match an existing export. Fetch a published content-hash when the bytes must match.
- `quantize_mtp: false` means keep MTP in BF16 inside `mtp.safetensors`. It does not mean delete the tensors.
- License is AGPL-3.0-or-later (`LICENSE`, Copyright 2026 Donz). Leave that grant in place.
- The pipeline has been industrially validated. The concrete runs are the 5090 production PTQ and the finished GPQA journals in `docs/qwen3.8-27b.md`. Do not invent customers, certifications, or extra scores.

## Schemes

| Scheme | What it is | Export | SGLang |
|---|---|---|---|
| `nvfp4_w4a8` | Default. Same map as `nvfp4_mixed`: NVFP4 group 16 on MLP + `lm_head`, FP8 on self-attn and linear-attn | `MIXED_PRECISION` + `quantized_layers` | `modelopt_mixed` |
| `nvfp4_mixed` | Same encoding. Quality YAML is Local-Hessian; `mixed.5090.yaml` is the 32 GB production run (`max`) | same | `modelopt_mixed` |
| `nvfp4_w4a4` | Uniform NVFP4 block 16, weights and activations, including attention | `NVFP4` | `modelopt_fp4` |
| `nvfp4_w4a16_mixed` | Optional Marlin export. MLP activations stay BF16 | MLP entry `W4A16_NVFP4` | not the Spark fast path |

5090 PTQ is `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`: ModelOpt **0.46.1** (pinned), algorithm `max`, ultrachat 256×1024, batch 1. Batch 4 can OOM at `lm_head` on a 32 GB instance. The export restores BF16 vision tensors into `vision.safetensors` and BF16 MTP tensors into `mtp.safetensors`; the HF index includes both. It writes `outputs/Qwen3.8-27B-NVFP4-W4A8` with 401 `quantized_layers` entries (193 NVFP4 + 208 FP8), which eval / serve use. Quality Local-Hessian (`mixed.yaml`) still writes `outputs/Qwen3.8-27B-NVFP4-mixed`. NVIDIA's public `nvidia/Qwen3.8-27B-NVFP4` uses the same layer map with Local-Hessian, Nemotron v3, 2048×2048, modelopt 0.48.0 (`recipes/qwen3.8-27b-nvfp4-mixed.yaml`). Card GPQA Diamond: 88.92 BF16 / 88.01 NVFP4 on GB300 vLLM. Qwen thinking card: 89.2. DGX Spark serves this mixed map at about 12 tok/s decode (bandwidth ceiling about 14); speculative MTP is the faster path.

Pinned in `pyproject.toml` and `docker/requirements-gpu.txt`: `nvidia-modelopt[hf]==0.46.1`. Serve image pins `sglang==0.5.20`.

Qwen3.5/3.8 PTQ supports `transformers>=5.8,<5.15`; CPU tests pin 5.14.1.
Transformers 4.x lacks the required model API. If the GDN hook reports an
unsupported API, install the declared range before retrying; do not bypass
the error and continue calibration with unpatched GDN calls.

Spark full-27B W4A8/W4A4 PTQ with vision and MTP passed 256-sample calibration
and export on 2026-09-29 (ModelOpt 0.47.0, Transformers 5.12.1). Environment,
artifact checks and limits: `docs/validation/dgx-spark-pr46-20260929.md`.
This validation does not establish SGLang inference quality or MTP acceptance.
Subsequent text/image and embedded MTP checks are recorded in
`docs/validation/dgx-spark-serving-20260930.md`. Spark serving uses FP32 SSM
states: BF16 states reproduced a W4A4 greedy-output mismatch between ordinary
decode and MTP verification. This changes the state cache, not W4A4 weights or
activations. Use `scripts/check_sglang_multimodal.py` to check image answers,
actual accepted drafts and exact baseline token IDs; its MTP gate must fail on
token mismatches or zero accepted drafts. The recorded checks are small TP1
tests, not a general quality or determinism guarantee.

## Eval

Default YAML sends temperature **0** plus the rest of the Qwen thinking card (`top_p=0.95`, `top_k=20`, thinking on and preserved, `reasoning_effort=xhigh`). Default, 5090, 6000D and Spark full GPQA recipes use 262144 context. The separate Spark vision/MTP checks retain 32768. `max_new_tokens: 0` fills the selected recipe's remaining window. Continuations cannot enlarge that window; final length finishes count as wrong. Journal: `<model>/gpqa_diamond/gpqa_diamond.jsonl`, or an explicit `--output` directory, resume by `item_id`.

Spark full GPQA uses `recipes/eval-gpqa-diamond.spark-gpqa.yaml` for both mixed
W4A8 and uniform W4A4, with 262144 context, the full remaining output budget,
seed 0, shuffled choices, 16 requests, 64 FP32 SSM slots and MTP disabled. The
vision/MTP validation recipes remain at 32768 context and four requests.
Check available shared RAM and actual server cache allocations before running
GPQA; do not merge 32k/256k or four-/16-request journals into one score.
Run the two exports sequentially on the single GPU. Use separate dated output
directories for both formats and every changed protocol; resume only the same
run with the same inputs and settings. Keep the official CSV unchanged,
including repeated answer options. Commands and limits are in
`docs/DGX_SPARK.md#full-gpqa-diamond-at-temperature-0`.

The original v0.1.2 baked eval client predates the raw HTTP parameter repair.
Use a current client that sends SGLang extensions as top-level JSON fields,
reserves chat-template tokens and applies strict truncation scoring. An older
serving image may use a separate updated CPU client; record both source
revisions and the serving image digest. The stdlib result checker accepts
`--csv`, `--run-dir`, `--output-json`, `--seed` and `--no-shuffle-choices`; it
requires complete evidence and emits hashes/aggregate checks without dataset
questions, response text, item IDs or local paths. Do not publish raw journals.

Finished journals:

- Mixed / default W4A8, temperature **1.0**, 80 GB SM120, SGLang 64-way, 2026-09-22: **178/198** (truncated 0, unparsed 0). Not a rerun of the temperature-0 YAML.
- Uniform W4A4, temperature **0**, Marlin, 2026-09-22: **172/198** (truncated 4, unparsed 6).
- Historical **32k trial**: Spark mixed W4A8 with quantized vision/MTP, temperature **0**, GB10 TP1, FlashInfer, 16 requests, 32768 context, MTP disabled during GPQA, 2026-10-02: **157/198 (79.29%)**, truncated 30, unparsed 32 (overlapping counts). All 198 dataset-matching records and the summary passed the strict checker. Evidence: `docs/validation/dgx-spark-gpqa-w4a8-20261002.json`. This is a length-limited 32k result, not the user's requested 262144-context result.
- The 32k Spark uniform W4A4 trial was stopped without a complete score. Both new 256k W4A8/W4A4 results remain pending; publish only after their separate complete runs pass verification. Use `w4a8-temp0-ctx262144-c16` and `w4a4-temp0-ctx262144-c16` under a new dated run directory; never resume those runs from 32k journals.

| Box | Recipe |
|---|---|
| CUDA ≥ 12.9 | `recipes/eval-gpqa-diamond.yaml` (FlashInfer, HiCache 12 GiB, concurrency 1) |
| 32 GB SM120 / CUDA 12.8 | `recipes/eval-gpqa-diamond.5090.yaml` (Triton + Marlin, HiCache 64 GiB, 24-way, CUDA graph off) |
| 80 GB SM120 / CUDA 12.8 | `recipes/eval-gpqa-diamond.6000d.yaml` (Triton + Marlin + CUTLASS, KV on GPU, 64-way, CUDA graph on, SiLU+FP4 fusion off) |
| DGX Spark ARM64 / SM121 / CUDA 13 | Full GPQA: `recipes/eval-gpqa-diamond.spark-gpqa.yaml` (262144 context, 16 requests, 64 FP32 SSM slots, 0.70 memory, no MTP/HiCache/CUDA graphs). Vision/MTP checks retain `eval-gpqa-diamond.spark.yaml` and `eval-gpqa-diamond.spark-mtp.yaml` at 32768 context and four requests. |

Compose default image is CUDA **12.8.1**. That build does not include a CUDA 13 FlashInfer toolchain. Optional rebuild: `SGLANG_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04`. Leave the 12.8 default in place unless asked.

`megaquant serve` does not add `--speculative-draft-model-path`.

Spark serving uses `Dockerfile.sglang.spark` and the `serve-sglang-spark` Compose
service, independently of the PTQ image. The base is pinned by digest; the
version-specific loader patch must fail on unrecognized sources. Do not claim
vision or quantized MTP serving from export coverage or an import check alone.
Spark CPU and GPU share RAM: do not enable automatic host HiCache allocation.
The Spark recipe is TP1; a two-node TP2 deployment needs explicit distributed
configuration. Optional `serve.speculative_*` fields enable embedded MTP, without
adding an external draft path.

The 2026-10-01 v0.1.2 release rerun at `ac53833` is recorded in
`docs/validation/dgx-spark-release-20261001.md`. Both images passed two offline
clean builds and isolated save/load checks. Fresh full W4A8/W4A4 quantization,
strict export audits, vision/OCR and quantized embedded MTP passed; each format
matched 205 baseline/MTP output IDs across five fixed cases. This remains a
single-host TP1 result, without quality, performance or TP2 claims. The public
stdlib auditor is `scripts/audit_spark_export.py`; pass the full PTQ image
revision with `--expected-revision` when checking export provenance.

## MTP

After export, `megaquant.mtp_export` copies BF16 `mtp.*` from the in-memory module or from the original HF source. Only shards whose index entries are `mtp.*` are read. The shard is `mtp.safetensors`. A sibling `<export>-draft` directory is the 1-layer SGLang draft (`--speculative-algorithm NEXTN`). Do not point the draft path at the 64-layer export.

Language-only Qwen3.5 PTQ skips constructing the vision tower. `megaquant.vision_export` copies the source BF16 `model.visual.*` tensors into `vision.safetensors` and adds them to the HF index. Existing exports missing vision can run `megaquant restore-vision <export> --source <BF16 checkpoint>`.

```bash
megaquant restore-vision outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
```

## OSS

Layout: `<prefix>/<scheme>/<content-hash>/`, prefix `Mega-Quantification`. Content-hash is SHA256 of sorted `{safetensors-name} {sha256}\n` lines. Publish one mixed export twice, `--scheme mixed` and `--scheme w4a8`, same hash. Bucket and endpoint come from the environment or `.oss.env`, never from git. Anonymous auth when access keys are unset.

## Commands

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
MEGAQUANT_LOW_MEMORY=1 megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
```

Tests are CPU-only: `python3 -m pytest` and `python3 -m ruff check src tests scripts`.

## Where code lives

- `src/megaquant/pipeline.py` — load, calibrate, quantize, export
- `src/megaquant/backends/modelopt.py` — quant config and HF export
- `src/megaquant/sglang_export.py` — `MIXED_PRECISION` rewrite; MTP stays out of `quantized_layers`
- `src/megaquant/mtp_export.py` — BF16 MTP restore and 1-layer draft
- `src/megaquant/eval_gpqa.py` — GPQA client and SGLang argv
- `recipes/` — YAML; index in `recipes/README.md`
- `docs/ARCHITECTURE.md` — schema and pipeline steps
- `docs/qwen3.8-27b.md` — human runbook
