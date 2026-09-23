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
- A GPQA score is `correct/198` only when `summary.json` exists and the journal has 198 rows. Truncated and unparsed rows count as wrong. Do not quote a partial journal.
- Do not commit `.env`, `.oss.env`, `docker-compose.override.yml`, SSH material, bucket names, tenant DNS, or host paths.
- Do not publish the `*-draft` directory as scheme `w4a8`, `w4a4`, or `mixed`.
- Do not implement or document ModelOpt `W4A8_NVFP4_FP8` (NVFP4 block 32) as a supported scheme. SGLang rejects that tag.
- Re-running PTQ recomputes activation scales. It will not byte-match an existing export. Fetch a published content-hash when the bytes must match.
- `quantize_mtp: false` means keep MTP in BF16 inside `mtp.safetensors`. It does not mean delete the tensors.
- License is AGPL-3.0-or-later (`LICENSE`, Copyright 2026 Donz). Leave that grant in place.

## Schemes

| Scheme | What it is | Export | SGLang |
|---|---|---|---|
| `nvfp4_w4a8` | Default. Same map as `nvfp4_mixed`: NVFP4 group 16 on MLP + `lm_head`, FP8 on self-attn and linear-attn | `MIXED_PRECISION` + `quantized_layers` | `modelopt_mixed` |
| `nvfp4_mixed` | Same encoding. Quality YAML is Local-Hessian; `mixed.5090.yaml` is the 32 GB production run (`max`) | same | `modelopt_mixed` |
| `nvfp4_w4a4` | Uniform NVFP4 block 16, weights and activations, including attention | `NVFP4` | `modelopt_fp4` |
| `nvfp4_w4a16_mixed` | Optional Marlin export. MLP activations stay BF16 | MLP entry `W4A16_NVFP4` | not the Spark fast path |

5090 production PTQ is `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`: ModelOpt **0.46.1** (pinned), algorithm `max`, ultrachat 256×1024, batch 4. The W4A8 output directory is that mixed export (401 `quantized_layers` entries: 193 NVFP4 + 208 FP8). NVIDIA's public `nvidia/Qwen3.8-27B-NVFP4` uses the same layer map with Local-Hessian, Nemotron v3, 2048×2048, modelopt 0.48.0 (`recipes/qwen3.8-27b-nvfp4-mixed.yaml`). Card GPQA Diamond: 88.92 BF16 / 88.01 NVFP4 on GB300 vLLM. Qwen thinking card: 89.2. DGX Spark serves this mixed map at about 12 tok/s decode (bandwidth ceiling about 14); speculative MTP is the faster path.

Pinned in `pyproject.toml` and `docker/requirements-gpu.txt`: `nvidia-modelopt[hf]==0.46.1`. Serve image pins `sglang==0.5.20`.

## Eval

Default YAML sends temperature **0** plus the rest of the Qwen thinking card (`top_p=0.95`, `top_k=20`, thinking on, `reasoning_effort=xhigh`). `max_new_tokens: 0` fills the remaining 262144 context. Journal: `<model>/gpqa_diamond/gpqa_diamond.jsonl`, resume by `item_id`.

Finished journals:

- Mixed / default W4A8, temperature **1.0**, 80 GB SM120, SGLang 64-way, 2026-09-22: **178/198** (truncated 0, unparsed 0). Not a rerun of the temperature-0 YAML.
- Uniform W4A4, temperature **0**, Marlin, 2026-09-22: **172/198** (truncated 4, unparsed 6).
- No finished 198-row mixed score at temperature 0.

| Box | Recipe |
|---|---|
| CUDA ≥ 12.9 | `recipes/eval-gpqa-diamond.yaml` (FlashInfer, HiCache 12 GiB, concurrency 1) |
| 32 GB SM120 / CUDA 12.8 | `recipes/eval-gpqa-diamond.5090.yaml` (Triton + Marlin, HiCache 64 GiB, 24-way, CUDA graph off) |
| 80 GB SM120 / CUDA 12.8 | `recipes/eval-gpqa-diamond.6000d.yaml` (Triton + Marlin + CUTLASS, KV on GPU, 64-way, CUDA graph on, SiLU+FP4 fusion off) |

Compose default image is CUDA **12.8.1**. That build does not include a CUDA 13 FlashInfer toolchain. Optional rebuild: `SGLANG_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04`. Leave the 12.8 default in place unless asked.

`megaquant serve` does not add `--speculative-draft-model-path`.

## MTP

After export, `megaquant.mtp_export` copies BF16 `mtp.*` from the in-memory module or from the original HF source. Only shards whose index entries are `mtp.*` are read. The shard is `mtp.safetensors`. A sibling `<export>-draft` directory is the 1-layer SGLang draft (`--speculative-algorithm NEXTN`). Do not point the draft path at the 64-layer export.

```bash
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
```

## OSS

Layout: `<prefix>/<scheme>/<content-hash>/`, prefix `Mega-Quantification`. Content-hash is SHA256 of sorted `{safetensors-name} {sha256}\n` lines. Publish one mixed export twice, `--scheme mixed` and `--scheme w4a8`, same hash. Bucket and endpoint come from the environment or `.oss.env`, never from git. Anonymous auth when access keys are unset.

## Commands

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
```

Tests are CPU-only: `python -m pytest` and `python -m ruff check src tests scripts`.

## Where code lives

- `src/megaquant/pipeline.py` — load, calibrate, quantize, export
- `src/megaquant/backends/modelopt.py` — quant config and HF export
- `src/megaquant/sglang_export.py` — `MIXED_PRECISION` rewrite; MTP stays out of `quantized_layers`
- `src/megaquant/mtp_export.py` — BF16 MTP restore and 1-layer draft
- `src/megaquant/eval_gpqa.py` — GPQA client and SGLang argv
- `recipes/` — YAML; index in `recipes/README.md`
- `docs/ARCHITECTURE.md` — schema and pipeline steps
- `docs/qwen3.8-27b.md` — human runbook
