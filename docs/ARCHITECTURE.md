# Mega-Quantification architecture

管线把 YAML recipe 转成量化计划，再从 BF16 模型导出 checkpoint。模型适配由 family adapter、backend 和 recipe 共同完成；本仓库的完整量化与服务实测以 Qwen/Qwen3.8-27B 为主。

本文说明配置、模块和导出流程。运行命令见 [Qwen3.8 手册](qwen3.8-27b.md)、[Spark 容器手册](DGX_SPARK.md) 和 [Docker 手册](../docker/README.md)；实测结果与适用范围见 [验证记录索引](validation/README.md)。更多文档见 [文档索引](README.md)，代理入口为 [SKILL.md](../SKILL.md)。

目录：[量化格式](#schemes) · [仓库布局](#repository-layout) · [配置](#recipe-schema-megaquantconfig) · [模型适配](#model-family-protocol) · [ModelOpt 映射](#modelopt-mapping) · [导出流程](#pipeline-steps) · [容器](#docker--compose-rtx-5090) · [CLI](#cli)

## Schemes

| Scheme | Weights | Activations | Export `quant_algo` | Runtime |
|---|---|---|---|---|
| `nvfp4_w4a8`, `nvfp4_mixed` | MLP + `lm_head`: NVFP4 E2M1 group 16. Attention projections: FP8 E4M3 | MLP + `lm_head`: NVFP4 group 16. Attention: FP8 | `MIXED_PRECISION`; per layer `NVFP4` or `FP8` | SGLang `modelopt_mixed` |
| `nvfp4_w4a4` | NVFP4 group 16 on every targeted LM linear | NVFP4 group 16 | `NVFP4` | SGLang `modelopt_fp4` |
| `nvfp4_w4a16_mixed` | MLP + `lm_head`: NVFP4 group 16. Attention: FP8 | MLP activations stay BF16. Attention activations: FP8 | MLP entry `W4A16_NVFP4` | optional Marlin export |

均匀 W4A4 走 `NVFP4_DEFAULT_CFG`。本仓库的方案停在 group 16；ModelOpt 的 `W4A8_NVFP4_FP8`（NVFP4 block 32）不在表里，SGLang 也不收那个标签。

上表描述语言模型线性层。embeddings、GDN `conv1d` / `in_proj_a` / `in_proj_b` 等忽略项保留 BF16。通用配方默认保留 BF16 视觉和 MTP；Spark 配方显式启用两者量化，仍保留视觉位置与 patch embedding。KV cache 精度由服务配方控制，不属于权重量化格式。

## Repository layout

```
src/megaquant/          library
  cli.py                quantize / serve / eval / rewrite-sglang
  config.py             Pydantic recipe schema
  pipeline.py           load → calibrate → quantize → export
  eval_gpqa.py          GPQA Diamond client (Qwen thinking, SGLang, KV offload)
  sglang_export.py      MIXED_PRECISION rewrite for SGLang
  calibration.py        calibration batches
  backends/             ModelOpt and llm-compressor
  models/               family adapters (qwen3_5, qwen3, llama, generic)
  schemes/              named quant schemes
recipes/                quant and GPQA YAML (index: recipes/README.md)
tests/                  CPU tests; dry-run does not download weights
docs/                   runbooks, architecture, releases and dated validation
docker/                 image helpers; Dockerfiles stay at the repo root
scripts/                gpu-pod, OSS publish/fetch, serve and eval wrappers
SKILL.md                agent entry (humans: README and the runbook)
```

Weights, Hub caches, journals, `.env`, `.oss.env`, and `docker-compose.override.yml` stay out of git.

## Recipe schema (`megaquant.config`)

量化配置来自 [config.py](../src/megaquant/config.py)，使用 Pydantic 严格校验，未知字段会报错。下面保留主要字段；服务与评测使用 [eval_gpqa.py](../src/megaquant/eval_gpqa.py) 中的独立 schema。

```python
class PrecisionSpec(BaseModel):
    format: Literal["nvfp4", "fp8", "mxfp4", "bf16", "int4"]
    bits: int
    group_size: int | None = None   # W4A4 and SGLang/NVIDIA mixed NVFP4 layers: 16
    scale_dtype: str = "float8_e4m3fn"
    dynamic: bool | str = False
    strategy: str | None = None     # tensor | channel | group | tensor_group | token
    observer: str | None = None

class LayerGroup(BaseModel):
    name: str
    targets: list[str]              # glob / regex against module names
    weights: PrecisionSpec
    activations: PrecisionSpec | None = None
    ignore: list[str] = []

class CalibrationSpec(BaseModel):
    dataset: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    num_samples: int = 512
    max_seq_length: int = 2048
    batch_size: int = 1
    seed: int = 42
    with_images: bool = False
    text_field: str | None = None

class ExportSpec(BaseModel):
    output_dir: str
    format: Literal["hf", "compressed-tensors"] = "hf"
    pack: bool = True

class ModelSpec(BaseModel):
    source: str                     # HF id or local path
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"
    quantize_vision: bool = False
    quantize_mtp: bool = False
    ignore_lm_head: bool | None = None

class Recipe(BaseModel):
    name: str
    model: ModelSpec
    backend: Literal["auto", "modelopt", "llmcompressor"] = "auto"
    scheme: str                     # catalog key, e.g. nvfp4_w4a8
    groups: list[LayerGroup] | None = None  # optional overrides
    algorithm: str = "max"          # max | local_hessian | mse
    kv_cache: str | None = "fp8"
    family: str | None = None       # qwen3_5 | qwen3 | llama | generic
    extra_ignore: list[str] = []
    ignore_lm_head: bool | None = None
    calibration: CalibrationSpec
    export: ExportSpec
```

Named schemes live in `megaquant.schemes.catalog.SCHEME_CATALOG`.
`groups` in a YAML recipe override the catalog defaults.

## Backend protocol

```python
class QuantBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def supports(self, scheme_name: str) -> bool: ...
    def build_quant_cfg(self, plan) -> Any: ...
    def quantize(self, model, plan, calib_iter) -> Any: ...
    def export(self, model, recipe, tokenizer=None) -> Path: ...
```

`auto` backend: prefer `modelopt` for NVFP4 (mixed W4A8 and uniform W4A4),
else `llmcompressor`.

Dry-run (no GPU, missing optional deps) must still:

1. Validate the recipe
2. Resolve family ignore lists
3. Print the plan (targets, ignores, backend, algorithm, calib size)
4. Exit 0 with `--dry-run`

## Model family protocol

```python
class ModelFamily(Protocol):
    name: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    def default_ignore(self, recipe) -> list[str]: ...
    def load_kwargs(self, recipe) -> dict: ...
```

`qwen3_5` covers Qwen3.5 / 3.6 / 3.8 dense VLMs (`Qwen3_5ForConditionalGeneration`).

Default ignore for `qwen3_5` (language-model W4A8):

- vision tower: `*visual*`, `*vision*`
- embeddings: `*embed_tokens*`, `*embed_positions*`
- GDN extras: `*linear_attn.conv1d*`, `*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*`
- MTP: `*mtp*` (leave BF16 **in the export** as `mtp.safetensors`; do not
  drop the tensors. `quantize_mtp: true` calibrates and quantizes the attached MTP module)
- lm_head: **do not ignore** for mixed NVIDIA recipe; ignore only if recipe says so
- norms stay unquantized automatically (not Linear)

After `export_hf_checkpoint`, `megaquant.vision_export.restore_vision` copies
the skipped Qwen3.5 `model.visual.*` BF16 weights from the original HF source
into `vision.safetensors`, restores the image/video processor configs, and
sets `language_model_only: false` in the exported config.
`megaquant.mtp_export.restore_bf16_mtp` copies ignored `mtp.*` weights back
from the in-memory module or original HF source.
Both restored shards are indexed in `model.safetensors.index.json`.
When vision/MTP quantization is enabled, ModelOpt exports those modules into
the main checkpoint and the restoration helpers retain their quantized tensors
and metadata. Image calibration must execute the vision tower; MTP calibration
uses target hidden states and next-token embeddings. The export records actual
activation-quantizer calls in `calibration_coverage.json`.

The BF16 MTP restoration path also writes a sibling `<export>-draft` directory
for an optional external draft. Spark uses quantized embedded MTP in the full
export, configured with `serve.speculative_*` fields. `megaquant serve` does
not add `--speculative-draft-model-path`; embedded MTP must keep the full
checkpoint's quantization metadata. See [Spark vision/MTP inference](DGX_SPARK.md#vision-and-mtp-inference)
for the measured serving path.

## Qwen3.8-27B facts

- HF id: `Qwen/Qwen3.8-27B`
- `model_type`: `qwen3_5`
- `architectures`: `Qwen3_5ForConditionalGeneration`
- 27B dense VLM, 64 layers, hidden 5120, FFN 17408
- Layout: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- Native context 262,144; MTP present; vision encoder present
- NVIDIA mixed NVFP4 (`nvidia/Qwen3.8-27B-NVFP4`): Local-Hessian, 2048 samples,
  `Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0
- 5090 production mixed: ModelOpt 0.46.1, algorithm `max`, ultrachat 256×1024,
  batch 1 (`qwen3.8-27b-nvfp4-mixed.5090.yaml`). It uses the same language-layer
  map with a different calibration dataset and algorithm. Recorded runs and
  quality comparisons belong in the [runbook](qwen3.8-27b.md).
- Mixed encoding: NVFP4 **group_size 16** on MLP + `lm_head`; FP8 on self-attn +
  linear-attn. Default `nvfp4_w4a8` matches this for SGLang. Not
  `W4A8_NVFP4_FP8` / NVFP4 block 32.

## ModelOpt mapping

| Our scheme | ModelOpt object / qformat | NVFP4 block |
|---|---|---|
| `nvfp4_w4a8` | mixed overrides + `MIXED_PRECISION` export / `mixed_nvfp4_fp8` | **16** on MLP + `lm_head`; FP8 on attention (SGLang) |
| `nvfp4_mixed` | same mixed cfg / `mixed_nvfp4_fp8` | **16** on MLP + `lm_head` (NVIDIA public mapping) |
| `nvfp4_w4a4` | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | **16** weights and activations, uniform |
| `nvfp4_w4a16` | `mtq.W4A16_NVFP4_CFG` / `w4a16_nvfp4` | 16, weight-only |
| `nvfp4_w4a16_mixed` | mixed W4A16 MLP + FP8 attn / `W4A16_NVFP4` on MLP | 16 MLP weights, BF16 MLP activations, FP8 attention. Optional Marlin export, not the Spark fast path |
| `fp8_w8a8` | `mtq.FP8_DEFAULT_CFG` / `fp8` | n/a (FP8) |

SGLang-serving NVFP4 uses **group_size 16** on MLP + `lm_head`. Do not
describe `nvidia/Qwen3.8-27B-NVFP4` as ModelOpt `W4A8_NVFP4_FP8` (NVFP4
block 32). SGLang rejects that tag, and this repo does not quantize to it.
A loaded `W4A8_NVFP4_FP8_CFG` is only a fallback base that mixed quant
rewrites to group_size 16.

Export: `modelopt.torch.export.export_hf_checkpoint(model, export_dir)` then
`megaquant.sglang_export.rewrite_sglang_mixed_export` for `nvfp4_w4a8` /
`nvfp4_mixed`. ModelOpt 0.46 often writes a single `NVFP4` /
`W4A8_NVFP4_FP8` tag; the rewriter rebuilds `quantized_layers` from the
weight map.

Quantize: `mtq.quantize(model, quant_cfg, forward_loop)`.

Algorithm mapping (`_apply_algorithm`):

| Recipe `algorithm` | ModelOpt `cfg["algorithm"]` | Used on |
|---|---|---|
| `max` (default) | `"max"` — amax / RTN, no Hessian | 5090 production mixed / W4A8 / W4A4 |
| `local_hessian` | `{method: local_hessian, fp8_scale_sweep: True, block_size: 16}` | Mixed quality recipe and optional Spark W4A4 experiment |
| `mse` | `{method: mse, fp8_scale_sweep: True}` for NVFP4; `{method: mse}` for other formats | Optional Spark W4A4 experiment |

For NVFP4, `_apply_algorithm` switches weight block scales to `static` for
MSE/Local-Hessian before ModelOpt calibration. Inputs retain dynamic scales,
activation calibration remains max, and the packed group-16 export stays the
same. An algorithm name alone would leave the default dynamic weight scales
outside the intended scale search. The `max` configuration retains its existing
dynamic scales.

The controlled Spark recipes are `qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml` and
`qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml`. Check actual GPU search behavior
with `scripts/check_nvfp4_calibration.py`; complete results and stopping limits
are recorded in the [W4A4 calibration report](validation/dgx-spark-w4a4-quality-20261008.md).

VLM: quantize the language model; keep the vision encoder in BF16 unless
`quantize_vision: true`.

## llm-compressor mapping

There is **no** stock `NVFP4A8` preset. Default `nvfp4_w4a8` uses two custom
groups matching the NVIDIA mixed map. This backend does not build ModelOpt
`W4A8_NVFP4_FP8` (NVFP4 block 32 + FP8 activations).

- weights: FP4, `TENSOR_GROUP`, `group_size=16` for uniform W4A4 and for
  NVIDIA/SGLang mixed MLP + `lm_head`, `scale_dtype=float8_e4m3fn`
- activations: NVFP4 (group 16) for uniform W4A4; mixed follows the NVIDIA
  mapping (NVFP4 on MLP + `lm_head`, FP8 on attention)
- ignore: family ignore list + `lm_head` when the recipe says so
- save: `model.save_pretrained(dir, save_compressed=True)`

The output is a compressed-tensors checkpoint (`quant_method: compressed-tensors`).
It is not the ModelOpt `MIXED_PRECISION` checkpoint that SGLang `modelopt_mixed`
loads, and `megaquant rewrite-sglang` refuses it. Use ModelOpt for the SGLang
serving path; the published scores all come from ModelOpt exports.

## Pipeline steps

1. Load YAML → `Recipe`
2. Detect family from `config.json` `model_type` / `architectures` (or recipe.family)
3. Resolve scheme catalog + family ignore + extra_ignore
4. If `--dry-run`, print plan and stop
5. Load tokenizer/processor + model in BF16, attaching MTP when requested
6. Build text or image/text calibration iterator
7. `backend.quantize(...)`, including opt-in vision/MTP calibration coverage
8. `backend.export(...)`; preserve quantized vision/MTP when enabled, otherwise
   restore BF16 tensors and index them. The BF16 MTP path also writes an
   optional `<export>-draft`. Rewrite the mixed ModelOpt layer map for SGLang.
9. Write `provenance.json` (base model, scheme, backend, calib, git sha, timestamp)

## Docker / Compose (RTX 5090+)

Compose provides the packaged Blackwell workflow. GPU pods without Docker use
the repository's host wrapper. Spark uses separate ARM64 PTQ and serving
images; build and provenance checks are in the [Spark runbook](DGX_SPARK.md).

- Image: `Dockerfile` (CUDA 12.8 devel + PyTorch cu128 + ModelOpt + llm-compressor + megaquant)
- Orchestration: `docker-compose.yml` — `megaquant` service dry-runs; `quantize`,
  `w4a4`, and `mixed` are behind `--profile gpu`. `mixed` defaults to
  `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` (**production**: `max` +
  ultrachat 256); NVIDIA quality Local-Hessian is
  `MIXED_RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml`. 5090 GPQA is
  `recipes/eval-gpqa-diamond.5090.yaml` (24-way, HiCache 64 GiB, bf16 GDN).
- GPU pod (no Docker): `bash scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`
  (starts a job; does not mean W4A4 / mixed PTQ or OSS upload has already finished).
  `serve`/`eval` default to SGLang (`:30000`, HiCache KV → RAM).
- After mixed / default-W4A8 export: `megaquant rewrite-sglang <export_dir>`
  before SGLang serve if `hf_quant_config.json` is bare NVFP4 without
  `quantized_layers`. A `W4A8_NVFP4_FP8` export is not SGLang-loadable, and
  this repo does not quantize to that tag.
- Publish: `python scripts/oss_publish.py <export> --scheme w4a4|mixed|w4a8`
  and/or `bash scripts/gpu-pod.sh publish [scheme]`. Bucket/endpoint from
  `OSS_BUCKET` / `OSS_ENDPOINT` (optional `.oss.env`). Mixed encoding **is**
  default W4A8 — publish the same mixed export twice as `mixed` and `w4a8`
  (same content-hash); do not run a second PTQ.
- Host check: `bash docker/host-check.sh` (driver 570+, NVIDIA Container Toolkit)
- Move the box: `make image-tar` then `docker image load` on the 5090
- Weights stay on the host: `./.cache/huggingface`, `./models`, `./outputs`
- Runbook: [docker/README.md](../docker/README.md)

## CLI

```
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-mixed
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
megaquant write-mtp-draft outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant serve -c recipes/eval-gpqa-diamond.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.yaml --dry-run
megaquant serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
megaquant schemes
megaquant families
```

CLI overrides: `--model`, `--output`, `--backend`, `--num-samples`, `--algorithm`.
`serve` / `eval` default to SGLang `:30000`; cache settings come from the selected recipe. Pass `--engine vllm`
only when you want the optional GB300 vLLM flags.
