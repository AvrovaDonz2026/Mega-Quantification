# Mega-Quantification architecture

Generic post-training quantization (PTQ) pipeline. First production target:
**Qwen/Qwen3.8-27B BF16 → SGLang-serving NVFP4** (recipe `nvfp4_w4a8`).

The pipeline is model-agnostic. A recipe YAML plus a model-family adapter should
be enough to quantize any Hugging Face causal / VLM checkpoint.

中文要点：默认 `nvfp4_w4a8` 对齐 NVIDIA 混合**层图**（MLP + `lm_head` NVFP4
gs16，注意力 FP8，导出 `MIXED_PRECISION`）。5090 生产 PTQ 是
`mixed.5090.yaml`：ModelOpt 0.46.1、算法 **`max`**、ultrachat 256×1024。
NVIDIA 公开权重才是 Local-Hessian + Nemotron v3 2048（modelopt 0.48.0）。
5090 GPQA 24 路 / HiCache 64 GiB / bf16 GDN；journal 跑完前不要报 198 分。

## Goal

- Default W4A8 (`nvfp4_w4a8`): NVIDIA mixed map so **SGLang can serve it**.
  NVFP4 (E2M1, group_size **16**) on MLP + `lm_head`; FP8 E4M3 on self-attn +
  linear-attn. Export `quant_algo=MIXED_PRECISION` + `quantized_layers`.
- Uniform W4A4 (`nvfp4_w4a4`): NVFP4 block **16** weights and activations.
- TensorRT-LLM uniform W4A8 (`w4a8_nvfp4_fp8`): NVFP4 block **32** + FP8
  activations (`W4A8_NVFP4_FP8`). SGLang rejects that `quant_algo`.

NVIDIA's public `nvidia/Qwen3.8-27B-NVFP4` checkpoint is mixed NVFP4/FP8:
**NVFP4 group_size 16** on MLP + `lm_head`, **FP8** on self-attn + linear-attn.
Default `nvfp4_w4a8` matches that map. Card: Local-Hessian, 2048 samples,
`Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0.

This repo ships four Qwen3.8-27B schemes. Uniform W4A4 is `NVFP4_DEFAULT_CFG`
block 16. Default W4A8 and mixed share the NVIDIA gs16/FP8 map.

**5090 production PTQ is `max`, not Hessian.** Compose `mixed` and
`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` use ModelOpt **0.46.1** (PyPI;
0.48 is not published there), algorithm **`max`** (amax / RTN),
`HuggingFaceH4/ultrachat_200k` 256×1024 batch 4. Same mixed encoding as
`nvfp4_w4a8`; the W4A8 export dir is that checkpoint. Quality Local-Hessian
is `recipes/qwen3.8-27b-nvfp4-mixed.yaml` and needs a larger GPU.

5090 GPQA (`recipes/eval-gpqa-diamond.5090.yaml`): 24-way, HiCache 64 GiB
(split by GPU KV vs GDN pool), bf16 mamba 96 slots, Triton + Marlin. Do not
quote a partial GPQA journal as `correct/198`.

| Recipe | Meaning |
|---|---|
| `nvfp4_w4a8` | SGLang mixed: NVFP4 group_size **16** on MLP + `lm_head`, FP8 on attention; `MIXED_PRECISION` export |
| `nvfp4_w4a4` | Uniform NVFP4 W4A4 (`NVFP4_DEFAULT_CFG`, block **16** weights and activations) |
| `nvfp4_mixed` | Same encoding as `nvfp4_w4a8`; quality recipe uses Local-Hessian |
| `w4a8_nvfp4_fp8` | TensorRT-LLM uniform NVFP4 block **32** + FP8 activations |

## Package layout (file ownership)

```
src/megaquant/config.py            # Agent Core — Pydantic recipe schema
src/megaquant/exceptions.py        # Agent Core
src/megaquant/registry.py          # Agent Core
src/megaquant/pipeline.py          # Agent Core
src/megaquant/cli.py               # Agent Core
src/megaquant/eval_gpqa.py         # GPQA Diamond (Qwen thinking + SGLang serve + KV CPU offload)
src/megaquant/sglang_export.py     # MIXED_PRECISION rewrite for SGLang
src/megaquant/calibration.py       # Agent Core
src/megaquant/__init__.py          # Agent Core
src/megaquant/backends/base.py     # Agent ModelOpt — Protocol
src/megaquant/backends/modelopt.py # Agent ModelOpt
src/megaquant/backends/__init__.py # Agent ModelOpt
src/megaquant/backends/llmcompressor.py  # Agent LLMCompressor
src/megaquant/schemes/             # Agent LLMCompressor
src/megaquant/models/              # Agent Recipes
recipes/                           # Agent Recipes
tests/                             # Agent Recipes
```

Do not edit files outside your ownership list. Do not `git commit`.

## Recipe schema (`megaquant.config`)

```python
class PrecisionSpec(BaseModel):
    format: Literal["nvfp4", "fp8", "mxfp4", "bf16", "int4"]
    bits: int
    group_size: int | None = None   # W4A4 and SGLang/NVIDIA mixed NVFP4 layers: 16; TRT-LLM W4A8_NVFP4_FP8: 32
    scale_dtype: str = "float8_e4m3fn"
    dynamic: bool = False
    strategy: str | None = None     # tensor | channel | group | tensor_group | token

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
    def quantize(self, model, resolved, calib_iter) -> Any: ...
    def export(self, model, recipe, tokenizer=None) -> Path: ...
```

`auto` backend: prefer `modelopt` for NVFP4 (mixed W4A8 / W4A4 / `W4A8_NVFP4_FP8`),
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
- MTP: `*mtp*`
- lm_head: **do not ignore** for mixed NVIDIA recipe; ignore only if recipe says so
- norms stay unquantized automatically (not Linear)

## Qwen3.8-27B facts

- HF id: `Qwen/Qwen3.8-27B`
- `model_type`: `qwen3_5`
- `architectures`: `Qwen3_5ForConditionalGeneration`
- 27B dense VLM, 64 layers, hidden 5120, FFN 17408
- Layout: `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`
- Native context 262,144; MTP present; vision encoder present
- NVIDIA mixed NVFP4 (`nvidia/Qwen3.8-27B-NVFP4`): Local-Hessian, 2048 samples,
  `Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0
- 5090 production mixed: ModelOpt 0.46.1, algorithm `max`, ultrachat 256×1024
  batch 4 (`qwen3.8-27b-nvfp4-mixed.5090.yaml`). Same layer map; weaker
  calibrator. 32 GB cannot hold Hessian 2048.
- Mixed encoding: NVFP4 **group_size 16** on MLP + `lm_head`; FP8 on self-attn +
  linear-attn. Default `nvfp4_w4a8` matches this for SGLang. Not
  `W4A8_NVFP4_FP8` / NVFP4 block 32.

## ModelOpt mapping

| Our scheme | ModelOpt object / qformat | NVFP4 block |
|---|---|---|
| `nvfp4_w4a8` | mixed overrides + `MIXED_PRECISION` export / `mixed_nvfp4_fp8` | **16** on MLP + `lm_head`; FP8 on attention (SGLang) |
| `nvfp4_mixed` | same mixed cfg / `mixed_nvfp4_fp8` | **16** on MLP + `lm_head` (NVIDIA public mapping) |
| `w4a8_nvfp4_fp8` | `mtq.W4A8_NVFP4_FP8_CFG` / `w4a8_nvfp4_fp8` | **32** weights + FP8 E4M3 activations, uniform (TRT-LLM) |
| `nvfp4_w4a4` | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | **16** weights and activations, uniform |
| `nvfp4_w4a16` | `mtq.W4A16_NVFP4_CFG` / `w4a16_nvfp4` | 16, weight-only |
| `fp8_w8a8` | `mtq.FP8_DEFAULT_CFG` / `fp8` | n/a (FP8) |

SGLang-serving NVFP4 uses **group_size 16** on MLP + `lm_head`. TensorRT-LLM
uniform W4A8 uses **block size 32** (`nvfp4_bs32`). Do not describe
`nvidia/Qwen3.8-27B-NVFP4` as `W4A8_NVFP4_FP8` / block 32.

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
| `local_hessian` | `{method: local_hessian, fp8_scale_sweep: True, block_size: 16}` (32 for `w4a8_nvfp4_fp8`) | NVIDIA quality `mixed.yaml` |
| `mse` | `{method: mse}` | optional |

Local Hessian (quality recipe only):

```python
cfg["algorithm"] = {"method": "local_hessian", "fp8_scale_sweep": True}
```

VLM: quantize the language model; keep the vision encoder in BF16 unless
`quantize_vision: true`.

## llm-compressor mapping

There is **no** stock `NVFP4A8` preset. Default `nvfp4_w4a8` uses two custom
groups matching the NVIDIA mixed map. TensorRT-LLM uniform W4A8
(`w4a8_nvfp4_fp8`) builds a custom `QuantizationScheme`:

- weights: FP4, `TENSOR_GROUP`, `group_size=32` for TRT-LLM W4A8 (16 for
  uniform W4A4 and for NVIDIA/SGLang mixed MLP + `lm_head`),
  `scale_dtype=float8_e4m3fn`
- activations: FP8 for TRT-LLM W4A8; NVFP4 (group 16) for uniform W4A4;
  mixed follows the NVIDIA mapping (NVFP4 on MLP + `lm_head`, FP8 on attention)
- ignore: family ignore list + `lm_head` when the recipe says so
- save: `model.save_pretrained(dir, save_compressed=True)`

Prefer ModelOpt for SGLang-serving mixed NVFP4 (`MIXED_PRECISION` rewrite).

## Pipeline steps

1. Load YAML → `Recipe`
2. Detect family from `config.json` `model_type` / `architectures` (or recipe.family)
3. Resolve scheme catalog + family ignore + extra_ignore
4. If `--dry-run`, print plan and stop
5. Load tokenizer + model in BF16
6. Build calibration iterator
7. `backend.quantize(...)`
8. `backend.export(...)`
9. Write `provenance.json` (base model, scheme, backend, calib, git sha, timestamp)

## Docker / Compose (RTX 5090+)

The supported way to run this pipeline on a Blackwell box is Compose, not a host venv.

- Image: `Dockerfile` (CUDA 12.8 devel + PyTorch cu128 + ModelOpt + llm-compressor + megaquant)
- Orchestration: `docker-compose.yml` — `megaquant` service dry-runs; `quantize`,
  `w4a4`, and `mixed` are behind `--profile gpu`. `mixed` defaults to
  `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` (**production**: `max` +
  ultrachat 256); NVIDIA quality Local-Hessian is
  `RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml`. 5090 GPQA is
  `recipes/eval-gpqa-diamond.5090.yaml` (24-way, HiCache 64 GiB, bf16 GDN).
- GPU pod (no Docker): `bash scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`
  (starts a job; does not mean W4A4 / mixed PTQ or OSS upload has already finished).
  `serve`/`eval` default to SGLang (`:30000`, HiCache KV → RAM).
- After mixed / default-W4A8 export: `megaquant rewrite-sglang <export_dir>`
  before SGLang serve if `hf_quant_config.json` is bare NVFP4 without
  `quantized_layers`. Uniform `w4a8_nvfp4_fp8` is TensorRT-LLM only — not
  SGLang-loadable.
- Publish: `python scripts/oss_publish.py <export> --scheme w4a4|mixed|w4a8`
  and/or `bash scripts/gpu-pod.sh publish [scheme]`. Bucket/endpoint from
  `OSS_BUCKET` / `OSS_ENDPOINT` (optional `.oss.env`). Mixed encoding **is**
  default W4A8 — publish the same mixed export twice as `mixed` and `w4a8`
  (same content-hash); do not run a second PTQ.
- Host check: `bash docker/host-check.sh` (driver 570+, NVIDIA Container Toolkit)
- Move the box: `make image-tar` then `docker image load` on the 5090
- Weights stay on the host: `./.cache/huggingface`, `./models`, `./outputs`
- Runbook: `docker/README.md`

## CLI

```
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8-trtllm.yaml
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-mixed
megaquant serve -c recipes/eval-gpqa-diamond.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.yaml --dry-run
megaquant serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
megaquant schemes
megaquant families
```

CLI overrides: `--model`, `--output`, `--backend`, `--num-samples`, `--algorithm`.
`serve` / `eval` default to SGLang `:30000` (HiCache). Pass `--engine vllm`
only when you want the optional GB300 vLLM flags.
