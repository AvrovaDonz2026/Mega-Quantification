# Mega-Quantification architecture

Generic post-training quantization (PTQ) pipeline. First production target:
**Qwen/Qwen3.8-27B BF16 → NVFP4 W4A8**.

The pipeline is model-agnostic. A recipe YAML plus a model-family adapter should
be enough to quantize any Hugging Face causal / VLM checkpoint.

## Goal

- Weights: **NVFP4** (E2M1 micro-blocks, FP8 E4M3 local scales).
- Activations: **FP8 E4M3** → this is **W4A8**, not W4A4.
- Export a Hugging Face unified checkpoint for TensorRT-LLM / vLLM / SGLang.

NVIDIA's public `nvidia/Qwen3.8-27B-NVFP4` checkpoint is mixed NVFP4/FP8:
**NVFP4 group_size 16** on MLP + `lm_head`, **FP8** on self-attn + linear-attn.
It is **not** uniform W4A8 and **not** NVFP4 block 32. Card: Local-Hessian,
2048 samples, `Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0.

This repo ships three Qwen3.8-27B schemes. Uniform W4A8 stays block 32
(`W4A8_NVFP4_FP8`). Uniform W4A4 is `NVFP4_DEFAULT_CFG` block 16 (weights and
activations).

| Recipe | Meaning |
|---|---|
| `nvfp4_w4a8` | Uniform NVFP4 weights (block **32**) + FP8 activations on language-model linears |
| `nvfp4_w4a4` | Uniform NVFP4 W4A4 (`NVFP4_DEFAULT_CFG`, block **16** weights and activations) |
| `nvfp4_mixed` | NVIDIA mapping: NVFP4 group_size **16** on MLP + `lm_head`, FP8 on self-attn + linear-attn |

## Package layout (file ownership)

```
src/megaquant/config.py            # Agent Core — Pydantic recipe schema
src/megaquant/exceptions.py        # Agent Core
src/megaquant/registry.py          # Agent Core
src/megaquant/pipeline.py          # Agent Core
src/megaquant/cli.py               # Agent Core
src/megaquant/eval_gpqa.py         # GPQA Diamond (Qwen thinking + NVIDIA vLLM + KV CPU offload)
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
    group_size: int | None = None   # W4A4 and NVIDIA mixed NVFP4 layers: 16; W4A8_NVFP4_FP8: 32
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

`auto` backend: prefer `modelopt` for NVFP4 W4A8 (native `W4A8_NVFP4_FP8_CFG`),
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
- Mixed encoding: NVFP4 **group_size 16** on MLP + `lm_head`; FP8 on self-attn +
  linear-attn. Not uniform W4A8. Not NVFP4 block 32.

## ModelOpt mapping

| Our scheme | ModelOpt object / qformat | NVFP4 block |
|---|---|---|
| `nvfp4_w4a8` | `mtq.W4A8_NVFP4_FP8_CFG` / `w4a8_nvfp4_fp8` | **32** weights + FP8 E4M3 activations, uniform |
| `nvfp4_w4a4` | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | **16** weights and activations, uniform |
| `nvfp4_w4a16` | `mtq.W4A16_NVFP4_CFG` / `w4a16_nvfp4` | 16, weight-only |
| `fp8_w8a8` | `mtq.FP8_DEFAULT_CFG` / `fp8` | n/a (FP8) |
| `nvfp4_mixed` | custom: `NVFP4_DEFAULT_CFG` on `*mlp*` + `*lm_head*`; FP8 on `*self_attn*` + `*linear_attn*` | **16** on MLP + `lm_head` (NVIDIA public mapping) |

Uniform W4A8 NVFP4 weights use **block size 32** (`nvfp4_bs32`). Uniform W4A4
and the NVFP4 layers of NVIDIA mixed PTQ use **block size 16**. Do not describe
`nvidia/Qwen3.8-27B-NVFP4` as `W4A8_NVFP4_FP8` / block 32.

Export: `modelopt.torch.export.export_hf_checkpoint(model, export_dir)`.

Quantize: `mtq.quantize(model, quant_cfg, forward_loop)`.

Local Hessian:

```python
cfg["algorithm"] = {"method": "local_hessian", "fp8_scale_sweep": True}
```

VLM: quantize the language model; keep the vision encoder in BF16 unless
`quantize_vision: true`.

## llm-compressor mapping

There is **no** stock `NVFP4A8` preset. Build a custom `QuantizationScheme`:

- weights: FP4, `TENSOR_GROUP`, `group_size=32` for uniform W4A8 (16 for
  uniform W4A4 and for NVIDIA mixed MLP + `lm_head`),
  `scale_dtype=float8_e4m3fn`
- activations: FP8 for uniform W4A8; NVFP4 (group 16) for uniform W4A4;
  mixed follows the NVIDIA mapping (NVFP4 on MLP + `lm_head`, FP8 on attention)
- ignore: family ignore list + `lm_head` when the recipe says so
- save: `model.save_pretrained(dir, save_compressed=True)`

vLLM kernels for NVFP4+FP8 activations are weaker than TensorRT-LLM; document that
ModelOpt is the first-class W4A8 path.

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
  `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`; NVIDIA quality Local-Hessian is
  `RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml`.
- GPU pod (no Docker): `bash scripts/gpu-pod.sh plan|quantize|serve|eval [w4a8|w4a4|mixed]`
  (starts a job; does not mean W4A4 or mixed PTQ has already finished).
- Host check: `bash docker/host-check.sh` (driver 570+, NVIDIA Container Toolkit)
- Move the box: `make image-tar` then `docker image load` on the 5090
- Weights stay on the host: `./.cache/huggingface`, `./models`, `./outputs`
- Runbook: `docker/README.md`

## CLI

```
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
megaquant schemes
megaquant families
```

CLI overrides: `--model`, `--output`, `--backend`, `--num-samples`, `--algorithm`.
