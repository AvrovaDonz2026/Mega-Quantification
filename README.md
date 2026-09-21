# Mega-Quantification

Generic post-training quantization (PTQ) for Hugging Face checkpoints.
First production target: **Qwen/Qwen3.8-27B BF16 → NVFP4 W4A8**
(NVFP4 weights + FP8 activations — not W4A4).

A recipe YAML plus a model-family adapter is enough to quantize any causal LM
or VLM. This repo ships Qwen3.8-27B recipes, a `qwen3_5` family adapter, and
thin `qwen3` / `llama` / `generic` adapters so the pipeline is not a one-off
script.

---

## English

### What Mega-Quantification is

Mega-Quantification loads a BF16 Hugging Face model, calibrates it, quantizes
selected linears, and exports a unified HF checkpoint for TensorRT-LLM / vLLM
/ SGLang.

| Recipe | Meaning |
|---|---|
| `nvfp4_w4a8` | Uniform NVFP4 weights + FP8 activations on language-model linears |
| `nvfp4_mixed` | NVIDIA-style mixed: NVFP4 MLP + `lm_head`, FP8 self-attn + linear-attn |
| `nvfp4_w4a4` | Uniform NVFP4 W4A4 (comparison; block size 16) |

NVIDIA's public [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
checkpoint is the **mixed** recipe (Local-Hessian, 2048 samples, ModelOpt 0.48.0).
The user-requested export is **uniform W4A8**.

### Qwen3.8-27B W4A8 quickstart

Always dry-run first. Dry-run validates the YAML, resolves the `qwen3_5`
ignore list, and prints the plan. It does **not** download 27B weights.

```bash
# from the repo root, after an editable install
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
# equivalent:
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
```

On a machine with enough GPU memory (see Hardware):

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
```

NVIDIA-matched mixed checkpoint:

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
```

End-to-end notes: [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md).

### Install extras

Core (recipe schema, CLI dry-run, family adapters) needs no GPU:

```bash
pip install -e '.[dev]'
```

Real PTQ needs Hugging Face loaders **and** a backend:

```bash
# First-class W4A8 path (ModelOpt native W4A8_NVFP4_FP8)
pip install -e '.[hf,modelopt]'

# Alternative compressed-tensors / vLLM-oriented path
pip install -e '.[hf,llmcompressor]'
```

`backend: modelopt` is set in the Qwen3.8 recipes. `backend: auto` prefers
ModelOpt for NVFP4 W4A8, else llm-compressor.

### Why ModelOpt is the default for W4A8

- ModelOpt ships `mtq.W4A8_NVFP4_FP8_CFG` / qformat `w4a8_nvfp4_fp8` with
  NVFP4 **block size 32** (`nvfp4_bs32`).
- llm-compressor has **no** stock `NVFP4A8` preset; it must emit a custom
  `QuantizationScheme` (FP4 `TENSOR_GROUP` `group_size=32` + FP8 activations).
- vLLM kernels for NVFP4 weights + FP8 activations are weaker than
  TensorRT-LLM. Treat ModelOpt export as the production W4A8 path; use
  llm-compressor when you specifically want a compressed-tensors checkpoint.

W4A4 (block size **16**) is a different scheme (`nvfp4_w4a4`). Do not mix
block sizes.

### How to add a new model

1. **Family adapter** in `src/megaquant/models/<family>.py` implementing
   `ModelFamily` (`name`, `model_types`, `architectures`, `default_ignore`,
   `load_kwargs`). Subclass `BaseFamily` if you only need a custom ignore list.
2. Register on import (`Family().register()`), and import the module from
   `src/megaquant/models/__init__.py` so `import megaquant.models` registers it.
3. **Recipe YAML** under `recipes/` with the nested schema: `name`, `model`,
   `backend`, `scheme`, `algorithm`, `kv_cache`, `family`, `extra_ignore`,
   `calibration`, `export`. Point `model.source` at the HF id or a local path.
4. Dry-run: `python -m megaquant.cli plan -c recipes/<your>.yaml`.

`default_ignore` should list modules that stay BF16 (embeddings, vision,
conv1d, …). Do **not** glob away every `mlp` — those are the GEMMs you want
to quantize. Pipeline merges `family.default_ignore` with `extra_ignore`.

### Docker Compose (RTX 5090+)

The supported way to run PTQ on a Blackwell box is Compose, not a host venv.
Full runbook: [`docker/README.md`](docker/README.md).

```bash
bash docker/host-check.sh
cp .env.example .env
docker compose build
docker compose run --rm megaquant plan
docker compose --profile gpu run --rm quantize
```

`compose up` only dry-runs. Real PTQ is `--profile gpu`. Copy a built image
with `make image-tar` then `docker image load` on the 5090. Weights stay on
the host (`./.cache/huggingface`, `./models`, `./outputs`).

### Hardware

| Stage | What you need |
|---|---|
| Dry-run / unit tests | CPU. No weights, no GPU. |
| PTQ calibration | Hopper (H100/H200) or Blackwell. 27B BF16 is ~54 GiB; a single 32 GB RTX 5090 needs `device_map: auto` + `MEGAQUANT_LOW_MEMORY=1` (CPU offload). Prefer 2×5090 or a 96 GB-class card. |
| NVFP4 **inference** | Blackwell SM100+ / sm_120 (RTX 5090, B200, GB300). NVIDIA tested `nvidia/Qwen3.8-27B-NVFP4` on Grace Blackwell GB300. |

---

## 中文

### Mega-Quantification 是什么

通用的 Hugging Face **训练后量化（PTQ）** 流水线。配方 YAML + 模型族适配器
即可量化任意因果 LM / VLM。首个生产目标是把 **Qwen/Qwen3.8-27B** 从 BF16
量化成 **NVFP4 W4A8**（权重量 NVFP4，激活量 FP8），导出统一 HF 权重，供
TensorRT-LLM / vLLM / SGLang 使用。

NVIDIA 公开的 `nvidia/Qwen3.8-27B-NVFP4` 是 **混合** 配方（MLP + `lm_head`
走 NVFP4，注意力走 FP8）。本仓库同时提供均匀 W4A8 与混合两套 recipe。

### Qwen3.8-27B W4A8 快速开始

先 dry-run，确认忽略列表、backend、校准条数，**不会**拉取 27B 权重：

```bash
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
```

GPU 量化：

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
```

对齐 NVIDIA 公开 checkpoint 的混合配方：

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
```

完整手册见 [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md)。

### 安装 extras

```bash
pip install -e '.[dev]'                 # 仅 schema / dry-run / 单测
pip install -e '.[hf,modelopt]'         # 推荐：W4A8 走 ModelOpt
pip install -e '.[hf,llmcompressor]'    # 备选：compressed-tensors / vLLM
```

### 为什么 W4A8 默认用 ModelOpt

ModelOpt 原生支持 `W4A8_NVFP4_FP8`（NVFP4 **block size 32**）。llm-compressor
没有现成的 `NVFP4A8` preset，只能拼自定义 scheme；vLLM 对「NVFP4 权重 +
FP8 激活」的 kernel 也不如 TensorRT-LLM。生产 W4A8 请走 ModelOpt。

W4A4 的 block size 是 **16**，和 W4A8 不是同一套。

### 如何接入新模型

1. 在 `src/megaquant/models/` 写 family 适配器（`ModelFamily` / `BaseFamily`）。
2. import 时 `register()`，并在 `models/__init__.py` 里导入以完成注册。
3. 在 `recipes/` 写 YAML（字段见 `docs/ARCHITECTURE.md`）。
4. `python -m megaquant.cli plan -c recipes/<your>.yaml` 检查计划。

忽略列表只放 **保持 BF16** 的模块（视觉、embedding、GDN 的 conv1d 等），
不要用 `*mlp*` 一把全忽略。

### Docker Compose（5090+）

推荐不要在宿主机装 CUDA 轮子，直接：

```bash
bash docker/host-check.sh
cp .env.example .env
docker compose build
docker compose run --rm megaquant plan
docker compose --profile gpu run --rm quantize
```

手册：[`docker/README.md`](docker/README.md)。单卡 32 GB 5090 放不下 27B BF16，
默认 CPU offload；双卡或更大 Blackwell 更合适。

### 硬件

- **推理 NVFP4**：需要 Blackwell（SM100+ / 5090 的 sm_120）。
- **PTQ 校准**：Hopper 显存够就可以跑；27B BF16 大约 54 GiB，需要多卡或
  CPU offload。单卡 5090 32 GB 必须 offload。
- **单测 / dry-run**：纯 CPU，不下载权重。
