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
| `nvfp4_w4a8` | Uniform NVFP4 weights (block **32**) + FP8 activations on language-model linears |
| `nvfp4_w4a4` | Uniform NVFP4 W4A4 (`NVFP4_DEFAULT_CFG`; block **16** weights and activations) |
| `nvfp4_mixed` | NVIDIA mapping: NVFP4 **group_size 16** on MLP + `lm_head`, FP8 on self-attn + linear-attn |

NVIDIA's public [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
is mixed NVFP4/FP8 (Local-Hessian, 2048 samples,
`Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0). NVFP4 layers
use **group_size 16**. It is **not** uniform W4A8 and **not** NVFP4 block 32.
The user-requested export in this repo is **uniform W4A8** (block 32 /
`W4A8_NVFP4_FP8`).

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

Uniform W4A4 (block 16 weights and activations):

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
```

NVIDIA-matched mixed checkpoint (group_size 16 on MLP + `lm_head`, FP8 on
attention; Local-Hessian + Nemotron v3):

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

Uniform W4A4 (`nvfp4_w4a4`, `NVFP4_DEFAULT_CFG`) uses block size **16** for
weights and activations. NVIDIA mixed NVFP4 layers also use **group_size 16**;
do not encode them as W4A8 block 32. Do not mix block sizes.

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
docker compose --profile gpu run --rm w4a4
docker compose --profile gpu run --rm mixed
```

`compose up` only dry-runs. Real PTQ is `--profile gpu`. The `mixed` service
uses `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`. NVIDIA quality Local-Hessian:

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

On a k8s GPU pod (no Docker): `bash scripts/gpu-pod.sh plan|quantize|serve|eval [w4a8|w4a4|mixed]`.
Those commands start a job; they do not mean W4A4 or mixed PTQ has already
finished on the pod.

### GPQA Diamond (after an export)

Match the Qwen thinking card and the NVIDIA vLLM card. Generation uses the
remaining 262144-token window (`max_new_tokens: 0`) and continues on length.
On a 32 GB card, vLLM **native-offloads KV into host RAM** (MemTotal − 6 GiB)
instead of truncating.

```bash
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml --dry-run
bash scripts/gpu-pod.sh serve w4a8
MEGAQUANT_VLLM_BASE_URL=http://127.0.0.1:8000/v1 bash scripts/gpu-pod.sh eval w4a8
```

Copy a built image with `make image-tar` then `docker image load` on the 5090.
Weights stay on the host (`./.cache/huggingface`, `./models`, `./outputs`).

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

| 配方 | 含义 |
|---|---|
| `nvfp4_w4a8` | 均匀：语言模型线性层 NVFP4 权重（block **32**）+ FP8 激活 |
| `nvfp4_w4a4` | 均匀：`NVFP4_DEFAULT_CFG`，权重和激活均为 NVFP4 block **16** |
| `nvfp4_mixed` | NVIDIA 映射：MLP + `lm_head` 为 NVFP4 **group_size 16**，self-attn + linear-attn 为 FP8 |

NVIDIA 公开的 `nvidia/Qwen3.8-27B-NVFP4` 是 **混合 NVFP4/FP8**：MLP + `lm_head`
为 **NVFP4 group_size 16**，self-attn + linear-attn 为 **FP8**。它 **不是**
均匀 W4A8，也 **不是** NVFP4 block 32。模型卡：Local-Hessian、2048 条、
`Nemotron-Post-Training-Dataset-v3`、`nvidia-modelopt` v0.48.0。本仓库均匀
W4A8 仍是 block 32（`W4A8_NVFP4_FP8`）；均匀 W4A4 是 `NVFP4_DEFAULT_CFG`
（权重和激活均为 block 16）。

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

均匀 W4A4：

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
```

对齐 NVIDIA 公开 checkpoint 的混合配方（MLP + `lm_head` 为 NVFP4 group_size
16，注意力为 FP8；Local-Hessian + Nemotron v3）：

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

W4A4（`NVFP4_DEFAULT_CFG`）的 block size 是 **16**（权重和激活），和均匀 W4A8
的 32 不是同一套。NVIDIA 混合配方里的 NVFP4 层也是 **group_size 16**，不要写成
W4A8 block 32。

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
docker compose --profile gpu run --rm w4a4
docker compose --profile gpu run --rm mixed
```

`mixed` 服务默认 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`。NVIDIA 质量
Local-Hessian：

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

K8s GPU 容器（没有 Docker）：`bash scripts/gpu-pod.sh plan|quantize|serve|eval [w4a8|w4a4|mixed]`。
这只是启动命令，不表示 W4A4 / mixed PTQ 已经在 pod 上跑完。

量化产物评测 GPQA Diamond 时使用官方 thinking 采样 + NVIDIA vLLM 参数；
`max_new_tokens: 0` 表示用完剩余 262k 窗口，length 后再续写。32 GB 显存
放不下 262k KV 时走 vLLM native **CPU offload**（MemTotal − 6 GiB），不要
靠截断生成来省显存。

手册：[`docker/README.md`](docker/README.md)。单卡 32 GB 5090 放不下 27B BF16，
默认 CPU offload；双卡或更大 Blackwell 更合适。

### 硬件

- **推理 NVFP4**：需要 Blackwell（SM100+ / 5090 的 sm_120）。
- **PTQ 校准**：Hopper 显存够就可以跑；27B BF16 大约 54 GiB，需要多卡或
  CPU offload。单卡 5090 32 GB 必须 offload。
- **单测 / dry-run**：纯 CPU，不下载权重。
