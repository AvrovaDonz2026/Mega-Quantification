# Mega-Quantification

Generic post-training quantization (PTQ) for Hugging Face checkpoints.
First production target: **Qwen/Qwen3.8-27B BF16 → SGLang-serving NVFP4**
(default recipe name `nvfp4_w4a8`: NVFP4 gs16 MLP + FP8 attention, not
uniform W4A4 and not ModelOpt `W4A8_NVFP4_FP8`).

A recipe YAML plus a model-family adapter is enough to quantize any causal LM
or VLM. This repo ships Qwen3.8-27B recipes, a `qwen3_5` family adapter, and
thin `qwen3` / `llama` / `generic` adapters so the pipeline is not a one-off
script.

---

## English

### What Mega-Quantification is

Mega-Quantification loads a BF16 Hugging Face model, calibrates it, quantizes
selected linears, and exports a unified HF checkpoint for SGLang / vLLM /
TensorRT-LLM.

| Recipe | Meaning |
|---|---|
| `nvfp4_w4a8` | **SGLang default.** Same mixed map as `nvidia/Qwen3.8-27B-NVFP4`: NVFP4 **group_size 16** on MLP + `lm_head`, FP8 on self-attn + linear-attn. Export writes `quant_algo=MIXED_PRECISION` + `quantized_layers`. |
| `nvfp4_w4a4` | Uniform NVFP4 W4A4 (`NVFP4_DEFAULT_CFG`; block **16** weights and activations). SGLang `quant_algo=NVFP4`. |
| `nvfp4_mixed` | Same encoding as `nvfp4_w4a8`; quality recipe uses Local-Hessian + Nemotron v3. |

NVIDIA's public [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
is mixed NVFP4/FP8 (Local-Hessian, 2048 samples,
`Nemotron-Post-Training-Dataset-v3`, `nvidia-modelopt` v0.48.0). NVFP4 layers
use **group_size 16**. Default `nvfp4_w4a8` in this repo matches that map so
SGLang `modelopt_mixed` can serve the checkpoint. This repo does not ship a
recipe for ModelOpt `W4A8_NVFP4_FP8` (block 32); SGLang rejects that tag.

Bits on that default checkpoint (`nvfp4_w4a8` / `nvfp4_mixed`). The recipe
name says W4A8; the MLP is NVFP4 weights and NVFP4 activations:

| Tensor | Weights | Activations | Export entry |
|---|---|---|---|
| `mlp.{gate,up,down}_proj`, `lm_head` | NVFP4 E2M1, group 16 | NVFP4, group 16 | `quant_algo: NVFP4` |
| `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| vision, MTP, embeddings, `linear_attn.conv1d` / `in_proj_a` / `in_proj_b`, norms | BF16 | BF16 | omitted |
| KV at serve time | | fp8_e4m3 | recipe `kv_cache: fp8` |

Top-level `hf_quant_config.json` is `quant_algo: MIXED_PRECISION` plus `quantized_layers`. Layer-by-layer notes: [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md#quantization-format).

### What this repo actually quantized (5090)

The live 32 GB RTX 5090 PTQ is **not** NVIDIA's public recipe. It used
[`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`](recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml)
(scheme `nvfp4_mixed`; same encoding as default `nvfp4_w4a8`). The W4A8 export
directory is that mixed checkpoint, not a second PTQ.

| Knob | 5090 production | NVIDIA public `nvidia/Qwen3.8-27B-NVFP4` |
|---|---|---|
| Layer map | NVFP4 gs16 MLP + `lm_head`, FP8 attn | Same |
| Algorithm | ModelOpt **`max`** (per-tensor amax / RTN) | **Local-Hessian** (`fp8_scale_sweep`, Hessian `block_size` 16) |
| nvidia-modelopt | **0.46.1** (PyPI; 0.48 is not published there) | **0.48.0** |
| Calib | `HuggingFaceH4/ultrachat_200k` **256×1024**, batch 4 | `Nemotron-Post-Training-Dataset-v3` **2048×2048**, batch 1 |
| Export | `quant_algo=MIXED_PRECISION` + `quantized_layers` | Same mixed HF layout |
| GPQA (published) | Do **not** quote a 198-row score until the journal finishes | **88.92** BF16 / **88.01** NVFP4 on GB300 **vLLM** |

`max` is the cheap PTQ path and is what fits a 32 GB card with CPU offload.
Local-Hessian at 2048 samples does not. Quality requant:
`recipes/qwen3.8-27b-nvfp4-mixed.yaml` on a larger GPU. Compose `mixed`
defaults to the 5090 `max` recipe; override with
`RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml`.

5090 GPQA (`recipes/eval-gpqa-diamond.5090.yaml`): **24-way**, HiCache
**64 GiB** (SGLang splits that host pool by GPU KV vs GDN size), **bf16**
GDN with `max_mamba_cache_size: 96`, Triton attention + Marlin NVFP4,
CUDA graph off (CUDA 12.8 cannot FlashInfer-JIT SM 12.0, and 32 GB OOMs
during graph capture). An 80 GB SM120 uses
`recipes/eval-gpqa-diamond.6000d.yaml`: **64-way**, KV on GPU, CUDA graph
on, SiLU+FP4 fusion off. NVIDIA's mixed card used vLLM on GB300; their
SGLang cookbook uses `extra_buffer` + float32 mamba. Keep the FlashInfer,
32 GB, and 80 GB recipes distinct.

DGX Spark serves this same mixed W4A4 checkpoint (NVFP4 activations on the
MLP, FP8 attention, FP8 KV). Plain GB10 decode sits near 12 tok/s under a
~14 tok/s bandwidth ceiling; the extra speed is speculative decode on that
checkpoint. `recipes/qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml` is an optional
Marlin export, not that fast path.

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
attention; Local-Hessian + Nemotron v3). A 32 GB 5090 cannot run this;
use `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` (`max` + ultrachat) instead:

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
# 5090 production:
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
```

End-to-end notes: [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md).

### Install extras

Core (recipe schema, CLI dry-run, family adapters) needs no GPU:

```bash
pip install -e '.[dev]'
```

Real PTQ needs Hugging Face loaders **and** a backend:

```bash
# First-class SGLang W4A8 path (mixed NVFP4/FP8 + MIXED_PRECISION export)
pip install -e '.[hf,modelopt]'

# Alternative compressed-tensors / vLLM-oriented path
pip install -e '.[hf,llmcompressor]'
```

`backend: modelopt` is set in the Qwen3.8 recipes. `backend: auto` prefers
ModelOpt for NVFP4 W4A8, else llm-compressor.

### Why ModelOpt is the default for W4A8

- SGLang's ModelOpt loader accepts `quant_algo` in `{FP8, NVFP4, NVFP4_AWQ}`
  **or** `MIXED_PRECISION` with a non-empty `quantized_layers` map. It
  **rejects** `W4A8_NVFP4_FP8`.
- Default `nvfp4_w4a8` therefore uses the NVIDIA mixed map (NVFP4 gs16 MLP +
  `lm_head`, FP8 attention) and rewrites `hf_quant_config.json` after export.
- llm-compressor has **no** stock `NVFP4A8` preset. Mixed W4A8 is two custom
  groups. This repo does not ship a uniform `W4A8_NVFP4_FP8` recipe.

Uniform W4A4 (`nvfp4_w4a4`, `NVFP4_DEFAULT_CFG`) uses block size **16** for
weights and activations. Default W4A8 NVFP4 layers also use **group_size 16**.
Do not encode SGLang W4A8 as `W4A8_NVFP4_FP8` block 32.

### Repository layout

```
src/megaquant/   library (CLI, pipeline, schemes, model families)
recipes/         quant and GPQA YAML; index in recipes/README.md
tests/           CPU tests; dry-run does not download weights
docs/            architecture and the Qwen3.8-27B runbook
docker/          image helpers; Dockerfiles stay at the repo root
scripts/         gpu-pod, OSS publish/fetch, serve and eval wrappers
```

Copy `.env.example` to `.env`. Bucket credentials go in gitignored `.oss.env`.
Do not commit `docker-compose.override.yml`. Weights and journals live under
`outputs/`, `.cache/`, `models/`, and `data/`. License: Apache-2.0 (`LICENSE`).

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

On a k8s GPU pod (no Docker): `bash scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`.
Those commands start a job; they do not mean W4A4 or mixed PTQ has already
finished on the pod.

### GPQA Diamond (SGLang)

Two terminals: **`megaquant serve`**, then **`megaquant eval`**. Both use
SGLang on `:30000`. Sampling is the Qwen thinking card
(`temperature=0`, `top_p=0.95`, `top_k=20`, thinking on,
`reasoning_effort=xhigh`). `max_new_tokens: 0` fills the remaining 262144
context and `continue_on_length` continues until EOS. The journal is
`<model>/gpqa_diamond/gpqa_diamond.jsonl` (full text, resume by `item_id`).
The score is `correct/198` after every Diamond row is in the file.

| Box | Recipe | Serve shape |
|---|---|---|
| CUDA ≥ 12.9, FlashInfer | `recipes/eval-gpqa-diamond.yaml` | FlashInfer, HiCache 12 GiB, concurrency 1, CUDA graph off |
| 32 GB SM120 / CUDA 12.8 (5090) | `recipes/eval-gpqa-diamond.5090.yaml` | Triton + Marlin, HiCache 64 GiB, 24-way bf16 GDN, CUDA graph off |
| 80 GB SM120 / CUDA 12.8 (6000D) | `recipes/eval-gpqa-diamond.6000d.yaml` | Triton + Marlin + CUTLASS, KV on GPU, 64-way, CUDA graph on, SiLU+FP4 fusion off |

`megaquant serve` writes the recipe env (`SGLANG_FORCE_FP8_MARLIN`, and on
the 80 GB recipe `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION`) before it starts
`sglang serve`. Full flag table:
[`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md#sglang-inference-and-gpqa).

`--dry-run` plans against the local export `outputs/Qwen3.8-27B-NVFP4-W4A8`
and does **not** download `Qwen/Qwen3.8-27B`. Compose `eval-gpqa` is HTTP-only
(no `gpus:` / deploy devices, empty `NVIDIA_VISIBLE_DEVICES`,
`MEGAQUANT_SKIP_GPU_REPORT=1`); `serve-sglang` holds the GPU.
Default recipe is FlashInfer (`recipes/eval-gpqa-diamond.yaml`). RTX 5090 /
CUDA 12.8 uses Triton (`recipes/eval-gpqa-diamond.5090.yaml`) — keep them
distinct.

```bash
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
# Terminal 1
bash scripts/gpu-pod.sh serve w4a8
# Terminal 2
MEGAQUANT_SGLANG_BASE_URL=http://127.0.0.1:30000/v1 bash scripts/gpu-pod.sh eval w4a8
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
量化成 **SGLang 能直接推理的 NVFP4**（默认配方名仍叫 `nvfp4_w4a8`：MLP +
`lm_head` 为 NVFP4 group_size 16，注意力为 FP8；导出 `MIXED_PRECISION`），
供 SGLang / vLLM / TensorRT-LLM 使用。

| 配方 | 含义 |
|---|---|
| `nvfp4_w4a8` | **SGLang 默认。** 与 `nvidia/Qwen3.8-27B-NVFP4` 同一套混合图：MLP + `lm_head` NVFP4 **group_size 16**，self-attn + linear-attn 为 FP8。导出 `quant_algo=MIXED_PRECISION`。 |
| `nvfp4_w4a4` | 均匀：`NVFP4_DEFAULT_CFG`，权重和激活均为 NVFP4 block **16** |
| `nvfp4_mixed` | 与 `nvfp4_w4a8` 同一套编码；质量配方用 Local-Hessian + Nemotron v3 |

NVIDIA 公开的 `nvidia/Qwen3.8-27B-NVFP4` 是 **混合 NVFP4/FP8**：MLP + `lm_head`
为 **NVFP4 group_size 16**，self-attn + linear-attn 为 **FP8**。默认 W4A8
对齐这套图，好让 SGLang `modelopt_mixed` 加载。本仓库不提供 SGLang 拒收的
均匀 `W4A8_NVFP4_FP8`（block 32）配方。

默认 checkpoint（`nvfp4_w4a8` / `nvfp4_mixed`）的位宽。配方名叫 W4A8，MLP 是 NVFP4 权重加 NVFP4 激活：

| 张量 | 权重 | 激活 | 导出条目 |
|---|---|---|---|
| `mlp.{gate,up,down}_proj`、`lm_head` | NVFP4 E2M1，group 16 | NVFP4，group 16 | `quant_algo: NVFP4` |
| `self_attn.{q,k,v,o}_proj`、`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| 视觉、MTP、embedding、`linear_attn.conv1d` / `in_proj_a` / `in_proj_b`、norm | BF16 | BF16 | 不写入 |
| 推理时 KV | | fp8_e4m3 | 配方 `kv_cache: fp8` |

`hf_quant_config.json` 顶层是 `quant_algo: MIXED_PRECISION`，外加 `quantized_layers`。逐层说明见 [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md#量化格式)。

### 这台 5090 实际跑的量化

线上 32 GB RTX 5090 的 PTQ **不是** NVIDIA 公开配方。实际用的是
[`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`](recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml)
（`nvfp4_mixed`，与默认 `nvfp4_w4a8` 同一套编码）。W4A8 导出目录就是这份
mixed checkpoint，没有再跑一遍 PTQ。

| 项 | 5090 生产 | NVIDIA 公开 `nvidia/Qwen3.8-27B-NVFP4` |
|---|---|---|
| 层图 | MLP + `lm_head` NVFP4 gs16，注意力 FP8 | 相同 |
| 算法 | ModelOpt **`max`**（amax / RTN） | **Local-Hessian**（`fp8_scale_sweep`，Hessian `block_size` 16） |
| nvidia-modelopt | **0.46.1**（PyPI；0.48 未上 PyPI） | **0.48.0** |
| 校准 | `HuggingFaceH4/ultrachat_200k` **256×1024**，batch 4 | `Nemotron-Post-Training-Dataset-v3` **2048×2048**，batch 1 |
| 导出 | `MIXED_PRECISION` + `quantized_layers` | 同一套 mixed HF 布局 |
| GPQA（已发表） | 198 题 journal 跑完前 **不要报总分** | GB300 **vLLM**：**88.92** BF16 / **88.01** NVFP4 |

`max` 省显存，单卡 32 GB 只能走这条。Local-Hessian 2048 需要更大卡：
`recipes/qwen3.8-27b-nvfp4-mixed.yaml`。Compose `mixed` 默认就是 5090 的
`max` 配方。

5090 GPQA（`recipes/eval-gpqa-diamond.5090.yaml`）：**24 路**，HiCache
**64 GiB**（按 GPU 上 KV / GDN 池比例切主机内存），GDN **bf16** 且
`max_mamba_cache_size: 96`，Triton 注意力 + Marlin NVFP4，CUDA graph 关闭
（CUDA 12.8 编不了 SM 12.0 的 FlashInfer JIT，32 GB 抓 graph 会 OOM）。
80 GB 级 SM120 用 `recipes/eval-gpqa-diamond.6000d.yaml`：**64 路**，KV 在
GPU 上，CUDA graph 开着，SiLU+FP4 融合关掉。NVIDIA 模型卡用 GB300 上的
vLLM；他们的 SGLang cookbook 是 `extra_buffer` + float32 mamba。FlashInfer、
32 GB、80 GB 三套配方不要混用。

DGX Spark 推理用的就是这份混合 W4A4 checkpoint（MLP 为 NVFP4 激活，注意力
FP8，KV FP8）。GB10 普通 decode 大约 12 tok/s，带宽上限大约 14 tok/s；
再快是这份权重上的投机解码。`qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml` 是可选
Marlin 导出，不是这条快路径。

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
16，注意力为 FP8；Local-Hessian + Nemotron v3）。32 GB 5090 跑不了 Hessian
2048，生产用 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`（`max` + ultrachat）：

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
# 5090 生产：
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
```

完整手册见 [`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md)。

### 安装 extras

```bash
pip install -e '.[dev]'                 # 仅 schema / dry-run / 单测
pip install -e '.[hf,modelopt]'         # 推荐：W4A8 走 ModelOpt
pip install -e '.[hf,llmcompressor]'    # 备选：compressed-tensors / vLLM
```

### 为什么 W4A8 默认用 ModelOpt

SGLang 只认 `NVFP4` / `MIXED_PRECISION`（外加非空 `quantized_layers`），
**拒收** `W4A8_NVFP4_FP8`。所以默认 `nvfp4_w4a8` 走 NVIDIA 混合图
（MLP NVFP4 gs16 + 注意力 FP8），导出后再改写 `hf_quant_config.json`。
本仓库不提供 SGLang 拒收的均匀 `W4A8_NVFP4_FP8` 配方。

均匀 W4A4（`NVFP4_DEFAULT_CFG`）的 block size 是 **16**。默认 W4A8 的 NVFP4
层也是 **group_size 16**，不要写成 `W4A8_NVFP4_FP8` block 32。

### 仓库目录

```
src/megaquant/   库（CLI、流水线、量化方案、模型族）
recipes/         量化与 GPQA 的 YAML，索引在 recipes/README.md
tests/           CPU 测试；dry-run 不下载权重
docs/            架构说明和 Qwen3.8-27B 手册
docker/          镜像辅助脚本；Dockerfile 在仓库根目录
scripts/         gpu-pod、OSS 发布/拉取、推理和评测包装
```

`.env.example` 复制成 `.env`。桶凭证放在被 git 忽略的 `.oss.env`。
不要提交 `docker-compose.override.yml`。权重和 journal 在 `outputs/`、
`.cache/`、`models/`、`data/`。许可证是 Apache-2.0（`LICENSE`）。

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

K8s GPU 容器（没有 Docker）：`bash scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`。
这只是启动命令，不表示 W4A4 / mixed PTQ 已经在 pod 上跑完。

量化产物评测 GPQA Diamond：两个终端，先 **`megaquant serve`** 再
**`megaquant eval`**。推理走 **SGLang** `:30000`，采样是 Qwen thinking 卡
（`temperature=0`，`top_p=0.95`，`top_k=20`，thinking 开，
`reasoning_effort=xhigh`）。`max_new_tokens: 0` 用完剩余 262144 上下文。
完整轨迹在 `<model>/gpqa_diamond/gpqa_diamond.jsonl`，按 `item_id`
续跑。198 行都在文件里之后，分数才是 `correct/198`。

| 机器 | 配方 | 推理形态 |
|---|---|---|
| CUDA ≥ 12.9，FlashInfer | `recipes/eval-gpqa-diamond.yaml` | FlashInfer，HiCache 12 GiB，并发 1，CUDA graph 关 |
| 32 GB SM120 / CUDA 12.8（5090） | `recipes/eval-gpqa-diamond.5090.yaml` | Triton + Marlin，HiCache 64 GiB，24 路 bf16 GDN，CUDA graph 关 |
| 80 GB SM120 / CUDA 12.8（6000D） | `recipes/eval-gpqa-diamond.6000d.yaml` | Triton + Marlin + CUTLASS，KV 在 GPU，64 路，CUDA graph 开，SiLU+FP4 融合关 |

`megaquant serve` 会先写入配方环境变量（`SGLANG_FORCE_FP8_MARLIN`，80 GB
配方还有 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION`），再启动 `sglang serve`。
完整标志表见
[`docs/qwen3.8-27b.md`](docs/qwen3.8-27b.md#sglang-推理与-gpqa)。
`--dry-run` 只看本地导出 `outputs/Qwen3.8-27B-NVFP4-W4A8`，不会去拉
`Qwen/Qwen3.8-27B`。Compose `eval-gpqa` 不挂 GPU；`serve-sglang` 才挂。

手册：[`docker/README.md`](docker/README.md)。单卡 32 GB 5090 放不下 27B BF16，
默认 CPU offload；双卡或更大 Blackwell 更合适。

### 硬件

- **推理 NVFP4**：需要 Blackwell（SM100+ / 5090 的 sm_120）。
- **PTQ 校准**：Hopper 显存够就可以跑；27B BF16 大约 54 GiB，需要多卡或
  CPU offload。单卡 5090 32 GB 必须 offload。
- **单测 / dry-run**：纯 CPU，不下载权重。
