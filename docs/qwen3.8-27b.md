# Qwen3.8-27B NVFP4 runbook

End-to-end notes for quantizing **Qwen/Qwen3.8-27B** (BF16) with Mega-Quantification.

## 中文要点

这是 27B 稠密 VLM（`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`）。
默认量化语言模型线性层；视觉、MTP、embedding、GDN 的 `conv1d` / `in_proj_a` /
`in_proj_b` 留 BF16。

NVIDIA 公开 [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
是 **混合 NVFP4/FP8**：MLP + `lm_head` 为 **NVFP4 group_size 16**，self-attn +
linear-attn 为 **FP8**。它 **不是** 均匀 W4A8，也 **不是** NVFP4 block 32。
模型卡：Local-Hessian（`fp8_scale_sweep`）、2048 条 × 2048、
`Nemotron-Post-Training-Dataset-v3`、`nvidia-modelopt` v0.48.0；GPQA Diamond
**88.92** BF16 / **88.01** NVFP4（GB300 上 **vLLM**）。Qwen 卡片 thinking 分
**89.2**。

本仓库默认 W4A8（`nvfp4_w4a8`）对齐这套**层图**，导出
`quant_algo=MIXED_PRECISION` + `quantized_layers`，好让 SGLang 推理。
**层图相同 ≠ 校准相同。** 32 GB RTX 5090 实际跑的是
`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`：ModelOpt **0.46.1**（PyPI 上没有
0.48）、算法 **`max`**（amax / RTN，不是 Hessian）、
`HuggingFaceH4/ultrachat_200k` **256×1024** batch 4。W4A8 导出目录就是这份
mixed checkpoint。Local-Hessian 2048 需要更大卡：
`recipes/qwen3.8-27b-nvfp4-mixed.yaml`。

均匀 W4A4 是 `NVFP4_DEFAULT_CFG`（权重和激活均为 NVFP4 block 16）。
TensorRT-LLM 均匀 W4A8（`W4A8_NVFP4_FP8`，block 32）见
`recipes/qwen3.8-27b-nvfp4-w4a8-trtllm.yaml`。NVFP4 推理需要 Blackwell；校准
可以在 Hopper 上用多卡 / offload 做。

5090 GPQA：`recipes/eval-gpqa-diamond.5090.yaml`，**24 路**，HiCache **64 GiB**
（按 GPU KV / GDN 池比例切主机内存），GDN **bf16**、96 个 mamba slot，Triton
+ Marlin，CUDA graph 关掉（32 GB 抓 graph 会 OOM）。80 GB 级 SM120
（RTX PRO 6000 / 6000D）用 `recipes/eval-gpqa-diamond.6000d.yaml`：**64 路**，
KV 留在 GPU，CUDA graph 开着，并关掉 SiLU+FP4 融合（CUDA 12.8 的 FlashInfer
JIT 编不了 SM 12.x）。float32 64-slot mamba 会把 HBM KV 吃到只剩不到 1 GB，16 路 HTTP
会排队。198 题 journal 跑完前不要报总分；eval 客户端按 `item_id` 续跑，不要
用 `open("w")` 清空 jsonl。

DGX Spark（GB10，约 273 GB/s）上该用的就是这份混合 W4A4 checkpoint
（MLP NVFP4 gs16 含 FP4 激活，注意力 FP8，KV FP8）。普通 decode 大约
12 tok/s，带宽上限大约 14 tok/s；再快靠这份权重上的投机解码，不靠再量化一次。
`nvfp4_w4a16_mixed` 只是 32 GB 机器上的可选 Marlin 导出，不是 Spark 快路径。

## What this repo actually quantized (5090)

The 32 GB RTX 5090 PTQ used
[`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`](../recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml).
That is the production checkpoint. Default `nvfp4_w4a8` is the **same mixed
encoding**; the W4A8 directory is this mixed export, not a second PTQ.

| Knob | 5090 production | NVIDIA public `nvidia/Qwen3.8-27B-NVFP4` |
|---|---|---|
| Layer map | NVFP4 gs16 MLP + `lm_head`, FP8 self-attn + linear-attn | Same |
| PTQ | ModelOpt **`max`** (per-tensor amax / round-to-nearest) | **Local-Hessian**, layerwise, `fp8_scale_sweep: true`, Hessian `block_size` 16 |
| nvidia-modelopt | **0.46.1** (`docker/requirements-gpu.txt`; 0.48 is not on PyPI) | **0.48.0** |
| Calib | `HuggingFaceH4/ultrachat_200k`, **256** samples × **1024** tokens, batch **4** | `nvidia/Nemotron-Post-Training-Dataset-v3`, **2048** × **2048**, batch **1** |
| Recipe | `qwen3.8-27b-nvfp4-mixed.5090.yaml` | `qwen3.8-27b-nvfp4-mixed.yaml` (not run on the 5090) |
| Export | `MIXED_PRECISION` + `quantized_layers` (401 entries on the 5090 run: 193 NVFP4 + 208 FP8) | Same mixed HF layout |
| Serve / GPQA | SGLang 0.5.20 on 5090, Triton + Marlin, bf16 GDN, `extra_buffer_lazy` | Card: vLLM on GB300, `temp=1.0 top_p=0.95`, `max_new_tokens=65536` |

`max` is not a weaker *format*. It is a weaker *calibrator*: it records
activation amax and quantizes, with no Local-Hessian reconstruction or
FP8 scale sweep. A 32 GB 5090 cannot hold Hessian 2048 with 27B BF16
offload. Quality requant is `mixed.yaml` on Hopper / larger Blackwell.

Do **not** treat an in-flight GPQA journal as a 198-row score. Truncated or
unparsed answers count as wrong; resume appends by `item_id`.

## Model facts

| | |
|---|---|
| Hugging Face id | `Qwen/Qwen3.8-27B` |
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` (`text_config.model_type`: `qwen3_5_text`) |
| Size / shape | 27B dense VLM, hidden 5120, 64 layers, FFN 17408 |
| Block layout | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| Extra modules | Vision encoder + MTP |
| Native context | 262144 |
| Public mixed ckpt | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |

Family adapter: `qwen3_5` (`src/megaquant/models/qwen3_5.py`). This is **not**
vanilla Qwen3 (`qwen3` / `qwen3_moe`).

## Recipes

| File | Scheme | Algorithm | Calib | Output |
|---|---|---|---|---|
| `recipes/qwen3.8-27b-nvfp4-w4a8.yaml` | `nvfp4_w4a8` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A8` |
| `recipes/qwen3.8-27b-nvfp4-w4a8.5090.yaml` | `nvfp4_w4a8` | `max` | ultrachat 256×1024, **batch 4** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-w4a8.public-calib.yaml` | `nvfp4_w4a8` | `max` | ultrachat 512 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-w4a8-trtllm.yaml` | `w4a8_nvfp4_fp8` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A8-TRTLLM` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.yaml` | `nvfp4_w4a4` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A4` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.5090.yaml` | `nvfp4_w4a4` | `max` | ultrachat 256×1024, **batch 4** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-w4a4.public-calib.yaml` | `nvfp4_w4a4` | `max` | ultrachat 512 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-mixed.yaml` | `nvfp4_mixed` | `local_hessian` | **2048**, `nvidia/Nemotron-Post-Training-Dataset-v3` | `outputs/Qwen3.8-27B-NVFP4-mixed` |
| `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` | `nvfp4_mixed` | `max` | ultrachat 256×1024, **batch 4** | same; **5090 production PTQ** |
| `recipes/qwen3.8-27b-nvfp4-mixed.public-calib.yaml` | `nvfp4_mixed` | `local_hessian` | ultrachat 2048 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml` | `nvfp4_w4a16_mixed` | `max` | ultrachat 256×1024, **batch 4** | `outputs/Qwen3.8-27B-NVFP4-W4A16-mixed`; optional Marlin export, not the Spark fast path |

All of these set `backend: modelopt`, `kv_cache: fp8`, `family: qwen3_5`,
`model.quantize_vision: false`, and `model.quantize_mtp: false`.

Uniform quality recipes (`w4a8.yaml`, `w4a4.yaml`) use
`nvidia/Nemotron-Post-Training-Dataset-v2` (gated; set `HF_TOKEN`). Mixed
**quality** (`mixed.yaml`) follows the NVIDIA card and uses
**`nvidia/Nemotron-Post-Training-Dataset-v3`**. Anonymous Hub access: the
`*.public-calib.yaml` files and all `*.5090.yaml` files use
`HuggingFaceH4/ultrachat_200k`.

## Ignore list (why these stay BF16)

`Qwen35Family.default_ignore` (unless the recipe flips the flags):

| Pattern | Why |
|---|---|
| `*visual*`, `*vision*` | Vision encoder is not the language-model PTQ target. NVIDIA mixed PTQ also left it in BF16. Set `model.quantize_vision: true` to include it. |
| `*embed_tokens*`, `*embed_positions*` | Embedding tables are poor NVFP4 candidates; keep BF16. |
| `*linear_attn.conv1d*` | Gated DeltaNet depthwise conv — not a standard Linear GEMM; ModelOpt / compressed-tensors NVFP4 paths do not treat it as `q/k/v/o` or `gate/up/down`. |
| `*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*` | GDN extras (not `in_proj_qkv` / `in_proj_z` / `out_proj`). Mixed FP8 attention still quantizes the real GDN projections; these two stay BF16. |
| `*mtp*` | Multi-Token Prediction heads. Off unless `model.quantize_mtp: true`. |
| **not** `*mlp*` | MLP `gate/up/down_proj` are the main NVFP4 targets. |
| **not** `*lm_head*` | NVIDIA mixed NVFP4 quantizes `lm_head`. Uniform W4A8 / W4A4 do too. Add it in `extra_ignore` only if you want BF16 logits. |

Norms are not `Linear` and stay unquantized automatically.

## Mixed vs SGLang W4A8 vs uniform W4A4 vs TRT-LLM W4A8

Default `nvfp4_w4a8` is the same encoding as the public NVIDIA checkpoint
(and as `nvfp4_mixed`). That is what SGLang can serve.

| | SGLang W4A8 (default) | Uniform W4A4 | Mixed quality | TRT-LLM W4A8 |
|---|---|---|---|---|
| Scheme | `nvfp4_w4a8` | `nvfp4_w4a4` | `nvfp4_mixed` | `w4a8_nvfp4_fp8` |
| ModelOpt | mixed overrides + `MIXED_PRECISION` rewrite | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | same mixed cfg as W4A8; Local-Hessian | `mtq.W4A8_NVFP4_FP8_CFG` / `w4a8_nvfp4_fp8` |
| NVFP4 group / block | **16** on MLP + `lm_head` | **16** uniform | **16** on MLP + `lm_head` | **32** (`nvfp4_bs32`) |
| Weights / activations | NVFP4 gs16 on MLP + `lm_head`; **FP8** on self-attn + linear-attn | NVFP4 weights **and** activations (block 16) | same map as SGLang W4A8 | NVFP4 weights + FP8 E4M3 activations on every targeted LM linear |
| SGLang | `modelopt_mixed` (`MIXED_PRECISION`) | `modelopt_fp4` (`NVFP4`) | `modelopt_mixed` | **rejected** (`W4A8_NVFP4_FP8`) |
| Public HF id | — | — | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | — |

`nvidia/Qwen3.8-27B-NVFP4` is mixed NVFP4/FP8. It is **not**
`W4A8_NVFP4_FP8` and **not** NVFP4 block 32.

Accuracy on NVIDIA's card (vLLM, 262k context, mixed checkpoint):

| Benchmark | BF16 | NVFP4 mixed |
|---|---|---|
| GPQA Diamond | 88.92 | 88.01 |
| Terminal-Bench | 75.56 | 74.02 |
| AA-LCR | 72.63 | 73.38 |
| MMMU-Pro | 75.14 | 74.86 |
| SciCode | 47.93 | 48.41 |
| IFBench | 80.07 | 78.93 |

## Calibration

| | SGLang W4A8 | Uniform W4A4 | Mixed (NVIDIA quality) | 5090 packed (`*.5090.yaml`) |
|---|---|---|---|---|
| Recipe | `qwen3.8-27b-nvfp4-w4a8.yaml` | `qwen3.8-27b-nvfp4-w4a4.yaml` | `qwen3.8-27b-nvfp4-mixed.yaml` | `*.5090.yaml` for `w4a8` / `w4a4` / `mixed` |
| Algorithm | `max` | `max` | `local_hessian` (`fp8_scale_sweep: true` in ModelOpt) | `max` |
| Samples | 512 | 512 | 2048 | 256 × 1024, **batch 4** |
| Dataset | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v3` | `HuggingFaceH4/ultrachat_200k` |
| Images | `with_images: false` (text-only; vision is ignored) | same | same | same |

`max` is cheaper and is the default for SGLang W4A8, uniform W4A4, and the
5090 mixed profile. It records activation amax and round-to-nearest
quantizes (RTN). Local-Hessian is the quality knob NVIDIA used for mixed
NVFP4 (`fp8_scale_sweep: true`, Hessian `block_size` 16); it is slower and
does not fit a 32 GB 5090 at 2048 samples. The 5090 production run **is**
`mixed.5090.yaml` (`max` + ultrachat 256), not `mixed.yaml`. You can
override without editing YAML:

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml \
  --algorithm local_hessian --num-samples 2048
```

## Commands

```bash
pip install -e '.[hf,modelopt]'

# No GPU, no 27B download:
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml

# PTQ (Hopper or better, multi-GPU / CPU offload for 27B BF16):
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
# 5090 production (max + ultrachat; not Hessian):
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
```

Export is a Hugging Face unified checkpoint
(`modelopt.torch.export.export_hf_checkpoint` on the ModelOpt path, then
`rewrite_sglang_mixed_export` for `nvfp4_w4a8` / `nvfp4_mixed`). Destination
is `export.output_dir`. The pipeline also writes `provenance.json` (base
model, scheme, backend, calib, git sha, timestamp).

To patch an already-exported mixed / default-W4A8 directory without
re-running PTQ:

```bash
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-mixed
```

Do this **before SGLang serve** when `hf_quant_config.json` is still a
bare `NVFP4` tag without `quantized_layers` (ModelOpt 0.46 often writes
that). Uniform `w4a8_nvfp4_fp8` is TensorRT-LLM only — do not rewrite it
and call it SGLang-loadable.

27B BF16 ≈ 54 GiB of weights plus activations. Set `model.device_map` (`auto`
by default) or CUDA_VISIBLE_DEVICES; expect multiple 80 GB Hopper GPUs or
heavy CPU offload. NVFP4 **inference** still needs Blackwell SM100+.

## Single 5090 + 64 GB RAM

The 27B BF16 does not fit in 32 GB. Mega-Quantification packs the box
instead of leaving headroom idle:

| Resource | Packing |
|---|---|
| GPU | Weights fill **VRAM − 1 GiB** (`MEGAQUANT_GPU_HEADROOM_GIB`). |
| RAM | Weights that do not fit on GPU stay in **MemTotal − 6 GiB** (~56 GiB on a 64 GB pod). Disk `offload_folder` is spill-only. |
| CPU | `nproc` threads via `OMP_NUM_THREADS` / `torch.set_num_threads`. Calib tensors are `pin_memory`'d. |
| PCIe | Idle 5090 reports **gen1 x16**. PTQ runs a pinned H2D/D2H warmup so the link trains to **gen5 x16** (~50 GiB/s DMA). CPU-resident weights are pinned; accelerate copies use `non_blocking=True`; calib prefetches the next batch on a CUDA copy stream. `CUDA_DEVICE_MAX_CONNECTIONS=16`. |
| Calib | 5090 recipes use `batch_size: 4` so one CPU↔GPU weight walk covers 4 samples. |

K8s GPU pod (already a container). Scheme argument selects the
5090-packed recipe (`w4a8` default):

```bash
bash scripts/gpu-pod.sh plan
bash scripts/gpu-pod.sh quantize
bash scripts/gpu-pod.sh plan w4a4
bash scripts/gpu-pod.sh quantize w4a4
bash scripts/gpu-pod.sh plan mixed
bash scripts/gpu-pod.sh quantize mixed
```

Usage is `scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`.
These commands start a job; they do not imply that W4A4 / mixed PTQ or an
OSS upload has already finished on the pod.

Compose on a machine that has Docker:

```bash
docker compose --profile gpu run --rm quantize
docker compose --profile gpu run --rm w4a4
docker compose --profile gpu run --rm mixed
```

The `mixed` Compose service uses `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`
(max, ultrachat 256×1024, batch 4). NVIDIA quality Local-Hessian:

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

Install GDN fused kernels for the next run (`kernels` + `flash-linear-attention`);
without them transformers falls back to a PyTorch Gated DeltaNet and SM% stays
single-digit. Do not compile `causal-conv1d` while another PTQ is on the GPU.

Caps if you need them: `MEGAQUANT_MAX_MEMORY=0:29GiB,cpu:56GiB`,
`MEGAQUANT_NUM_THREADS`, `MEGAQUANT_BATCH_SIZE`, `MEGAQUANT_GPU_HEADROOM_GIB`.

## OSS publish

After a scheme finishes, upload the export as
`<prefix>/<scheme>/<content-hash>/` (`w4a8` / `w4a4` / `mixed`).
`<content-hash>` is SHA256 of sorted `{safetensors-name} {sha256}\\n` lines.
Bucket and endpoint come from `OSS_BUCKET` / `OSS_ENDPOINT` or `--bucket` /
`--endpoint` (optional gitignored `.oss.env`). Objects larger than 5 GiB use
multipart upload.

Mixed encoding **is** the default W4A8. A finished mixed export can be
published twice (`--scheme mixed` and `--scheme w4a8`) under the **same
content-hash** — do not run a second PTQ. Uniform `w4a8_nvfp4_fp8` is
TensorRT-LLM only; do not publish or advertise it as SGLang-loadable `w4a8`.

```bash
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A4 --scheme w4a4
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-mixed --scheme mixed
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-mixed --scheme w4a8
# intended gpu-pod wrapper (same env / prefix):
bash scripts/gpu-pod.sh publish w4a4
bash scripts/gpu-pod.sh publish mixed
bash scripts/gpu-pod.sh publish w4a8
```

## Serve (SGLang / vLLM / TensorRT-LLM)

Default inference is **SGLang**. `megaquant serve --dry-run` prints the
resolved argv (cookbook flags + HiCache size). Point `--model-path` at
your export dir or at `nvidia/Qwen3.8-27B-NVFP4`.

### SGLang

Needs a recent `sglang` / `lmsysorg/sglang:dev` that routes
`MIXED_PRECISION` + NVFP4 `quantized_layers` to `modelopt_mixed`
(SGLang PR #28099). Older builds treated non-NemotronH `MIXED_PRECISION`
as `w4afp8` and will not run this checkpoint.

```sh
# Recipe-built argv (adds HiCache so 262k fits a 32 GB card):
bash scripts/serve_sglang.sh outputs/Qwen3.8-27B-NVFP4-W4A8
# or: python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
#        --model outputs/Qwen3.8-27B-NVFP4-W4A8

sglang serve \
  --trust-remote-code \
  --model-path outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype float32 \
  --attention-backend flashinfer \
  --context-length 262144 \
  --enable-hierarchical-cache \
  --hicache-size 58 \
  --host 0.0.0.0 \
  --port 30000
```

Docker: `lmsysorg/sglang:dev` or `make serve-sglang` (not `make serve-vllm`
as the primary path). If `hf_quant_config.json` still says
`quant_algo: W4A8_NVFP4_FP8` or a bare `NVFP4` without `quantized_layers`,
run `megaquant rewrite-sglang <export_dir>` first. GPQA client:
`MEGAQUANT_SGLANG_BASE_URL` (default `http://127.0.0.1:30000/v1`).

### vLLM

Optional (`--engine vllm`). NVIDIA mixed card on GB300 used these flags:

```sh
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
  --engine vllm --model outputs/Qwen3.8-27B-NVFP4-W4A8
```

### TensorRT-LLM

NVIDIA documented vLLM + SGLang for the 27B NVFP4 checkpoint. TRT-LLM's
published Qwen3.8 guides currently target the **2.4T MoE**, not this 27B dense
VLM — treat the snippet as a starting point on Blackwell, not a certified
profile:

```sh
trtllm-serve outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --backend pytorch \
  --tp_size 4 \
  --kv_cache_dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

Prefer TensorRT-LLM over vLLM when you need fused NVFP4+FP8 W4A8 kernels;
vLLM support for that combo is limited.

## GPQA Diamond (official card protocol)

Evaluate each NVFP4 export with the **same sampling as the Qwen thinking
card** on a **SGLang** server (NVIDIA Qwen3.8 cookbook flags). Do not
greedy-decode. Do not cap generation at 512/2048 tokens.

| Knob | Value |
|---|---|
| Sampling | `temperature=1.0 top_p=0.95 top_k=20 min_p=0 presence_penalty=0 repetition_penalty=1.0 do_sample=true` |
| Thinking | `enable_thinking=true preserve_thinking=true reasoning_effort=xhigh` |
| Context | `context-length=262144`; `max_new_tokens=0` means the remaining window |
| Truncation | fill remaining context; `continue_on_length` keeps going until EOS (up to 8 continuations) |
| KV on 32 GB | SGLang `--enable-hierarchical-cache` + `--hicache-size`. Cookbook ~58 GiB on a 64 GB box. This 5090 VM (94 GiB) pins **64 GiB** HiCache (`eval-gpqa-diamond.5090.yaml`). SGLang `_split_hicache_size` splits that host pool by the **GPU** Mamba vs KV pool sizes — a fat GPU mamba cache also steals host KV. |
| Concurrency | default recipe 1; 5090 recipe **24** (`max_running_requests: 24`, `max_mamba_cache_size: 96` bf16 GDN slots so 24×4); 80 GB recipe **64** (`max_mamba_cache_size: 256`). float32 64-slot mamba used ~9.3 GB HBM and left ~0.88 GB GPU KV, so 16 HTTP workers queued behind 3–4 decode slots. |
| Mamba / GDN | 5090 and 80 GB: `mamba_ssm_dtype: bfloat16`, `mamba_radix_cache_strategy: extra_buffer_lazy`. NVIDIA SGLang cookbook: `extra_buffer` + float32. |
| Attention / GEMM | default FlashInfer (`eval-gpqa-diamond.yaml`); CUDA 12.8 SM120: Triton attn + Marlin NVFP4 + `SGLANG_FORCE_FP8_MARLIN`. 32 GB recipe also disables CUDA graph and uses HiCache 64 GiB (`eval-gpqa-diamond.5090.yaml`). 80 GB recipe keeps CUDA graph, leaves KV on GPU, uses CUTLASS FP8, and sets `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION` (`eval-gpqa-diamond.6000d.yaml`). |
| Compose eval | no GPU (`NVIDIA_VISIBLE_DEVICES=""`, no `gpus:`); client talks to `serve-sglang:30000` |
| Headline | `correct/198` on GPQA Diamond (full denominator; truncated/unparsed count as wrong). Do not quote a partial journal as the score. |

NVIDIA's mixed card reports GPQA Diamond **88.92 BF16 / 88.01 NVFP4** on
GB300 **vLLM** (`temp=1.0 top_p=0.95`, `max_new_tokens=65536`). The Qwen
card reports **89.2** (thinking, `top_k=20`). Those numbers are not greedy.
A 5090 SGLang run of our `max`+ultrachat checkpoint is a different PTQ
**and** a different eval stack; treat it as a separate measurement.

On a 32 GB 5090 the 262k window does not fit in HBM. **Offload KV into host
RAM** rather than shrinking `max_new_tokens`. Qwen GPQA traces can run tens
of thousands of tokens; a short cap is not an official-card eval. Host
`docker-compose.override.yml` (not in git) that hardcodes `sglang` argv must
match the recipe (`--hicache-size 64`, `--max-mamba-cache-size 96`,
`--mamba-ssm-dtype bfloat16`) or the YAML never reaches the server. Journals
resume by `item_id`; do not truncate `gpqa_diamond.jsonl`.

```bash
# Terminal 1 — serve (SGLang HiCache KV → RAM)
bash scripts/gpu-pod.sh serve w4a8
# or: python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
#        --model outputs/Qwen3.8-27B-NVFP4-W4A8

# Terminal 2 — client (full 198; needs HF_TOKEN or GPQA_CSV for gated Hub)
MEGAQUANT_SGLANG_BASE_URL=http://127.0.0.1:30000/v1 \
  bash scripts/gpu-pod.sh eval w4a8
# same for w4a4 / mixed after those exports exist

# No GPU, no 27B, no Hub (local export, not Qwen/Qwen3.8-27B):
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
```

Compose: `make serve-sglang` then `make eval-gpqa`. `eval-gpqa` does not
attach a GPU; `serve-sglang` does. On CUDA 12.8 / SM 12.0 set
`EVAL_RECIPE=recipes/eval-gpqa-diamond.5090.yaml`. The client talks to
`MEGAQUANT_SGLANG_BASE_URL` (default `http://127.0.0.1:30000/v1`). Journals
land in `outputs/eval/gpqa_diamond-<scheme>/` (`gpqa_diamond.jsonl` keeps
the full text; stdout is a one-line status). Optional `--engine vllm`
keeps the NVIDIA GB300 vLLM flags on port 8000.

`Idavidrein/gpqa` is gated. Set `HF_TOKEN` or point `GPQA_CSV` at a local
CSV with the Hub columns (`Question`, `Correct Answer`,
`Incorrect Answer 1/2/3`, `Record ID`).

## Troubleshooting

- Dry-run prints `family 'qwen3_5' is not registered`: `import megaquant.models`
  (the CLI pipeline loads plugins lazily; adapters register on that import).
- Plan tries to download `config.json`: set `family: qwen3_5` in the YAML
  (already set) so resolve() skips Hub lookup.
- Do not pass vanilla `qwen3` as `family` for this checkpoint — GDN extras
  would not be ignored.
- Block size 16 vs 32: default SGLang W4A8 (`nvfp4_w4a8`) is NVFP4
  **group_size 16** on MLP + `lm_head` with FP8 attention
  (`MIXED_PRECISION`). Uniform W4A4 is NVFP4 **16** (`NVFP4_DEFAULT_CFG`).
  TensorRT-LLM uniform W4A8 is NVFP4 **32** (`W4A8_NVFP4_FP8` /
  scheme `w4a8_nvfp4_fp8`). SGLang rejects that tag. Using block 32 for
  mixed will not match `nvidia/Qwen3.8-27B-NVFP4`.
