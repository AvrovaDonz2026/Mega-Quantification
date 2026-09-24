# Qwen3.8-27B

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 是 27B 稠密 VLM（`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`）。这条管线经过工业级验证：量化在 32 GB RTX 5090 上跑完，GPQA Diamond 在 80 GB SM120 上用 SGLang 跑完。量化打在语言模型的线性层上。视觉塔、embedding、MTP，以及 GDN 的 `conv1d` / `in_proj_a` / `in_proj_b` 留在 BF16。MTP 写进 `mtp.safetensors`。投机解码时，把 `--speculative-draft-model-path` 指到旁边那个 1 层的 `*-draft` 目录。`megaquant serve` 不会替你加上这个参数。

默认方案 `nvfp4_w4a8` 和 `nvfp4_mixed` 是同一张层图，跟 NVIDIA 公开的 [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) 对齐：MLP 和 `lm_head` 是 NVFP4 group 16（权重和激活都是），self-attn 和 linear-attn 是 FP8。导出写成 `quant_algo=MIXED_PRECISION`，并带上 `quantized_layers`。均匀 W4A4 是另一张图，选中的线性层全部是 NVFP4 block 16。NVFP4 推理要 Blackwell；校准可以在 Hopper 上多卡或 offload 完成。

32 GB 5090 上跑完的配方是 [`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`](../recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml)：ModelOpt 0.46.1（PyPI 上的版本；公开卡写的是 0.48.0）、算法 `max`（记下激活 amax，再 round-to-nearest）、`HuggingFaceH4/ultrachat_200k` 256×1024、batch 4。`outputs/Qwen3.8-27B-NVFP4-W4A8` 就是这次导出。那次 `quantized_layers` 有 401 项，193 个 NVFP4、208 个 FP8。`max` 动的是校准，层的比特布局和 Hessian 那版相同，只是没有 Local-Hessian 重建，也没有 FP8 scale sweep。想贴近公开卡，用 [`recipes/qwen3.8-27b-nvfp4-mixed.yaml`](../recipes/qwen3.8-27b-nvfp4-mixed.yaml)：Local-Hessian（`fp8_scale_sweep`，Hessian `block_size` 16）、`nvidia/Nemotron-Post-Training-Dataset-v3`、2048×2048、batch 1。32 GB 放不下这次校准。公开卡在 GB300 的 vLLM 上报 GPQA Diamond 88.92（BF16）/ 88.01（NVFP4），`max_new_tokens=65536`。Qwen 自己的 thinking 分是 89.2。

DGX Spark（GB10，大约 273 GB/s）用的也是这份混合权重：MLP 带 NVFP4 激活，注意力和 KV 是 FP8。普通 decode 大约 12 tok/s，带宽上限大约 14 tok/s，再快靠这上面的 MTP 投机解码。`nvfp4_w4a16_mixed` 是 32 GB 机器上的可选 Marlin 导出，MLP 激活留 BF16。

代理从仓库根目录的 [SKILL.md](../SKILL.md) 读硬约定。

## Quantization format

| 张量 | 权重 | 激活 | `quantized_layers` |
|---|---|---|---|
| `mlp.{gate,up,down}_proj` | NVFP4 E2M1，group_size 16，scale 为 FP8 E4M3 | NVFP4，group_size 16 | `quant_algo: NVFP4`, `group_size: 16` |
| `lm_head` | 同上 | 同上 | 同上 |
| `self_attn.{q,k,v,o}_proj` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| `linear_attn.conv1d` / `in_proj_a` / `in_proj_b` | BF16 | BF16 | 不写入 |
| 视觉塔、MTP、embedding、norm | BF16 | BF16 | 不写入；MTP 张量在 `mtp.safetensors` |
| 推理时的 KV | | fp8_e4m3 | 配方 `kv_cache: fp8` |

`hf_quant_config.json` 顶层是 `MIXED_PRECISION`，`quantized_layers` 非空。SGLang 0.5.20 按 `modelopt_mixed` 加载。导出如果还是裸的 `NVFP4`，先跑 `megaquant rewrite-sglang <export_dir>`。本仓库的方案停在 group 16 的混合图和均匀 W4A4；ModelOpt 那个 block 32 的 `W4A8_NVFP4_FP8` 标签 SGLang 不收，这里也不量化成它。

| 方案 | 和上表的差别 | 导出 | 运行时 |
|---|---|---|---|
| `nvfp4_w4a4` | 选中的 LM 线性层全部是 NVFP4 W4A4 group 16，含注意力 | `quant_algo: NVFP4` | SGLang `modelopt_fp4` |
| `nvfp4_w4a16_mixed` | MLP + `lm_head` 为 NVFP4 group 16 权重、BF16 激活；注意力仍是 FP8 | MLP 条目 `quant_algo: W4A16_NVFP4` | 可选 Marlin 导出 |

## SGLang inference and GPQA

推理和 GPQA 都走 SGLang。评测客户端打 `http://127.0.0.1:30000/v1` 的 OpenAI chat。仓库里的 YAML 现在发 temperature 0，其余仍是 Qwen thinking 卡：`top_p=0.95`，`top_k=20`，`min_p=0`，`presence_penalty=0`，`repetition_penalty=1`，`enable_thinking` 与 `preserve_thinking`，`reasoning_effort=xhigh`。`max_new_tokens: 0` 用完剩下的 262144 上下文，`continue_on_length` 一直续到 EOS，最多 8 次续写。HTTP 超时 21600 秒。完整轨迹在 `<model>/gpqa_diamond/gpqa_diamond.jsonl`，旁边有 `summary.json`，按 `item_id` 续写。198 行都在、并且 `summary.json` 也在时，分数才是 `correct/198`。截断和解析失败算错。

跑完的两次：

| 权重 | 采样 | 分数 | 机器 | 日期 |
|---|---|---|---|---|
| 混合 / 默认 W4A8（5090 的 `max` + ultrachat） | temperature 1.0，其余同 thinking 卡 | **178/198**（截断 0，解析失败 0） | 80 GB SM120，SGLang，64 路，KV 在 GPU | 2026-09-22 |
| 均匀 W4A4 | temperature 0，Marlin | **172/198**（截断 4，解析失败 6） | 同一天 | 2026-09-22 |

混合权重还没有跑完的 temperature 0 总分。第一行是 temperature 1 的测量，当前 YAML 发的是 0。

| | 默认 | 32 GB SM120（5090） | 80 GB SM120（6000D） |
|---|---|---|---|
| 配方 | `eval-gpqa-diamond.yaml` | `eval-gpqa-diamond.5090.yaml` | `eval-gpqa-diamond.6000d.yaml` |
| 注意力 | FlashInfer | Triton | Triton |
| NVFP4 GEMM | FlashInfer（CUDA ≥ 12.9） | Marlin | Marlin |
| FP8 GEMM | FlashInfer | Triton，`SGLANG_FORCE_FP8_MARLIN=1` | CUTLASS，`SGLANG_FORCE_FP8_MARLIN=1` |
| CUDA graph | 关 | 关（32 GB 上抓 graph 会 OOM） | 开 |
| KV | HiCache，主机 12 GiB | HiCache，主机 64 GiB，按 GPU 上 KV / GDN 池的比例切 | 只在 GPU |
| 并发 | 1 | 24（bf16 GDN，96 slot） | 64（bf16 GDN，256 slot） |
| `mem-fraction-static` | 0.85 | 0.95 | 0.90 |
| chunked prefill | 2048 | 2048 | 4096 |
| 额外环境变量 | | | `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1`，`SGLANG_IS_FLASHINFER_AVAILABLE=0`，`SGLANG_ENABLE_JIT_DEEPGEMM=0` |
| Radix cache | 开 | 开 | 关 |

CUDA 12.8 编不了 SM 12.x 的 FlashInfer JIT。80 GB 配方把 SiLU+FP4 融合关掉，因为那条路径即使 GEMM 走 Marlin，import 时仍会拉 FlashInfer。float32、64 个 mamba slot 大约占 9.3 GB HBM，GPU 上的 KV 剩不到 1 GB，16 路 HTTP 会排在 3 到 4 个 decode slot 后面。

```bash
# 80 GB SM120，CUDA 12.8
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.6000d.yaml \
  --model outputs/Qwen3.8-27B-NVFP4-W4A8
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.6000d.yaml \
  --base-url http://127.0.0.1:30000/v1

# 32 GB 5090
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.5090.yaml \
  --model outputs/Qwen3.8-27B-NVFP4-W4A8
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.5090.yaml \
  --base-url http://127.0.0.1:30000/v1
```

`megaquant serve` 先把配方里的环境变量导出，再执行 `sglang serve`。`--dry-run` 只打印 argv，不会下载 `Qwen/Qwen3.8-27B`。要和 GB300 上的 vLLM 卡片对照时，加 `--engine vllm`。

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
| `recipes/qwen3.8-27b-nvfp4-w4a4.yaml` | `nvfp4_w4a4` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A4` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.5090.yaml` | `nvfp4_w4a4` | `max` | ultrachat 256×1024, **batch 4** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-w4a4.public-calib.yaml` | `nvfp4_w4a4` | `max` | ultrachat 512 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-mixed.yaml` | `nvfp4_mixed` | `local_hessian` | **2048**, `nvidia/Nemotron-Post-Training-Dataset-v3` | `outputs/Qwen3.8-27B-NVFP4-mixed` |
| `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` | `nvfp4_mixed` | `max` | ultrachat 256×1024, **batch 4** | `outputs/Qwen3.8-27B-NVFP4-W4A8`; **5090 production PTQ** |
| `recipes/qwen3.8-27b-nvfp4-mixed.public-calib.yaml` | `nvfp4_mixed` | `max` | ultrachat 512 (anonymous Hub) | `outputs/Qwen3.8-27B-NVFP4-mixed` |
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
| `*mtp*` | Multi-Token Prediction stays **BF16 in the export** (`mtp.safetensors`). `quantize_mtp: false` means do not quantize it, not delete it. See [MTP export](#mtp-export). |
| **not** `*mlp*` | MLP `gate/up/down_proj` are the main NVFP4 targets. |
| **not** `*lm_head*` | NVIDIA mixed NVFP4 quantizes `lm_head`. Uniform W4A8 / W4A4 do too. Add it in `extra_ignore` only if you want BF16 logits. |

Norms are not `Linear` and stay unquantized automatically.

## Mixed vs SGLang W4A8 vs uniform W4A4

三列是三份配方，比特布局见 [Quantization format](#quantization-format)。`nvfp4_w4a8` 和 `nvfp4_mixed` 导出同一张混合图；质量配方把校准换成 Local-Hessian。

| | SGLang W4A8 (default) | Uniform W4A4 | Mixed quality |
|---|---|---|---|
| Scheme | `nvfp4_w4a8` | `nvfp4_w4a4` | `nvfp4_mixed` |
| ModelOpt | mixed overrides + `MIXED_PRECISION` rewrite | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | same mixed cfg as W4A8; Local-Hessian |
| NVFP4 group / block | 16 on MLP + `lm_head` | 16 uniform | 16 on MLP + `lm_head` |
| Weights / activations | NVFP4 gs16 on MLP + `lm_head`; FP8 on self-attn + linear-attn | NVFP4 weights and activations (block 16) | same map as SGLang W4A8 |
| SGLang | `modelopt_mixed` (`MIXED_PRECISION`) | `modelopt_fp4` (`NVFP4`) | `modelopt_mixed` |
| Public HF id | — | — | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |

NVIDIA 公开卡上的其他基准（vLLM，262k，混合权重）：

| Benchmark | BF16 | NVFP4 mixed |
|---|---|---|
| GPQA Diamond | 88.92 | 88.01 |
| Terminal-Bench | 75.56 | 74.02 |
| AA-LCR | 72.63 | 73.38 |
| MMMU-Pro | 75.14 | 74.86 |
| SciCode | 47.93 | 48.41 |
| IFBench | 80.07 | 78.93 |

本仓库自己的 GPQA 在 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。

## Calibration

| | SGLang W4A8 | Uniform W4A4 | Mixed (NVIDIA quality) | 5090 packed (`*.5090.yaml`) |
|---|---|---|---|---|
| Recipe | `qwen3.8-27b-nvfp4-w4a8.yaml` | `qwen3.8-27b-nvfp4-w4a4.yaml` | `qwen3.8-27b-nvfp4-mixed.yaml` | `*.5090.yaml` for `w4a8` / `w4a4` / `mixed` |
| Algorithm | `max` | `max` | `local_hessian` (`fp8_scale_sweep: true` in ModelOpt) | `max` |
| Samples | 512 | 512 | 2048 | 256 × 1024, **batch 4** |
| Dataset | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v3` | `HuggingFaceH4/ultrachat_200k` |
| Images | `with_images: false` (text-only; vision is ignored) | same | same | same |

`max` 记下激活的 amax，再 round-to-nearest。它是 SGLang W4A8、均匀 W4A4，以及 5090 混合配方的默认算法。Local-Hessian（`fp8_scale_sweep: true`，Hessian `block_size` 16）是 NVIDIA 混合卡用的质量档，2048 条在 32 GB 上放不下。5090 上跑完的是 `mixed.5090.yaml`（`max` + ultrachat 256）。`mixed.yaml` 留给更大的卡。不改 YAML 也可以临时换算法：

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

ModelOpt 0.46 经常只写一个裸的 `NVFP4` 标签、没有 `quantized_layers`。这种目录在 `sglang serve` 之前先 rewrite。要加载的是 `MIXED_PRECISION` 那份导出。

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
MIXED_RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

Install GDN fused kernels for the next run (`kernels` + `flash-linear-attention`);
without them transformers falls back to a PyTorch Gated DeltaNet and SM% stays
single-digit. Do not compile `causal-conv1d` while another PTQ is on the GPU.

Caps if you need them: `MEGAQUANT_MAX_MEMORY=0:29GiB,cpu:56GiB`,
`MEGAQUANT_NUM_THREADS`, `MEGAQUANT_BATCH_SIZE`, `MEGAQUANT_GPU_HEADROOM_GIB`.

## MTP export

`quantize_mtp: false` keeps MTP in BF16. ModelOpt `export_hf_checkpoint`
often drops CPU-pinned `mtp.*`, so the backend copies them back after
export:

1. In-memory `mtp` module if it is still on the quantized model
2. Else the original HF source — **only** shards whose index entries are
   `mtp.*` (Qwen3.8: the last BF16 shard). Never the full 27B.

The tensors land in `mtp.safetensors` and stay out of `quantized_layers`
(`exclude_modules` already lists `mtp*`). SGLang's Qwen3.5 target loader
skips `mtp`; the draft loader wants those exact top-level `mtp.*` names.

SGLang 0.5.20 does not shrink Qwen3.5 hybrid depth for the draft
`ModelConfig`, so export also writes a sibling **1-layer** directory
`<export_dir>-draft`. Point speculative decode there:

```bash
# already-exported checkpoint that lost MTP:
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
# or just the draft helper:
megaquant write-mtp-draft outputs/Qwen3.8-27B-NVFP4-W4A8

sglang serve --model-path outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path outputs/Qwen3.8-27B-NVFP4-W4A8-draft \
  ...
```

`--speculative-draft-model-path` 指 `*-draft`。64 层那个目录是 target。`megaquant serve` 和 `megaquant eval` 不加投机参数，要在 `sglang serve` 的命令行上自己写。`w4a8` / `w4a4` / `mixed` 上传的是完整导出，`*-draft` 只是旁边的 1 层助手。`model.quantize_mtp: true` 会把 draft 量化，而不是拷回 BF16。

## OSS publish

After a scheme finishes, upload the export as
`<prefix>/<scheme>/<content-hash>/` (`w4a8` / `w4a4` / `mixed`).
`<content-hash>` is SHA256 of sorted `{safetensors-name} {sha256}\\n` lines.
Bucket and endpoint come from `OSS_BUCKET` / `OSS_ENDPOINT` or `--bucket` /
`--endpoint` (optional gitignored `.oss.env`). Objects larger than 5 GiB use
multipart upload. Real uploads need `oss2`: `pip install oss2` or
`pip install megaquant[oss]`. `--dry-run` does not.

混合导出就是默认 W4A8。同一份目录用 `--scheme mixed` 和 `--scheme w4a8` 各发一次，content-hash 相同，不用再跑一遍 PTQ。`w4a8` 对应的是 `MIXED_PRECISION` 导出。

```bash
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A4 --scheme w4a4
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A8 --scheme mixed
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A8 --scheme w4a8
# intended gpu-pod wrapper (same env / prefix):
bash scripts/gpu-pod.sh publish w4a4
bash scripts/gpu-pod.sh publish mixed
bash scripts/gpu-pod.sh publish w4a8
```

Local-Hessian `mixed.yaml` still writes `outputs/Qwen3.8-27B-NVFP4-mixed`. Publish that directory the same way if that is the export you ran.

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

The block above is the cookbook default (`eval-gpqa-diamond.yaml`). The 32 GB and 80 GB SM120 boxes use `eval-gpqa-diamond.5090.yaml` and `eval-gpqa-diamond.6000d.yaml`. See [SGLang inference and GPQA](#sglang-inference-and-gpqa).

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

比特布局和三份机器配方在 [Quantization format](#quantization-format) 和 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。这里是采样锁和启动时容易踩的地方。

每份 NVFP4 导出都在 SGLang 上评（NVIDIA Qwen3.8 cookbook 的旗标）。这份 eval 发 temperature 0，其余字段跟 Qwen thinking 卡一致。公开卡是 temperature 1.0。生成长度保持 262144 的剩余窗口，512 或 2048 的截断不是这张卡的协议。

| Knob | Value |
|---|---|
| Sampling | `temperature=0 top_p=0.95 top_k=20 min_p=0 presence_penalty=0 repetition_penalty=1.0 do_sample=true`. Published Qwen / NVIDIA cards use temperature 1.0. |
| Thinking | `enable_thinking=true preserve_thinking=true reasoning_effort=xhigh` |
| Context | `context-length=262144`; `max_new_tokens=0` means the remaining window |
| Truncation | fill remaining context; `continue_on_length` keeps going until EOS (up to 8 continuations) |
| KV on 32 GB | SGLang `--enable-hierarchical-cache` + `--hicache-size`. Cookbook ~58 GiB on a 64 GB box. This 5090 VM (94 GiB) pins **64 GiB** HiCache (`eval-gpqa-diamond.5090.yaml`). SGLang `_split_hicache_size` splits that host pool by the **GPU** Mamba vs KV pool sizes — a fat GPU mamba cache also steals host KV. |
| Concurrency | default recipe 1; 5090 recipe **24** (`max_running_requests: 24`, `max_mamba_cache_size: 96` bf16 GDN slots so 24×4); 80 GB recipe **64** (`max_mamba_cache_size: 256`). float32 64-slot mamba used ~9.3 GB HBM and left ~0.88 GB GPU KV, so 16 HTTP workers queued behind 3–4 decode slots. |
| Mamba / GDN | 5090 and 80 GB: `mamba_ssm_dtype: bfloat16`, `mamba_radix_cache_strategy: extra_buffer_lazy`. NVIDIA SGLang cookbook: `extra_buffer` + float32. |
| Attention / GEMM | default FlashInfer (`eval-gpqa-diamond.yaml`); CUDA 12.8 SM120: Triton attn + Marlin NVFP4 + `SGLANG_FORCE_FP8_MARLIN`. 32 GB recipe also disables CUDA graph and uses HiCache 64 GiB (`eval-gpqa-diamond.5090.yaml`). 80 GB recipe keeps CUDA graph, leaves KV on GPU, uses CUTLASS FP8, and sets `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION` (`eval-gpqa-diamond.6000d.yaml`). |
| Compose eval | no GPU (`NVIDIA_VISIBLE_DEVICES=""`, no `gpus:`); client talks to `serve-sglang:30000` |
| Headline | `correct/198` once `summary.json` exists and the journal has every row. Truncated and unparsed rows count as wrong. |

分数见上面的 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。32 GB 上 262k 窗口放不进 HBM，用 HiCache 把 KV 放到主机内存，保持 `max_new_tokens`。Qwen 的 GPQA 轨迹会到几万 token。宿主机上的 `docker-compose.override.yml`（不进 git）如果写死了 `sglang` argv，要和配方一致：`--hicache-size 64`、`--max-mamba-cache-size 96`、`--mamba-ssm-dtype bfloat16`，否则 YAML 到不了进程。journal 按 `item_id` 续写，保留 `gpqa_diamond.jsonl` 里已有的行。

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
land in `<model>/gpqa_diamond/` (`gpqa_diamond.jsonl` keeps the full text;
stdout is a one-line status). Optional `--engine vllm`
keeps the NVIDIA GB300 vLLM flags on port 8000.

`Idavidrein/gpqa` is gated. Set `HF_TOKEN` or point `GPQA_CSV` at a local
CSV with the Hub columns (`Question`, `Correct Answer`,
`Incorrect Answer 1/2/3`, `Record ID`).

## Troubleshooting

- Dry-run prints `family 'qwen3_5' is not registered`: `import megaquant.models`
  (the CLI pipeline loads plugins lazily; adapters register on that import).
- Plan tries to download `config.json`: set `family: qwen3_5` in the YAML
  (already set) so resolve() skips Hub lookup.
- 这个 checkpoint 的 `family` 用 `qwen3_5`。写成 vanilla `qwen3` 时，GDN 那些额外模块不会进忽略列表。
- group size 见 [Quantization format](#quantization-format)。混合图和均匀 W4A4 都是 16。
