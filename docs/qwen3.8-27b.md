# Qwen3.8-27B NVFP4 runbook

End-to-end notes for quantizing **Qwen/Qwen3.8-27B** (BF16) with Mega-Quantification.

中文要点：这是 27B 稠密 VLM（`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`）。
默认量化语言模型线性层；视觉、MTP、embedding、GDN 的 `conv1d` / `in_proj_a` /
`in_proj_b` 留 BF16。

NVIDIA 公开 [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
是 **混合 NVFP4/FP8**：MLP + `lm_head` 为 **NVFP4 group_size 16**，self-attn +
linear-attn 为 **FP8**。它 **不是** 均匀 W4A8，也 **不是** NVFP4 block 32。
模型卡：Local-Hessian、2048 条、`Nemotron-Post-Training-Dataset-v3`、
`nvidia-modelopt` v0.48.0。

本仓库的均匀 W4A8 仍是 block 32（`W4A8_NVFP4_FP8`）。均匀 W4A4 是
`NVFP4_DEFAULT_CFG`（权重和激活均为 NVFP4 block 16）。均匀配方用 `max` + 512
条；对齐 NVIDIA 公开混合权重用 `recipes/qwen3.8-27b-nvfp4-mixed.yaml`
（`local_hessian` + 2048 + Nemotron v3）。NVFP4 推理需要 Blackwell；校准可以在
Hopper 上用多卡 / offload 做。

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
| `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` | `nvfp4_mixed` | `max` | ultrachat 256×1024, **batch 4** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-mixed.public-calib.yaml` | `nvfp4_mixed` | `local_hessian` | ultrachat 2048 (anonymous Hub) | same |

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

## Mixed vs uniform W4A8 vs uniform W4A4

These are three different encodings. The public NVIDIA checkpoint is only the
mixed column.

| | Uniform W4A8 | Uniform W4A4 | Mixed (NVIDIA public) |
|---|---|---|---|
| Scheme | `nvfp4_w4a8` | `nvfp4_w4a4` | `nvfp4_mixed` |
| ModelOpt | `mtq.W4A8_NVFP4_FP8_CFG` / `w4a8_nvfp4_fp8` | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | custom: `NVFP4_DEFAULT_CFG` on `*mlp*` + `*lm_head*`, FP8 on `*self_attn*` + `*linear_attn*` |
| NVFP4 group / block | **32** (`nvfp4_bs32`) | **16** | **16** on MLP + `lm_head` (**not** 32) |
| Weights / activations | NVFP4 weights + FP8 E4M3 activations on every targeted LM linear | NVFP4 weights **and** activations (block 16) on every targeted LM linear | NVFP4 (group_size 16) on MLP + `lm_head`; **FP8** on self-attn + linear-attn |
| Public HF id | — | — | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |

`nvidia/Qwen3.8-27B-NVFP4` is mixed NVFP4/FP8. It is **not** uniform W4A8 and
**not** NVFP4 block 32.

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

| | Uniform W4A8 | Uniform W4A4 | Mixed (NVIDIA quality) | 5090 packed (`*.5090.yaml`) |
|---|---|---|---|---|
| Recipe | `qwen3.8-27b-nvfp4-w4a8.yaml` | `qwen3.8-27b-nvfp4-w4a4.yaml` | `qwen3.8-27b-nvfp4-mixed.yaml` | `*.5090.yaml` for `w4a8` / `w4a4` / `mixed` |
| Algorithm | `max` | `max` | `local_hessian` (`fp8_scale_sweep: true` in ModelOpt) | `max` |
| Samples | 512 | 512 | 2048 | 256 × 1024, **batch 4** |
| Dataset | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v3` | `HuggingFaceH4/ultrachat_200k` |
| Images | `with_images: false` (text-only; vision is ignored) | same | same | same |

`max` is cheaper and is the default for uniform W4A8, uniform W4A4, and the
5090 mixed profile. Local-Hessian is the quality knob NVIDIA used for mixed
NVFP4; it is slower and more memory hungry. You can override without editing
YAML:

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

# PTQ (Hopper or better, multi-GPU / CPU offload for 27B BF16):
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
```

Export is a Hugging Face unified checkpoint
(`modelopt.torch.export.export_hf_checkpoint` on the ModelOpt path, or
`save_pretrained(..., save_compressed=True)` on llm-compressor). Destination
is `export.output_dir`. The pipeline also writes `provenance.json` (base
model, scheme, backend, calib, git sha, timestamp).

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

Compshare / k8s GPU pod (already a container). Scheme argument selects the
5090-packed recipe (`w4a8` default):

```bash
bash scripts/gpu-pod.sh plan
bash scripts/gpu-pod.sh quantize
bash scripts/gpu-pod.sh plan w4a4
bash scripts/gpu-pod.sh quantize w4a4
bash scripts/gpu-pod.sh plan mixed
bash scripts/gpu-pod.sh quantize mixed
```

Usage is `scripts/gpu-pod.sh plan|quantize [w4a8|w4a4|mixed]`. These commands
start a job; they do not imply that W4A4 or mixed PTQ has already finished on
the pod.

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

## Serve (vLLM / SGLang / TensorRT-LLM)

Patterns below follow the NVIDIA mixed model card (tested on Grace Blackwell
GB300 with vLLM nightly). Point `--model` / `--model-path` at your export dir
or at `nvidia/Qwen3.8-27B-NVFP4`.

### vLLM

```sh
vllm serve outputs/Qwen3.8-27B-NVFP4-W4A8 \
    --port 8000 \
    --quantization modelopt \
    --kv-cache-dtype fp8_e4m3 \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --mm-encoder-tp-mode data \
    --seed 0 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill
```

Docker: `vllm/vllm-openai:nightly`. NVIDIA's mixed card omits
`--quantization modelopt` and serves `nvidia/Qwen3.8-27B-NVFP4` directly.

### SGLang

```sh
sglang serve \
  --trust-remote-code \
  --model-path outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --host 0.0.0.0 \
  --port 30000 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

Docker: `lmsysorg/sglang:dev`.

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

## Troubleshooting

- Dry-run prints `family 'qwen3_5' is not registered`: `import megaquant.models`
  (the CLI pipeline loads plugins lazily; adapters register on that import).
- Plan tries to download `config.json`: set `family: qwen3_5` in the YAML
  (already set) so resolve() skips Hub lookup.
- Do not pass vanilla `qwen3` as `family` for this checkpoint — GDN extras
  would not be ignored.
- Block size 16 vs 32: uniform W4A8 is NVFP4 **32** (`W4A8_NVFP4_FP8`).
  Uniform W4A4 is NVFP4 **16** (`NVFP4_DEFAULT_CFG`, weights and activations).
  NVIDIA mixed NVFP4 layers (MLP + `lm_head`) are also **group_size 16**, with
  FP8 on self-attn + linear-attn. Using block 32 / `w4a8_nvfp4_fp8` for mixed
  will not match `nvidia/Qwen3.8-27B-NVFP4`.
