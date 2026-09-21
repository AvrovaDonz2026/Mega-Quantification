# Qwen3.8-27B NVFP4 runbook

End-to-end notes for quantizing **Qwen/Qwen3.8-27B** (BF16) with Mega-Quantification.

中文要点：这是 27B 稠密 VLM（`Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`）。
默认量化语言模型线性层；视觉、MTP、embedding、GDN 的 `conv1d` / `in_proj_a` /
`in_proj_b` 留 BF16。均匀 W4A8 用 `max` + 512 条校准；要对齐 NVIDIA 公开
`nvidia/Qwen3.8-27B-NVFP4` 则用混合配方 `local_hessian` + 2048 条。NVFP4 推理
需要 Blackwell；校准可以在 Hopper 上用多卡 / offload 做。

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

| File | Scheme | Algorithm | Samples | Output |
|---|---|---|---|---|
| `recipes/qwen3.8-27b-nvfp4-w4a8.yaml` | `nvfp4_w4a8` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A8` |
| `recipes/qwen3.8-27b-nvfp4-mixed.yaml` | `nvfp4_mixed` | `local_hessian` | 2048 | `outputs/Qwen3.8-27B-NVFP4-mixed` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.yaml` | `nvfp4_w4a4` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A4` |

All three set `backend: modelopt`, `kv_cache: fp8`, `family: qwen3_5`,
`model.quantize_vision: false`, `model.quantize_mtp: false`, and
`calibration.dataset: nvidia/Nemotron-Post-Training-Dataset-v2`.

NVIDIA's model card calibrated on **Nemotron-Post-Training-Dataset-v3**. This
repo follows the architecture default (**v2**). Swap `calibration.dataset` if
you are reproducing the public mixed checkpoint bit-for-bit.

## Ignore list (why these stay BF16)

`Qwen35Family.default_ignore` (unless the recipe flips the flags):

| Pattern | Why |
|---|---|
| `*visual*`, `*vision*` | Vision encoder is not the W4A8 language-model target. NVIDIA mixed PTQ also left it in BF16. Set `model.quantize_vision: true` to include it. |
| `*embed_tokens*`, `*embed_positions*` | Embedding tables are poor NVFP4 candidates; keep BF16. |
| `*linear_attn.conv1d*` | Gated DeltaNet depthwise conv — not a standard Linear GEMM; ModelOpt / compressed-tensors NVFP4 paths do not treat it as `q/k/v/o` or `gate/up/down`. |
| `*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*` | GDN extras (not `in_proj_qkv` / `in_proj_z` / `out_proj`). Mixed FP8 attention still quantizes the real GDN projections; these two stay BF16. |
| `*mtp*` | Multi-Token Prediction heads. Off unless `model.quantize_mtp: true`. |
| **not** `*mlp*` | MLP `gate/up/down_proj` are the main NVFP4 targets. |
| **not** `*lm_head*` | NVIDIA mixed NVFP4 quantizes `lm_head`. Uniform W4A8 does too. Add it in `extra_ignore` only if you want BF16 logits. |

Norms are not `Linear` and stay unquantized automatically.

## Mixed vs uniform W4A8

**Uniform `nvfp4_w4a8`** (this repo's default request): every targeted language
linear gets NVFP4 weights (block size **32**) and FP8 E4M3 activations.
ModelOpt object: `mtq.W4A8_NVFP4_FP8_CFG` / qformat `w4a8_nvfp4_fp8`.

**Mixed `nvfp4_mixed`**: NVFP4 on `*mlp*` + `*lm_head*`, FP8 on `*self_attn*` +
the GDN GEMMs (`in_proj_qkv`, `in_proj_z`, `out_proj`). This matches
`nvidia/Qwen3.8-27B-NVFP4`. Accuracy on NVIDIA's card (vLLM, 262k context):

| Benchmark | BF16 | NVFP4 mixed |
|---|---|---|
| GPQA Diamond | 88.92 | 88.01 |
| Terminal-Bench | 75.56 | 74.02 |
| AA-LCR | 72.63 | 73.38 |
| MMMU-Pro | 75.14 | 74.86 |
| SciCode | 47.93 | 48.41 |
| IFBench | 80.07 | 78.93 |

**W4A4** (`nvfp4_w4a4`) uses NVFP4 activations as well, with block size **16**.
It is only a comparison recipe.

## Calibration: 512 vs 2048

| | Uniform W4A8 | Mixed (NVIDIA intent) |
|---|---|---|
| Algorithm | `max` (minmax / absmax scales) | `local_hessian` (`fp8_scale_sweep: true` in ModelOpt) |
| Samples | 512 | 2048 |
| Seq length | 2048 | 2048 |
| Images | `with_images: false` (text-only; vision is ignored) | same |

`max` is cheaper and is the W4A8 default. Local-Hessian is the quality knob
NVIDIA used for mixed NVFP4; it is slower and more memory hungry. You can
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

# PTQ (Hopper or better, multi-GPU / CPU offload for 27B BF16):
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
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

Docker: `vllm/vllm-openai:nightly`.

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
- Block size 16 vs 32: W4A8 is 32; W4A4 is 16. Wrong size will not match
  ModelOpt `w4a8_nvfp4_fp8`.
