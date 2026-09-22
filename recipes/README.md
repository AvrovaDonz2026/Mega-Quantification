# Recipes

YAML stays in this directory so `megaquant -c recipes/<file>.yaml` does not
change when a hardware profile is added. Groups below are the public index.

中文：配方文件都放在这一层，命令行路径保持 `recipes/<file>.yaml`。下表是索引。

## Quantize

| File | Scheme | When to use |
|---|---|---|
| `qwen3.8-27b-nvfp4-w4a8.yaml` | `nvfp4_w4a8` | Default. SGLang mixed map: NVFP4 group 16 on MLP + `lm_head`, FP8 attention, `MIXED_PRECISION` export. |
| `qwen3.8-27b-nvfp4-w4a8.5090.yaml` | `nvfp4_w4a8` | Same map, packed for a 32 GB GPU (ultrachat 256×1024, batch 4). |
| `qwen3.8-27b-nvfp4-w4a8.public-calib.yaml` | `nvfp4_w4a8` | Same map, anonymous ultrachat calibration. |
| `qwen3.8-27b-nvfp4-w4a4.yaml` | `nvfp4_w4a4` | Uniform NVFP4 W4A4, group 16. SGLang `modelopt_fp4`. |
| `qwen3.8-27b-nvfp4-w4a4.5090.yaml` | `nvfp4_w4a4` | Uniform W4A4 packed for 32 GB. |
| `qwen3.8-27b-nvfp4-w4a4.public-calib.yaml` | `nvfp4_w4a4` | Uniform W4A4, anonymous ultrachat. |
| `qwen3.8-27b-nvfp4-mixed.yaml` | `nvfp4_mixed` | Same layer map as `nvfp4_w4a8`. Local-Hessian + Nemotron v3. Needs a larger GPU than 32 GB. |
| `qwen3.8-27b-nvfp4-mixed.5090.yaml` | `nvfp4_mixed` | 32 GB production PTQ: algorithm `max`, ultrachat 256×1024. |
| `qwen3.8-27b-nvfp4-mixed.public-calib.yaml` | `nvfp4_mixed` | Local-Hessian with anonymous ultrachat. |
| `qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml` | `nvfp4_w4a16_mixed` | Optional Marlin export (BF16 MLP activations). Not the DGX Spark fast path. |

## GPQA (SGLang)

| File | Machine |
|---|---|
| `eval-gpqa-diamond.yaml` | CUDA ≥ 12.9, FlashInfer, HiCache 12 GiB, concurrency 1. |
| `eval-gpqa-diamond.5090.yaml` | 32 GB SM120 / CUDA 12.8. Triton + Marlin, HiCache 64 GiB, 24-way, CUDA graph off. |
| `eval-gpqa-diamond.6000d.yaml` | 80 GB SM120 / CUDA 12.8. Triton + Marlin + CUTLASS, KV on GPU, 64-way, CUDA graph on. |

Sampling and the bit layout are in `docs/qwen3.8-27b.md`. A finished score is
`correct/198` after every Diamond row is in the journal.

Generated `recipes/*.pod.yaml` files are local and gitignored.
