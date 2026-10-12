# Recipes

YAML stays in this directory so command paths remain `recipes/<file>.yaml`.
Choose a recipe below, then follow the hardware guide for installation,
inputs, quantization and validation.

中文：配方文件统一放在这一层，命令行路径保持 `recipes/<file>.yaml`。
按硬件选择配方，再进入 [5090 / 模型手册](../docs/qwen3.8-27b.md)、
[Spark 流程](../docs/DGX_SPARK.md) 或 [Docker 指南](../docker/README.md)。
完整成绩和历史试跑见 [验证索引](../docs/validation/README.md)，
其他任务见 [文档导航](../docs/README.md)。

## Quantize

| File | Scheme | When to use |
|---|---|---|
| [qwen3.8-27b-nvfp4-w4a8.yaml](qwen3.8-27b-nvfp4-w4a8.yaml) | `nvfp4_w4a8` | Default. NVFP4 group 16 on MLP + `lm_head`, FP8 attention, `MIXED_PRECISION` export. |
| [qwen3.8-27b-nvfp4-w4a8.5090.yaml](qwen3.8-27b-nvfp4-w4a8.5090.yaml) | `nvfp4_w4a8` | Same map, packed for 32 GB (ultrachat 256×1024, batch 1). |
| [qwen3.8-27b-nvfp4-w4a8.public-calib.yaml](qwen3.8-27b-nvfp4-w4a8.public-calib.yaml) | `nvfp4_w4a8` | Same map, anonymous ultrachat calibration. |
| [qwen3.8-27b-nvfp4-w4a8.spark.yaml](qwen3.8-27b-nvfp4-w4a8.spark.yaml) | `nvfp4_w4a8` | GB10 mixed map, quantized vision and MTP, ModelOpt 0.47.0. |
| [qwen3.8-27b-nvfp4-w4a4.yaml](qwen3.8-27b-nvfp4-w4a4.yaml) | `nvfp4_w4a4` | Uniform NVFP4 W4A4, group 16; SGLang `modelopt_fp4`. |
| [qwen3.8-27b-nvfp4-w4a4.5090.yaml](qwen3.8-27b-nvfp4-w4a4.5090.yaml) | `nvfp4_w4a4` | Uniform W4A4 packed for 32 GB. |
| [qwen3.8-27b-nvfp4-w4a4.spark.yaml](qwen3.8-27b-nvfp4-w4a4.spark.yaml) | `nvfp4_w4a4` | GB10 language/vision/MTP, ModelOpt 0.47.0, local `/models` and `/data` mounts. |
| [qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml](qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml) | `nvfp4_w4a4` | Optional controlled weight-MSE search; separate export. See experiments below. |
| [qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml](qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml) | `nvfp4_w4a4` | Optional controlled Local-Hessian search; separate export. See experiments below. |
| [qwen3.8-27b-nvfp4-w4a4.public-calib.yaml](qwen3.8-27b-nvfp4-w4a4.public-calib.yaml) | `nvfp4_w4a4` | Uniform W4A4, anonymous ultrachat. |
| [qwen3.8-27b-nvfp4-mixed.yaml](qwen3.8-27b-nvfp4-mixed.yaml) | `nvfp4_mixed` | W4A8 layer map, Local-Hessian + Nemotron v3; needs more than 32 GB. |
| [qwen3.8-27b-nvfp4-mixed.5090.yaml](qwen3.8-27b-nvfp4-mixed.5090.yaml) | `nvfp4_mixed` | 5090 production profile: max, ultrachat 256×1024, batch 1; BF16 vision/MTP. Writes `outputs/Qwen3.8-27B-NVFP4-W4A8`. |
| [qwen3.8-27b-nvfp4-mixed.public-calib.yaml](qwen3.8-27b-nvfp4-mixed.public-calib.yaml) | `nvfp4_mixed` | Same map, max, anonymous ultrachat 512. |
| [qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml](qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml) | `nvfp4_w4a16_mixed` | Optional Marlin export with BF16 MLP activations. |

The W4A8 name refers to the mixed map above: MLP activations are NVFP4.
Generic recipes retain BF16 vision/MTP; Spark recipes explicitly quantize both.
See the [format table](../docs/qwen3.8-27b.md#quantization-format) for details.

### Optional W4A4 calibration experiments

MSE and Local-Hessian retain quantized vision/MTP, activation max calibration,
256×1024 samples, batch 1 and seed 42. Static NVFP4 weight block scales with
FP8 scale search make the searches execute; input block scales stay dynamic
and the packed group-16 format is retained. Export KV quantizers are unset;
serving uses FP8 KV and FP32 SSM states.

Both completed full PTQ, strict export audits, fixed vision/MTP checks and
256k GPQA. They remain optional: a single run does not establish general
quality or stopping recovery. Use the [experiment workflow](../docs/DGX_SPARK.md#controlled-w4a4-calibration-experiments)
and [verified results](../docs/validation/README.md#完整-spark-gpqa-diamond).

## GPQA (SGLang)

| File | Context / requests | Hardware and use |
|---|---|---|
| [eval-gpqa-diamond.yaml](eval-gpqa-diamond.yaml) | 262144 / 1 | CUDA ≥ 12.9, FlashInfer, HiCache 12 GiB. |
| [eval-gpqa-diamond.5090.yaml](eval-gpqa-diamond.5090.yaml) | 262144 / 24 | 32 GB SM120 / CUDA 12.8, Triton + Marlin, HiCache 64 GiB, CUDA graph off. |
| [eval-gpqa-diamond.6000d.yaml](eval-gpqa-diamond.6000d.yaml) | 262144 / 64 | 80 GB SM120 / CUDA 12.8, Triton + Marlin + CUTLASS, GPU KV, CUDA graph on. |
| [eval-gpqa-diamond.spark.yaml](eval-gpqa-diamond.spark.yaml) | 32768 / 4 | GB10 ARM64 / CUDA 13 / SM121, fixed vision checks, FP32 SSM, CUDA graphs/HiCache off. |
| [eval-gpqa-diamond.spark-gpqa.yaml](eval-gpqa-diamond.spark-gpqa.yaml) | 262144 / 16 | Full Spark GPQA, 64 FP32 SSM slots, temperature 0, preserved `xhigh` thinking, MTP disabled. |
| [eval-gpqa-diamond.spark-mtp.yaml](eval-gpqa-diamond.spark-mtp.yaml) | 32768 / 4 | Fixed Spark checks with quantized embedded EAGLE MTP; no external draft checkpoint. |

Check available GPU/host memory before loading. Spark CPU and GPU share RAM;
the request limit does not reserve a full context window for every request.
Use separate result directories when weights or protocol change.

完整 Spark GPQA 使用 256k 配方；32k 配方用于独立功能检查。正式分数要求
198 个唯一题目、summary 和结果检查器全部通过，截断及未解析回答计错。
启动与验分命令见 [Spark GPQA 流程](../docs/DGX_SPARK.md#full-gpqa-diamond-at-temperature-0)。
四组完整成绩、32k 历史试跑和三题诊断的范围见 [验证索引](../docs/validation/README.md)。
代理的硬性约定见 [SKILL.md](../SKILL.md)。

Generated `recipes/*.pod.yaml` files are local and gitignored.
