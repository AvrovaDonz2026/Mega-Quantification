# Mega-Quantification

把 Hugging Face BF16 模型做训练后量化（PTQ），导出 SGLang 可加载的 checkpoint。
当前完整验证的模型是 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)：
32 GB RTX 5090 上完成量化，80 GB SM120 和 DGX Spark 上完成 GPQA Diamond。
Spark 配方还包含视觉塔和 MTP 量化、图像/OCR 与内嵌 MTP 功能验证。

**当前版本：[v0.1.3](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.3)**
· [文档导航](docs/README.md) · [配方索引](recipes/README.md) · [验证结果](docs/validation/README.md)
· [English](#english)

## 快速开始

使用 Python 3.10 或更新版本，从仓库根目录执行。查看计划只需要基础依赖，
不需要 GPU，也不会下载 27B 权重。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

megaquant plan -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant serve -c recipes/eval-gpqa-diamond.spark-gpqa.yaml --dry-run
megaquant eval -c recipes/eval-gpqa-diamond.spark-gpqa.yaml --dry-run
```

实际量化需要 GPU、完整 BF16 模型与校准数据。先按机器选择指南：

| 环境 | 使用入口 |
|---|---|
| RTX 5090，32 GB | [单卡量化与内存配置](docs/qwen3.8-27b.md#single-5090--64-gb-ram) |
| DGX Spark，ARM64 / GB10 | [构建、量化、视觉/MTP 和完整 GPQA](docs/DGX_SPARK.md) |
| x86 CUDA / Docker / GPU Pod | [镜像和宿主机配置](docker/README.md) |
| 添加模型或量化后端 | [架构与扩展接口](docs/ARCHITECTURE.md)、[开发检查](CONTRIBUTING.md) |

## 选择量化方案

| 方案 / Scheme | 权重与激活布局 | 导出 / SGLang |
|---|---|---|
| `nvfp4_w4a8`（默认）、`nvfp4_mixed` | MLP 和 `lm_head` 的权重、激活均为 NVFP4 group 16；注意力投影为 FP8 | `MIXED_PRECISION` / `modelopt_mixed` |
| `nvfp4_w4a4` | 所有目标线性层的权重、激活均为 NVFP4 group 16，包括注意力投影 | `NVFP4` / `modelopt_fp4` |
| `nvfp4_w4a16_mixed`（可选） | MLP 权重为 NVFP4，激活保留 BF16；注意力为 FP8 | Marlin 导出；详见[配方索引](recipes/README.md) |

仓库的 W4A8 名称指上述混合布局；其中 MLP 的激活仍是 NVFP4。
默认配方保留 BF16 视觉塔与 MTP，Spark 配方显式量化这两部分。
MSE 和 Local-Hessian W4A4 校准实验是可选配方，详见
[Spark 实验流程](docs/DGX_SPARK.md#controlled-w4a4-calibration-experiments)。

导出 Spark 的均匀 W4A4 时，`hf_quant_config.json` 和 `config.json` 必须包含
真实 safetensors 中视觉与 MTP NVFP4 线性层的 `quantized_layers`。导出器会从
权重 header 自动补齐这份 metadata，并排除恢复为 BF16 的层；缺少它会让 SGLang
按 BF16 构造视觉层，随后在加载 packed NVFP4 权重时产生 shape mismatch。

32 GB 5090 上使用过的量化命令：

```bash
python -m pip install -e '.[hf,modelopt]'
MEGAQUANT_LOW_MEMORY=1 megaquant quantize \
  -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
```

这份配方使用 ModelOpt 0.46.1、`max` 校准、ultrachat 256×1024、batch 1。
Spark 使用独立的 ARM64 / CUDA 13 镜像和 ModelOpt 0.47.0，按 Spark 指南构建。
推理时传入量化后的本地目录；[模型手册](docs/qwen3.8-27b.md)包含导出恢复、服务与 OSS 命令。

## 已验证结果

以下为单台 DGX Spark / GB10、TP1、temperature 0、262144 上下文、16 并发、
seed 0、打乱选项、保留 `xhigh` thinking、关闭 MTP 的完整 GPQA Diamond 结果。
每轮 198 个唯一题目均通过独立校验，长度截断和未解析回答计错。

| 权重 / Export | 正确 / Correct | 准确率 / Accuracy | 长度截断 / Length | 未解析 / Unparsed |
|---|---:|---:|---:|---:|
| 混合 W4A8，max | 177/198 | 89.39% | 0 | 4 |
| 均匀 W4A4，max | 174/198 | 87.88% | 0 | 4 |
| 均匀 W4A4，MSE | 178/198 | 89.90% | 3 | 4 |
| 均匀 W4A4，Local-Hessian | 170/198 | 85.86% | 3 | 5 |

长度截断包含在未解析数中，两列不能相加。MSE 本轮较 max 多对 4 题，仍有长输出与
长度截断，普遍质量和停止行为恢复尚未证实。视觉/OCR 与 MTP 另有固定功能检查；
这些检查不代表广泛视觉质量、256k MTP 质量或 TP2 已验证。
[验证索引](docs/validation/README.md)提供完整报告、机器可读证据与历史运行的区别。

## 复现与发布

仓库和 [GitHub Release](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.3)
提供代码、配方、安装包与验证记录。模型权重、校准数据和原始评测轨迹单独保存。
重新量化会重新估计 scale；需要相同权重字节时，应获取相同的 content-hash 导出。

v0.1.3 安装包通过离线安装、源码包重建一致性及 CLI 计划检查。
GPU 和容器验证记录保留实际运行版本与冻结 revision，范围见
[发布说明](docs/releases/v0.1.3.md)。密钥和机器配置放在被 git 忽略的 `.env`、`.oss.env`
与 `docker-compose.override.yml`。

## English

Mega-Quantification performs post-training quantization of BF16 Hugging Face
models and exports checkpoints for SGLang. Qwen/Qwen3.8-27B has completed PTQ
on a 32 GB RTX 5090 and GPQA Diamond on an 80 GB SM120 and DGX Spark.
Spark recipes also quantize vision and MTP; fixed-case image/OCR and embedded
MTP functional checks are recorded separately from the text benchmark above.

Run the quick-start commands from the repository root to install the base
package and inspect plans without a GPU or model download. For execution, use
the [Spark runbook](docs/DGX_SPARK.md), [Docker guide](docker/README.md),
[recipe index](recipes/README.md) or [model guide](docs/qwen3.8-27b.md).
The [architecture guide](docs/ARCHITECTURE.md) covers extension points.

The default `nvfp4_w4a8` uses NVFP4 group 16 for MLP/LM-head weights and
activations, and FP8 for attention. `nvfp4_mixed` uses the same encoding.
Uniform `nvfp4_w4a4` applies NVFP4 weights and activations to all targeted
linears. Generic recipes retain BF16 vision/MTP; Spark opts into their
quantization. MSE and Local-Hessian W4A4 recipes remain optional. For Spark
uniform W4A4 exports, both `hf_quant_config.json` and `config.json` must name
the serialized visual and MTP NVFP4 linears in `quantized_layers`; the exporter
derives this map from safetensors headers and excludes BF16 restores.

The table records complete, independently verified 256k GPQA runs with MTP
disabled. Length finishes are included in unparsed counts. MSE's four extra
correct answers in this run do not establish general quality or stopping
recovery. See the [validation index](docs/validation/README.md) and
[v0.1.3 release notes](docs/releases/v0.1.3.md) for frozen revisions and scope.
Recalibration does not reproduce an older export byte for byte.

License: [GNU AGPL version 3 or later](LICENSE) (`AGPL-3.0-or-later`). Copyright 2026 Donz.
Agent instructions: [SKILL.md](SKILL.md).
