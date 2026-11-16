# 主线配方矩阵与 DGX Spark PTQ 验证：2026-11-16

**状态：配置矩阵完成，四个 Spark PTQ 配方完成。** 本记录绑定合并后的
`c1b3756a1ce9bf9aa6a5895f7dd13765e48cde14`（`origin/main`）。它把“配方能否被
当前镜像接受”和“在目标 GPU 上完成了完整量化”分开记录，避免用 dry-run 代替
GPU 实测。

## 配方矩阵

在 Spark 的主线镜像中逐项执行了 14 个量化配方的 `megaquant plan`，以及 6 个
评测配方各自的 `megaquant serve --dry-run` 和 `megaquant eval --dry-run`。远端
`summary.json` 已独立读取并校验：`schema=megaquant-recipe-matrix-v2`、
`all_passed=true`、`expected_rows=26`；26 行均为退出码 0、引用日志均存在，且量化与
评测配方清单分别有 14 和 6 项。汇总文件 SHA-256 为
`e5afebac9e9a8959412b9a8c7c9ba45c2de57432f75b0db17b23858a36ab5799`。本次 rerun
目录没有 `status.tsv`；较早目录中的同名文件不属于本次 rerun，不能作为本次逐项
状态证据。文件位置为：

```text
/home/zhiman/megaquant-validation/c1b3756-recipe-matrix-rerun-20261116/summary.json
/home/zhiman/megaquant-validation/c1b3756-recipe-matrix-rerun-20261116/status.tsv  # 未生成
```

量化配方为：

* `qwen3.8-27b-nvfp4-w4a8.yaml`
* `qwen3.8-27b-nvfp4-w4a8.5090.yaml`
* `qwen3.8-27b-nvfp4-w4a8.public-calib.yaml`
* `qwen3.8-27b-nvfp4-w4a8.spark.yaml`
* `qwen3.8-27b-nvfp4-w4a4.yaml`
* `qwen3.8-27b-nvfp4-w4a4.5090.yaml`
* `qwen3.8-27b-nvfp4-w4a4.public-calib.yaml`
* `qwen3.8-27b-nvfp4-w4a4.spark.yaml`
* `qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml`
* `qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml`
* `qwen3.8-27b-nvfp4-mixed.yaml`
* `qwen3.8-27b-nvfp4-mixed.5090.yaml`
* `qwen3.8-27b-nvfp4-mixed.public-calib.yaml`
* `qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml`

评测配方为：

* `eval-gpqa-diamond.yaml`
* `eval-gpqa-diamond.5090.yaml`
* `eval-gpqa-diamond.6000d.yaml`
* `eval-gpqa-diamond.spark.yaml`
* `eval-gpqa-diamond.spark-gpqa.yaml`
* `eval-gpqa-diamond.spark-mtp.yaml`

这 26 项只证明 YAML、CLI 参数、镜像内路径和计划生成在该环境中可接受；不证明
5090、6000D 或通用配方已经在其目标硬件上完成量化或评测，也不下载 27B 权重。
本机另外完成了 20 个 YAML 的语法解析、Python 编译和 shell 语法检查；本机缺少
完整 GPU/CLI 依赖，因此完整模型测试以 Spark 记录为准。

## Spark 上的完整 27B PTQ

以下四个配方在 `aitopatom-aacd` 的 GB10 上按主线镜像实际运行，校准集为 256
条图文样本（其中 128 张图片）。每个作业均完成校准、导出和独立导出审计，退出
码为 0；远端日志中的最终标记为 `ALL_PASS`。证据根目录为：

```text
/home/zhiman/megaquant-validation-20261116-c1b3756/
```

| 配方 | 目标 | 结果 |
|---|---|---|
| `qwen3.8-27b-nvfp4-w4a8.spark.yaml` | 混合 W4A8，视觉与 MTP 量化 | PTQ、导出审计、校准检查通过 |
| `qwen3.8-27b-nvfp4-w4a4.spark.yaml` | 均匀 W4A4，max 校准 | PTQ、导出审计、校准检查通过 |
| `qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml` | 均匀 W4A4，MSE 搜索 | PTQ、导出审计、校准检查通过 |
| `qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml` | 均匀 W4A4，Local-Hessian 搜索 | PTQ、导出审计、校准检查通过 |

主线镜像身份已复核为：

```text
sha256:16657528650800627d436e4244a1a4ac52fd3244cb2fb0657c7bd7ff3fe4c467
revision=c1b3756
```

本次实测建立了四个 Spark 配方的 GPU 量化和导出证据，但没有把它外推到
5090、6000D、Marlin 或需要更大显存的 `mixed.yaml`。这些配方仍需在对应硬件
上进行真实量化；本报告中的 26 项计划检查不能替代该验证。

## 视觉、MTP 与长生成边界

本轮额外重复 static MAX/MSE 的视觉/MTP serving 检查没有形成正式通过结果：
SGLang JIT 共享对象在容器的 `/tmp/home/.cache/sglang/jit` 报
`failed to map segment from shared object`，容器随后退出。该故障发生在 serving
启动阶段，不是量化或导出审计失败。后续应
使用全新的 evidence 目录和唯一容器名，并把 `HOME`/`TMPDIR` 指向 `/cache` 下的
主机可写目录，再独立执行 baseline、embedded MTP 和长生成 gate。本次四个 c1b3756
PTQ 输出尚未各自完成视觉/MTP serving 或长生成验证。

仓库中的 [DGX Spark static MAX/MSE TP1 实测](dgx-spark-static-tp1-20261116.md)
记录的是另一组已完成的 static 导出 serving 证据，不能冒充本报告四个 c1b3756
PTQ 导出的视觉/MTP 或长生成结果。它所记录的固定五题 MTP gate 与长生成周期
分析仍可作为协议参考，但两组证据的导出身份必须分别保留。

因此，本报告的结论是：**主线上的 14 个量化配方和 6 个评测配方已由独立校验的
`summary.json` 报告配置级检查通过（本次 rerun 未生成 `status.tsv`）；四个 Spark 配方完成了真实 27B PTQ 与导出审计；c1b3756 新导出
的视觉/MTP serving 和长生成仍待在修复 JIT 可写缓存后的独立运行中确认。**
