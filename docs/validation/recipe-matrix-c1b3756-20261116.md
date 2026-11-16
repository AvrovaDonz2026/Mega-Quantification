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

## 视觉与 embedded MTP

把 `HOME`、`TMPDIR` 及 SGLang、TileLang、Triton、TorchInductor 缓存全部放入
挂载的可写 `/cache` 后，W4A4 MSE 与 Local-Hessian 在新 serving 镜像上完成了
固定五题的 baseline 与 embedded EAGLE MTP 对照。五题包括算术、Python、红色块、
蓝色块和 OCR `3729`：baseline 的功能和有限 logprob 检查全部通过，MTP 每题的
native output IDs 均与相应 baseline 完全一致。

| 实际配方 | runner 槽位 | 接受 draft 数 | verify 次数 | 五题结果 |
|---|---|---:|---:|---|
| W4A4 MSE | `static-max` | 135 | 72 | baseline/MTP 检查及 output IDs parity 通过 |
| W4A4 Local-Hessian | `static-MSE` | 134 | 74 | baseline/MTP 检查及 output IDs parity 通过 |

该次容器 `megaquant-c1b3756-mm-mse-hessian-safe-20261116` 退出码为 0；结果通过
独立 `validate_multimodal_result()`，pair 为 `358425443e056dba2bd11444`，结果 SHA-256 为
`0c781ee118939b933e6bd242a622429ad7b5ceea4b78444ae6723dcc12d3062c`。证据文件为：

```text
/home/zhiman/megaquant-validation/c1b3756-mm-mse-hessian-safe-evidence-20261116/multimodal/pair-serving-result.json
```

结束后检查到 `pair-multimodal-*` 原始临时报告无残留，30000/30001/30080 端口无监听。
这些结果仅覆盖所测导出与五个固定用例，不代表泛化视觉质量、TP2、GPQA 或长上下文
MTP 质量已经通过。

### 权重身份与失败尝试

此前 JIT 共享对象曾在 `/tmp/home/.cache/sglang/jit` 报
`failed to map segment from shared object`；显式将上述缓存移到 `/cache` 后，后续
serving 成功运行。旧的一次 checker 非零退出未保留失败用例，因此不能确定原失败原因。

MAX/W4A8 的一次安全重跑虽然退出码为 0，且容器实际挂载本次新导出，但其计划和
结果仍引用旧 static MAX/MSE 的 source job 和树哈希。它的功能输出不能作为身份绑定
正确的本次正式通过证据；已安排重新运行。

此外，之前导出清单的树哈希与本次显式算法的结果不同，尚未证明两者采用相同算法，
不能据此断言文件发生变化。为避免混用，后续队列在运行前重新核验四棵完整导出树，
并将计划、source job 和结果重新绑定到以下算法与结果，再重复四个导出的视觉/MTP：

```text
sha256(concat(sorted(relative_path + NUL + binary_sha256(file))))
```

其中按相对路径排序，每个文件使用其原始内容 SHA-256 的 32 字节摘要。
2026-11-16 23:02 CST 回读的完整树哈希如下：

| 导出目录后缀 | 文件数 | SHA-256 |
|---|---:|---|
| `W4A4-spark` | 16 | `d2854d6f43b20a33de664bbf6d1ca597ddcc0be6436fca3af00dd9eb9b6e74ba` |
| `W4A8-spark` | 17 | `be7da417c6cad88492ceea79f45fdb07fcf804b007eb097409b37eab15add115` |
| `W4A4-spark-mse` | 16 | `93971d74ac78b659175bd672ed9a1663f878894227b0bdedd5d4a80124fe0ce3` |
| `W4A4-spark-hessian` | 16 | `74d45b686c55411c0943ed96789915f9f653665eb5f88f3c267e8945c66fb7a1` |

## Spark 后台复验队列

队列使用 digest 固定的 serving 镜像：

```text
megaquant@sha256:7c591ade0bdaf84d6abb342419f08ba0f8ddab92987f05a3af063b3237379b42
```

队列先独立校验已完成的 MSE/Hessian 结果，再核验上述四棵导出树；按
MAX/W4A8、MSE/Hessian 两组串行重复视觉/MTP gate。两组均通过后，各执行一次
TP1 长生成配对：262,144 context、固定 288-token 输入、temperature 0、seed
213087260、MTP 关闭、FP8 KV、FP32 SSM、冷启动并 flush cache，单题最多生成
49,152 tokens。每组长生成的内部总时限为 7,200 秒，外层另留 180 秒及停止宽限。
持久结果只保留计数、哈希、周期检测与停止元数据；视觉/MTP 原始临时报告结束后清理。

后台进程已由 SSH 返回 `STARTED 829659`。随后的状态读取因 Teleport 证书于
2026-11-16 23:04:47 CST 到期而未能完成；**尚未回读该队列的状态文件，也未确认新
容器是否进入 GPU 推理**。不能把成功启动后台进程写成整条验证队列通过。
队列按脱离 SSH 会话的方式启动，当前是否存活仍待续证后确认。续证后应读取以下产物：

```text
/home/zhiman/megaquant-validation/c1b3756-queued-validation-20261116-2302/status.json
/home/zhiman/megaquant-validation/c1b3756-queue-2302.log
/home/zhiman/megaquant-validation/c1b3756-queue-2302.pid
/home/zhiman/megaquant-validation/megaquant-c1b3756-spark-queue-20261116-2302/
```

本地归档的[队列身份清单](recipe-matrix-c1b3756-20261116-queue.json)保存镜像、
四棵树、验证包及其文件的 SHA-256，不包含 prompt 或模型响应。

容器名前缀为 `megaquant-c1b3756-q2255-`。每阶段均需核对退出码与计划绑定，并独立
执行对应结果校验；结束后检查 GPU 进程、服务端口与原始临时报告清理情况。

仓库中的 [DGX Spark static MAX/MSE TP1 实测](dgx-spark-static-tp1-20261116.md)
属于另一组 static 导出，不替代本报告四个配方的长生成结果。`length` 只表示达到
49,152 输出预算，不等于耗尽 262K 上下文；`stop` 不能单独证明 EOS。
目前没有新的长生成通过结果，也不能宣称重复问题已修复。
