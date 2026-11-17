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

后续 `main` 提交 `eaae979` 的 [GitHub Actions CI](https://github.com/AvrovaDonz2026/Mega-Quantification/actions/runs/43896392287)
通过：Ruff 检查通过，pytest 为 823 passed、8 skipped（29.93 秒）。这是后续源码的 CI
证据，不改变本报告绑定的 `c1b3756` GPU 运行身份，也不替代其远端 PTQ/serving 证据。

本次实测建立了四个 Spark 配方的 GPU 量化和导出证据，但没有把它外推到
5090、6000D、Marlin 或需要更大显存的 `mixed.yaml`。这些配方仍需在对应硬件
上进行真实量化；本报告中的 26 项计划检查不能替代该验证。

## 视觉与 embedded MTP

早期一次 MAX/W4A8 运行因 plan/result 引用旧导出身份而不被接受。随后首次身份正确的
运行中，Python 固定用例在第 16 个 token 出现 MTP 与 baseline mismatch。之后
MAX/W4A8 repeat2 按当前四棵导出树身份完成独立多模态复验：容器退出码为 0；baseline 与 embedded MTP 的五个固定用例
均通过，逐题 native output IDs parity 通过，draft 接受数和 verify 次数均为正数。
该轮结果 SHA-256 为
`8b4bd5a68f63f89f7b6e5cbd7f0f314a50fbf926ef3564ee0e915aee8fdd79d7`。
其独立只读 `validate_plan` 与 `validate_multimodal_result` 校验均通过，`validated=true`，
pair ID 为 `c4aa749615092a23af51e97f`。旧身份运行、身份正确首轮的 Python token mismatch
及早期失败均保留为历史；repeat2 不证明重复运行确定性或普遍 parity。

续跑队列的 MSE/Local-Hessian 多模态结果状态为 `exit_code=0`、`validated=true`，
结果 SHA-256 为 `73ed94c70806f035dbca578c483c969692833a1bbfe400e2b9c4d42ec1a5bd97`。
该 MSE/Hessian 续跑 pair ID 为 `1ddc44624608209ff48568f1`；MSE 为 135 accepted / 144
proposed drafts、72 verify，Local-Hessian 为 134 / 148、74。两臂分别完成 baseline 与
MTP 五题（baseline 14/14 checks、MTP 19/19 checks），独立结果校验及功能验证通过；
每臂内部 MTP↔baseline 均为 5/5 exact output IDs、无首个 mismatch 且 greedy outputs
identical。这是臂内 parity，不表示 MAX/W4A8 与 MSE/Hessian 跨臂输出相同。该队列结果与
下方先前安全运行记录分别保留；哈希用于绑定结果，不包含原始视觉请求或模型回答。

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

MAX/W4A8 首次身份正确的尝试中，Python 固定用例在第 16 个 token 出现 MTP 与 baseline
mismatch。后续 repeat2 通过的是该固定用例及当前结果校验；它不抹去此前 mismatch，
也不构成重复运行确定性或普遍 token parity 证明。

另一次安全重跑虽然退出码为 0，且容器实际挂载本次新导出，但其计划和结果仍引用旧
static MAX/MSE 的 source job 和树哈希。它的功能输出不能作为身份绑定正确的本次正式
通过证据；该身份不匹配记录仍作为历史保留。

此外，之前导出清单的树哈希与本次显式算法的结果不同，尚未证明两者采用相同算法，
不能据此断言文件发生变化。为避免混用，后续队列在运行前重新核验四棵完整导出树，
并将计划、source job 和结果重新绑定到以下算法与结果，用于四个导出的视觉/MTP 复验：

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

队列独立校验多模态结果并核验上述四棵导出树；W4A8 与 W4A4 MAX 多模态复验，
以及 MSE/Hessian 续跑多模态阶段均已完成。随后按计划执行两组 TP1 长生成：262,144 context、固定
288-token 输入、temperature 0、计划记录 seed `213087260`、MTP 关闭、FP8 KV、FP32 SSM、
冷启动并 flush cache，单题最多生成
49,152 tokens。每组长生成的内部总时限为 7,200 秒，外层另留 180 秒及停止宽限。
持久结果只保留计数、哈希、周期检测与停止元数据；视觉/MTP 原始临时报告结束后清理。
长生成各组只有一轮、两臂各一次，没有重跑一致性要求。`completion_tokens` 是当前单臂
单次请求的累计值；`elapsed_seconds` 是整组启动以来的经过时间，包含加载阶段，不能据此
计算 decode 速度。seed 仅存在于计划元数据；当前 runtime 未将其传入 HTTP `sampling_params`。

续跑队列位于 `c1b3756-queued-validation-cont-20261117-041456`。W4A8/W4A4 MAX 长生成
配对已完成：任务状态为 `validated=true`、容器退出码 0，pair ID 为
`640a797b25687df4bd0a5638`，结果 SHA-256 为
`cb4b7d76bb3545ec1fa3722a14d6a215c3373f0cfe4e4821416ccd51aa5679fd`。14:25:34 CST 又按
当前 bundle 中的 plan/result 独立执行 `validate_result` 并通过。

| 请求/量化臂 | completion tokens | native finish | 额外记录 |
|---|---:|---|---|
| `static-max`（W4A4 MAX） | 46,784 | `stop` | `matched_kind=integer`；周期搜索 966 candidates / 85,339 comparisons |
| `static-MSE`（W4A8） | 33,399 | `stop` | matched hash `54363ddee68f4a5db81c9d37e5fb738d28f5b67dc7f725ad7333172b1ea157da`；周期搜索 183 candidates / 13,130 comparisons |

两臂 `period.found=false`，但 `complete=false`、`exhaustive=false`；各自覆盖数与 copies
均为 0、period 为 null。这个有限检测结果不表示没有重复。实际 EOS stop token 未核查，
`effective_stop=false`；native `stop` 不能据此解释为 EOS 或自然停止恢复。该实验每臂仅运行
一次，且不是质量 gate，也不证明 MSE 根因或重复问题已修复。

队列于 2026-11-17 06:23:43 UTC 自动进入 MSE/Hessian 长生成阶段。06:28:43 UTC 的
[历史运行快照](recipe-matrix-c1b3756-20261117-long-status.json)记录 MSE 第一臂已开始解码，
Hessian 尚未开始；该快照不含 MSE 第一臂的最终 finish、长度或周期摘要。

2026-11-17 14:05:39.992445 UTC 的终态回读显示队列以 `error_code=validation_container_nonzero`
失败。MSE/Hessian 容器于 06:23:43.891602041 UTC 启动、08:23:51.130518426 UTC 退出，
exit code 1、`OOMKilled=false`；runtime 报 `ServingRuntimeError: incomplete streaming response`。
失败发生在 7,200 秒共享时限附近，但现有证据不能确定具体是哪一个调用因超时抛错。
`pair-serving-result.json` 不存在，runner 已退出。失败前的后续回读确认 MSE 第一臂已结束、
Hessian 第二臂已开始生成；但运行时只将臂级 summary 保存在内存，直到两臂均结束才落盘，
所以两臂最终 summary 均未保存。终态没有 MSE 最终 finish、输出长度或周期统计，因此不推测
其 token 总数或 `length`/`stop` 结果；Hessian 也无最终结果。该组长生成验证未完成。
终态主机观察显示 Aitop 与 Robo 的 GPU 利用率均为 0%、compute 进程数均为 0，且没有新
验证任务运行。这是该时点的观察，不表示 GPU 资源预留。

终态证据见[脱敏终态快照](recipe-matrix-c1b3756-20261117-terminal-status.json)。06:28 的
运行快照仍保留为历史记录；更早的[启动快照](recipe-matrix-c1b3756-20261117-status.json)
保留 05:10 状态。本地[旧队列身份清单](recipe-matrix-c1b3756-20261116-queue.json)只对应旧包，
不是续跑队列的完整 manifest；续跑 queue 脚本哈希不同于旧包，但 runtime、`run_tp1` 和
四份 plan 哈希与仓库 manifest 匹配。

仓库中的 [DGX Spark static MAX/MSE TP1 实测](dgx-spark-static-tp1-20261116.md)
属于另一组 static 导出，不替代本报告四个配方的长生成结果。`length` 只表示达到
49,152 输出预算，不等于耗尽 262K 上下文；`stop` 不能单独证明 EOS。
MSE/Hessian 长生成尚未完成，不能据此给出该组配对结论、质量排序或重复修复结论。
