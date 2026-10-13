# 验证结果与证据 / Validation

这里汇总已完成的运行与验证范围。执行当前代码见
[Spark 指南](../DGX_SPARK.md)和[模型手册](../qwen3.8-27b.md)。
日期命名的报告与相邻 JSON 保留实际运行的版本、源码 revision、输入及文件哈希。

[配置哈希口径勘误](dgx-spark-server-info-hash-erratum-20261013.md)确认：题目 3 的
MSE / max 与题目 5 BF16 三组完整 server-info 对象均匹配生成前记录的 canonical
摘要。saved 原始字节哈希与 actual 对象哈希口径不同；此前因此限制完整对象绑定
的表述应与本勘误合读。它不证明生成全过程配置不变，旧日期报告保留原字节。

最新[自然三权重对照](dgx-spark-repetition-case3-natural-20261013.md)确认：九组诊断
全部完成并通过启动 gate，9 条 validated、0 条 incomplete、0 条 invalid。匿名题目 3
的 MSE W4A4 在零缓存启动后出现周期 2032 的循环；max 在 22569 tokens 停止，
后续正式解析器判定可解析但答错；源 BF16 权重在共同 **FP8 KV / FP32 SSM** 下
于 16685 tokens 停止，后续判定可解析且正确。此为三条诊断轨迹，未修改正式分数，
不同种子、容量与数值路径仍限制量化因果和算法排序。旧 dated 报告保留各自时点。

此前匿名题目 5 的源 BF16 权重在 **FP8 KV / FP32 SSM** 下也出现
持续的原生 token 精确循环，覆盖输出末尾 11441 tokens，周期为 4。MSE 权重量化
不是这类循环出现的必要条件；KV 根因及量化对发生概率的影响仍未确定。九组计划
截至该快照有 6 条 validated、3 条 incomplete，见[BF16 独立证据](dgx-spark-repetition-case5-bf16-20261012.md)。

## 完整 Spark GPQA Diamond

统一协议：单台 ARM64 DGX Spark / GB10、TP1、temperature 0、262144 上下文、
16 并发上限、seed 0、固定种子打乱选项、保留 `xhigh` thinking、MTP 关闭。
每轮有 198 个与官方 CSV 对应的唯一题目，summary 与 journal 通过正式结果检查器。
截断和未解析回答均计错。

| 权重 | 正确/198 | 准确率 | 长度截断 | 未解析 | 证据 |
|---|---:|---:|---:|---:|---|
| 混合 W4A8，max | 177/198 | 89.39% | 0 | 4 | [报告](dgx-spark-gpqa-256k-20261002.md) · [JSON](dgx-spark-gpqa-256k-20261002.json) |
| 均匀 W4A4，max | 174/198 | 87.88% | 0 | 4 | [报告](dgx-spark-gpqa-256k-20261002.md) · [JSON](dgx-spark-gpqa-256k-20261002.json) |
| 均匀 W4A4，MSE | 178/198 | 89.90% | 3 | 4 | [报告](dgx-spark-w4a4-quality-20261008.md) · [JSON](dgx-spark-w4a4-quality-20261008.json) |
| 均匀 W4A4，Local-Hessian | 170/198 | 85.86% | 3 | 5 | [报告](dgx-spark-w4a4-quality-20261008.md) · [JSON](dgx-spark-w4a4-quality-20261008.json) |

MSE 和 Local-Hessian 的 3 条长度截断各自包含在未解析数中，两列不能相加。
MSE 本轮较 max 多对 4 题，Local-Hessian 少对 4 题；单轮结果尚不能证明普遍质量
或停止行为恢复。后续完整 trace 检查已确认 6 条截断均有精确循环；固定前缀与精度
对照见[重复输出诊断](dgx-spark-mse-repetition-20261012.md)。六组 72 次诊断全部完成，
该次执行中恢复 BF16 输出头、注意力投影或两者均未打断选定的 3 条循环。
后续[启动补充审计](dgx-spark-repetition-startup-audit-20261012.md)确认：客户端 c1
未建立全部请求的预热隔离 gate，18 条续写全部复用缓存；它不是排除启动与缓存
影响的 c1 对照，也没有建立循环的量化根因。新的原题起点实验保留独立快照。
默认方案继续为混合 `nvfp4_w4a8`，两种 W4A4 实验配方作为可选项。

每次请求使用剩余上下文预算，续写不会扩大 262144 窗口。并发上限 16 不代表
16 路完整上下文能够同时驻留；请求共享有限 KV 池。客户端耗时保留作诊断，
没有据此建立吞吐比较。MSE 的恢复计时只覆盖重跑的 12 条请求，整轮耗时未知。

## 功能与容器验证

| 验证 | 已完成范围 | 报告与记录 |
|---|---|---|
| MSE / Local-Hessian 完整 PTQ 与服务 | 各完成全 27B 语言/视觉/MTP 量化、519 项权重搜索和严格导出审计；五个固定文本/图像/OCR 用例通过，各自 baseline/MTP 的 205 个输出 token ID 一致 | [2026-10-08 报告](dgx-spark-w4a4-quality-20261008.md) · [JSON](dgx-spark-w4a4-quality-20261008.json) |
| v0.1.2 完整容器重验 | 两种镜像各两次离线 clean build、全新 W4A8/W4A4 PTQ、严格审计、固定视觉/MTP 检查与独立空 Docker store 回装 | [2026-10-01 报告](dgx-spark-release-20261001.md) · [JSON](dgx-spark-release-20261001.json) |
| 较早的 clean-container 排查 | 保留最初构建、量化、服务故障及修复过程 | [2026-09-30 报告](dgx-spark-clean-container-20260930.md) · [JSON](dgx-spark-clean-container-20260930.json) |
| 首轮完整 Spark PTQ | W4A8/W4A4 的 256 样本校准、视觉/MTP 覆盖与导出检查 | [2026-09-29 报告](dgx-spark-pr46-20260929.md) · [JSON](dgx-spark-pr46-20260929.json) |
| 首轮视觉与 MTP 服务 | 固定文本/图像/OCR、实际接受 draft 与 FP32 SSM 修复 | [2026-09-30 报告](dgx-spark-serving-20260930.md) · [JSON](dgx-spark-serving-20260930.json) |

视觉/MTP 功能检查使用 32768 上下文，GPQA 关闭 MTP。固定用例通过不代表广泛视觉
质量、256k MTP 质量、任意提示的逐 token 确定性或速度提升。上述 Spark GPU 运行
均为单机 TP1；跨机器 TP2 尚无对应验证。

## 版本与冻结运行

| 内容 | 实际来源 |
|---|---|
| v0.1.3 发布源码与安装包 | `a68e542341e0ee69fe30ee0357554b2ea99c62f3`；主线 CI 为 456 通过、5 跳过；安装包 12 项检查通过，源码包重建 wheel 字节一致 |
| 新 W4A4 候选 PTQ | `14d5ad6f9d4ac30fc91a55b2ff4697f2ab90a76d` |
| 候选服务、GPQA 配方与客户端 | `79b5dfef54618d0e385c5974d0ea508cd2fbdd40` |
| v0.1.2 容器完整重验 | `ac53833dc5af3f03a10087449ca80e9484deba48` |

候选 GPU 镜像报告 MegaQuant 0.1.2。它们验证了 v0.1.3 所含的运行时修复；
最终标注 0.1.3 的镜像没有另外完成一次 GPU 重验或双次 clean build。
发布包验证记录及 SHA256SUMS 位于
[v0.1.3 Release](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.3)，
版本变更与附件说明见[发布说明](../releases/v0.1.3.md)。

## 历史试跑与诊断

- **固定前缀探针补充**：共同前缀 9 请求与语义历史 6 请求均完成，独立 CPU 逐位置核验通过。零缓存 cold prefill 和跨预算首步一致性不恢复自然轨迹的 incremental KV / SSM 状态，也不能定位自然首次分叉、循环根因或证明停止恢复。[2026-10-13 报告](dgx-spark-repetition-prefix-probes-20261013.md) · [JSON](dgx-spark-repetition-prefix-probes-20261013.json)。
- **KV 路径对照补充**：同一检查点、种子、容量和请求协议下，`fp8_e4m3` 与 `bfloat16` KV 在两个固定历史上完成重复单步与 512-token 续写；独立复核确认重复 gate 和跨路径分数差异。续写均为长度截断，不能据此证明停止恢复或循环因果。[2026-10-13 报告](dgx-spark-repetition-kv-recovery3-20261013.md) · [JSON](dgx-spark-repetition-kv-recovery3-20261013.json)。
- **匿名题目 3 的完成自然三权重对照**：截至 `2026-10-13T02:08:02.781567Z`，九组自然诊断全部完成，启动 gate 有 9 条 validated、0 条 incomplete、0 条 invalid。同题 MSE 精确循环并耗尽 49152 输出预算；max 在 22569 tokens 停止、后续判定答错；源 BF16 权重在共同 FP8 KV / FP32 SSM 下于 16685 tokens 停止、后续判定答对。不是新 GPQA 分数，也不建立量化根因或普遍重复恢复。[报告](dgx-spark-repetition-case3-natural-20261013.md) · [JSON](dgx-spark-repetition-case3-natural-20261013.json)。
- **配置哈希口径勘误**：三组完整 server-info 对象、六份 natural / decoded 记录的字节与 canonical 摘要复算通过；纠正因两种编码哈希不同而限制完整对象身份绑定的表述，不建立全过程运行认证，不改变停止、循环或判分结论。[2026-10-13 报告](dgx-spark-server-info-hash-erratum-20261013.md) · [JSON](dgx-spark-server-info-hash-erratum-20261013.json)。
- **匿名题目 3 的 max 原生停止**：截至 `2026-10-12T13:03:12.992792Z`，max 输出 22569 tokens，以 EOS 停止，thinking-end 与 EOS 各一次，未检出支持阈值内精确循环，零缓存启动与终止审计通过。后续 CPU 判定最终答案可解析但答错，正式分数未改。同题 MSE 已循环，BF16 未完成；九组计划有 8 条 validated、1 条 incomplete，配对行为不建立算法排序或因果。[报告](dgx-spark-repetition-case3-max-20261012.md) · [JSON](dgx-spark-repetition-case3-max-20261012.json)。
- **匿名题目 3 的 MSE 长周期循环**：截至 `2026-10-12T12:29:22.878411Z`，MSE 输出 49152 tokens，原生周期 2032、稳定后缀起点 33131、连续覆盖 16021 tokens，零缓存启动 gate 与终止审计通过。本题 max / BF16 未完成，九组计划有 7 条 validated、2 条 incomplete，尚不能比较算法或归因。[报告](dgx-spark-repetition-case3-mse-20261012.md) · [JSON](dgx-spark-repetition-case3-mse-20261012.json)。
- **匿名题目 5 的源 BF16 精确循环**：截至 `2026-10-12T11:59:00.794562Z`，本条输出 24576 tokens，原生周期 4、稳定后缀起点 13135、覆盖 11441 tokens，启动与终止审计通过。共同 FP8 KV / FP32 SSM 条件下，MSE 不是此类循环的必要条件，根因仍未知；九组计划有 6 条 validated、3 条 incomplete。[报告](dgx-spark-repetition-case5-bf16-20261012.md) · [JSON](dgx-spark-repetition-case5-bf16-20261012.json)。
- **匿名题目 5 的 MSE / max 对照**：两条各用尽 24576 输出预算，停止结果仍未知，未检出支持阈值内精确循环。修正分块 prefill 审计后，九组计划有 5 条 validated、4 条 incomplete；BF16 尚无完成结果。配对语义审阅确认两条都有转录、分组、计数与对齐返工，量化根因仍未知。[报告](dgx-spark-repetition-case5-controls-20261012.md) · [JSON](dgx-spark-repetition-case5-controls-20261012.json)。
- **匿名题目 4 的 BF16 权重对照**：MSE / max / 源 BF16 三条自然生成与严格启动 gate 均通过；三条都用尽 16384 输出预算，没有 EOS、thinking-end 或符合阈值的精确循环，结果均为 `budget_exhausted_unknown`。九组计划完成 3/9；这不是新分数，也不建立量化因果。[报告](dgx-spark-repetition-case4-control-20261012.md) · [JSON](dgx-spark-repetition-case4-control-20261012.json)。
- **重复诊断启动补充审计**：旧 54 条 next-token 请求中，13 条 prefill 全在最终 ready 前、1 条跨标记、40 条全在标记后；18 条续写均在标记后但全部复用缓存。新的原题起点对照截至 `2026-10-12T07:48:47Z` 完成严格审计 2/9：匿名题目 4 的 MSE / max 都耗尽 16384 输出预算，停止结果未知；其余未完成，不构成新分数。[报告](dgx-spark-repetition-startup-audit-20261012.md) · [JSON](dgx-spark-repetition-startup-audit-20261012.json)。
- **W4A4 重复输出诊断**：MSE / Local-Hessian 的 6 条正式截断覆盖 5 道匿名题目，均确认精确循环。六组 72 次诊断完成，原执行中恢复 BF16 输出头、注意力投影或两者均未打断选定的 3 条循环；启动与缓存条件以上述补充审计限定。实际 logprob 服务异常已修复，不修改正式分数。[2026-10-12 报告](dgx-spark-mse-repetition-20261012.md) · [JSON](dgx-spark-mse-repetition-20261012.json)。
- **32k W4A8 试跑：157/198**，30 条长度截断、32 条未解析（包含关系）。这是单独保留的长度受限运行；完整 256k 成绩见上表。[JSON](dgx-spark-gpqa-w4a8-20261002.json)。32k W4A4 未完成，没有正式成绩。
- **三题 W4A8 诊断重试**：并发改为 3，三条都提取到最终答案，其中两条正确；没有修改正式 16 并发 journal 或分数，也不证明逐 token 重复输出一致。[报告](dgx-spark-gpqa-unparsed-20261003.md) · [JSON](dgx-spark-gpqa-unparsed-20261003.json)。
- **MSE 186 题 / Local-Hessian 197 题快照**：为中断与慢请求排查保留，最终两轮均已完成 198 题。快照不是正式成绩，见[候选验证报告](dgx-spark-w4a4-quality-20261008.md)。
- **5090 / 80 GB SM120 运行**：硬件、temperature、后端和截断情况不同，见[模型手册](../qwen3.8-27b.md#sglang-inference-and-gpqa)，不与上表合并为同一协议。

## English

The first table contains complete, formally verified 198-item GPQA results on
one GB10 Spark at temperature 0 and 256k context, with MTP disabled. Length
finishes are included in unparsed counts and count as wrong. The MSE improvement
is a single-run result; general quality and stopping recovery remain unestablished.

PTQ coverage, fixed vision/MTP functionality, text benchmark accuracy, container
build/transfer and package checks have distinct scopes. Dated reports and their
JSON preserve the actual tested revisions. GPU evidence uses the frozen 0.1.2
runtimes above; it does not claim a new GPU rerun of final 0.1.3 images, TP2,
broad vision quality or throughput gains.
