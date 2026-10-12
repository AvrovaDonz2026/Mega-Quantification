# 验证结果与证据 / Validation

这里汇总已完成的运行与验证范围。执行当前代码见
[Spark 指南](../DGX_SPARK.md)和[模型手册](../qwen3.8-27b.md)。
日期命名的报告与相邻 JSON 保留实际运行的版本、源码 revision、输入及文件哈希。

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
或停止行为恢复。两种新权重仍有长输出，聚合统计也不能证明发生了重复思考。
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
