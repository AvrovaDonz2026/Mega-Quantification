# DGX Spark 重复输出诊断：KV 路径对照，2026-10-13

**这是一个文本、MTP 关闭的受控 KV 路径对照。** 两种 KV dtype 在两个固定历史上
各做重复的单步请求，并各做一次 512-token 续写；独立 CPU 审计通过。结果可以说明
本次矩阵中的首步选择和处理后分数如何变化，不能证明原始自然增量状态、停止恢复，
或旧的长输出循环已经被修复。

自然三权重结果见[题目 3 自然对照](dgx-spark-repetition-case3-natural-20261013.md)，
固定前缀探针见[前缀补充](dgx-spark-repetition-prefix-probes-20261013.md)。本文与相邻
JSON 是新的独立快照，不修改正式 GPQA journal、分数或旧日期证据。

## 运行范围

同一检查点、随机种子、容量和请求协议下，仅切换请求的 KV dtype（`fp8_e4m3` 与
`bfloat16`）。每个 KV 路径在两个固定历史上各发送两次 `max_new_tokens=1` 请求，
再对其中一条历史各发送一次 512-token 续写，共 **10 个 generate 请求**（每个 arm
5 个）。每个 arm 按顺序执行 5 次 flush 和 5 次 generate，两臂合计各 10 次；捕获
记录的 cached tokens、retractions 均为 0。

独立复算确认 78 个捕获文件、**1032 个生成位置、5160 个候选位置分数、20640 个
返回 top 项**与原批次报告一致。审计只使用服务返回的处理后 logprobs，没有发布完整
logits、题目、回答或 token 序列。

## 结果

- 在每个 KV 路径内，两个固定历史的重复单步请求均保持相同的 selected token、top-20
  集合和全部返回候选分数；两种路径的续写首步也分别与该路径的单步结果相同。
- 跨 KV 路径的两个历史中，首步 selected token 相同，但分数路径不同：共享 top-20
  集合为 19 个；最大共享项处理后 logprob 差异约为 `2.107895`（历史 1）和
  `0.769880`（历史 2），返回候选中的最大绝对差异约为 `1.107897` 和 `1.073870`。
  这些是本次响应的数值比较，不是质量分数。
- MSE 历史的两条 512-token 续写 native LCP 为 **9 tokens**；两条均以 `length` 结束，
  没有 thinking-end、EOS 或检查阈值内的精确周期。单条续写未复制，因此不能从这项
  结果估计更长续写的稳定性；`budget_exhausted_unknown` 也不等于自然停止恢复。

## 结论边界

本矩阵的重复 gate 支持把跨 KV 的分数差异作为受控观察，但 KV dtype 同时会影响存储
与注意力运行路径，不能进一步拆分为单一数值舍入因素。固定历史是 cold/imposed replay，
没有重建原始自然请求的增量 KV 或 SSM 状态；不同 KV 路径的续写在第 9 个 token 后
已经不是同一历史，不能对齐比较后续位置。结果不证明量化是循环原因、不能证明停止行为
恢复、不能解释旧题目的长周期循环，也不能外推到视觉或 MTP。根因仍未知。

## 证据绑定

下列哈希按源文件原始字节复算；审计 JSON 没有内容级创建时间字段时保持未知。

| 证据 | SHA-256 |
| --- | --- |
| 捕获清单（78 文件） | `e91e0fe219aa166fac301889eca637e4546fc296257a0184ac6404edabe20b3f` |
| recovery3 原始比较 | `d33d24ae2e500e73835865531ca811ff83464fe1cdcd15377b9f488d92240bc7` |
| watcher 完成记录 | `38428516b525f0030b5e82cd382f3be479dc1f1c925e5fd4daefebfbc8dc47e0` |
| 独立 CPU 最终复核 | `e9c5fac5ed92340a0eccda7acf7acb2969f4b82902a020d5152d10d799482e55` |
| 独立 native-score 日志复核 | `11cf1e82b9608c9dd7e77c09ac01db5b82092983acc4ee8d499d980e4ffa79a0` |
| 环境差异独立复核 | `83e20223e97cab9ba5a71e9bfe1d884b72bbf3cda531df8756f29d01443c01fd` |
