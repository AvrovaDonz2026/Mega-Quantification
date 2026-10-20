# DGX Spark 重复输出诊断：匿名题目 3 的 MSE 长周期循环，2026-10-12

**截至 `2026-10-12T12:29:22.878411Z` 的冻结快照：匿名题目 3 的 MSE W4A4 自然
生成确认持续的原生 token 精确循环，终止、native 解码与启动 gate 审计通过。**
本条输出 49152 tokens，`finish=length`，没有 EOS 或 thinking-end，结果为
`budget_exhausted_with_exact_cycle`。九组计划共 **7 条 validated、2 条 incomplete、
0 条 invalid**；本题 max / 源 BF16 未完成，尚无配对终止结果。

本文及[相邻 JSON](dgx-spark-repetition-case3-mse-20261012.json)保留独立快照。
[case5 源 BF16 循环](dgx-spark-repetition-case5-bf16-20261012.md)和
[固定前缀诊断](dgx-spark-mse-repetition-20261012.md)保留原记录；没有新增 GPQA 分数。

## 原生循环与运行范围

| 原生输出统计 | 结果 |
| --- | ---: |
| 最短精确周期 | 2032 tokens |
| 最终稳定后缀，0-based | [33131, 49152) |
| 连续覆盖 | 16021 tokens，约 32.59% 输出 |
| 完整圈数 / 余 token | 7 / 1797 |
| 首次阈值确认检查点 | 41472 tokens |

周期检查要求至少 4 圈且连续覆盖 512 tokens，每 256 tokens 及最终点执行。
33131 是最终稳定后缀能向前延伸的最早位置，41472 是首次达到检查阈值的采样点；
两者都不表示已定位循环的因果决策点。原分析与独立 reverse-Z 检查一致，31 个
达到阈值的检查点保持同一起点与周期。原生 IDs、终止事件和已记录的解码证据
相互绑定；后续 CPU 审阅未独立重跑 tokenizer，公开只保留匿名计数、位置与哈希。

单台 Spark / GB10、TP1、FlashInfer、temperature=0、top_p=0.95、top_k=20，
无重复惩罚；CUDA graphs 与 MTP 关闭，共同协议使用 FP8 KV (`fp8_e4m3`) 与
FP32 SSM。262144 是服务上下文窗口，49152 是本题输出上限；其他匿名题目的
输出预算为 16384 / 24576。length 不表示用尽 256k 窗口，MTP 关闭的文本诊断
也不验证视觉/MTP。

## 启动 gate 的有界结论

本条在最终 ready 后成功 idle flush，只记录一次 generate POST、一次 flush POST，
没有额外健康检查生成。288-token 输入一次 prefill 完成，cached=0、pending=0；
native cached tokens 与 retractions 均为 0。1228 条 decode 日志样本均为
running=1、queue=0，启动 gate 通过。

因此本条循环在经过最终就绪、idle flush、观测零缓存命中的条件下仍能出现。
这些记录不能证明整台主机/GPU 完全隔离或所有执行都无缓存影响，也不定位
量化、KV、调度或停止实现的根因。九组 gate 的 exit 2 表示结果尚未齐，
不表示本条 validated 运行无效。

客户端 c1，服务 `max_running_requests=16`，正式 GPQA 为 c16；这不是同一调度
条件下的正式重跑。服务种子为 213087260，token 池容量为 1738591，deterministic
inference 关闭。未固定种子、资源和计算路径的单次轨迹不建立算法质量排序；
本题 max / BF16 尚未完成，更不能由当前结果比较重复发生概率或认定 MSE 原因。

实际关键配置保留在响应的 `actual_server_settings` 中，种子另外在 server 与
readiness 日志核对。捕获的完整 saved ServerArgs 文件与所记录的完整 actual
ServerArgs 哈希不同；后者原文件不在此捕获中，未独立重建该完整配置哈希，
不能将 saved 文件当作整个 actual 配置的身份验证。

## 后续 CPU 审阅

有界语义审阅创建于 `2026-10-12T12:40:36.710325Z`，是对冻结证据的后续 CPU
分析。它抽样审阅新轨迹 37439 字符，其中循环前 27669 字符，占循环前部分约
25.28%；另审阅历史正式轨迹的 6898 字符，不是穷尽的回答质量检查。

本条循环内容为长段选项复核，反复重新打开同一未决知识点；所阅片段没有正式
收尾。它与 case5 的短符号有限逆变换循环具有不同内容。新轨迹的最短字符周期
为 6770，历史同题正式 MSE 为 4306，具体周期哈希也不同；有界语义主题相似
不表示同一精确轨迹复现。旧正式输出缺少实际生成 IDs，重新分词不能替代历史
native 证据。这些观察不定位第一次错误或循环根因，也不建立质量恢复结论。

## 文件绑定

| 证据 | SHA-256 |
| --- | --- |
| after07 捕获清单 | `6d2bc2e1e7f1bb63c29f3e8e796ddbfa4bbab836988518d28de2ae9da83fc752` |
| after07 比较 | `8a42b10752ac19d8119ada44b0be74eef253d1cd5f847047231bd5a76b6cbc0d` |
| after07 启动 gate | `a007451992c61b98009525d0df10e59565459b844f9b110ebd5ad53edd7b65c5` |
| MSE 自然输出证据 | `8e151d76fdd25603216aeb79c59a39d637ad25c47efd87a3581f6b65b3dc7218` |
| MSE 原生循环分析 | `f41537fd234ca17f79234e0cca8ab7866aeca37043602892418b8485e195f044` |
| 独立原生周期检查 | `cfe408b58a0c5c4329d991a3599fba8fc2b472c2851405826c3ea2ddb646be2a` |
| 后续 CPU 有界语义审阅 JSON | `d8d3a18d2d51491d82a47df4a444f38d3a5dfd57e2d6724e2fbcb21edbe78131` |
| 后续 CPU 有界语义审阅说明 | `33ec4ec587a86d24a923f99720b5ab5f0416f624635babd4a72bef939d871cd3` |

本快照不修改正式分数，也不证明重复输出已修复。题目、回答文本、原生 token
序列与私有部署信息均不公开。
