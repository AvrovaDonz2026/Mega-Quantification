# DGX Spark 重复输出诊断：匿名题目 5 的 MSE / max 对照，2026-10-12

**截至 `2026-10-12T09:46:31.617054Z` 的冻结快照：匿名题目 5 的 MSE / max 均完成
24576-token 自然生成，终止、native 解码及修正后的启动 gate 审计通过。** 两条均
`finish=length`，没有 EOS、thinking-end 或符合阈值的精确循环，结果均为
`budget_exhausted_unknown`。九组计划共 **5 条 validated、4 条 incomplete、0 条
invalid**；本题 BF16 未完成，快照没有其终止结果。

本文及[相邻 JSON](dgx-spark-repetition-case5-controls-20261012.json)是独立快照。
[case4 对照](dgx-spark-repetition-case4-control-20261012.md)、
[启动补充审计](dgx-spark-repetition-startup-audit-20261012.md)和
[固定前缀报告](dgx-spark-mse-repetition-20261012.md)保留原记录。没有新增 GPQA 分数。

## 两条结果与运行范围

| 权重 | native 输出 tokens | finish | EOS / thinking-end | 支持的精确循环 | 结果 |
| --- | ---: | --- | --- | ---: | --- |
| MSE W4A4 | 24576 | length | 均无 | 0 | budget_exhausted_unknown |
| max W4A4 | 24576 | length | 均无 | 0 | budget_exhausted_unknown |

两条使用同一冻结输入的 2848 个 native IDs，temperature=0、top_p=0.95、top_k=20，
无重复惩罚；单台 Spark / GB10、TP1、FlashInfer、CUDA graphs 关闭、MTP 关闭，
共同使用 FP8 KV 与 FP32 SSM。基础镜像与只读单文件 processor 修复与 case4 相同。
这不是新镜像 clean build 或视觉/MTP 重验。

262144 是服务窗口；24576 是本次输出上限，length 不表示耗尽 256k 窗口。
精确周期检查每 256 tokens 及最终点执行，最大周期 16384，要求至少 4 圈且连续
覆盖 512 tokens；未检出不排除近似、短暂或语义反复，也不证明较长预算会正常停止。
两次实际 native 输出前 7 tokens 相同，第 8 个不同（0-based 偏移 7），解码共同
前缀为 25 字符。这个开头措辞分歧不定位推理失败或量化根因。

客户端 c1、原正式 GPQA c16、两份 PTQ 的目标及 block-scale 配置均存在比较限制。
实际服务种子也不同，deterministic inference 均关闭；尚未证明种子影响温度 0 输出。
本次不是只改变权重精度或 MSE 目标的单因素实验，根因仍未知。

## 分块 prefill 的审计修正

此前 gate 将每个 prefill chunk 的 pending tokens 都要求为 0，不能正确处理本题
2848-token 输入。v3 按冻结调度源码的计数语义核对当前 chunk 后的剩余输入：

| chunk | new tokens | cached tokens | pending tokens | 累计 new tokens |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2048 | 0 | 800 | 2048 |
| 2 | 800 | 0 | 0 | 2848 |

两组均在最终 ready 后成功 idle flush，各只有一次 generate POST 和一次 flush
POST，没有额外健康检查生成；native cached tokens 为 0。两次 chunk 的 new-seq
计数不等于两次 HTTP 请求。输入计数、pending 递减、零缓存及 prefill 先于 decode
均通过核对；各 614 条 decode 日志样本均为 running=1、queue=0。

这项修复验证本次静态分块、单请求、零命中缓存的记录，不证明整个主机/GPU
完全隔离，也不泛化到并发、mixed prefill/decode 或动态分块调度。冻结源码哈希
绑定的是源码审阅，未独立证明运行二进制与其逐字一致。全九组仍不完整，启动
gate 的 exit 2 表示剩余结果未齐；两条已完成运行不是 invalid。

## 后续 CPU 审阅与待运行探针

绑定文件哈希的配对语义审阅创建于 `2026-10-12T09:55:48.810046Z`，是对冻结输出
的后续分析。两组都反复转录、分组、计数和调整对齐，同时仍有局部进展；未观察
到先形成稳定最终答案再机械重复。单次轨迹和材料复述不能认证稳定语义循环，
也不支持“MSE 更早或更强”的重复程度排序。

一项可复核线索是 max 首次完整长材料转录相对最近的输入材料多出 **11 个符号**，
后续再次使用该变体；MSE 对应首次完整转录与输入匹配。这仅支持后续检验复制
误差传播的假设，不能证明它造成迟迟不结束、比较答案质量或认定量化原因。
公开证据只保留匿名计数和哈希，不发布材料、答案、文本或 token ID 序列。

启动 gate v3 的 69 项测试、watcher v2 的 78 项测试及独立审阅通过。common-prefix
与 semantic-entry 队列的 76 项测试已通过；快照中的队列分别等待九组自然审计与
前置探针，尚无 GPU 探针完成结果。诊断工具通过不等于重复输出已修复。

## 文件绑定

| 证据 | SHA-256 |
| --- | --- |
| after05 比较 | `135ca20274d85a451569498971f2527078e8bdf42ccf929e1417521269da0e47` |
| after05 启动 gate | `5a11b44c0e67ece011a216351710529df4e826dce5b3a60e20934e0a4dafe8bd` |
| 配对 CPU 语义审阅 JSON | `eb9088b84fda0961bce913a2bbc783d8bc04e302067bbe0b372aee84ee9e870d` |
| 配对 CPU 语义审阅说明 | `f9b59f01091695f158849c6a73c200c43e7d3e5396bbef9a658f8e222ccb1486` |

正式完整 trace 的精确循环仍已确认。本快照不改变正式分数；本题两条有界结果
仍不能建立循环进入原因、正常停止、质量恢复或全九组结论。
