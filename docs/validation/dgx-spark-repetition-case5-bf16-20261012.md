# DGX Spark 重复输出诊断：匿名题目 5 的源 BF16 精确循环，2026-10-12

**截至 `2026-10-12T11:59:00.794562Z` 的冻结快照：匿名题目 5 的源 BF16 权重运行
确认持续的原生 token 精确循环，终止、native 解码与启动 gate 审计通过。** 本条输出
24576 tokens，`finish=length`，没有 EOS 或 thinking-end，结果为
`budget_exhausted_with_exact_cycle`。九组计划共 **6 条 validated、3 条 incomplete、
0 条 invalid**，尚未完成全九组。

本文及[相邻 JSON](dgx-spark-repetition-case5-bf16-20261012.json)是新增的独立证据。
[本题 MSE / max 快照](dgx-spark-repetition-case5-controls-20261012.md)、
[case4 对照](dgx-spark-repetition-case4-control-20261012.md)和
[固定前缀报告](dgx-spark-mse-repetition-20261012.md)保留原记录；没有新增 GPQA 分数。

## 原生循环与对照范围

| 权重 | 输出 tokens | finish | EOS / thinking-end | 支持的精确循环 | 结果 |
| --- | ---: | --- | --- | ---: | --- |
| MSE W4A4 | 24576 | length | 均无 | 0 | budget_exhausted_unknown |
| max W4A4 | 24576 | length | 均无 | 0 | budget_exhausted_unknown |
| 源 BF16 | 24576 | length | 均无 | 1 | budget_exhausted_with_exact_cycle |

BF16 的最终稳定后缀为原生输出偏移 **[13135, 24576)**，最短周期 **4 tokens**，
连续覆盖 **11441 tokens**（约 46.55% 的输出），包含 **2860 圈及 1 个余 token**。
每 256 tokens 检查一次，首次满足至少 4 圈、连续 512 tokens 的确认点为 **13824**。
原分析与独立 CPU 检查一致，核对了最短周期、前一位置破坏周期、原生 IDs 哈希
和终止事件绑定；已记录的原生输出解码检查通过，后续审阅未独立重跑 tokenizer。
13135 是最终稳定后缀能向前延伸的最早位置，
不是已证明的因果决策点；13824 是检查点，不是循环开始位置。

三组使用同一冻结输入的 2848 个 native IDs，temperature=0、top_p=0.95、top_k=20，
无重复惩罚；单台 Spark / GB10、TP1、FlashInfer、CUDA graphs 与 MTP 关闭。
262144 是服务上下文窗口，24576 是本题输出上限；其他匿名题目的预算分别为
16384 与 49152，不能将 length 解释为耗尽 256k 窗口。客户端只发一路请求，服务
`max_running_requests=16`。MSE / max 未检出支持阈值内精确循环，仍可能存在语义
反复，且正常停止结果未知。

**源 BF16 权重也出现这类精确循环，因此 MSE 权重量化不是其出现的必要条件。**
三组共同使用 **FP8 KV (`fp8_e4m3`) 与 FP32 SSM**；BF16 是权重对照，并非全部
计算与缓存都用 BF16。不同启动种子、KV 容量和 BF16 / W4A4 计算路径限制了
单因素比较，deterministic inference 均关闭；尚未证明种子或资源导致温度 0 的
差异。本结果不证明 KV 是根因、全部 BF16 运行都会失败、量化不影响发生概率，
也不构成三种权重的质量排序。

## 启动 gate 与后续语义审阅

本条在最终 ready 后成功 idle flush，只记录一次 generate POST、一次 flush POST、
零额外健康检查生成；native cached tokens 与 retractions 均为 0。2848-token 输入
分为 2048 / 800 两次 prefill，pending 相应为 800 / 0，cached 均为 0；614 条 decode
日志样本均为 running=1、queue=0。分块计数及顺序通过 v3 gate，覆盖本次静态分块
单请求记录，不证明整个主机/GPU 完全隔离或任意调度下成立。全九组 gate 的
exit 2 表示结果尚未齐，不表示本条 validated 运行无效。

后续 CPU 语义审阅创建于 `2026-10-12T12:12:53.889513Z`，不倒填为上述快照时间；
它结合片段审阅与有限操作核对，不是穷尽的回答质量检查。
循环出现在对有限符号序列进行逆变换的展开过程中：生成的输入有 **468 个符号**，
展开输出达 **17181 个符号**，仅前 **9 个符号**与正确有限变换吻合。因此这条轨迹
还包含有限操作结果本身的错误，问题范围超过最终停止标记是否出现。

首次完整材料转录增加 **11 个符号**，与 max 的首次转录变体相同；max 本次却未
出现支持阈值内精确循环，这不能建立复制误差充分导致循环。旧正式 MSE 轨迹也在
同类有限操作中出现字符周期 8 的循环，但具体周期哈希不同；它不是本条循环的
精确重放，也不能把历史文本重新分词当作历史原生 IDs。根因仍待 KV 与入口探针
对照。MTP 关闭的文本诊断不验证视觉/MTP，诊断工具通过也不等于故障已修复。

## 文件绑定

| 证据 | SHA-256 |
| --- | --- |
| after06 比较 | `d606621e4d70b9ad2924094ee196fc99bd06e1d8ea84860ec556ff20ef6fd636` |
| after06 启动 gate | `49b637c970504ade9ade0dc996be0effc33531d7c1077d1c394d32eb8396ab16` |
| after06 捕获清单 | `40c356033666ab28f20b23d498d85122beec71ee3f69da8f1abef90fb7c17704` |
| BF16 自然输出证据 | `3b2a69dd777f703ec85d5ca6d2906ff4601dc62cf38b2d6794a6407c21da6cf0` |
| BF16 原生循环分析 | `752729b8e3c8fa3c048f0e5fbf54bd3e8a60e9a4254e210dbf3a18f6f218b392` |
| 第二份独立周期检查 | `1e97a8ba47e16f95d6ebaf1a5aaf91f207ab7e8692d7698469f14d4464a6aa61` |
| 后续 CPU 语义审阅 JSON | `cadb4088de6f131a6f37854e61852cb350d214e1829141973df2f834f57fc9fa` |
| 后续 CPU 语义审阅说明 | `79a1b6f6f3fbe61c2eef2e439acd49bd8e8c497e0e6bf5e487c280a038f5b35c` |

公开证据只包含匿名统计、位置与哈希，不发布题目、材料、回答文本或 token ID 序列。
