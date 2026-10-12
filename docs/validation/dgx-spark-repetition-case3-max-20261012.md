# DGX Spark 重复输出诊断：匿名题目 3 的 max 原生停止，2026-10-12

**截至 `2026-10-12T13:03:12.992792Z` 的冻结快照：匿名题目 3 的 max W4A4 输出
22569 tokens 后以 EOS 停止，native 解码、终止与启动 gate 审计通过。** 原生输出
thinking-end 位于 0-based 偏移 **22129**，EOS 位于 **22568**，各出现一次；本条
没有达到阈值的精确循环。九组计划共 **8 条 validated、1 条 incomplete、0 条
invalid**，同题源 BF16 尚未完成。

本文及[相邻 JSON](dgx-spark-repetition-case3-max-20261012.json)是独立快照。
[同题 MSE 长周期循环](dgx-spark-repetition-case3-mse-20261012.md)与其他日期报告
保留原记录；不改正式 GPQA journal 或分数。

## 本次配对行为

| 权重 | 输出 tokens | finish | thinking-end / EOS 次数 | 支持的精确循环 | 终止结果 |
| --- | ---: | --- | --- | ---: | --- |
| MSE W4A4 | 49152 | length | 0 / 0 | 1，周期 2032 | budget_exhausted_with_exact_cycle |
| max W4A4 | 22569 | stop | 1 / 1 | 0 | audited_eos_stop_after_thinking_end |
| 源 BF16 | 未完成 | 未知 | 未知 | 未知 | incomplete |

两条使用同一冻结输入的 288 个 native IDs，同为 temperature=0、top_p=0.95、
top_k=20，无重复惩罚；单台 Spark / GB10、TP1、FlashInfer，CUDA graphs 与 MTP
关闭，共同使用 FP8 KV (`fp8_e4m3`) 与 FP32 SSM。262144 是服务上下文窗口，
49152 是本题输出预算；max 在该上限前停止，不能将 MSE length 解释为耗尽 256k。

直接比较本次 MSE 与 max 的原生输出，前 7 tokens 相同，第 8 个不同（0-based
偏移 7），解码共同前缀为 25 字符；这里没有使用历史正式输出的 retokenized IDs。
这个开头分歧不定位第一次错误或循环原因。精确循环检查每 256
tokens 及最终点执行，最大周期 16384，要求至少 4 圈且连续覆盖 512 tokens；
max 未检出不排除近似或语义反复。

## 启动与配置限制

max 在最终 ready 后成功 idle flush，只记录一次 generate POST、一次 flush POST，
没有额外健康检查生成；288-token 输入一次 prefill 完成，cached=0、pending=0，
native cached tokens 与 retractions 均为 0。564 条 decode 日志样本均为 running=1、
queue=0，单项 gate 通过。全九组 gate 的 exit 2 表示剩余 BF16 结果未齐，
不表示两条 validated 运行无效；这些记录也不证明全主机/GPU 完全隔离。

客户端 c1，服务 `max_running_requests=16`，正式 GPQA 为 c16。MSE / max 的
服务启动种子分别为 **213087260 / 781300540**，token 池容量分别为
**1738591 / 1737732**，deterministic inference 均关闭。PTQ 的目标与 block-scale
配置、计算路径也存在差异，未建立严格单因素比较。此次 MSE 循环而 max 停止
只是一对轨迹的行为证据，不证明算法质量排序、普遍停止恢复或量化根因。

实际关键配置记录在响应内；captured saved ServerArgs 的完整哈希与 recorded
actual 完整配置哈希不同，后者原文件不在此捕获中，不能将 saved 文件视为完整
actual 配置身份验证。基础镜像加单文件只读 processor 修复仍是先前冻结运行时，
本条没有新增镜像 clean build 或视觉/MTP 检查。

## 后续 CPU 解析与语义审阅

冻结的原分析将 `final_answer_parseable` 与 `final_answer_correct` 保留为 `null`，
表示当时未评估解析及正确性。后续 CPU 审计创建于
`2026-10-12T13:14:21.662944Z`：仓库正式解析器可以提取最终答案，**但该答案
与绑定的参考答案不符，按正式判分规则为错**。当前与冻结的正式 evaluator 源码
字节相同；这是一条诊断轨迹的后续判定，没有改写正式 journal 或分数。

原始 EOS 字面文本紧接答案行，使解析器使用 thinking-end 后的最后独立选项
fallback；仅在内存中移除已确认的终端 EOS 文本后，严格答案行匹配仍得到同一
预测，排除了此次错误只是 fallback 误取尾字母的解释。参考绑定来自历史正式
MSE 记录，不是历史 max；本审计没有重新哈希完整历史 journal。

独立 CPU 周期核验覆盖 86 个推理检查点，未发现符合上述阈值的精确循环。
有界语义审阅覆盖推理的 18500 / 72138 字符（25.6453%），以及 thinking-end
至 EOS 的完整 1442 字符区间。两条都反复检查同一未决冲突；max 后段接受另一
种解释、继续复核并最终选定答案。该答案仍错，因此本条停止及精确循环为零
不能证明普遍质量改善、没有语义反复或重复故障已修复。

新两条原生轨迹的直接前缀审计创建于 `2026-10-12T13:17:02.852648Z`；上述
后续审计时间均不倒填为生成快照时间。独立核验检查既有解码绑定，没有重新运行
tokenizer。

计数勘误创建于 `2026-10-12T13:23:02.461820Z`：两个终端标记严格之间有
438 tokens；原私有摘要的 439 包含 EOS。旧审计保留原字节，此项算术校正不改变
停止、解析、答错或周期结论。

## 文件绑定

| 证据 | SHA-256 |
| --- | --- |
| after08 捕获清单 | `fdaf1457699a339f271398613236f893bfa1302bc0f77eefe3bc3481117e9ffb` |
| after08 比较 | `7d65d62c42c3ab22ed8e985625f4fc202fd60110c04270da7813261d11ee390a` |
| after08 启动 gate | `316f17d00135c38b2a425c7e13bc34810a4a42a11f5a0e5efa14edfe8aa80b19` |
| max 自然输出证据 | `07c5c51a77b3333051da79948cceb9694fc9664f9ba69c8bd7fe05672c338ae9` |
| max 原生分析 | `3bed1e9e5f56e14d037a5ff1da63220a833ffa3bd9b29c7ed9b48e843f24e584` |
| max 终止审计 | `57cb9af6873fed9ccf22933b9ee4553c8d6e83195bce1370c7dd5a61ab3baf1d` |
| 后续 CPU 解析与语义审计 JSON | `7d48885b9783cf84b8ea0ad16b1a512978e7dca1154cad2df08bda8e76bb4d3e` |
| 后续 CPU 审计 MD | `fa40fb198d09529ca769fa83aabef5ba2a6d549b1bcdd0f15a5b1fd73eddd2e8` |
| 独立原生停止核验 | `15b571620a313eeb9a3b06e4a43ed78cc1003f7d43884647e5b332db9449c303` |
| 本次 MSE / max 直接前缀审计 | `d4abb565952b02e8c5650eb06c43a694ae51df15db262dca53a6aec8aeeca4bc` |
| 后续停止区间计数勘误 | `fd5061c60226432a0e053fcf1b1242cc7d414483d4a1c687f7bc22e0a6db9328` |

公开证据只包含匿名统计、位置与哈希，不发布题目、回答文本、原生 ID 序列或
私有部署信息。根因与重复输出修复仍待验证。
