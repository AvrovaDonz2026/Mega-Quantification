# DGX Spark 重复输出诊断：匿名题目 4 的 BF16 权重对照，2026-10-12

**匿名题目 4 的 MSE W4A4、max W4A4 与源 BF16 三条自然生成已完成，严格启动
gate 均通过。三条都用尽 16384-token 输出预算，没有 EOS、thinking-end 或符合
检查阈值的精确循环；停止结果均为 `budget_exhausted_unknown`。** 这说明当前有界
运行中的长度结束并非 MSE 独有，不能据此把截断判为循环、量化根因或停止恢复。

本文是截至 **`2026-10-12T08:30:30.618183Z`** 的独立 after03 文件冻结快照；此前的
[2/9 启动补充审计](dgx-spark-repetition-startup-audit-20261012.md)与
[固定前缀报告](dgx-spark-mse-repetition-20261012.md)保留原记录。对应的
[机器可读证据](dgx-spark-repetition-case4-control-20261012.json)仅包含匿名计数、
位置与哈希。九组计划中 **3/9 完成且严格审计通过，6/9 未完成，0 条 invalid**；
完整实验仍为 incomplete。以下 CPU 输入审计与语义审阅分别绑定文件哈希；
生成快照时间不表示后续解释已在当时完成。

## 当前运行条件

| 项目 | 三条运行的设置 |
| --- | --- |
| 平台 | 单台 DGX Spark / GB10，TP1，FlashInfer，CUDA graphs 关闭 |
| 镜像 | `sha256:ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25` |
| runtime | 冻结镜像只读覆盖一个 logprob processor 文件；不是新镜像 clean build 验证 |
| 输入 | 同一冻结 thinking prompt 的 497 个 native IDs；仅公开输入哈希 |
| 采样 | temperature=0，top_p=0.95，top_k=20，min_p=0，无重复惩罚 |
| 上下文与输出 | 服务窗口 262144；本题输出上限 16384；自动截断关闭 |
| 调度 | 客户端 c1，每条运行启动新服务，最终 ready → 成功 idle flush → cached tokens=0 |
| 状态精度 | FP8 KV、FP32 SSM；BF16 对照只指源权重，不能称全部计算与缓存均为 BF16 |
| 缓存与确定性 | radix cache 启用，实际本次缓存命中为 0；deterministic inference 关闭 |
| MTP | 关闭；本次纯文本诊断不重新验证视觉或 MTP |

262144 是服务窗口，16384 是这条诊断的输出预算，不能将本次 length 结束称为
耗尽 256k 窗口。后两题计划的输出上限分别为 24576 和 49152，尚无完成结果。
新运行 c1 与正式 GPQA 的 c16 不同；本次没有重新评测 198 题，也不产生新分数。

## 三条完成记录

| 权重 | native 输出 tokens | finish | EOS / thinking-end | 支持的精确循环 | 结果 |
| --- | ---: | --- | --- | ---: | --- |
| MSE W4A4 | 16384 | length | 均无 | 0 | budget_exhausted_unknown |
| max W4A4 | 16384 | length | 均无 | 0 | budget_exhausted_unknown |
| 源 BF16 权重 | 16384 | length | 均无 | 0 | budget_exhausted_unknown |

三条均通过终止审计与冻结 tokenizer 的 native 解码检查。精确周期检查每 256
tokens 及最终点执行，最大周期 16384，要求至少 4 圈且连续覆盖 512 tokens。
未发现符合条件的精确循环，只限定这些实际生成与检查阈值；较短、近似或语义
层面的反复仍需独立检查，不能由“精确循环为 0”推断完全没有重复。
BF16 于 `2026-10-12T08:27:32.317538Z` 完成。

三组实际 native 输出的共同前缀如下。首次分歧定位轨迹开始变化，不证明该 token
是停止异常或后续循环的原因。

| 自然生成比较 | 相同前缀 tokens | 首次不同的 token（0-based） | 解码 reasoning 共同前缀字符 |
| --- | ---: | ---: | ---: |
| MSE / max | 7 | 7 | 25 |
| MSE / BF16 | 7 | 7 | 25 |
| max / BF16 | 22 | 22 | 92 |

这里直接比较三次新生成的 native IDs。旧正式 journal 没有保存实际生成 IDs；
与旧文本重分词的比较不等于原 native 首次分歧，也不恢复原 c16 的增量状态。

## 严格启动范围与后续审阅

三条运行各有一次 idle flush POST 和一次 generate POST，最终 ready 先于两者，
没有额外健康检查生成。每条实际 native cached tokens 为 0，497-token prefill 的
缓存计数也为 0；各 409 条 decode 日志样本均为一个 running request、零个排队
请求。启动 gate v2 的 CRLF / LF 处理仅应用 Python universal newlines 规范化。
这些记录核验当前服务条件，不证明整个主机或 GPU 完全隔离，也不能排除日志
样本之间的其他工作。

比较程序成功生成 partial 比较报告并返回 0；启动 gate 因九组尚未全部完成返回
预期的 exit 2。这不是三条已验证运行失败，后续仍需完成匿名题目 5 与 3 的六条运行。

独立 CPU 审计只加载冻结镜像中的 metadata/tokenizer，重建原正式输入处理，未发
GPU 请求。当前文件重建的输入与诊断 reference 一致，没有找到当前 template
漂移；本题输入为 497 tokens，三条自然请求的输入哈希相同。但正式 journal 没有
保存历史实际输入 IDs，当前重建不能替代该证据，也不能将旧 c16 与新 c1 的分歧
归因于调度或量化。

输入审计的记录创建时间为 `2026-10-12T08:30:25.433059Z`。三向语义审阅是对该冻结
快照的后续 CPU 分析，其文件没有记录创建时间，不倒填时间证据。

## 三向语义审阅的有界观察

对早、中、晚多个连续片段的独立审阅发现，**BF16 也反复回到未决解释、知识与
构造问题**；三条同时仍加入新条件和细节。它们呈现语义回返和迟迟不作最终决定，
但这不是逐字符人工证明，也不足以认证稳定的语义循环或 MSE 独有停止退化。

| 权重 | 符合规则的长句 | 与先前长句相似 | 其中规范化后完全相同 |
| --- | ---: | ---: | ---: |
| MSE W4A4 | 433 | 34 | 12 |
| max W4A4 | 435 | 42 | 13 |
| 源 BF16 权重 | 408 | 56 | 13 |

规则保留至少 60 字符且 8 个单词的句子，统一大小写与空白后，对先前句子要求
SequenceMatcher 相似度至少 0.90、长度比至少 0.80。完全相同的计数包含在相似
计数中。命中也包括原题材料复述和正常自查；这些描述性统计不排列答案质量，
不定位失稳入口，也不能证明知识幻觉或量化因果。max 与 MSE 的 weight-block scale
配置也不同，因此本次不能单独归因于 MSE 目标。

## 文件冻结与结论边界

| 证据 | SHA-256 |
| --- | --- |
| 三条 natural 输出比较 | `0ac33ef82644423794d22279aec6c0eb5f2e1dd68149fb8f69c95e88707e88b8` |
| 三条严格启动 gate 审计 | `222ecce45675f9e0dd525e2e34d9d6bf3811b3f22536fcd39a789615e956c892` |
| 当前输入重建审计 JSON | `2719ea9a65a1aa1967cb760a287cda2678b5310c039333a1624dba7cb6686906` |
| 当前输入重建审计说明 | `6f5428239611579f99ae09ac1418f2ddb1560c329232dbefca2409139392b33e` |
| 三向语义审阅 JSON | `c287eca6aea68f6db36f69e0aeead8b3593059ef804b7f992efc1f2d66319c50` |
| 三向语义审阅说明 | `038d88a8d2b4f3ae5d597e91036138a70fc5b2d62036ca07c544817bd26250dd` |
| 只读 processor 覆盖 | `22ae6a64ab021d93f826ee6a8edc0befad14dfb5fd04213e31de762bfd5e4dd2` |

正式完整 trace 的精确循环仍已确认。本次三条有界自然生成没有建立其进入原因，
也没有证明量化导致循环、warmup 改变数值、正常停止或质量恢复。BF16 权重同样
耗尽当前诊断预算，只支持这道题、这些设置与这个预算下的观察。
