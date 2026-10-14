# DGX Spark 重复输出诊断：启动条件补充审计，2026-10-12

**截至 `2026-10-12T07:48:47Z`：六组固定前缀运行的启动范围审计完成；新的原题起点
对照有 2/9 完成生成与严格启动 gate 审计，7 组未完成。** 本文补充
[原固定前缀报告](dgx-spark-mse-repetition-20261012.md)的执行条件；原报告、JSON、
响应和正式 GPQA 成绩保持不可变。详见[机器可读证据](dgx-spark-repetition-startup-audit-20261012.json)。

原诊断客户端按 concurrency=1 顺序发送请求，但 `/server_info` 在内部启动 warmup
最终就绪前已经响应。六组服务的首个 probe prefill 均早于最终 ready 日志。因此，
这些请求的预热隔离 gate 没有成立，客户端 c1 不能扩展为已排除启动工作影响。
新原题起点编排改为等待最终 ready
标记、成功 idle flush，再发请求；实际响应必须确认 cached tokens 为 0。

## 已归属的固定前缀请求

306 条 prefill 日志全部匹配：6 条 122-token warmup 与 300 个诊断 prefill chunks，
覆盖 72 次请求。按顺序请求及 `prompt_tokens − cached_tokens` 核对，每条 chunk
计数匹配、无剩余事件。同一日志的行序解决同秒标记的先后，不依赖未经独立确认的
日志时区。下表只说明 **prefill 相对于最终 ready 标记的位置**，不等于 HTTP 提交、
完成或全部 decode 的时间边界。

| 权重 | 9 次 next-token prefill 全在标记前 | 跨越标记 | 全在标记后 | 3 次 512-token 续写 prefill |
| --- | ---: | ---: | ---: | --- |
| MSE | 3 | 0 | 6 | 全在标记后 |
| max | 2 | 0 | 7 | 全在标记后 |
| MSE mixed control | 2 | 0 | 7 | 全在标记后 |
| BF16 输出头 | 2 | 0 | 7 | 全在标记后 |
| BF16 注意力投影 | 2 | 0 | 7 | 全在标记后 |
| BF16 输出头 + 注意力投影 | 2 | 1 | 6 | 全在标记后 |
| 合计 | **13** | **1** | **40** | **18 次全在标记后** |

54 次 next-token 请求中，13 次 prefill 全在 ready 前，1 次跨越标记，40 次全在
ready 后。18 次续写 prefill 全在 ready 后，但 **18 次均复用了前缀缓存**；其中匿名
题目 3 的缓存状态来自 ready 前的请求。因此，续写也不是全部从零缓存开始的隔离
对照，不能仅按“在 ready 后执行”消除其前驱状态影响。

ready 前的日志同时记录一个新 sequence 和另一个 running request，但 running
计数可能包括完成后尚未清理的请求，ready 标记还在 `freeze_gc` 之后。因此，这只能
确认诊断 prefill 早于最终就绪，不能独立证明 warmup kernel 当时仍在计算或 GPU
并发。`enable_mixed_chunk=false`；也没有证明两条请求进入同一个 forward batch、
同一次 GEMM 或两行矩阵计算。没有证据单独证明 warmup 改变输出或导致循环。

12 份远程报告与冻结本地报告的内容一致；72 次 native 输出的 token 数及哈希均与
摘要匹配，全部用尽各自诊断预算，没有 EOS 或 thinking-end。6 份私有续写文件的
哈希与原周期分析来源一致。原记录中的同输出与循环持续仍是那次执行的观察；
本次修正限制其调度及缓存条件，不能将其称为排除 warmup 和缓存影响的 c1 对照。

## 新的原题起点诊断

新实验仅输入原题 thinking prompt，分别用 MSE W4A4、max W4A4 与源 BF16 自然
生成，保留 native IDs 供匿名比较。每题每权重启动新服务，协议如下。

| 项目 | 当前设置 |
| --- | --- |
| 平台及后端 | 单台 Spark / GB10，TP1，FlashInfer，CUDA graphs 关闭 |
| 基础镜像 | `sha256:ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25` |
| runtime | 冻结镜像上的单文件 logprob 修复覆盖；没有新镜像 clean build 结论 |
| 上下文 | 262144；自动截断关闭 |
| 输入与采样 | 原 thinking prompt，seed=0 打乱选项，xhigh；temperature=0，top_p=0.95，top_k=20，无重复惩罚 |
| 调度与缓存 | 客户端 c1；最终 ready → idle flush 成功 → 实际 cached tokens 为 0 |
| 状态精度 | FP8 KV、FP32 SSM；源 BF16 对照指权重精度，不是所有缓存与计算均为 BF16 |
| MTP | 关闭，纯文本诊断不验证视觉或 MTP 功能 |

| 匿名题目 | 每套权重输出上限 | 当前完成生成 |
| --- | ---: | --- |
| 4 | 16384 | MSE / max 生成与严格 gate 已通过；BF16 已生成 6465 tokens，未完成 |
| 5 | 24576 | 三套权重待运行 |
| 3 | 49152 | 三套权重待运行 |

262144 是服务窗口；16384 / 24576 / 49152 是本次输出上限，长度结束只表示耗尽
该诊断预算。它不代表耗尽正式 256k 窗口，也不计入新的 GPQA 分数。c1 与正式 c16
的调度条件不同，不能直接比较其输出差异并归因于量化。

## 两条完成记录与启动 gate

匿名题目 4 的 MSE 与 max 均输出 **16384 tokens**，`finish=length`，没有 thinking-end
或 EOS，两条停止审计与冻结 tokenizer 的 native 解码均通过。周期检查每 256 tokens
及最终点执行，最大周期 16384，要求至少 4 圈且连续覆盖 512 tokens；两条均未发现
满足阈值的精确循环。结果都是 **`budget_exhausted_unknown`**，不表示正常停止、
循环修复或质量恢复。MSE 于 `2026-10-12T06:52:31Z` 完成。

两套权重的自然输出前 **7 个 native tokens 相同，第 8 个开始不同**（0-based 偏移 7），
可见 reasoning 共同前缀为 25 字符。这里比较的是两次新生成的实际 native IDs，
首次分歧只定位轨迹开始变化，不能证明这个 token 是后续停止问题的原因。

新 MSE 输出与单独保留的最终 ready 前提交记录的 16384 个 native IDs 完全一致。这个
单次匹配不证明 warmup 必然无数值影响，也不构成 warmup 的因果实验。相对于旧
正式 MSE journal，新记录有 25 个共同前缀字符；新 native IDs 与旧文本重分词 IDs
的共同前缀为 7 tokens。旧正式生成 IDs 没有保存，不能称为原 native 首次分歧。

独立启动 gate v2 经日志 CRLF / LF 规范化后通过，**29 项测试通过**。实际记录确认
两条运行的最终 ready 均先于 idle flush 和诊断提交，各只有一个 flush POST 和一个
generate POST，没有额外健康检查生成，native cached tokens 为 0。两条各 409 条
decode 日志样本均为 1 个 running request、0 个排队请求。该 gate 核验记录到的
服务条件，不证明整个主机完全隔离或日志间不存在其他工作。BF16 的 6465-token
进度没有终止记录，不能计作完成结果；九组总实验仍为 incomplete。

这份 2/9 快照绑定以下独立文件哈希，后续结果不在此快照中推断：

| 证据 | SHA-256 |
| --- | --- |
| 两条 natural 输出比较 | `c2d04e4cec56276b52cdbe6b4061bcc25a677c9b002e0c877f400d4b47b00d72` |
| 两条严格启动 gate 审计 | `f51f5f2bb3aeb323385834727b0d473e70458732f15060ed8f954e2eb08827a0` |

## 探测终止检查与结论范围

公开修复 [`69bc5f9`](https://github.com/AvrovaDonz2026/Mega-Quantification/commit/69bc5f9)
让固定前缀 probe 验证实际保留的 EOS 与匹配字段，并要求 length 正好耗尽诊断预算，
拒绝提前长度结束或不符合协议的停止。相关 **42 项测试通过，Ruff 通过**。
这是诊断可信度修复，不是模型重复输出修复。

完整正式 trace 中的循环已确认；本次没有建立量化根因、warmup 因果关系、停止恢复
或新的质量分数。原六组实验保留为带启动与缓存限制的观察；后续需完成新九组
生成、实际启动 gate 审计及模型间 native 分歧和循环起点比较。所有公开证据仅含
匿名计数、位置和哈希，不含原题、答案、响应文本或 token ID 序列。
