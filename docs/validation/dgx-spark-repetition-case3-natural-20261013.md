# DGX Spark 重复输出诊断：匿名题目 3 的自然三权重对照，2026-10-13

**截至 `2026-10-13T02:08:02.781567Z` 的冻结捕获，九组自然生成全部完成：
9 条 validated、0 条 incomplete、0 条 invalid，启动 gate 全部通过。** 匿名题目 3
的源 BF16 权重在 16685 tokens 后以 EOS 停止；后续 CPU 正式解析器判定最终答案
可解析且正确。同题 MSE 出现精确循环，max 停止但答错。这三条轨迹支持本次行为
差异，尚不建立量化根因或算法质量排序。

本文及[相邻 JSON](dgx-spark-repetition-case3-natural-20261013.json)是新的完成快照。
此前 [MSE 循环](dgx-spark-repetition-case3-mse-20261012.md)、
[max 停止](dgx-spark-repetition-case3-max-20261012.md)等日期报告保留当时状态；
本次没有改写正式 GPQA journal 或分数。

## 本题完整自然结果

| 权重 | 输出 tokens | finish | thinking-end / EOS 次数 | 支持的精确循环 | 后续正式解析器判定 |
| --- | ---: | --- | --- | ---: | --- |
| MSE W4A4 | 49152 | length | 0 / 0 | 1，周期 2032 | 无终端最终答案 |
| max W4A4 | 22569 | stop | 1 / 1 | 0 | 可解析，答错 |
| 源 BF16 | 16685 | stop | 1 / 1 | 0 | 可解析，答对 |

BF16 的 thinking-end / EOS 位于 0-based 偏移 **16375 / 16684**，各一次，EOS 是
最后一个原生 token；两个标记严格之间有 **308 tokens**。本条生成完成时间为
`2026-10-12T14:09:01.237534Z`，后续捕获与 CPU 审阅分别记录自己的时间。native 终止审计通过，
没有非零 speculative 元数据。MSE 的周期 2032 稳定后缀从偏移 33131 到 49152，
连续覆盖 16021 tokens；max 的终止和后续答错审阅见其独立报告。

三组使用同一冻结题目的 288 个 native 输入 tokens、temperature=0、top_p=0.95、
top_k=20、repetition_penalty=1；单台 Spark / GB10、TP1、FlashInfer，关闭 CUDA
graphs 与 MTP。共同服务上下文为 **262144**，本题输出上限为 **49152**；另外
两题上限分别为 16384、24576。MSE 的 length 不等于耗尽 256k 上下文。
三组同用当前冻结输入；历史正式运行的 actual native 输入未保留，当前 metadata
重建不认证历史逐 token 重现。

直接比较三条新原生输出，MSE / max 共同前缀为 7 tokens，MSE / BF16 为
12 tokens，max / BF16 为 7 tokens；对应推理解码共同前缀为 25、46、25 字符。
这里没有用历史正式输出的 retokenized IDs。这些早期分歧不定位第一次错误、循环
入口或因果分叉。

## BF16 启动、配置与比较限制

BF16 在最终 ready 后成功 idle flush，日志与响应记录一次 generate POST、一次
flush POST、零次额外健康检查生成。288-token 输入一次 prefill 完成，cached=0、
pending=0；native cached tokens 与 retractions 均为 0。417 条 decode 日志样本
均为 running=1、queue=0。启动 gate 通过，但日志采样不证明全主机/GPU 完全隔离。

源 BF16 只表示权重对照：三组仍共同使用 **FP8 KV (`fp8_e4m3`) / FP32 SSM**。
MSE / max / BF16 的服务种子分别为 213087260 / 781300540 / 234258400，token
池容量分别为 1738591 / 1737732 / 620136，deterministic inference 均关闭。
PTQ 目标、block-scale、权重及激活数值路径和运行历史也有差异，不能称严格只差
量化目标的单因素比较；temperature=0 本身不证明种子造成差异。

BF16 saved server-info 的原始字节摘要与生成前记录的 canonical JSON 对象摘要
分别复算通过，绑定了完整保存对象。编码口径见
[配置哈希勘误](dgx-spark-server-info-hash-erratum-20261013.md)。该绑定不证明
每个字段在整个生成期间不变，也不是独立的远端 GPU 运行认证。基础镜像加单个
只读 processor 文件修复仍是冻结运行时；本次没有新增 clean image、视觉或 MTP
验证。诊断客户端 c1 与正式 GPQA c16 的调度条件也不同。

## 后续 CPU 解析与循环核验

BF16 原生分析保留 `final_answer_parseable` / `final_answer_correct` 为 `null`，
表示当时未评估；后续 CPU 审计创建于 `2026-10-13T02:21:33.404130Z`，使用与
冻结版本字节一致的仓库正式解析器，提取结果与绑定参考答案一致。原始回答使用
thinking-end 后的最后独立选项 fallback，仅在内存去除已确认的终端 EOS 文本后
预测不变。没有公开预测选项或参考答案，也没有回写原分析、正式 journal 或分数。
最终选项正确不认证每一步推理事实。

精确周期核验每 256 tokens 及最终推理点执行，最大周期 16384，要求至少 4 圈且
连续覆盖 512 tokens。BF16 初始分析报告 63 个检查点。
后续独立 CPU 核验使用另一实现：63 个常规检查点加最终点，共 64 个，均无符合
阈值的精确循环；此计数口径单独保留。它不能排除短周期、近似重复、语义反复或
检查点之间的周期。

独立原生停止、请求、周期与配置对象核验创建于 `2026-10-13T02:44:29.722725Z`，
另行确认三组当前输入与 sampling 参数相同、终端事件与记录响应一致、EOS 后无
续写，完整配置对象的 canonical 摘要匹配。它没有重新运行 tokenizer 解码，
也没有进行解析器或语义判分；本报告没有纳入 BF16 的有界推理语义审阅。
共同前缀与语义入口的 cold prefill 探针不恢复自然轨迹的 incremental KV / SSM
状态，其证据范围与本题自然停止、最终答案判分分开。

## 文件绑定

| 证据 | SHA-256 |
| --- | --- |
| 完成捕获清单 | `00f87af1fa3e2608b2b9f6b9a4072e39885361b46f5ee3e18879121f5b1a6099` |
| after09 比较 | `6849adb34c3418c934d365fb14e91e3c69d817e36887d6114188bb60c639180c` |
| after09 启动 gate | `4d99017b68b9d4fcd7442db758acbc7a8187f232d2028d5448e8a13b607fd2b9` |
| BF16 自然输出证据 | `a5f30f0bfd22883f75e10285f50ed4a4a12216e61a415e8ebcdc5c82e7cb6cf1` |
| BF16 原生解码证据 | `627a06b6dfe26ae94fa8f84d55011c326e663bd235432124113081242c8350bf` |
| BF16 原生分析 | `5e45f38eebd9be160487bcd876edc25ffc55145041c4c83d664377a9617ef49c` |
| BF16 终止审计 | `7ca6a967225dde7d4966da8b4f3ba0178c06005dea627859fbfac2422a509485` |
| BF16 初始 CPU 解析与周期审计 | `e24a685016d0f95c4e7800b87d8149e1bb72e2ae3bf5f4a050f3b24399eddd9a` |
| BF16 后续独立原生核验 | `46162765f44bdf6d07c978c46d2f595f7fa06c693f6c031114e13d46ad0ef728` |

公开证据只保留匿名统计、位置与哈希。根因、通用重复恢复与整轮质量比较仍待验证。
