# DGX Spark W4A4 重复输出诊断：2026-10-12

**状态：六组独立重验完成，恢复 BF16 输出头、注意力投影或两者均未打断选定循环。**
全部 72 次请求成功，六组均在 9 个固定前缀上选择原 MSE 后继 token；三条循环内
续写各生成 512 tokens，逐 token 与原循环一致，没有 thinking-end 或 EOS。
相同权重的 mixed loader 对照与原 MSE 记录的首步分数和输出完全一致；恢复 BF16 改变了
logprob，但没有改变这些请求的 greedy 选择或续写。

完整 GPQA journal 中，MSE 和 Local-Hessian 的 6 条长度截断覆盖 5 道匿名题目，
均包含精确重复循环，最终回答为空。本轮修复了诊断中实际遇到的 SGLang logprob
列表转换异常；没有建立重复输出的修复或初始量化根因。最终比较仅使用修复后的
六组新结果，首轮未完成记录保持独立。详见[机器可读证据](dgx-spark-mse-repetition-20261012.json)。

本报告保留匿名统计和哈希。原始 GPQA 问题、答案、生成文本和 token ID 不公开。
诊断使用独立输出，没有修改正式 journal、checkpoint 或成绩；完整评测见
[256k 基线](dgx-spark-gpqa-256k-20261002.md)和
[MSE / Local-Hessian 完整验证](dgx-spark-w4a4-quality-20261008.md)。

## 完整 trace 中确认了什么

四组正式成绩继续为：max W4A8 177/198、max W4A4 174/198、MSE W4A4 178/198、
Local-Hessian W4A4 170/198。两组新权重各有 3 条长度截断，这些记录均计错，并已
包含在各自未解析数中。在 262144 上下文下仍出现这些循环。

对完整 reasoning 按空白规范化后检查精确循环，6 条截断的周期及末尾占比如下。
匿名编号只用于本次诊断；同一编号表示同一道题，不对应公开数据集 item ID。

| 匿名题目 | 权重 | 周期字符数 | 完整重复次数 | 周期性后缀占 reasoning 比例 |
| --- | --- | ---: | ---: | ---: |
| 1 | Local-Hessian | 14813 | 68 | 91.94% |
| 2 | Local-Hessian | 549 | 1346 | 85.22% |
| 3 | MSE | 4292 | 170 | 83.84% |
| 4 | MSE | 22 | 31082 | 93.64% |
| 4 | Local-Hessian | 2655 | 308 | 93.69% |
| 5 | MSE | 8 | 59557 | 90.72% |

这 6 条记录共生成 **1,566,831 completion tokens**，占 MSE 和 Local-Hessian 两轮
合计 **7,424,763 tokens 的 21.10%**。这是输出 token 占比，不能换算为耗时占比。
长输出中只有少数题目出现这种严重循环，不能概括为全部 MSE 回答都会重复。

在完整 198 题上，以重复 256 字符片段的覆盖率衡量，MSE 有 4 条覆盖率达到 50%，
4 条均未解析，其中 3 条截断；Local-Hessian 也有 4 条达到 50%，4 条均未解析，
其中 3 条截断。旧 max W4A4 有 2 条达到 50%，没有长度截断。
这项指标描述文本重复，不衡量答案正确性，也不单独证明量化算法造成循环。

## 为什么进行精度对照

MSE 搜索优化权重重建误差，没有直接约束 next-token 排序、thinking 结束、EOS 或
长序列生成的稳定性。本轮激活仍使用 max 校准，静态权重 scale 不会在每个生成步骤
重新变化。因此，较低的权重误差与较高的单轮 GPQA 分数不能保证停止行为稳定。

当前实验检查：在相同循环前缀下，恢复输出头或语言注意力投影的 BF16 精度，是否
改变原循环后继 token 的分数及短程续写。它不预设 `lm_head` 或注意力就是根因。
旧 max 用于检查另一套已评测权重面对这些前缀的行为；相同 MSE 张量的 mixed loader
对照用于区分精度变化和加载路径变化。

旧 max 与 MSE 也不是只更换优化目标的实验：仓库的 max 保留动态权重 block scale，
MSE 使用静态权重 block scale 并开启搜索；每次完整 PTQ 都重新采集激活统计。
两组都使用 max 激活校准，不代表两次导出的全部激活 global scale 字节相同。
这里的动态 / 静态指 PTQ 时权重 block scale 的处理；两组成品的 packed 权重在
推理时均保持固定，不会逐 token 重新量化。
因此，正式评测中重复行为的差别不能直接归因于 MSE 目标；本次派生精度对照才是
从相同 MSE 导出保留其余张量的干预。

## 冻结协议与派生权重

| 项目 | 实际设置 |
| --- | --- |
| 硬件 | 单台 ARM64 DGX Spark，GB10 / SM121，TP1 |
| 基础服务镜像 | `sha256:ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25` |
| 冻结服务及配方 revision | `79b5dfef54618d0e385c5974d0ea508cd2fbdd40` |
| 重验 runtime 修复 | `d2bac63` 的 `megaquant-spark-v7` logprob 修复，以单个只读源码文件覆盖运行 |
| MSE PTQ revision | `14d5ad6f9d4ac30fc91a55b2ff4697f2ab90a76d` |
| MSE PTQ 镜像 | `sha256:d3f9770ae061eeedd10100971ce77e2ac253cf5a45914b3d65c1bee86847dfe1` |
| 服务配方 | `recipes/eval-gpqa-diamond.spark-gpqa.yaml`，context=262144 |
| 执行后端 | FlashInfer，FP8 KV，FP32 SSM，CUDA graphs 关闭 |
| 前缀缓存与确定性开关 | radix cache 开启，`enable_deterministic_inference=false` |
| 服务容量设置 | `max_running_requests=16`，`max_mamba_cache_size=64` |
| 诊断请求 | 顺序执行，concurrency=1；native `/generate` 输入 token IDs |
| 采样 | temperature=0，top_p=0.95，top_k=20，无重复惩罚 |
| 截断与 MTP | `allow_auto_truncate=false`，MTP 关闭 |
| 每个权重的请求 | 9 个固定前缀，各生成 1 token；3 个循环内前缀，各续写最多 512 tokens |

六组权重顺序运行，模型之间释放诊断服务。四个派生 checkpoint 直接从已评测 MSE
导出复制，按选择恢复源 BF16 权重，没有重新校准或 PTQ。每个保留张量的 payload
SHA-256、每个恢复张量的源 BF16 payload 均已验证；删除相应量化 scale，依据实际
序列化张量重建 mixed 精度映射。

| 诊断标签 | 与 MSE 的差异 | BF16 恢复投影数 | 保留 NVFP4 投影数 |
| --- | --- | ---: | ---: |
| `mse` | 原始均匀 W4A4 MSE，原 loader | 0 | 519 |
| `max` | 原始已评测均匀 W4A4 max 权重 | 0 | 519 |
| `mse_mixed_control` | 全部 MSE 张量保持一致，改用 mixed loader | 0 | 519 |
| `lm_head_bf16` | mixed loader，恢复 `lm_head` | 1 | 518 |
| `attention_bf16` | mixed loader，恢复语言 self-attn / linear-attn 投影 | 208 | 311 |
| `combined_bf16` | mixed loader，同时恢复上述两组 | 209 | 310 |

四个派生权重都保留 **110 个视觉、8 个 MTP 量化投影**及其原始 payload。
相同张量对照的 2756 个张量全部与 MSE 一致，源和输出的 tensor inventory SHA-256
均为 `f615a477133ef97e8694515654e18ddd74144bb0cafb020be1b99e13a4c4b8f6`。
本次文本诊断关闭 MTP；张量保留不等于重新验证了派生模型的视觉或 MTP 服务。
派生 manifest 的 `gpu_serving_verified=false` 保留为导出时的离线标志；实际 GPU
文本诊断的成功与验证范围另记于本轮 JSON。

注意力对照恢复了投影权重及其输入激活的 BF16 路径，不能分别归因于权重或激活。
FP8 KV 和 FP32 SSM 仍按冻结配置运行；已为 BF16 的 GDN 卷积、门控和归一化组件
保留 MSE 原始字节。因此，它也不是全部注意力计算均为 BF16 的实验。

## 实际服务故障与独立重验

首轮 `mse_mixed_control` 的第一个单 token 请求成功，随后 SGLang 的
`batch_result_processor.py:1113` 抛出 `AttributeError: 'list' object has no attribute 'tolist'`。
服务于 `2026-10-12T04:28:31Z` 退出，原编排于 `2026-10-12T04:32:44Z` 停止。
这是一项实际服务异常，不能将失败或缺失的诊断结果解释为模型循环或停止行为。

冻结 SGLang 的 prefill / decode 处理方法对 top logprob 的值与索引、指定 token
的 logprob 值直接调用 `.tolist()`，但这些行可能已经是主机列表；原代码没有转换
指定 token 索引。当前 `megaquant-spark-v7` 补丁仅对 tensor 行调用 `.tolist()`，
保留已为列表的行，并补充指定 token 索引转换，覆盖两种阶段的值与索引。
从实际冻结 runtime 提取的方法在 CPU 上 **2/2 复现原错误**，补丁后的相同输入
保留全部值与索引。修复提交为
[`d2bac63`](https://github.com/AvrovaDonz2026/Mega-Quantification/commit/d2bac63)，
公开实现及回归测试为 `scripts/patch_sglang_spark.py` 和
`tests/test_sglang_spark_patch.py`。

全部六组独立重验于 `2026-10-12T04:37:54.900632Z` 启动，
于 `2026-10-12T05:18:11.649923Z` 完成，继续使用相同 checkpoint、
前缀和采样协议，所有输出写入全新文件。它使用上表的原冻结服务镜像，单独只读覆盖
`batch_result_processor.py`，并更换为带 abort 检查的 probe。首轮 MSE / max 结果及
失败记录保持独立，最终精度比较只采用这次重验的六组结果。

| 重验源码文件 | SHA-256 |
| --- | --- |
| 原冻结 `batch_result_processor.py` | `e4fb82ba6bc501f2e8f29a0fc76ce240d1d128210122619c899917433c2b6ddb` |
| v7 修复后的覆盖文件 | `22ae6a64ab021d93f826ee6a8edc0befad14dfb5fd04213e31de762bfd5e4dd2` |
| 重验 probe | `16149139fc98bbf4ee841ef956c3f8aec9278a80ff23fbd00141a3961212b515` |

这是冻结镜像上的 runtime 源码覆盖验证，没有证明新镜像完成 clean build 或 GPU 重验。
全部六组 GPU 文本诊断已完成，72 次请求没有 error 或 abort。冻结 driver 在每组
ready 标记之前核验实际运行的镜像和 processor SHA-256；该运行门检查不等于另一次
独立复核已停止容器，也不等于新镜像构建验证。

## 固定前缀的来源与限制

从 MSE 的 3 条截断记录分别取循环前、循环起点、循环内三个位置，合计 9 个前缀。
循环内前缀含原循环的两次完整重复，续写从第三轮开始。
使用冻结 evaluator、CSV、seed=0 的选项打乱规则和同一 tokenizer 重建 thinking prompt，
保留原始空白，仅去除 evaluator 添加的外层包装，并在 UTF-8 / tokenizer 边界截取。
9 个重建 prompt 的 token 数均与正式 journal 匹配，整段编码与 prompt 加 reasoning
的连接编码也逐项一致。

| 匿名题目 | 未规范化周期字符数 | 循环前 reasoning tokens | 循环起点 reasoning tokens | 循环内 reasoning tokens | 原题 prompt tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 4306 | 42543 | 42629 | 45203 | 288 |
| 4 | 22 | 12680 | 12769 | 12785 | 497 |
| 5 | 8 | 20940 | 21040 | 21048 | 2848 |

这里使用原始空白，所以题目 3 的周期为 4306 字符，区别于上表空白规范化后的 4292。
tokenizer SHA-256 为
`d40b12f2eef705e5017a5f4fc6c0c25b174a594485c1d7515624daf208212950`。
前缀是 journal 文本重新分词得到的 token IDs；正式生成时的原始 IDs 与增量 decode
状态没有保存。当前以这些 IDs 调用 native generate，服务的 radix cache 开启，
不能称为每个请求都完整重新 prefill，也不能恢复原运行的增量 KV / SSM 状态。

`loop` 分数组代表原始 MSE trace 在该位置的后继 token；循环前的位置未必已经重复。
`thinking_end` 和 `eos` 为单 token 候选。报告中的组分数是组内最高单 token logprob，
不是完整结束序列概率，也不是未经处理的 logits 张量。

512-token 诊断预算中的 `finish_type=length` 表示本次短续写用尽预算，不能与正式
262144 窗口的截断混为一谈。长段落循环的一整轮超过 512 tokens，单独检查续写内
是否自我重复会漏检；因此还检查其与原循环后继片段及之前 reasoning 的匹配。
即便出现新的词句、下一 token 改变或短循环消失，也不能据此认定完整停止行为恢复。

## 六组独立重验结果

以下结果全部来自修复后独立重验：**12 份探测报告、72 次成功请求**，其中 54 次
单 token、18 次 512-token 续写；18 条周期检查全部完成。输入、候选、协议、服务
设置和源码哈希逐项匹配；没有 error 或 abort，总计生成 9270 tokens。

| 权重 | 单 token 选择原后继 | 续写与原 MSE 后继完全一致 | thinking-end / EOS | 单 token `loop − thinking_end` 范围（nats） |
| --- | ---: | ---: | --- | ---: |
| `mse` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 22.6250–31.1719 |
| `max` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 21.4688–35.7617 |
| `mse_mixed_control` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 22.6250–31.1719 |
| `lm_head_bf16` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 23.1094–31.6406 |
| `attention_bf16` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 21.7500–32.0781 |
| `combined_bf16` | 9/9 | 3/3，均 512/512 tokens | 0 / 0 | 21.8437–32.9063 |

六组在每道题上都生成相同的 512 个输出 token IDs，且全部字符跟随原循环。

| 匿名题目 | 六组相同的 512-token 续写 |
| --- | --- |
| 3 | 1721 字符，覆盖原长周期 0.3997 轮；每个 32-token 窗口均出现于此前 reasoning |
| 4 | 最短周期 8 tokens，完整重复 64 次，512 tokens 全部周期化 |
| 5 | 最短周期 4 tokens，完整重复 128 次，512 tokens 全部周期化 |

题目 3 的一轮长循环超过诊断预算，续写内部没有短周期不能解释为退出循环。
题目 4/5 输入仅含两轮短循环，此前 reasoning 中没有完整 32-token 窗口匹配，也
不能解释为退出；原后继 512/512 token 和原字符周期匹配确认了循环持续。
全部请求以 `length` 用尽本次诊断预算，这不同于正式评测的 262144 窗口截断。

相同张量的 mixed loader 对照与原 MSE 在两阶段记录的所有首 token 候选及摘要
分数上差值均为 0；本轮没有比较 512 步的全部 logprob。
恢复 BF16 后分数发生变化，但 greedy 选择和续写保持一致。下表列出候选组分数、
selected / top1 / top2 logprob 中的最大绝对差值；它不是全部词表 logits 的差异。

| 比较 | 9 个单 token 请求最大差值（nats） | 3 个续写首 token 最大差值（nats） |
| --- | ---: | ---: |
| max vs MSE | 7.8125 | 1.2302 |
| mixed control vs MSE | 0 | 0 |
| BF16 输出头 vs mixed control | 1.1933 | 1.1933 |
| BF16 注意力投影 vs mixed control | 3.0000 | 1.0312 |
| 两者同时恢复 vs mixed control | 3.5000 | 1.7412 |

三个精度恢复方案均未在这些固定历史上打断循环，不能据此排除它们对自然入环
概率的影响。旧 max 也继续这些由 MSE 生成的循环前缀；旧 max 正式 198 题零截断
并不保证从任意已循环前缀退出。旧 max 与 MSE 的正式 reasoning 在题目 3/4/5 上
的共同前缀分别为 8/84/25 字符，当前比较不能衡量从原题起点自然进入循环的概率。

两阶段相同循环内前缀的首 token **18/18 一致**，分数却有变化。例如题目 3 的
`loop − thinking_end` 在 MSE 相差约 0.375 nats，max 相差约 1.625 nats。
缓存与执行条件没有通过重复或清缓存运行单独控制，不能把分数差异直接归为精度
变化的因果效果。当前 `stopping_recovery_established=false`，没有新增正式 GPQA 分数。

## 后续候选与验证顺序

下一步优先补充这 3 道原题的 BF16 基线，并在相同协议下从 prompt 起点与 max、MSE
比较最早分歧、自然入环位置和最终停止情况；先定位分歧，再选择新的精度干预。
继续延长已含循环的前缀不能回答初始入环原因，也不能验证精度恢复是否降低自然
入环概率。这项原题对照尚未运行。

要单独检验 MSE 搜索目标，需要另做 static-max / static-MSE 对照，控制初始权重、
校准输入和激活 global scale。当前 max / MSE 成品不满足这个条件；仓库 NVFP4
映射也没有把通用 `PrecisionSpec.dynamic` 用作该开关，不能只改 YAML 就声称完成
了控制实验。

长推理校准是后续候选。已评测校准输入的实际长度仅为 **154–501 tokens**，均值
247.69；单独增大 `max_seq_length` 不会让这些输入变长。可在保留 128 条图像、
128 条文本及视觉 / MTP 量化的范式下，将文本替换为独立的较长且有完整收尾的推理
样本，再检查激活范围及生成行为。它主要检验激活 global range 的覆盖，权重 MSE
目标本身没有学习停止行为。这项候选尚未量化或验证，不能作为本轮已解决循环的结论。

## 使用公开工具复核

离线工具默认只读取 checkpoint header 并生成计划，`--write` 才写入一个全新目录。
可选 variant 为 `control`、`lm-head`、`attention`、`both`。先查看计划，再写出需要
的独立诊断权重；输入 export 和 BF16 source 不会被修改。

```bash
python3 scripts/prepare_precision_ablation.py \
  --export "$MSE_EXPORT" --bf16 "$BF16_SOURCE" --variant lm-head

python3 scripts/prepare_precision_ablation.py \
  --export "$MSE_EXPORT" --bf16 "$BF16_SOURCE" --variant lm-head \
  --output artifacts/repetition/lm-head --write
```

`probe_fixed_prefix.py` 需要独立准备的私有 JSON bundle。schema_version=1，包含
context_length、tokenization、tokenizer_sha256、source_journal_sha256 和 cases；
每个 case 包含匿名 case/prefix、stage、input_ids 和 loop/thinking_end/eos 候选 IDs。
本报告使用 `retokenized_journal` 来源。准备 bundle 必须匹配实际 tokenizer、冻结
evaluator、数据集及选项规则；公开 probe 不会自动取得或重建本次私有评测输入。

服务须包含 v7 logprob 修复，关闭 MTP、自动截断，并使用 FP32 SSM。以下命令调用 SGLang native API，
保存私有响应及只含哈希和统计的报告；每个输出文件必须不存在。

```bash
python3 scripts/probe_fixed_prefix.py \
  --bundle "$PREFIX_BUNDLE" --base-url http://127.0.0.1:30000 \
  --server-info-path /server_info --model-label mse \
  --max-new-tokens 1 \
  --private-output artifacts/repetition/mse.next.private.json \
  --report-output artifacts/repetition/mse.next.report.json

python3 scripts/probe_fixed_prefix.py \
  --bundle "$INSIDE_PREFIX_BUNDLE" --base-url http://127.0.0.1:30000 \
  --server-info-path /server_info --model-label mse \
  --max-new-tokens 512 \
  --private-output artifacts/repetition/mse.rollout.private.json \
  --report-output artifacts/repetition/mse.rollout.report.json
```

更换服务权重后，用相应 `--model-label` 和全新的输出文件再次探测；可添加
`--reference-report artifacts/repetition/mse.rollout.report.json` 生成比较统计。
reference report 必须来自同一 bundle、协议与探测脚本。判断 BF16 对照时，还必须
比较 `mse_mixed_control`，避免把 loader 路径差异当作精度效应。
相同输入逐 token 一致仅说明这些固定请求的结果，不能推广到任意提示或并发设置。

首轮远程冻结 probe SHA-256 为
`35e140e4a7a277108337f1a2bfbfe694180f376075b269377bf21df03a7668c9`。
当前提交版增加了拒绝 `finish_type=abort` 的检查；修复后的全部六组重验使用上表
SHA-256 为 `16149139…` 的同一新 probe。新增检查不改变成功 stop/length 响应的
统计值，本轮最终证据已逐项确认没有 abort。复核时使用同一 runtime、同一 probe
版本的 reference report，记录实际 SHA-256；首轮与重验报告不能直接拼接比较。

精度派生和 probe 的 CPU 测试 **34 项通过**。包含 v7 runtime 修复的相关检查为
**109 passed、21 skipped**；从干净源码快照运行完整测试，结果为
**469 passed、34 skipped**。`ruff check src tests scripts` 通过。
这些 CPU 检查验证 payload 保留、配置、logprob 列表 / tensor 转换和探测失败处理，
不代替 GPU 生成对照。

## 输入证据哈希

| 输入 | SHA-256 |
| --- | --- |
| GPQA CSV | `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305` |
| MSE 完整 journal | `7c7aa22be4f7a152380555b702e17fb3d4ef08255dc3181c680a5c6555a84336` |
| Local-Hessian 完整 journal | `a7b09345ccd65ba2db370bb3941ff5e59883c5b2c9c8de4ac34f83f10c30aa7b` |
| max W4A4 完整 journal | `3b1605dcd0c9fc59927795724daf05269f05ece1186d1f0c24c8aae851836007` |
| max W4A8 完整 journal | `0b6b1085707bbcbad6a76013c9b21bc9d45f77f7e5ec0497b7459157a20ce5dc` |

本次诊断确认了选定长输出中的循环以及六组权重在固定前缀后的循环持续。服务
序列化异常已修复，三种精度恢复方案没有打断这些循环；量化初始根因、完整停止
恢复和普遍质量提升仍未确立。验证后续修复需要从原题起点独立检查候选。
