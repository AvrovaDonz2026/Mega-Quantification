# DGX Spark GDN parity 验证：2026-10-19

**状态：W4A4/W4A8 候选功能矩阵完成，GPQA 和发布质量结论仍独立。** 本记录绑定 Aitop 的 GB10 / SM121
实验，不是新的 GPQA 分数，也不构成 release 结论。原始响应、token 序列和中间
张量未写入仓库；机器可读记录只保留脱敏后的长度、状态、哈希和标量摘要。

## 固定协议

实验使用固定容器
`sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf`，
NVIDIA GB10 / SM121、Torch `2.13.0+cu130`、Triton `3.7.1`、
`FLA_USE_FAST_OPS=0`。局部回归使用固定 fixture（SHA-256
`31fbf33194ea1f7429472bf02a9951bb08c9ba37edceb628e8a15f01b63ae3e0`），覆盖
isolated 和 random 两类输入、GQA `(4,8)` 与 `(16,48)`、32 steps、有限性和
guard 检查。候选矩阵使用 W4A4/W4A8、TP1、32k 上下文、temperature 0、thinking disabled、
FP8 KV、FP32 SSM、关闭 CUDA graphs；baseline 和 MTP 各重复两次。`python_code`
fixture 的预算是 768 tokens，四次响应必须自然 `stop`，并通过完整 AST gate。all-beta
MTP 实际完成 96 次 verify，提出 192 个 draft token，其中 181 个被接受（94.27%），
平均接受长度为 2.885。两种方案共 5 个 case、baseline/MTP 双重复，共 40 次请求。

## 局部 beta 回归

| 变体 | decode | prefill | random decode | 结论 |
|---|---|---|---|---|
| original (`4a667614…`) | mismatch | mismatch | mismatch | 预期负控，稳定复现 |
| decode-beta (`a3b5d837…`) | match | mismatch | match | 只修复 decode 路径 |
| all-beta (`49d3acfd…`) | match | match | match | 通过 |
| shared-decode (`c7c3f48f…`) | match | match | match | 通过；归档叠加 all-beta |

四组报告均通过 finite、guards、source/runtime identity 和 fixture gate。局部通过
说明 beta 修复在受控投影上有效，不能外推为完整模型或视觉路径已经修复。

## 整模型 r2

| 变体 | 输出长度 baseline/MTP | 首次分叉 | stop | AST | 状态 |
|---|---:|---:|---|---|---|
| decode-beta | 271 / 272 | 264 | stop / stop | 通过 | `parity_failed` |
| all-beta | 277 / 277 | 无 | stop / stop | 通过 | `passed` |
| shared-decode, force=0 | 277 / 277 | 无 | stop / stop | 通过 | `passed` |
| shared-decode, force=1 | 277 / 277 | 无 | stop / stop | 通过 | `passed` |

all-beta、shared-decode force=0 和 force=1 的 baseline/MTP 输出 token 完全一致，重复
请求稳定，prompt、fixture、image identity、自然停止和 AST gate 均通过。decode-beta
虽然两侧自然停止且 AST 有效，但在第 264 个 token 分叉，因此不能称为完整 parity
修复。编排器顶层仍显示 `parity_failed`，原因是保留了 decode-beta 的有意消融失败；
这不表示三个通过的候选作业失败。

shared-decode force 开关仅用于诊断：启用时必须关闭 ReplaySSM 的两个相关 flags，
否则 ring buffer 不写入会污染后续 verify。当前两组 shared-decode 对照均在关闭
ReplaySSM flags 的设置下执行。实际 all-beta 候选使用 force=0，不依赖此诊断开关。

首轮 r1 使用 192-token 上限并以 `length` 结束，只能证明截断前缀相同，已标记为
无效的完整回答证据；本报告不把它计入 r2 结论。

## 候选 W4A4/W4A8 矩阵

两种方案均完成 arithmetic、python_code、red_square、blue_square、ocr_3729 五个
固定 case。每个 case 的 baseline/MTP 输出 exact match、两次重复 stable、两侧自然
`stop`，prompt/fixture/image identity 和 expected-answer/AST gate 全部通过。

| 方案 | arithmetic | python_code | red_square | blue_square | ocr_3729 | 总体 |
|---|---|---|---|---|---|---|
| W4A4 | 4/4 stop, exact, stable | 277/277 stop, exact, stable | 2/2 stop, exact, stable | 2/2 stop, exact, stable | 5/5 stop, exact, stable | passed |
| W4A8 | 4/4 stop, exact, stable | 276/276 stop, exact, stable | 2/2 stop, exact, stable | 2/2 stop, exact, stable | 5/5 stop, exact, stable | passed |

MTP execution is independently checked. W4A4 `python_code` recorded 96 verify steps,
181/192 accepted draft tokens (94.27%), mean acceptance length 2.885. W4A8 recorded
96 verify steps, 180/192 accepted (93.75%), mean acceptance length 2.875. The shorter
cases also have finite positive verify counts. Baseline verify counts are zero or absent;
MTP verify counts are finite and positive.

This is a 32k, TP1, `thinking=false`, CUDA graphs off functional matrix. It is not a
256k GPQA result, does not infer anything about the historical GPQA long-loop behavior,
and does not provide an independent numerical quantized-apply error gate for vision/MTP.

## FP4 shape probe

SM121 的 Qwen3.x custom NVFP4 runtime gate 为 false（该 gate 只接受 SM120，
capability `(12,0)`）。因此本探针实际走 FlashInfer `mm_fp4`，测试
`5120x12288`、`5120x64`、`4096x5120` 三种形状及 M=`1,2,3,4,8`，共 15 个
输出比较；所有首行相对 M=1 均 exact，输出均 finite。输入 scale 在 M>1 时的
物理布局哈希可变化，但不能据此跨 M 比较数值；`physical_scale` 不作为模型根因
证据。该 synthetic probe 不能替代真实权重的完整模型验证。

## 身份、范围与发布门槛

四个 source commit、归档 SHA、模型 metadata、镜像、fixture 和运行时 import hash
均保存在相邻 JSON 的脱敏字段中；候选报告 SHA、四份 checkpoint SHA、四份 cleanup
SHA 和上传包 SHA 也保留。历史 `max` 权重在候选 metadata 中只是历史 metadata view，
不是 MSE 权重，也不转移 GPQA 结论。候选矩阵的视觉/OCR/MTP 功能 gate 已通过，但
不代表视觉/MTP quantized-apply 的独立数值误差验证、256k MTP 质量、TP2 或吞吐改善。

候选功能矩阵本身已完成；发布仍需单独评审 GPQA、长上下文停止行为、视觉/MTP 数值
误差和权重发布范围。当前结果不足以宣称原 GPQA 长循环已解决或普遍质量恢复。

## MSE native-prefix sensitivity（negative result）

独立固定前缀实验在同一份真正的 MSE 历史权重 view 上比较 original 与 all-beta 源码（PTQ revision
`14d5ad6f9d4ac30fc91a55b2ff4697f2ab90a76d`，不是候选矩阵使用的 max metadata
view）。前缀 41,472 tokens，加 288-token prompt，共 41,760 tokens；服务窗口
65,536，TP1、temperature 0、无 MTP、radix cache disabled、FP8 KV、FP32 SSM。
两臂各运行两次，512-token 输出窗口均与 native 旧周期后继逐 token exact，prompt
exact，cold cache 为 0，重复稳定；四次均因 `length` 结束，没有 EOS 或 thinking-end。

这是明确的 negative result：beta 补丁没有打破该固定前缀的 512-token 窗口。它不等同
自然生成轨迹已修复，也不能判断更长预算是否最终停止。native IDs 和原始响应不发布，
只保留相邻 JSON 中的摘要与 SHA；四个服务/client cleanup 均已移除且无残留容器。

## MLP-only BF16 recovery pilot（awaiting GPU）

该诊断目前只完成静态导出与 payload gate，尚未完成 GPU serving。MLP 变体从真正的
MSE 导出恢复 192 个语言 MLP 投影，移除 576 个 scale tensor，输出共 2,180 个
tensors 和 327 个 NVFP4 量化层；视觉 110 个投影和 MTP 8 个投影保持量化。weights
与 input activation 都属于 `modelopt_mixed` 恢复范围，不是 weight-only、新 PTQ 或
W4A8。逐张量 payload gate 通过，input inventory SHA 为
`f615a477133ef97e8694515654e18ddd74144bb0cafb020be1b99e13a4c4b8f6`，output
inventory SHA 为 `74591b904b9724b25556af2925bd3c28b8efa9cd0c6270e2824a5513c73420d1`。

自然起点 pilot controller 已准备 4,096-token 预算、两臂各两次、65,536 context、
all-beta、temperature 0、无 MTP、radix disabled、FP8 KV、FP32 SSM；脚本位于
[`mlp-pilot`](../../scripts/diagnostics/gdn_parity_20261019/mlp-pilot/)。AITOP 的
r1 因 GPU idle gate 被拦截且没有启动实验容器；r2 当前 `waiting_for_gpu`，证据显示
有 1 个活动 GPU 进程，最长等待 2 小时。Robo 与 AITOP 均有其他 GLM GPU 作业。

因此本节不宣称 MLP 修复、自然输出改善或视觉/MTP 功能通过；待 GPU 空闲后才可运行
pilot。详细脱敏 gate、工具 SHA、量化计数和队列证据见相邻 JSON。

机器可读脱敏结果：[dgx-spark-gdn-parity-20261019.json](dgx-spark-gdn-parity-20261019.json)。
