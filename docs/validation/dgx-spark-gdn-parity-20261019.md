# DGX Spark GDN parity 验证：2026-10-19

**状态：W4A4 `python_code` 矩阵完成，候选修复仍在验证。** 本记录绑定 Aitop 的 GB10 / SM121
实验，不是新的 GPQA 分数，也不构成 release 结论。原始响应、token 序列和中间
张量未写入仓库；机器可读记录只保留脱敏后的长度、状态、哈希和标量摘要。

## 固定协议

实验使用固定容器
`sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf`，
NVIDIA GB10 / SM121、Torch `2.13.0+cu130`、Triton `3.7.1`、
`FLA_USE_FAST_OPS=0`。局部回归使用固定 fixture（SHA-256
`31fbf33194ea1f7429472bf02a9951bb08c9ba37edceb628e8a15f01b63ae3e0`），覆盖
isolated 和 random 两类输入、GQA `(4,8)` 与 `(16,48)`、32 steps、有限性和
guard 检查。整模型 parity 使用 W4A4、TP1、temperature 0、thinking disabled、
FP8 KV、FP32 SSM、关闭 CUDA graphs；baseline 和 MTP 各重复两次。`python_code`
fixture 的预算是 768 tokens，四次响应必须自然 `stop`，并通过完整 AST gate。all-beta
MTP 实际完成 96 次 verify，提出 192 个 draft token，其中 181 个被接受（94.27%），
平均接受长度为 2.885。

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

## FP4 shape probe

SM121 的 Qwen3.x custom NVFP4 runtime gate 为 false（该 gate 只接受 SM120，
capability `(12,0)`）。因此本探针实际走 FlashInfer `mm_fp4`，测试
`5120x12288`、`5120x64`、`4096x5120` 三种形状及 M=`1,2,3,4,8`，共 15 个
输出比较；所有首行相对 M=1 均 exact，输出均 finite。输入 scale 在 M>1 时的
物理布局哈希可变化，但不能据此跨 M 比较数值；`physical_scale` 不作为模型根因
证据。该 synthetic probe 不能替代真实权重的完整模型验证。

## 身份、范围与发布门槛

四个 source commit、归档 SHA、模型 metadata、镜像、fixture 和运行时 import hash
均保存在相邻 JSON 的脱敏字段中；报告 SHA 也保留。完整 W4A8、视觉/OCR、MTP
候选矩阵仍待运行。语言 `python_code` parity 不代表视觉质量、256k MTP 质量、TP2
或吞吐改善。

在发布前，仍需要 W4A8 与 W4A4 的完整视觉/MTP 矩阵、
自然停止和重复稳定性证据，以及更新后的 README/运行记录。当前结果不足以发布新的
量化权重或宣称普遍质量恢复。

机器可读脱敏结果：[dgx-spark-gdn-parity-20261019.json](dgx-spark-gdn-parity-20261019.json)。
