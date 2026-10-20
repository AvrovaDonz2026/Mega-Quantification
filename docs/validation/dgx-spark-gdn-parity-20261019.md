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
10 月 20 日审计发现此冻结控制器检查了非标准的 `retraction_count` 并将缺失值当作零，
未保留实际 API 的 `num_retractions`；因此这四次请求没有直接的零回收计数证据。
新控制器改为严格校验并保留标准字段；原 continuation 哈希和上述限制保持记录。

## MLP-only BF16 recovery pilot（r3 已完成）

MLP 变体从真正的 MSE 导出恢复 192 个语言 MLP 投影，移除 576 个 scale tensor，
输出共 2,180 个 tensors 和 327 个 NVFP4 量化层；视觉 110 个投影和 MTP 8 个投影
保持量化。它同时恢复 MLP 权重与输入激活精度，使用 `modelopt_mixed`。
逐张量 payload gate 通过，input inventory SHA 为
`f615a477133ef97e8694515654e18ddd74144bb0cafb020be1b99e13a4c4b8f6`，output
inventory SHA 为 `74591b904b9724b25556af2925bd3c28b8efa9cd0c6270e2824a5513c73420d1`。

r1 被 GPU idle gate 拦截；r2 于 2026-10-19 11:40 UTC 等满两小时后以
`expired_without_launch` 退出，没有创建实验目录。2026-10-20 获得临时停止现有
GLM 服务并在实验后恢复的授权后，r3 已实际启动。

2026-10-20 04:16 UTC 的观察：短阶段两臂各完成两次 4,096-token 请求，臂内均逐
token 一致，冷缓存为 0、logprob 有限，全部因达到预算结束，没有 thinking-end。
control 耗时 289.24 / 288.87 秒，MLP 耗时 699.05 / 696.29 秒；MLP 本次短请求平均
耗时约为 control 的 2.41 倍。两臂输出在前 7 个 token 后分叉，各自与历史自然输出的
LCP 为 7 / 12。两臂都未检出确认的 2,032-token 周期，但该短预算无法覆盖历史循环
起点，因此不构成停止恢复或循环修复。短阶段两个容器均已清理。

监督进程于 04:05:30 UTC 接上 long 阶段。r3 于 08:04:06 UTC 完成，观察记录时间为
08:04:11 UTC。control 两次均在 34,189 tokens 后返回 `stop`，耗时
2,503.58 / 2,505.88 秒；两次输出 SHA 一致、repeat stable=true，
各出现一次 thinking-end（index 33,901），未检出确认的 2,032-token 周期。MLP 两次均在
25,000 tokens 后返回服务 `stop`，耗时 4,300.50 / 4,299.41 秒；
两次输出 SHA 一致、repeat stable=true，各出现一次 thinking-end（index 24,735），
同样未检出确认周期。所有 long 容器均已 removed；监督单元为 inactive/success，
supervisor 为 completed。观察点记录 GLM running=true；随后新的 source-control
supervisor 于 08:04:38 UTC 启动，临时暂停同一 GLM 后于 08:08:59 UTC 发起请求。

control 没有复现历史自然循环，因此这组结果不能提供 MLP 修复的对照意义；两臂均为
服务 `stop`，不能据此宣称自然停止恢复或答案正确。[原始源码对照](dgx-spark-natural-source-control-20261020.md)
已独立启动。

两条 control 记录的 `contains_eos=false`、`terminal_eos=false`、`natural_termination=false`
仅使用了运行中诊断器从 config 读取的 EOS 集合。只读元数据核对发现，control、MLP
和原始 BF16 都是 config 一种 EOS、generation config 两种 EOS，tokenizer 的 EOS
也不在 config 集合中。该判据漏读停止元数据，不能把此处的 false 当作没有发生
EOS 停止。原始响应与完整 matched stop reason 未留存，当前不追溯改标；远端 r3
代码保持原 SHA。新工具合并 config、generation config、tokenizer EOS 与固定
additional stops，并保存脱敏的 matched-stop 摘要，22 项相关回归测试通过。
05:54 UTC 通过 stdin 在三份实际模型元数据上核对，新 helper 的停止集合均为两种；
这只验证元数据读取，没有加载模型或修改远端文件，也不重标既有结果。
输出出现 thinking-end 也不证明答案正确。

两臂均从原题的 288-token prompt 起点运行，65,536 context、temperature 0、
seed 20261020、TP1、无 MTP、FP8 KV、FP32 SSM，关闭 radix cache 和 CUDA graphs。
两臂短阶段完成稳定重复并清理后，监督进程接上每臂两次、最多 49,152 tokens
的 long 阶段。历史周期后缀起点为 33,131；本次 control 在 34,189 tokens、MLP
在 25,000 tokens 均未检出确认周期。long 报告只探测这一固定的 2,032-token 周期，
并不把服务 `stop` 解释为自然停止。

远端由 systemd 托管，Python `finally` 与 `ExecStopPost` 负责清理本次诊断容器并
启动原 GLM 容器；r3 结束时恢复已完成，观察点为 running=true。随后新的 source-control
supervisor 于 08:04:38 UTC 启动，临时暂停同一 GLM 后于 08:08:59 UTC 发起请求。
early 5 小时、long 18 小时是 watchdog
上限。相关离线测试共 27 项通过，包括控制器失败和停止命令超时后仍尝试恢复服务。
复现脚本与配置见
[`mlp-pilot`](../../scripts/diagnostics/gdn_parity_20261019/mlp-pilot/)。

## R3 beta-store 单变量对照：已完成

R3 保留 R1 的 radix-on 协议，仅移除 beta store 的 `.to(b.dtype.element_ty)`；原分配
已为 FP32。单个冷请求输出 `49,152` tokens，以 `length` 结束，输出 SHA-256
`57ab62e0676d083e05f154b399cefe4d70e5633d2a3e4cecdc5b921afa128cfa`，与 R1/R2 不同。
lag `2,032` 的最大匹配计数为 `10`，起点 `10,959`、覆盖 `2,042` tokens；匹配不足
完整周期确认门槛，只能说未检出原已确认的长期 lag-2032 循环，不能排除其他 period，
也不能声称正常停止、答案正确或 MSE 修复。speculative verify 计数未观测，缓存 token
和 retractions 均为 0；容量摘要 `1,728,271` 只保留容量差异边界。microprobe 的
legacy-to-candidate maxabs/RMS 为 `0.0005898475646972656` / `0.0002853343030437827`，
g delta 为 0，candidate sigmoid maxerr 为 `5.960464477539063e-08`；literal
first-logits 未测量。实验后双端原 GLM exact/running，连续稳定约 120 秒；有限应用
元数据响应不等于 readiness，readiness 仍为 `unknown`，未作端到端推理结论。

当前不建立 MLP 修复、自然停止恢复、GPQA 或视觉/MTP 功能通过的结论。停止也不代表
回答正确；只有在同协议 control 复现相关退化后，才有对应的修复对照意义。
完整脱敏状态、工具 SHA 和控制臂摘要见相邻 JSON。

机器可读脱敏结果：[dgx-spark-gdn-parity-20261019.json](dgx-spark-gdn-parity-20261019.json)。
