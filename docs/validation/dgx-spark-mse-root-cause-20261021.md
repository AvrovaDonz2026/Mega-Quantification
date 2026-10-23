# DGX Spark MSE 根因排查：2026-10-21

**状态：scale 审计完成；整模因果结论仍未成立。**

## 已完成验证

- 原 PTQ 镜像中的 matched static runner 已通过无 GPU、无网络 CPU 预检。配置摘要、ModelOpt 0.47、校准输入、视觉和 MTP 开关均可构建。
- 磁盘上的 max 与 MSE 导出已完成只读 scale 审计。两者的 text scale 都是 401 个量化层，weight block/global scale 成对存在；所有 scale 均 finite 且为正，没有发现 NaN、零值或不完整 pair。
- MSE 导出另外包含 110 个视觉量化层和 8 个 MTP 量化层，二者的 scale pair 均完整。旧 max 导出没有视觉或 MTP scale。
- 因此，旧 max 与 MSE 的 benchmark 差异不能直接解释为 MSE scale 数值损坏；两份导出的量化覆盖范围不匹配，是总体比较中的一个已知 confounder。GPQA 和 R5 都是纯文本实验且关闭 MTP，不能把 vision/MTP 覆盖差异直接归因于文本重复或停止行为。后续 matched export 应固定 text 与 auxiliary（vision/MTP）开关，并使用相同 source、calibration 和 ModelOpt 版本，才能做有效的质量归因。

## R5 维护失败边界

decode beta BF16 对照 R5 在整模长测中被维护 heartbeat 错误中止，没有生成有效的模型结果。实验容器已清理，原 GLM 双端通过精确身份和约 120 秒进程稳定检查。只读 systemd 日志显示维护 worker 正常退出，但没有保留导致协调器失败的远端 SSH 细节，不能把这次失败归因于证书、网络、GPU 或模型。

新的维护包仅对 HEARTBEAT 做小于 45 秒 lease 的有界重试；ARM、STOP、LAUNCH、CLEANUP、START 等生命周期调用不重发，语义错误仍 fail-closed。该包先通过 6 个故障注入测试，随后用于下述独立 retry。原失败记录仍保留：它没有有效模型结果，也不能归因于证书、网络、GPU 或模型。

## R5 retry：已完成，未解决重复停止退化

Retry job `60348764d2634c4d68aaf486` 于 2026-10-21 22:49:32 CST 启动，模型实验阶段持续 3,962.954 秒；维护协调于 23:57:46 CST 结束。双端 heartbeat 健康，coordinator 为 `observation_passed`，恢复状态为 `process_stable`。两端均完成 CLEANED、STARTED 和 VERIFY_CLOSE `VERIFIED_CLOSED` 阶段，并通过超过 120 秒的进程稳定检查（peer 观测 121.242 秒，replay 观测 121.458 秒；RPC elapsed 分别为 122.893 和 123.306 秒）。实验容器清理及原 GLM 双端恢复已完成，armed=false，cleanup observed absent，failure=null，transport 非 success 事件为 0。

生成达到 49,152 output tokens 后以 `length` 结束；cached tokens 和 retractions 均为 0，未观测到 terminal stop match。beta imported/source SHA-256 为 `1090b1b624dd0606ac1c1aabb0ec2e0c2001ac4df5a2002ccddf5c9ae6aa8976`，生成前源码门禁通过。候选输出与历史参考输出的 SHA-256 分别为 `a54b083657e2e72653666057764f83bb4329bbea82b4a0cf83804ded1b74b066` 和 `075c5ca7a8fa642cc1d9086b84e21fbdf0cd8c33e0de37ddee2980ed9a265bb5`；双方长度均为 49,152，首次 token 分歧在 index 12，这表示生成轨迹不同，不是 logits 差异。

候选轨迹检测到周期 1,375（onset 38,469，匹配位置 9,308，覆盖 10,683 tokens，6 个完整重复）。历史参考周期为 2,032（onset 33,131，匹配位置 13,989，覆盖 16,021 tokens，6 个完整重复）。固定 lag-2032 比较只有 15 个匹配位置、extent 2,047 tokens，即 1 个完整周期；不得据此称历史 2,032 长循环已复现。新周期与 `length` 终止说明这一 beta BF16 单变量改动改变了输出轨迹，但没有解决重复或停止行为退化。

该 retry 不识别或修复 MSE 量化根因，也不证明答案质量、自然停止恢复或普遍重现性。它是单次固定运行对照；scale 覆盖范围不匹配这一既有混杂因素仍然存在。

## 当前判断

现有证据表明“导出覆盖范围不匹配”是总体 MSE 比较中的已知 confounder。GPQA/R5 均为纯文本且关闭 MTP，不能把 vision/MTP 差异直接归因于文本循环。R5 retry 显示 beta BF16 单变量改动改变生成轨迹，但候选仍检测到周期且以 `length` 结束；不能据此声称已找到 MSE 量化算法根因、修复了循环或恢复了停止行为。后续质量归因仍需固定 text 与 auxiliary 开关，并用相同 source、calibration、ModelOpt 版本生成 max/MSE 两臂 matched export 和重载摘要。

## 可复核证据

- CPU runner/adapter 测试：19 passed；scale collector 测试：11 passed；Ruff 全部通过。
- 无 GPU CPU 预检：return code 0，使用固定原 PTQ 镜像、runc、无网络和只读模型/校准目录。
- scale 审计：max 1,203 个 scale tensor；MSE 1,557 个 scale tensor；两个 arm 的 text scale 元素摘要一致。
- R5 retry：完成 49,152-token 单次生成与候选周期检查；源码门禁通过，双端服务恢复与稳定性验证通过。该统计不含题目、原始输出、token ID 或张量。

该报告只保留聚合统计，不包含主机、私有路径、请求内容、输出文本、token ID 或原始张量。

## 后续 matched export 与数值探针（2026-10-23）

后续 matched full-27B 导出已完成：text 401、vision 110、MTP 8 个量化模块，
auxiliary U8 条目为 118/118；配对 serving 的范围与限制见
[`dgx-spark-aux-dispatch-20261022.md`](dgx-spark-aux-dispatch-20261022.md)。
其 bounded long-prefix 试验没有复现自然循环：4,096-token 自然窗口早于历史
33,131-token 循环起点，supplied prefix 也不是旧增量 SSM/KV 状态。

同日 selected real-vision layer 数值探针观察到 fake/decode BF16 的有限差异，
但 decode 乘法顺序尚未与 vendor 对齐，scale、FP4 rounding 或重建顺序仍待细分。
该结果不定位本地导出 bug，也不能解释 MSE 独有重复；它不等同于 full-checkpoint、
full-model 或 SGLang kernel 数值 parity。

后续 v2 探针在同一层确认两臂有效 scale 的 248,832 个值均 exact（最大 ULP
差为 0），且 legacy 与 vendor 解码乘法顺序之间的 BF16 差异为 0；残差仍为 max 804、MSE 674
个元素，均属于 magnitude/signed-nonzero mismatch。effective-scale 数值和
decode association 因此不再是该探针中的解释，但 FP4 code decision 仍有差异。
v3 将除法精度假设限定在所选视觉层：path A 使用 Torch quotient，复现实际导出打包的编码选择；path B 使用默认 Triton quotient，精确复现 live fake quantization；path C 只把 B 的除法改为 `libdevice.div_rn`。默认除法在 0.75 FP4 tie 附近造成的 code 选择差异解释了该层观测到的 fake/export 差异；这不是整模 MSE 根因，也不表示重复问题已修复。

v3 两臂使用相同 source bytes，并复现 max 804、MSE 674 个残差；这些都位于
magnitude 从 1.0 向下跨过 0.75 tie 到 0.5 的位置。默认 ratio 相对 path A 有
1 ULP 差异，而 round-to-nearest 除法的 path C 在 ratio、magnitude 和非零符号上
与 path A 一致，仅有 318,729（max）和 335,136（MSE）个 signed-zero raw-code
差异，这些零符号码差异不改变解码数值。v2 与 v3 的 source hash 使用不同
计算规则，不应直接比较。数值 executor 以 exit code 0 完成，`observation_passed`，armed=false、
cleanup 为 observed-absent，恢复状态 process_stable。独立 readiness 检查确认
两个 GLM 容器运行且 restart count 为 0、无 OOM；服务入口的算术 smoke 在 0.219 秒
返回 2 和 native stop。该证据支持“默认近似除法造成所选层 fake/export code
mismatch”的局部解释；它不构成 full-model parity、MSE 独有重复根因或修复声明。
