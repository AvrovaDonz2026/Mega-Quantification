# DGX Spark MSE 根因排查：2026-10-21

**状态：scale 审计完成；整模因果结论仍未成立。**

## 已完成验证

- 原 PTQ 镜像中的 matched static runner 已通过无 GPU、无网络 CPU 预检。配置摘要、ModelOpt 0.47、校准输入、视觉和 MTP 开关均可构建。
- 磁盘上的 max 与 MSE 导出已完成只读 scale 审计。两者的 text scale 都是 401 个量化层，weight block/global scale 成对存在；所有 scale 均 finite 且为正，没有发现 NaN、零值或不完整 pair。
- MSE 导出另外包含 110 个视觉量化层和 8 个 MTP 量化层，二者的 scale pair 均完整。旧 max 导出没有视觉或 MTP scale。
- 因此，旧 max 与 MSE 的 benchmark 差异不能直接解释为 MSE scale 数值损坏；两份导出的量化覆盖范围也不匹配。需要同源校准、同一 ModelOpt 版本、相同视觉/MTP 开关的 matched export 才能做有效的质量归因。

## R5 维护失败边界

decode beta BF16 对照 R5 在整模长测中被维护 heartbeat 错误中止，没有生成有效的模型结果。实验容器已清理，原 GLM 双端通过精确身份和约 120 秒进程稳定检查。只读 systemd 日志显示维护 worker 正常退出，但没有保留导致协调器失败的远端 SSH 细节，不能把这次失败归因于证书、网络、GPU 或模型。

新的维护包仅对 HEARTBEAT 做小于 45 秒 lease 的有界重试；ARM、STOP、LAUNCH、CLEANUP、START 等生命周期调用不重发，语义错误仍 fail-closed。该包已经通过 6 个故障注入测试，但尚未用于重跑冻结 R5。

## 当前判断

现有证据支持“导出覆盖范围不匹配”是 MSE 分数比较中的首要混杂因素，不能支持“已找到 MSE 量化算法根因”或“beta 改动修复了重复输出”。下一步应先用 matched runner 完成 max/MSE 两臂的同源导出与重载摘要，再在成功的整模 R5 后决定是否采集同相状态摘要。

## 可复核证据

- CPU runner/adapter 测试：19 passed；scale collector 测试：11 passed；Ruff 全部通过。
- 无 GPU CPU 预检：return code 0，使用固定原 PTQ 镜像、runc、无网络和只读模型/校准目录。
- scale 审计：max 1,203 个 scale tensor；MSE 1,557 个 scale tensor；两个 arm 的 text scale 元素摘要一致。

该报告只保留聚合统计，不包含主机、私有路径、请求内容、输出文本、token ID 或原始张量。
