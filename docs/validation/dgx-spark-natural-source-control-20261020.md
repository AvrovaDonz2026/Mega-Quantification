# 原始 GDN 源码自然生成对照：2026-10-20

**状态：独立队列已启动，等待上一组实验完成；尚未开始新的生成。** 07:17:06 UTC
部署成功，07:26:44 UTC 只读回读确认新 systemd unit 仍为 active、队列为
`waiting_for_previous_run`，原 MLP 实验仍 active；新 supervisor 状态文件和长测
evidence 目录均未创建。13 个部署文件的哈希未变化。完整脱敏字段见[相邻 JSON](dgx-spark-natural-source-control-20261020.json)。

## 要排除的因素

[现有长测](dgx-spark-gdn-parity-20261019.md)使用 all-beta 源码。control 两次均在
34,189 tokens 后返回服务 `stop`，输出一致且未检出确认的 2,032-token 周期。MLP
恢复臂第一轮在 25,000 tokens 后返回服务 `stop`，第二轮仍在生成。control 未复现
历史循环，因此较早停止不足以证明恢复 MLP 精度修复了问题。

源码审计确认原始 GDN 的 prefill 和 packed decode 路径都会把 beta 降到 BF16；
all-beta 已修复这两处精度损失，并通过局部和短整模 parity。但原始 / all-beta
在固定 41,760-token 历史前缀后的 512-token 窗口均继续旧循环。该数值问题与历史
自然长循环的因果关系仍未确立。

本组保持 control 权重与推理协议，改用锁定的原始源码
`4a667614d810f7b5570be23eb75956c603a85c3a`，与 all-beta
`49d3acfd4d54b0cf1afffcfeb369c63af4f56499` 比较。只运行 control，两次冷缓存请求，
每次最多 49,152 tokens；从原始 288-token prompt 开始，无历史输出前缀。

## 固定条件与验收

上下文 65,536、temperature 0、seed 20261020、TP1、无 MTP、FP8 KV、FP32 SSM、
`modelopt_mixed` loader、关闭 radix cache 和 CUDA graphs、禁止自动截断。镜像沿用
`sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf`。
runner 沿用 r3 的原始字节，SHA 为
`338fce065587d36a05560b202c896eb2d78c6bf6f4133c297513a6318a5779cb`；
使用完整四变体 source lock，不能替换为仅含 all-beta 的候选矩阵 lock。

新控制器增加源码与单臂选择，并修正 EOS 元数据诊断；生成请求的参数保持一致。
它记录 source/model/runtime gates、prompt echo、零缓存、有限 logprob、输出长度与
SHA、thinking-end、脱敏 matched-stop 摘要，以及已知 lag-2032 的周期统计。旧 r3
只读 config EOS 的 false 字段保留原值与限制，不追溯改标。

队列持有跨 exec 的机器级租约。只有前组 unit 成功退出、两阶段报告重复稳定且容器
清理完成、原 GLM 恢复并通过精确身份校验后，才进入新 supervisor。实验后的服务
恢复由 Python `finally` 与 systemd `ExecStopPost` 托管。当前运行中的 r3 文件未修改。

控制器与 supervisor 的 38 项目标测试通过，涵盖原始/修正源码、单臂/双臂主流程、
错误 trial 元数据、失败恢复，以及按源码/作业/镜像/创建时间限定的清理。干净 tracked
副本完整 CPU 测试为 532 passed、34 skipped，Ruff 通过；本地跳过涉及缺失的
Torch、Pillow、PyArrow，可选依赖测试的 CI 结果另行记录。队列四种本地门禁场景通过。

## 结论边界

这是排查源码算术因素的对照，尚无新生成结果。历史循环的 262,144 上下文、开启
radix cache、seed 213087260 等条件与本组不同。如果两种源码都停止，下一步逐项
恢复历史条件，不能把多个协议变化一起解释为某个补丁的效果。

当前不建立循环修复、答案正确性、GPQA、视觉/MTP 质量、TP2、吞吐或 release 结论。
运行方法见[控制器说明](../../scripts/diagnostics/gdn_parity_20261019/mlp-pilot/README.md)。
