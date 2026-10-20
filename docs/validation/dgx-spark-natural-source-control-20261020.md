# 原始 GDN 源码自然生成对照：2026-10-20

**状态：原始源码 control 的两次长测均已完成并重复稳定。** 前组 MLP 实验完成、恢复
原 GLM 并通过身份校验后，队列于 08:04:07 UTC 接上新 supervisor。两次均输出
39,846 tokens、返回 `stop` 并由 EOS、terminal token 与脱敏 matched-stop 一致确认；
耗时为 2,931.67 / 2,930.05 秒，thinking-end index 均为 39,666，
输出哈希一致，未检出确认的 2,032-token 周期。09:48:12 UTC 观察时 supervisor 已完成，
实验清理完成、原 GLM 当时已恢复运行，systemd unit 为 inactive/success。10:01:21 UTC
独立复核确认 13 个部署文件哈希未变化、GLM 身份仍精确匹配，但 GLM 已退出。日志显示启动约 65 秒后因 NCCL 远端进程或网络错误退出，
exit code 为 1、非 OOM；10:21 UTC 对同一容器的恢复重试也未通过稳定性检查。
此复核时实验容器和 GPU 计算进程均为 0。
后续只读审计确认原服务跨两个节点运行 TP2，两端的 master 摘要、端口与 rank 配对一致；
另一端仍在运行。该次只读审计时成对恢复尚未执行，后续 GPU 实验尚未启动。此发现解释了为何不能
把单容器启动当作服务恢复，但尚不能证明底层网络故障的原因。
完整脱敏字段见[相邻 JSON](dgx-spark-natural-source-control-20261020.json)。

## 要排除的因素

[现有长测](dgx-spark-gdn-parity-20261019.md)使用 all-beta 源码。control 两次均在
34,189 tokens 后返回服务 `stop`，输出一致且未检出确认的 2,032-token 周期。MLP
恢复臂两轮均在 25,000 tokens 后返回服务 `stop`，输出也一致。control 未复现
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
冻结的 source-control 报告没有保留 `num_retractions`：旧控制器会拒绝存在的非零值，
但接受缺失或 null，不能将其标为直接的零回收计数观测。新运行要求标准字段为整数零，
并将该值保留在每次 trial 摘要中。

队列持有跨 exec 的机器级租约。前组 unit 已成功退出、两阶段报告重复稳定且容器
清理完成、原 GLM 恢复并通过精确身份校验，随后进入新 supervisor，再次临时暂停
同一 GLM 容器。实验后的服务恢复由 Python `finally` 与 systemd `ExecStopPost`
托管。r3 的部署文件保持冻结。

当前控制器与 supervisor 的 89 项目标测试通过，涵盖原始/修正源码、单臂/双臂主流程、
错误 trial 元数据、失败恢复，以及按源码/作业/镜像/创建时间限定的清理。干净 tracked
副本完整 CPU 测试为 583 passed、34 skipped，Ruff 通过；本地跳过涉及缺失的
Torch、Pillow、PyArrow。PR #52 在 `d36bb59` 的两项 pytest CI 检查已成功。队列四种本地门禁场景通过。新恢复检查要求精确身份与配置保持一致，连续运行 120 秒；
有 healthcheck 时须为 healthy，总等待上限为 240 秒。它防止将短暂启动误报为恢复，
不代表应用已就绪；原实验的冻结文件保持不变。单机 supervisor 还会在暂停前解析
受保护服务的启动拓扑，拒绝跨节点或无法明确审计的启动方式，防止只暂停 TP2 的一端。

## 历史重放的只读准备

11:52:26 UTC 在目标机重新读取完整 scored MSE 导出，确认原始 provenance、
config、HF quant config、索引及全部 2,756 个张量内容哈希匹配，包含 519 个量化模块；
原 288-token 测试输入及历史 49,152-token 输出夹具也通过身份检查。检查耗时 21.69 秒，
未启动容器或调用 GPU。这排除了候选文件与锁定历史导出的字节差异，仍须验证实际
loader 的运行时处理。

历史重放准备器另行锁定原始镜像、完整启动模板、实际 override 目标和流式请求格式。
该次预检未启动运行；目标机 bind、容器内覆盖文件、双机维护协调器和冷请求当时仍待验收。
两端 GLM 的临时暂停及实验后成对恢复已获授权；只读预检本身不能标为运行结果。

## 历史条件实测：进行中

15:57:27 UTC，双端维护协调器在两端原 GLM 均暂停后启动历史重放。部署时再次核验
全部 2,756 个张量、519 个量化模块、元数据、原始请求、镜像和五项 bind。
本轮恢复历史 262,144 上下文、开启 radix cache、64 个 FP32 SSM 槽位、16 个并发
上限、原始 `modelopt` loader 和 seed 213087260；只提交一个冷缓存请求，MTP 关闭，
输出预算 49,152 tokens。它与前面的 65,536 上下文原始源码 control 是不同实验。

16:32:18 UTC 的只读快照确认客户端仍在流式生成，累计经过 2,076.454 秒、800 条
读取行；读取行数包括 SSE 分隔行，**不是输出 token 数**。16:34:14 UTC 的只读
服务标量为约 12.96 token/s，容器仍运行、未发生 OOM。两端维护心跳继续有效。
终止结果、输出哈希与周期统计尚未产生，不能标为历史循环已复现或问题已修复。
原 GLM 的成对恢复及应用层就绪检查须在实验后单独确认；当前仍为待完成。

新协调器已通过五种真实 systemd 故障注入：worker 被杀且创建 helper 存活、worker
被杀且 child 存活、成功 child 退出状态、45 秒租约过期、首次恢复失败后的重试。
每种场景均实际连续观察 120 秒。注入使用 dummy 进程，没有操作 Docker、GPU 或
原 GLM；这些结果验证恢复机制，不构成实际服务已经恢复的证明。

## 结论边界

这是排查源码算术因素的对照；两次 EOS 停止表明在当前协议下停止不依赖 beta 补丁，
但未复现历史退化，不能排除历史根因，也不构成修复证明。原始与 all-beta control
在当前协议下都停止；历史循环的 262,144 上下文、开启 radix cache、seed 213087260
等条件与本组不同。若要定位历史差异，下一步逐项恢复这些历史条件，不能把多个协议
变化一起解释为某个补丁的效果。

当前不建立循环修复、答案正确性、GPQA、视觉/MTP 质量、TP2、吞吐或 release 结论。
运行方法见[控制器说明](../../scripts/diagnostics/gdn_parity_20261019/mlp-pilot/README.md)。
