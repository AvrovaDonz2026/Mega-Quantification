# 原始 GDN 源码自然生成对照：2026-10-20

**状态：历史固定案例已精确复现；radix-off 单变量对照正在生成，尚无修复结论。**
原始源码 control 的两次长测均已完成并重复稳定。前组 MLP 实验完成、恢复
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

## 历史条件实测：已完成

15:57:27 UTC，双端维护协调器在两端原 GLM 均暂停后启动历史重放。部署时再次核验
全部 2,756 个张量、519 个量化模块、元数据、原始请求、镜像和五项 bind。
本轮恢复历史 262,144 上下文、开启 radix cache、64 个 FP32 SSM 槽位、16 个并发
上限、原始 `modelopt` loader 和 seed 213087260；只提交一个冷缓存请求，MTP 关闭，
输出预算 49,152 tokens。它与前面的 65,536 上下文原始源码 control 是不同实验。

请求于 17:03 UTC 完成，运行耗时 3,936.556 秒（含启动）。输出达到 49,152-token
长度上限，终止类型为 `length`；输出 ID 摘要与历史轨迹完全一致
（SHA-256 `075c5ca7a8fa642cc1d9086b84e21fbdf0cd8c33e0de37ddee2980ed9a265bb5`）。
周期分析确认 lag 2,032、起点 33,131、周期计数 13,989、覆盖 16,021 tokens，含
7 个完整周期；缓存 token 为 0，请求回收计数为 0。历史固定案例的精确重放已证明，
但不能据此推断其他提示或条件。

17:05:14 UTC，双端原 GLM 均通过精确运行身份检查及连续 120 秒稳定性观察，并验证
维护协调关闭。后续只读应用观察中两端进程仍精确匹配且运行，但 observer framework
为空、readiness 为 `unknown`；因此这里只确认进程恢复，不声称应用已就绪。

## R2 单变量 radix-off 对照：已完成

R1 完成观察后，历史协议仅追加 server argv 的 `--disable-radix-cache`；metadata 的
`mamba` strategy 仍为 `extra_buffer_lazy`，其余历史条件保持不变。R2 作业于
17:15:42 UTC 启动，由双端维护协调器监管。

18:23:29 UTC 只读观察确认 R2 已完成；运行耗时 3,903.27 秒（含启动，不能由此推导请求完成时刻）。
唯一变量为 `--disable-radix-cache`，其余历史条件保持不变。输出达到 49,152-token
长度上限，输出 ID 摘要与 R1 及历史轨迹完全一致
（SHA-256 `075c5ca7a8fa642cc1d9086b84e21fbdf0cd8c33e0de37ddee2980ed9a265bb5`）。
周期分析同样确认 lag 2,032、起点 33,131、周期计数 13,989、覆盖 16,021 tokens，
含 7 个完整周期；缓存 token 为 0、回收计数为 0，speculative verify 计数未观测。
服务容量摘要为 1,731,408；该值低于 R1 的 1,737,140 和历史参考值 1,738,591，属于
有意的可移植容量差异，不能作为输出差异解释。

18:20:49/51 UTC 清理完成，18:20:53/54 UTC 原 GLM 成对启动，18:22:56/57 UTC
连续稳定性验证关闭，协调器于 18:22:57 UTC 通过。18:24 UTC 的应用观察确认两端
进程身份精确匹配且运行；一端的应用元数据探针得到结构化响应，但 observer
readiness 仍为 `unknown`，且没有端到端推理请求。因此这里只确认进程恢复和有限的
应用元数据响应，不声称应用已就绪或推理可用。

R2 与 R1 的固定案例输出完全一致，说明在此案例中关闭 radix cache 未观察到 token
序列影响，且 radix cache 不是该循环的必要条件；这不证明所有内部状态路径均无因果
作用，也不构成 MSE 修复结论。

R2 的只读 source capture 已收集历史容器中的全部十个允许文件；其中相关归档文件的
键哈希与旧归档一致，另有两个新增文件仅记录为已捕获。
归档历史源在相关 GDN 文件中没有 `TRACK_STATE`；当前 fork 的 snapshot 实现不能用于
解释 R1/R2，也不应把 current-fork all-beta 审计当作历史源码证据。

新协调器已通过五种真实 systemd 故障注入：worker 被杀且创建 helper 存活、worker
被杀且 child 存活、成功 child 退出状态、45 秒租约过期、首次恢复失败后的重试。
每种场景均实际连续观察 120 秒。注入使用 dummy 进程，没有操作 Docker、GPU 或
原 GLM；这些结果验证恢复机制，不构成实际服务已经恢复的证明。

## 结论边界

这是排查源码算术因素的对照。此前 65,536 上下文的原始 control 经 EOS 确认停止，
all-beta control 返回服务 `stop`，其有效 EOS 证据仍有限。历史条件重放则在固定案例中
精确复现 2,032-token 周期，且与历史输出逐 token
完全一致。R2 的 radix-off 结果表明该固定案例中 radix cache 未改变输出序列，不能
据此断言内部状态无因果作用，也不证明 MSE 已修复或确定循环根因。历史归档源与当前
fork 的 `TRACK_STATE` 观察必须分开解释。

当前不建立循环修复、答案正确性、GPQA、视觉/MTP 质量、TP2、吞吐或 release 结论。
运行方法见[控制器说明](../../scripts/diagnostics/gdn_parity_20261019/mlp-pilot/README.md)。
