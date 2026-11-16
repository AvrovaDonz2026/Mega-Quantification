# DGX Spark static MAX/MSE TP1 实测：2026-11-16

**状态：已完成。** 本轮在 `aitopatom-aacd` 上使用固定 serving 镜像和两棵已核验的 27B 导出，先完成视觉/MTP 对照，再完成 MTP 关闭的长生成对照。Spark 上原有 `qwen38-tp2` 容器在启动前已经是 `Exited (255)`，因此没有额外停止运行中的 Qwen 服务；Micius、embedding、网关和监控容器保持原状。

## 固定身份

- serving 镜像：`megaquant@sha256:ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25`
- `static-max` 导出：16 文件，树 SHA-256 `63ca89e911cabe4e04106ce7173f21853fdda93cc73ddbfac3cef22ba225a914`
- `static-MSE` 导出：16 文件，树 SHA-256 `eed63db900dfd4df60edcaa8c8a2a840a2d95b04eec19552553531dcabcbfb4b`
- pair ID：`84aea3442f486cea7a5d6f88`
- 两份计划均通过 `validate_plan()`；声明的计划摘要为：multimodal `c561ed72d68b683376ba5ace567dff4c21df46effb511bb3553761087c13c95b`，long `ae55e7556d7793a870208da83345aa603c0d04648a7cee6ac15154d5e597fb0`。

启动前重新读取了镜像 digest、两棵导出树、GPU 和监听端口。两棵树均与计划匹配，GB10 空闲，推理端口无监听。容器使用 `--network none`、`--ipc=host`、只读 bundle/权重挂载、独立可写 evidence/cache，并显式指定镜像的 `python3` entrypoint。

## 视觉与 MTP

固定 checker 覆盖算术、Python 代码、红色方块、蓝色方块和 OCR `3729` 五个用例。每个量化臂都分别冷启动 baseline 和 embedded EAGLE MTP；MTP gate 要求功能检查通过、greedy native output IDs 与该臂 baseline 完全一致，并且出现正的 draft/verify 计数。

| 臂 | gate | correct drafts | speculative verify | baseline IDs | MTP 原始报告 |
| --- | --- | ---: | ---: | --- | --- |
| static-max | passed | 136 | 72 | `56e78973f4f55fc419c598f75e61d74476cc7f5401e20389480007e7b9ffb5c1` | `25288b716085b88449d170761ad6abca5d887fd077771b9eebbe2b083dc9e3a0` |
| static-MSE | passed | 138 | 71 | `d43db3440fb2861353294ea37e1c767bd934d99b6a2340ea320cf891c466f01a` | `1a732144c12facfd19a3d903f9e550005121f23c718e0ffa750c7e32c0ccca84` |

视觉结果文件 SHA-256 为 `7ca5b597f328074576e3fba061bcf099bc13506b2a8a7420b38f988ec12c8366`。正常退出后，GPU 计算进程、推理端口和 `pair-multimodal-*` 临时目录均为空；原始 prompt、图片和 token IDs 没有被保留。

这只证明固定五题和当前 TP1 配置的功能/MTP gate 通过，不能外推泛化视觉质量、256K MTP 质量、TP2 或吞吐。

## 49,152-token 长生成

长计划固定 262,144 context、288-token prompt、temperature 0、seed `213087260`、MTP 关闭、FP8 KV、FP32 SSM、冷启动并先 flush cache。runner 只保留 token 数、输出 SHA、有限周期统计和停止元数据；结果顶层 `repetition_fixed=false`。

| 臂 | token 数 | finish | 输出 SHA-256 | 有界重复扫描 |
| --- | ---: | --- | --- | --- |
| static-max | 49,152 | `length`（到达上限） | `5fd23836d846f15da70f7e9a474541086057c42ed9487bb084774762de2e678e` | 检出 period `435`，onset `32667`，covered `16485`，copies `37`；`complete=false`、`exhaustive=false` |
| static-MSE | 31,039 | `stop`，matched kind `integer` | `9488191001a300562de18e90cb0a9361560ffa0913b39e5eafcb07dee3b4d7c2` | 未检出周期；扫描 `complete=false`、`exhaustive=false`，候选数有限 |

MAX 的结果说明输出后段存在强重复，并且在固定上限处被截断；它不能单独证明全程都是同一周期。MSE 在 31,039 tokens 处命中脱敏的 integer matched-stop；`eos.candidate_count=null`、`candidate_list_checked=false`、`effective_stop=false` 是协议要求的未验证标记，因此不能把这个 `stop` 解释成 EOS 自然停止，也不能据此声称 MSE 已无重复。

长结果 SHA-256 为 `a1785cd59103cc3c2635c5380b91491079be3fd86f684bda8a3d9677ac1d346b`，独立 `validate_result()` 复核通过。实验完成后容器退出码为 0，GPU 计算进程和 30000/8888/30001/30002 端口均为空；本轮 watchdog 已停用并删除退出容器。远端 evidence 保留在 `/tmp/megaquant-run-long-20261116-r3/evidence/long`，视觉 evidence 保留在 `/tmp/megaquant-run-mm-20261116-r3/evidence/multimodal`，均只含结构化结果和 progress 文件。

本轮建立了新权重在固定协议下的可复现 serving 证据，但没有建立重复根因修复或生产质量结论。后续若要判断根因，需要在保持相同权重、镜像、上下文和请求的条件下继续做有明确单变量的对照。
