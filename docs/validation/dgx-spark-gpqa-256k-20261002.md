# DGX Spark GPQA Diamond：temperature=0，256k 上下文

两种 Spark 量化权重均完成全部 198 题，远端检查和本地独立复核一致。最终完成时间：`2026-10-03T13:12:33Z`。
截断回答和无法提取选项的回答均计错，分母固定为 198。

| 权重 | 正确/198 | 准确率 | 截断 | 未解析 | 输出 tokens | 客户端耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed W4A8 | 177/198 | 89.39% | 0 | 4 | 3,030,800 | 14.91 小时 |
| uniform W4A4 | 174/198 | 87.88% | 0 | 4 | 2,745,811 | 9.28 小时 |

使用单台 DGX Spark（GB10、ARM64、SM121），TP=1。服务上下文为 262,144 tokens，
客户端并发上限 16，服务 max_running_requests=16，Mamba 缓存上限 64，FP8 E4M3 KV。
关闭 CUDA graph 和 host HiCache，HTTP 每次请求超时为 86,400 秒（24 小时）。
MTP 在本次评测中关闭。没有单独设置较小输出上限：每题使用剩余上下文；长度截断仍计错。
mixed W4A8 的实际 KV 池为 1,565,460 tokens，uniform W4A4 的实际 KV 池为 1,700,758 tokens；均超过 262,144+512 的检查下限。
容器没有正值 `SGLANG_MAX_NEW_TOKENS_LIMIT` 额外输出限制，且 `allow_auto_truncate=false`。

采样参数为 temperature=0、top_p=0.95、top_k=20、min_p=0、seed=0；
启用并保留 thinking，reasoning_effort=xhigh，使用仓库固定种子打乱选项。
两轮分别从空 journal 开始，最终均包含 198 个与 CSV 完全对应的唯一题目。
每轮启动前的空 journal、新输出目录及 262,144 上下文配置均保存了独立启动记录及 SHA-256。
早期 32,768 上下文的试跑结果单独保留，未纳入此表；两种格式均从新目录重新评测。

数据来自[官方 GPQA 仓库固定版本](https://raw.githubusercontent.com/idavidrein/gpqa/56686c06f5e19865c153de0fdb11be3890014df7/dataset.zip)，
CSV SHA-256：`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。
权重使用 2026-10-01 完整视觉/MTP 量化后的 Spark 导出；逐文件 SHA-256 和所有评测文件摘要
见[机器可读证据](dgx-spark-gpqa-256k-20261002.json)。

服务基础 runtime revision 为 `ac53833dc5af3f03a10087449ca80e9484deba48`；
本次 GPQA recipe、客户端和部署 revision 为 `79b5dfef54618d0e385c5974d0ea508cd2fbdd40`，
结果检查器 revision 为 `470bf3c2e341ac01f9a3a01e21462954ae824d9e`。
客户端和服务镜像 ID、manifest 摘要、实际服务参数及软件版本均保留在机器可读证据中。

复现时使用 `recipes/eval-gpqa-diamond.spark-gpqa.yaml` 启动服务，为两种权重分别设置新的输出目录。
GPQA CSV 通过 `GPQA_CSV` 指定；完成后运行：

```bash
python scripts/check_gpqa_results.py \
  --csv gpqa_diamond.csv --run-dir <完整评测输出目录> \
  --output-json verification.json
```

GPQA 仅验证文本答题质量。本次没有衡量视觉推理或 MTP 对 GPQA 的影响；相关容器加载、视觉输入、
MTP 接受及固定样例 token 一致性检查见[独立的 v0.1.2 验证报告](dgx-spark-release-20261001.md)。
每种格式只完成一轮评测，temperature=0 也不证明任意并发、硬件或后端上的完全确定性。
服务和客户端均设置 262,144 上下文，但并发上限不等于 16 路完整上下文同时驻留；
实际 KV 容量和调度可能限制同时运行的请求数。未据此作速度提升或严格质量优劣结论。
