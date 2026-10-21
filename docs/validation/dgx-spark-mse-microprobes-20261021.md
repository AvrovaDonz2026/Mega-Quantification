# DGX Spark MSE 与 BF16 数值微探针：2026-10-21

**状态：两个受限数值探针完成；结果只适用于所记录的合成夹具和单步 decode 条件。**

## Static-max / static-MSE 三臂对照

在 ModelOpt 0.47.0、seed 42 的 synthetic recipe-native 对照中，dynamic-max、static-max 和
static-MSE 三臂的 activation amax 哈希完全一致。static-max 与 dynamic-max 的权重 MSE 和
held-out output relative MSE 相同；static-MSE 相较 static-max 的权重 MSE 降低 **24.56%**，
held-out output relative MSE 降低 **12.23%**。这支持该合成设置中权重目标差异可与 activation
amax 分离。

| Arm | Weight scale | Weight MSE | Held-out output relative MSE |
| --- | --- | ---: | ---: |
| dynamic-max | dynamic | 0.0000136097778523 | 0.0193793047220 |
| static-max | static | 0.0000136097778523 | 0.0193793047220 |
| static-MSE | static | 0.0000102668527688 | 0.0170088298619 |

该 recipe-native ModelOpt 0.47.0 探针运行于 historical serving image 上，并使用 Torch
2.13.0+cu130 与 offline overlay；它不是对原 PTQ 环境 Torch 2.14 / nv26.8 的完全复刻。
夹具是 synthetic tensor，不是完整模型权重或真实校准集。结果不证明完整模型的因果关系、
部署效果、模型质量改善，也不构成权重修复证明。

## Beta BF16 单步 decode 对照

在 image SHA-256 `ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25` 上，
用精确锁定的 historical source SHA-256
`b90bb5b144e4815ebc3fbba074b06aecac4c6db8b27bef8e43c74e7ddfd6d2cd` 与 candidate source
SHA-256 `1090b1b624dd0606ac1c1aabb0ec2e0c2001ac4df5a2002ccddf5c9ae6aa8976` 比较。候选只在
beta 路径引入 BF16 roundtrip。针对 8 个值，Triton candidate roundtrip 与 Torch BF16
sigmoid 的最大绝对误差为 **0**；相对 FP32 sigmoid 的最大绝对误差为
**0.0011681914**。这是有限采样点上的 beta 数值结果，不代表全域等价，也不证明整个
Hugging Face kernel 等价。

小型 GPU 单步调用的 output delta 与 state delta 均有限且非零：output max-abs 为
0.0078125、RMS 为 0.0010358009；state max-abs 为 0.0028519630、RMS 为 0.0003067451。
因此该记录同时显示局部 beta roundtrip 一致和有限单步输出/状态差异；不能将其外推为
全序列行为或质量结论。

## 解释边界与元数据

- 这是独立的数值诊断，不是整模因果实验、全模型修复证明或发布验收。
- 先前失败的 probe 受诊断代码与缓存环境影响；相关问题已修复。此前失败不能归因为权重故障。
- 公开证据元数据、镜像 SHA、源码 SHA、探针版本及汇总数字见同名 JSON。

