# 文档导航 / Documentation

Mega-Quantification 的当前版本为
[v0.1.3](https://github.com/AvrovaDonz2026/Mega-Quantification/releases/tag/v0.1.3)。
从[项目 README](../README.md)了解方案与快速开始；按下面的任务选择详细指南。
文档中的命令默认从仓库根目录执行。

## 按任务查找

| 任务 | 文档 | 内容 |
|---|---|---|
| 选择量化和评测配方 | [配方索引](../recipes/README.md) | 默认混合 W4A8、均匀 W4A4、硬件配置和可选校准实验 |
| DGX Spark 完整流程 | [Spark 容器指南](DGX_SPARK.md) | ARM64 镜像构建、输入准备、完整 PTQ、导出审计、视觉/MTP 和 256k GPQA |
| RTX 5090 及 Qwen 模型操作 | [Qwen3.8-27B 手册](qwen3.8-27b.md) | 层布局、内存预算、校准、MTP 导出、推理、评测和 OSS |
| 安装 Docker、配置宿主机或 GPU Pod | [Docker 指南](../docker/README.md) | x86/Spark 镜像区别、Compose 和输入挂载 |
| 查看已验证结果 | [验证索引](validation/README.md) | 完整成绩、功能检查、容器复现与历史诊断的适用范围 |
| 查看版本变更和附件 | [发布索引](releases/README.md) | Release 说明、安装包及校验记录 |
| 扩展模型或后端 | [架构](ARCHITECTURE.md) | recipe schema、family adapter、后端协议和导出流程 |
| 开发和提交修改 | [贡献指南](../CONTRIBUTING.md) | CPU 测试依赖、Ruff、pytest 与提交约定 |
| 使用仓库代理约定 | [SKILL.md](../SKILL.md) | 量化、评测和发布的硬性要求 |

## 运行顺序

1. 按硬件选择配方，使用 `megaquant plan` 或 `--dry-run` 查看配置。
2. 安装 GPU 环境或构建镜像，准备完整 BF16 模型与校准数据。
3. 运行量化，保存 provenance、校准覆盖和导出审计结果。
4. 启动量化后的本地 checkpoint，分别检查文本、视觉/OCR 与 MTP。
5. 使用独立输出目录运行完整 GPQA，并通过结果检查器后记录成绩。

Spark 的 32768 上下文配方用于固定视觉/MTP 功能检查；完整 GPQA 使用
`eval-gpqa-diamond.spark-gpqa.yaml`，上下文为 262144、并发为 16、MTP 关闭。
两种用途的 journal 和验证范围分别保留。

## 阅读证据

运行指南描述如何执行当前代码。日期命名的验证报告描述实际运行过的源码、镜像、
输入和结果；相邻 JSON 保留摘要与哈希。历史版本号和 revision 对应当时的运行，
不能用新版标签替换。

完整 27B PTQ、固定视觉/MTP 功能、GPQA 文本质量、容器构建/迁移及安装包检查
各有自己的验证范围。具体分数和仍存在的问题见[验证索引](validation/README.md)。

## English

Start with the [project README](../README.md), then choose the task guide above.
The [Spark runbook](DGX_SPARK.md), [Docker guide](../docker/README.md),
[architecture](ARCHITECTURE.md) and [recipe index](../recipes/README.md) include
English instructions. The [validation index](validation/README.md) links every
dated report and its aggregate JSON; historical versions identify the actual
tested runtime. Commands assume the repository root.
