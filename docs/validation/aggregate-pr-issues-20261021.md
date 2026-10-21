# Issue/PR 聚合验证范围：2026-10-21

当前 `fix/mse-replay-eval-gates` 分支把以下问题修复合并到一个可审阅的变更集。
这份记录用于聚合 PR 的描述和复核，不替代各日期的 GPU 验证报告。

| Issue / PR | 变更 | 当前证据 | 范围边界 |
|---|---|---|---|
| #53 / PR #69 | recipe 的 `trust_remote_code` 同时传给 tokenizer、config 和 model | CPU 回归及 PR CI | 未声称远端模型下载或 GPU 重载已复现 |
| #54 / PR #67 | 删除不可达配置分支，统一 revision 大小写 | CPU 回归 | 不改变历史权重或服务结果 |
| #55/#56 | 严格 GPQA 答案解析、finish reason 门禁、未闭合 thinking 处理 | GPQA checker/CPU 测试 | 不重写历史 journal 或历史分数 |
| #57/#58/#64 / PR #65 | journal 缺字段告警、resume 协议校验、continuation budget 修复 | CPU 回归与故障注入 | 不证明某次远端评测已完成 |
| #59/#60 / PR #66 | OSS 下载流式校验、路径安全、失败清理、原子提交、dry-run 不写文件 | 单元测试与本地文件系统回归 | 不包含真实 bucket 凭据或权重发布证明 |
| #61 | x86 CUDA 12.8 / Python 3.12 依赖解析锁 | 依赖解析和安装检查 | 不是 27B GPU PTQ 成功或历史 5090 环境复现证明 |
| #62 / PR #68 | Docker digest 固定，SGLang patch manifest 缺失或漂移时 fail-closed | 容器静态门禁与 CPU 测试 | 不替代 Spark 上的完整模型启动、视觉/MTP 或 TP2 验证 |

## 合并前门槛

- `714 passed, 5 skipped` CPU 测试、Ruff 和 `git diff --check` 通过后，才可把该聚合
  PR 标记为代码层面可合并。
- GPQA 成绩必须同时具备 `summary.json`、198 个唯一题目、实际 CSV/seed/选项打乱
  策略匹配，并通过 `scripts/check_gpqa_results.py`；解析失败和长度结束计错。
- Spark 的视觉/MTP 功能检查、完整 256k GPQA、MSE 重复输出诊断和容器迁移分别绑定
  各自报告中的源码、镜像和权重身份；不能用 CPU 或合成 microprobe 结果替代。
- 当前 MSE 证据只确认 max/MSE 导出 scale 完整，且两者量化覆盖范围不匹配；R5 重放因
  heartbeat 维护错误中止。不要在 PR 或 release 中写成“已找到 MSE 根因”或“已修复循环”。

GitHub API 在本次文档整理时不可达，因此 PR/issue 的实时 CI 状态应在网络恢复后重新
核对；本表只记录仓库内已经存在的修复和本地验证证据。
