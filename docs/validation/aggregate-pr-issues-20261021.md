# Issue/PR 聚合验证范围：2026-10-21

当前 `fix/mse-replay-eval-gates` 分支把以下问题修复合并到一个可审阅的变更集。
这份记录用于聚合 PR 的描述和复核，不替代各日期的 GPU 验证报告。

| Issue / PR | 变更 | 当前证据 | 范围边界 |
|---|---|---|---|
| #53 / [PR #69](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/69) | recipe 的 `trust_remote_code` 同时传给 tokenizer、config 和 model | 本地 CPU 回归 | 未声称远端模型下载或 GPU 重载已复现 |
| #54 / [PR #67](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/67) | 删除不可达配置分支，统一 revision 大小写 | CPU 回归 | 不改变历史权重或服务结果 |
| #55/#56 | 严格 GPQA 答案解析、finish reason 门禁、未闭合 thinking 处理 | GPQA checker/CPU 测试 | 不重写历史 journal 或历史分数 |
| #57/#58/#64 / [PR #65](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/65) | journal 缺字段告警、resume 协议校验、continuation budget 修复 | CPU 回归与故障注入 | 不证明某次远端评测已完成 |
| #59/#60 / [PR #66](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/66) | OSS 下载流式校验、路径安全、失败清理、原子提交、dry-run 不写文件 | 单元测试与本地文件系统回归 | 不包含真实 bucket 凭据或权重发布证明 |
| #61 | x86 CUDA 12.8 / Python 3.12 依赖解析锁 | 依赖解析和安装检查 | 不是 27B GPU PTQ 成功或历史 5090 环境复现证明 |
| #62 / [PR #68](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/68) | Docker digest 固定，SGLang patch manifest 缺失或漂移时 fail-closed | 容器静态门禁与 CPU 测试 | 不替代 Spark 上的完整模型启动、视觉/MTP 或 TP2 验证 |
| 当前分支 | 量化 vision export 的 shard key 缺失或索引不一致时 fail-closed；BF16 restore 可从源 index/shard 恢复缺键 | Python 单元测试覆盖量化拒绝路径、BF16 缺键恢复和索引更新 | 仅覆盖 export 文件路径，不证明真实视觉权重已在 GPU 上加载或生成 |
| 当前分支 | MTP 4D packed/cached attention mask 诊断与兼容性边界 | 形状/类型诊断和明确错误路径 | 不把不支持的 4D packed/cached mask 转换成可用实现，也不证明 MTP 质量 |
| [commit 588afcb](https://github.com/AvrovaDonz2026/Mega-Quantification/commit/588afcbd1371af90e05cfed2a7b43f3e4a4f5c8d) | 校验 MTP calibration shapes，并从 safetensors headers 重建 export index `metadata.total_size` | Python 单元测试覆盖形状拒绝和 header-only total-size 修复 | 仍是 CPU/export metadata 证据，不是 GPU MTP 质量或 serving 结果 |
| 当前分支 | NVFP4 `static_max` 控制 selector：静态 weight block scale、动态 input scale，runtime 仍为 ModelOpt `max` | 代码、plan/provenance 元数据和单元测试已整合；本轮 CPU 门禁覆盖该改动 | 仅 CPU/config 语义，尚无 GPU PTQ 或同源 matched-export 结论，不代表质量/性能改善或 MSE 循环修复 |
| [聚合 PR #70](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/70) | GPQA continuation 高置信重复前缀保护；新 journal 的 `prompt_sha256` 与 CSV prompt 交叉校验，旧行缺字段兼容并单次告警；服务返回 usage 时 repetition row 保留该次实际 usage；vision restore 按源 shard 名恢复 BF16 缺键 | CPU 测试覆盖 continuation/checker/resume 和 vision shard 恢复路径 | 重复 chunk 不追加文本；checker 接受 `repetition` + `truncated=true` 并计错。此评测保护及 CPU 权重文件检查均不证明 GPU MSE 根因已修复或整模视觉已验证 |

## 合并前门槛

- 当前验证记录的 Python 3.11 CPU 门禁为 `PYTHONPATH=src /private/tmp/megaquant-clean-py311/bin/python -m pytest -q -o addopts=''`，结果 `742 passed, 5 skipped in 58.68s`；`ruff check .` 和 `git diff --check` 也通过。本轮门禁覆盖已整合的 `static_max` 代码、plan/provenance 元数据和测试；这是 CPU-only 验证，不是 GPU 验证，也没有 GPU matched-export 结论。
- GPQA 成绩必须同时具备 `summary.json`、198 个唯一题目、实际 CSV/seed/选项打乱
  策略匹配，并通过 `scripts/check_gpqa_results.py`；解析失败和长度结束计错。
- Spark 的视觉/MTP 功能检查、完整 256k GPQA、MSE 重复输出诊断和容器迁移分别绑定
  各自报告中的源码、镜像和权重身份；不能用 CPU 或合成 microprobe 结果替代。
- 当前 MSE 证据只确认 max/MSE 导出 scale 完整，且两者量化覆盖范围不匹配；R5 retry
  已完成但仍以 `length` 结束并检测到候选周期 1375，不能写成已修复循环或已定位根因。
  R6 仍为离线、未运行。不要在 PR 或 release 中写成“已找到 MSE 根因”或“已修复循环”。

## Pull request 状态

聚合入口为 [PR #70](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/70)。PR #65–#69
及 PR #70 的状态可能变化；合并前请以各 PR 页面当前 head 和 checks 为准，不能把旧 head
的成功结果转移给新提交。
