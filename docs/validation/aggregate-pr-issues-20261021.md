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

- 当前验证记录的 Python 3.11 CPU 门禁为 `PYTHONPATH=src python -m pytest -q -o addopts=''`（使用隔离的 Python 3.11 环境），结果 `745 passed, 5 skipped in 24.47s`；此全量包含 1 个 R6 wrapper，该 wrapper 在隔离子进程中验证 167 项 R6 tests，不能把 167 再加到全量计数。定向 parity/CLI 检查为 `10 passed in 11.47s`，仅覆盖 bounded-ID 输入的诊断路径；`ruff check .` 和 `git diff --check` 通过。本轮门禁覆盖已整合的 `static_max` 代码、plan/provenance 元数据和测试；这是 CPU-only 验证，不是 GPU 验证，也没有 GPU matched-export 结论。
- GPQA 成绩必须同时具备 `summary.json`、198 个唯一题目、实际 CSV/seed/选项打乱
  策略匹配，并通过 `scripts/check_gpqa_results.py`；解析失败和长度结束计错。
- Spark 的视觉/MTP 功能检查、完整 256k GPQA、MSE 重复输出诊断和容器迁移分别绑定
  各自报告中的源码、镜像和权重身份；不能用 CPU 或合成 microprobe 结果替代。
- 当前 MSE 证据只确认 max/MSE 导出 scale 完整，且两者量化覆盖范围不匹配；R5 retry
  已完成但仍以 `length` 结束并检测到候选周期 1375，不能写成已修复循环或已定位根因。
  R6 GPU PTQ 尚未开始。Aitop SSH 只读检查已成功；两台 Spark 的 BF16 `config.json` 和 index 哈希一致，
  均有 18 个 shards，总计 55,563,006,776 bytes。Aitop 有 256 行多模态校准集，SHA256
  `e2a48a43e732995c36a2096dd95f2edbad3f13eaa85a042ced7094a149af6eed`，图像资产齐全；
  Robo 上找到的 256 行文本校准候选 SHA256 不同，不可替代此输入；Aitop 本轮只读检查确认
  原 PTQ 镜像与该校准集匹配。固定镜像中的 CPU smoke 已通过：未加载模型、未暴露 GPU；两臂
  均使用 `native_selector`：一臂为 `static_max`，另一臂为 `mse`；两臂都启用
  `quantize_vision` 和 `quantize_mtp`。
  两臂同源模型 tree SHA256 为 `b096b397facea42fdf80dffe7eed3fb5c7881959011379deedd2b370562cf516`，
  校准集 SHA256 为 `e2a48a43e732995c36a2096dd95f2edbad3f13eaa85a042ced7094a149af6eed`；使用
  ModelOpt 0.47.0、Transformers 5.12.1。最终公共脚本 payload 的固定镜像 CPU smoke 已通过，
  实验容器确认清理；脱敏记录见[preflight JSON](dgx-spark-matched-ptq-preflight-20261022.json)
  （package SHA256 `52f1156d1006d9389133fa825b75c6f3cd6d515cca475894d03d25f61931895b`）。
  这不是 GPU PTQ 或 GPU export。两端 GLM TP2 服务仍在运行，
  本轮预检未暂停。R6 的本地 DockerAdapter、确定性 bootstrap、strict maintenance-plan 转换，
  以及 mount/entrypoint/tmpfs/env/budget 一致性校验已实现；native_response loopback 客户端
  仅通过 mock 验证。R6 公共离线套件 `tests/test_matched_ptq_r6.py` 的 wrapper 已通过，
  共 167 项 offline tests passed；尚未部署或执行完整模型 GPU PTQ/serving，因此没有完整模型 GPU export。R6 公共脚本
  位于 `scripts/diagnostics/matched-ptq-r6/`；私有部署 glue 未公开，公共工具不提供远程安装器或部署命令。CPU preflight
  不代表真实 serving 或 GPU PTQ 已验证。远端 stage-only 预检的模型、校准、18-shard、source、
  transport 和 bundle 哈希校验已通过，但 worker 未安装、未启动，服务未修改；deployment adapter
  未执行 Docker，仅完成只读 inspect。脱敏字段见[stage preflight JSON](dgx-spark-matched-ptq-stage-preflight-20261022.json)。
  不能把旧 stage bundle 当作 worker 最终版本或 GPU 运行证据。维护器现已覆盖有限启动等待、每端心跳、
  慢 STOP、失败时仍尝试两端恢复，以及 watchdog 竞态修复；这些是生命周期控制改进，不是应用健康检查。
  进程稳定（`process_stable`）不等于应用 ready，当前无 healthcheck 时仍应标记 application readiness unknown。
  不要在 PR 或 release 中写成“已找到 MSE 根因”或
  “已修复循环”。
  第二次真实 GPU microprobe 已通过，实际使用 ModelOpt 0.47.0；合成 Linear64x32 的 static-max
  与 static-MSE activation amax 完全相同，weight scale 均 finite-positive 但 fake weights 不同，
  heldout relative MSE 分别为 0.01958167366683483 与 0.017615284770727158。GLM 前后身份、启动时间、
  restart count 一致且仍运行，实验容器已清理、无 OOM。脱敏统计见[GPU microprobe JSON](dgx-spark-matched-ptq-gpu-microprobe-20261022.json)。
  这是合成微探针，不是 27B benchmark、真实 serving 或完整 vision/MTP PTQ，不证明 MSE 重复输出已修复。
- 视觉/MTP 的公开证据限于导出文件审计、quantizer coverage 检查，以及固定文本、图像、
  OCR 和嵌入式 MTP 功能用例；这些检查不等同于完整 GPU PTQ、256k MTP 质量、TP2 或
  广泛视觉质量验证。MTP 与视觉的实际量化仍需在具备同源权重和校准数据的 GPU 环境中复核。

## Pull request 状态

聚合入口为 [PR #70](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/70)。本次检查时，
commit `4bbe205` 已推送至 PR #70，且此前两项 CI checks 均成功；PR #70 为 OPEN、MERGEABLE。
PR #65–#69 仍为 OPEN。后续新提交需要重新核验 CI；状态和 checks 会变化，合并前请以各 PR
页面当前 head 和 checks 为准，不能把旧 head 的成功结果转移给新提交。
