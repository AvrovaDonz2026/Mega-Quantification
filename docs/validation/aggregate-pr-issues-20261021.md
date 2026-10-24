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
| 当前分支 | NVFP4 `static_max` 控制 selector：静态 weight block scale、动态 input scale，runtime 仍为 ModelOpt `max` | CPU 门禁通过；同源 matched GPU PTQ 完成，见下方 R6 记录；两臂 scoped five-case serving smoke 已完成 | MTP/五题 smoke 不等于广泛视觉质量、完整 256k GPQA 或 MSE 重复修复；49,152-token natural 对照未完成 |
| [聚合 PR #70](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/70) | GPQA continuation 高置信重复前缀保护；新 journal 的 `prompt_sha256` 与 CSV prompt 交叉校验，旧行缺字段兼容并单次告警；服务返回 usage 时 repetition row 保留该次实际 usage；vision restore 按源 shard 名恢复 BF16 缺键 | CPU 测试覆盖 continuation/checker/resume 和 vision shard 恢复路径 | 重复 chunk 不追加文本；checker 接受 `repetition` + `truncated=true` 并计错。此评测保护及 CPU 权重文件检查均不证明 GPU MSE 根因已修复或整模视觉已验证 |

## 合并前门槛

- 当前验证记录的 Python 3.11 CPU 门禁为 `PYTHONPATH=src python -m pytest -q -o addopts=''`（使用隔离的 Python 3.11 环境），结果 `745 passed, 5 skipped in 24.47s`；此全量包含 1 个 R6 wrapper，该 wrapper 在隔离子进程中验证 167 项 R6 tests，不能把 167 再加到全量计数。定向 parity/CLI 检查为 `10 passed in 11.47s`，仅覆盖 bounded-ID 输入的诊断路径；`ruff check .` 和 `git diff --check` 通过。本轮 CPU 门禁覆盖已整合的 `static_max` 代码、plan/provenance 元数据和测试；GPU PTQ 证据另见下方记录。
- GPQA 成绩必须同时具备 `summary.json`、198 个唯一题目、实际 CSV/seed/选项打乱
  策略匹配，并通过 `scripts/check_gpqa_results.py`；解析失败和长度结束计错。
- Spark 的视觉/MTP 功能检查、完整 256k GPQA、MSE 重复输出诊断和容器迁移分别绑定
  各自报告中的源码、镜像和权重身份；不能用 CPU 或合成 microprobe 结果替代。
- 历史 R5 仍以 `length` 结束并检测到候选周期 1375，不能写成已修复循环或已定位根因。
  当前 R6 两组 27B 整模 PTQ 已完成：256 个校准 batch 顺序一致，导出前 quantizer scope 两臂一致（core 401、vision 110、MTP 8、activation 574），
  519 个权重 amax 两臂不同而 global amax 一致，574 个 activation 哈希一致，导出索引 header keys 校验通过；
  `ptq_comparable=true` 且 activation confounder=false。[脱敏完成记录](dgx-spark-matched-ptq-completed-20261022.json)
  记录两臂均 return code 0、未超时，GLM 已恢复，固定 arithmetic probe 通过。static-max replay 已完成，
  但 static-MSE 未 ready/generate，故无 paired replay 或跨臂 token 比较；serving readiness 与 MSE 根因未定。旧 [stage 预检](dgx-spark-matched-ptq-stage-preflight-20261022.json)
  与 [GPU microprobe](dgx-spark-matched-ptq-gpu-microprobe-20261022.json) 仅作历史辅助记录；固定 probe 不代表 GLM 整体健康。
  Fresh serving 尝试的首组 `static-max` 在 6.018 秒因只读文件系统启动失败，题目请求与 MSE 均未开始；两轮均成对恢复并通过 120 秒进程稳定检查。
  容器可写路径修复后，`static-max` 完成真实冷启动回放：49,152 tokens、`length`，末尾 1,229-token 周期的 suffix match 达 20,988 tokens。`static-MSE` 在服务 identity/readiness 前以 -9 退出，题目请求未开始；日志关键字分类不足以确定启动故障原因，故跨臂 token 比较仍未完成。两端恢复后通过 120 秒进程稳定检查及固定算术 probe。见[脱敏回放尝试记录](dgx-spark-matched-replay-attempt-20261022.json)。该例说明无 MSE 搜索也会循环，不排除 MSE 改变轨迹或循环概率；根因及修复均未确定。
- 2026-10-22 启动 probe 对上述 export 结论作出明确勘误：`static-MSE` 在 142.042 秒、生成前以 `AssertionError` 退出（`param_data=[1152,576]`、`loaded=[1152,1152]`）。header 审计为 max `519 U8 + 519 F8` 且 vision/MTP（110/8）packed；MSE `401 U8 + 401 F8` 且 vision/MTP（110/8）为 BF16。MSE 配置为 `uniformNVFP4`，未声明 aux `quantized_layers`，loader 因而预期 packed vision；`visionqkv` 为 MSE `3456x1152 BF16` 对 max `3456x576 U8`，无 duplicate keys。matched PTQ comparable 仅是 quantizer/calibration 的导出前证据，不能证明导出的 aux scope 相同。见[脱敏启动 probe JSON](dgx-spark-mse-startup-probe-20261022.json)；无题目请求、token 或质量结果。GLM 成对恢复、120 秒稳定和固定 arithmetic probe（answer 2、`stop`）通过；MSE 重复根因仍未确定。历史 dated 记录保留，不将此勘误改写为质量结论。
- 本轮 MSE 视觉/MTP 导出未满足请求的量化范围。已有索引/key 检查不能证明实际量化范围一致；
  完整量化 serving、256k MTP 质量、TP2 和广泛视觉质量仍待验证。

- 2026-10-23 的后续脱敏记录见 [aux dispatch validation](dgx-spark-aux-dispatch-20261022.md)
  和 [JSON evidence](dgx-spark-aux-dispatch-20261022.json)。patched full-27B 的
  `static-max` 与 `static-MSE` 均完成五题 baseline/MTP smoke；每个 arm 的 MTP
  verify/accept 均覆盖五题，baseline 与 MTP output IDs 在 arm 内一致（accepted
  drafts 为 136；proposed drafts 为 max 144、MSE 146；verify count 为 max 72、MSE
  73）。`static-MSE` 五题通过，`static-max` 的 OCR 仍为 3720 而非 3729，因此
  不能写成双臂质量通过或广泛视觉验证。此前只读文件系统启动失败及 MSE
  `AssertionError` 的 dated records 保持不变。
  已完成的 bounded 4,096-token natural/prefix probe 仍只覆盖其限定窗口。随后
  49,152-token natural 对照的 coordinator 以 `MaintenanceError` 失败，且 max
  仅有陈旧 progress 记录；static-MSE 从未开始，因此没有可用的长输出、周期或
  跨臂比较结果，不能推断周期变化或长程验证通过。独立 after-run readiness
  证据显示两端 journal 均 `armed=false`、cleanup `observed-absent`、restoration
  `process_stable`，原 GLM 均 Running、restart count 为 0 且无 OOM；服务 API
  arithmetic smoke 返回 `2`、`stop`（0.22 秒）。这些是恢复/可用性观察，不是
  long-generation 结果；transport/lifecycle interruption 的具体原因仍待调查。
  selected-layer v3 仅在一层 NVFP4 视觉算术中显示默认近似除法解释
  fake/export code mismatch，不是整模 MSE 重复根因或修复证据。

- 2026-10-23 的 selected-layer candidate 已由协调器以 `observation_passed` 完成。
  该隔离 CUDA helper 对同一视觉层的 `max`/`MSE` 两臂各验证 3,981,312 个
  quantization codes（含 signed zero）全部 exact，并验证各自 1,990,656 个
  packed bytes 全部一致；weight/scale hash 的前后检查一致。两端 journal 均
  `armed=false`、cleanup `observed-absent`、restoration `process_stable`，服务 API
  返回 `2`、`stop`（0.211 秒）。这只证明 selected-layer CUDA helper 的候选
  数值结果；尚未集成 production/live export，不证明 full-model parity、整模
  serving 行为、重复原因或修复。

- 2026-10-23 strict CUDA matrix + tiny resident-export integration：strict matrix
  的 7/7 cases 通过，覆盖 random FP16/BF16/FP32、threshold 邻域、小正值与符号、
  subnormal positive scale 和 noncontiguous input；共 352 个元素，code/packed
  mismatch 均为 0，输入保持不变，且无 skip/error。候选 helper SHA-256 为
  `75eabea892299caa2db008f29275687dee438328431ecd115dd05d0c30eab308`，上下文
  source SHA-256 为
  `b3d070842b59916b47d964af5158a72bedadf0f1407277e190a864544aaa887b`；使用既有
  pinned ModelOpt 0.47.0 image。与此分开的真实 resident tiny-Llama export 仅执行
  seeded `static-max` 单臂，因 `TypeError`（脱敏错误摘要 SHA-256
  `51d3d9f64a62a84face3e10218f1819436d142e4faea3ce72c216e21ffad2542`）在 tiny gate
  失败；临时 checkpoint 已清理。因此整体 integration 的 tiny gate 未通过，不能
  宣称 tiny export 成功或 max/MSE 配对。独立 readback 确认两端 journal 均
  `armed=false`、cleanup `observed-absent`、restoration `process_stable` 且无 failure；
  原 GLM 均运行中、restart count 为 0、无 OOM，实验容器不存在，服务 API smoke
  通过。v2 仅完成 staging、未执行，已由 v3 替代。v3 的 strict CUDA matrix 再次
  7/7 通过；真实 resident tiny-Llama export 已通过 codec、oracle/candidate 比较和
  audit 计数/alias gates，但在 serialized contract 的 header-set 校验处失败
  （`RuntimeError`）：v3 测试夹具的 header allowlist 只允许 3 项，漏掉 pinned
  ModelOpt schema 中合法的第 4 项 `input_scale`。临时 checkpoint 已清理，独立
  readback 确认恢复通过。该 header gate 失败仅表示 v3 当时的 tiny export 未通过；
  这是测试夹具契约问题，不能与 v4 的四字段 contract pass 混淆。以上仍不证明
  production/live export 集成、full-model parity、serving quality 或 repetition
  修复；PR72 保持 draft。

- 2026-10-23 v4 codec integration（job `27b8490b4767d5e082cc571f`，plan SHA-256
  `63bb37c924489b5c99e96fd8c8c43f8b979c1d67a66b2b7dd049d905d68db245`）重新通过
  strict CUDA matrix 7/7，code/packed mismatch 均为 0。seeded `static-max` 的
  resident tiny export 这次通过四字段 serialized contract：weight、weight scale、
  `weight_scale_2` 和 input scale；candidate 与 oracle/packed 结果 exact，输入与
  scales 保持不变，alias restore 通过，临时 checkpoint 已删除。该结果只覆盖
  tiny 单臂，且 streaming 标记为 unsupported；不包含整模或 max/MSE 双臂，因此不
  证明 27B/full-model parity、production serving quality 或 repetition 修复。
  独立 readback 显示实验容器 absent，双端 GLM Running、restart count 为 0、无 OOM，
  journal 为 `armed=false`、cleanup `observed-absent`、restoration `process_stable`；
  服务 API 返回 `2`、`stop`（0.222 秒）。v3 的 serialized header gate 失败保留为
  历史记录；v4 仅说明本次 tiny contract gate 已通过，PR72 仍保持 draft。

## Pull request 状态

2026-10-23 当前核验显示，公开未合并 PR 仅有 [PR #72](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/72)
（draft）和 [PR #73](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/73)
（writable TileLang cache）。PR72 的 main 基线为 merge state `CLEAN`；head
`1b0df53` 的两项 pytest checks 均为 SUCCESS。该 head/check 记录只适用于当前
head，后续任何新 commit 都必须重新核验，不能继承旧 checks。PR72/PR73 的开放状态
也不代表 full-model parity、serving quality 或 repetition 修复已成立。

诊断 worker 的 17 项独立复审未发现 blocking issue，但尚未 bundle、stage 或部署；
这只是审阅状态，不是运行时验证。
