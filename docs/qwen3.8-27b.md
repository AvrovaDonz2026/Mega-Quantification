# Qwen3.8-27B

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 是 27B 稠密 VLM（`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`）。本页介绍模型、量化格式和通用操作；DGX Spark 的容器构建、图文/MTP 校准与评测命令集中在 [Spark 操作说明](DGX_SPARK.md)。当前版本的变更和验证范围见 [v0.1.3 发布说明](releases/v0.1.3.md)，历次实机证据见 [验证记录索引](validation/README.md)。

默认 `nvfp4_w4a8` 与 `nvfp4_mixed` 使用相同的混合层图：MLP 和 `lm_head` 是 NVFP4 group 16（权重和激活都是），self-attn 和 linear-attn 是 FP8。均匀 `nvfp4_w4a4` 把选中的语言模型线性层全部量化为 NVFP4 block 16，含注意力。默认配方保留 BF16 视觉塔和 MTP；Spark 配方同时量化语言、视觉和 MTP。BF16 外置 MTP helper 与 Spark 量化内嵌 MTP 的用法见 [MTP export](#mtp-export)。

| 任务 / 机器 | 从这里开始 |
|---|---|
| 选择量化方案与配方 | [Quantization format](#quantization-format)、[Recipes](#recipes)、[配方索引](../recipes/README.md) |
| 32 GB RTX 5090 PTQ | [Single 5090 + 64 GB RAM](#single-5090--64-gb-ram) |
| 5090 / 80 GB SM120 服务与 GPQA | [SGLang inference and GPQA](#sglang-inference-and-gpqa) |
| DGX Spark ARM64 / GB10 | [Spark 操作说明](DGX_SPARK.md)：[量化](DGX_SPARK.md#quantize-both-formats)、[视觉/MTP](DGX_SPARK.md#vision-and-mtp-inference)、[256k GPQA](DGX_SPARK.md#full-gpqa-diamond-at-temperature-0) |
| 查看完整成绩与验证边界 | [当前 Spark 结果](#sglang-inference-and-gpqa)、[验证记录索引](validation/README.md) |
| 上传已完成的导出 | [OSS publish](#oss-publish) |

## Contents

- [Quantization format](#quantization-format)
- [SGLang inference and GPQA](#sglang-inference-and-gpqa)
- [Model facts](#model-facts)
- [Recipes](#recipes)
- [Ignore list](#ignore-list-why-these-stay-bf16)
- [Mixed / W4A8 / W4A4 comparison](#mixed-vs-sglang-w4a8-vs-uniform-w4a4)
- [Calibration](#calibration)
- [Commands](#commands)
- [Single 5090 + 64 GB RAM](#single-5090--64-gb-ram)
- [MTP export](#mtp-export)
- [OSS publish](#oss-publish)
- [Serve](#serve-sglang--vllm--tensorrt-llm)
- [GPQA Diamond protocol](#gpqa-diamond-official-card-protocol)
- [Troubleshooting](#troubleshooting)

NVFP4 推理需要 Blackwell；校准可以在 Hopper 上多卡或 offload 完成。
代理从仓库根目录的 [SKILL.md](../SKILL.md) 读取硬约定。

Qwen3.5/3.8 量化使用 `transformers>=5.8,<5.15`，CPU 回归测试固定在
5.14.1。Transformers 4.x 缺少所需的模型接口。若启动时报告 GDN API 不兼容，
先重新安装仓库的 `.[hf,modelopt]` 依赖，再重跑；不要跳过错误继续校准。

## Quantization format

| 张量 | 权重 | 激活 | `quantized_layers` |
|---|---|---|---|
| `mlp.{gate,up,down}_proj` | NVFP4 E2M1，group_size 16，scale 为 FP8 E4M3 | NVFP4，group_size 16 | `quant_algo: NVFP4`, `group_size: 16` |
| `lm_head` | 同上 | 同上 | 同上 |
| `self_attn.{q,k,v,o}_proj` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | FP8 E4M3 | FP8 E4M3 | `quant_algo: FP8` |
| `linear_attn.conv1d` / `in_proj_a` / `in_proj_b` | BF16 | BF16 | 不写入 |
| 视觉塔的目标线性层 | 默认 BF16；Spark 配方启用 NVFP4 | 默认 BF16；Spark 配方启用 NVFP4 | 默认不写入；Spark 保留 ModelOpt 导出的量化张量 |
| MTP 的目标线性层 | 默认 BF16；Spark 混合 W4A8 为 FP8 注意力 + NVFP4 其他目标层 | 同权重布局 | 默认不写入；Spark 保留 ModelOpt 导出的量化张量 |
| embedding、norm | BF16 | BF16 | 不写入 |
| 推理时的 KV | | fp8_e4m3 | 配方 `kv_cache: fp8` |

上表是默认混合图，`hf_quant_config.json` 顶层为 `MIXED_PRECISION`，`quantized_layers` 非空，SGLang 0.5.20 按 `modelopt_mixed` 加载。若旧版混合导出仍是裸的 `NVFP4`，先跑 `megaquant rewrite-sglang <export_dir>`。均匀 W4A4 的顶层 `NVFP4` 是预期格式，见下表。ModelOpt 的 block 32 `W4A8_NVFP4_FP8` 标签不属于本仓库支持的方案。

| 方案 | 和上表的差别 | 导出 | 运行时 |
|---|---|---|---|
| `nvfp4_w4a4` | 选中的 LM 线性层全部是 NVFP4 W4A4 group 16，含注意力；Spark 的视觉/MTP 目标层也使用 NVFP4 | `quant_algo: NVFP4` | SGLang `modelopt_fp4` |
| `nvfp4_w4a16_mixed` | MLP + `lm_head` 为 NVFP4 group 16 权重、BF16 激活；注意力仍是 FP8 | MLP 条目 `quant_algo: W4A16_NVFP4` | 可选 Marlin 导出 |

## SGLang inference and GPQA

推理和 GPQA 都走 SGLang。评测客户端打 `http://127.0.0.1:30000/v1` 的 OpenAI chat。仓库里的 YAML 现在发 temperature 0，其余仍是 Qwen thinking 卡：`top_p=0.95`，`top_k=20`，`min_p=0`，`presence_penalty=0`，`repetition_penalty=1`，`enable_thinking` 与 `preserve_thinking`，`reasoning_effort=xhigh`。默认、5090、6000D 和 Spark 完整 GPQA 配方的上下文都是 262144；Spark 的独立视觉/MTP 功能验证配方保留 32768。`max_new_tokens: 0` 使用各自剩余窗口。`continue_on_length` 在窗口还有预算时续写，最多 8 次，不会扩大上下文；如果续写以高置信度重复已输出的前缀重新开始，客户端会停止追加该片段，将 `finish_reason` 设为 `repetition` 并保留 `truncated=true`；服务返回 usage 元数据时仍记录该次实际 usage。正式 checker 接受这种诊断组合后按严格规则计错：只有显式、非截断的 `stop` 才能答对。该门禁只防止重复内容继续放大评测轨迹，不代表已找到或修复 GPU MSE 根因。新 journal 行包含 `prompt_sha256`，恢复含该字段的 journal 时会用 CSV 重建题干指纹；旧 journal 缺字段仍兼容，但只发一次告警且无法进行题干一致性校验。`generation.http_timeout_seconds` 默认 86400 秒，每次 HTTP 请求（含各次续写）最多等待 24 小时，不是整套评测的总时限。完整轨迹默认在 `<model>/gpqa_diamond/gpqa_diamond.jsonl`，旁边有 `summary.json`，用 `--output` 可指定独立目录，按 `item_id` 续写。`summary.json` 存在、198 个唯一题目齐全并通过独立校验后，分数才是 `correct/198`。答案解析要求显式 `Answer: A` 至 `Answer: D`、等价 boxed 答案或单独答案字母；选项讨论中的字母不作为答案，整行反引号包裹的 Answer 格式也可解析。解析规则更新不重写既有 journal 或历史成绩。截断和解析失败算错。

GPQA journal 只应在相同的权重、CSV、seed、选项打乱策略、上下文和服务参数下续跑；修改任一输入或评测协议时使用新的 `--output` 目录。程序会按 `item_id` 复用已有行，并在追加前截掉最后一个不完整的 JSONL 行；中间损坏的行会使评测失败，避免静默跳过证据。新行的 `prompt_sha256` 必须匹配当前 CSV 题干；旧行缺少该字段时保留兼容性并发出一次告警。

### 当前 Spark 完整结果

四份 Spark 导出均完成 198 题并通过正式结果校验。它们使用量化视觉/MTP，
temperature 0、262144 上下文、GB10 TP1、FlashInfer、16 并发、seed 0、
选项洗牌、保留 `xhigh` thinking；GPQA 期间关闭 MTP。

| Spark 导出 | 正确 / 198 | 准确率 | 最终 length | 解析失败 | 完整报告 |
|---|---:|---:|---:|---:|---|
| 混合 W4A8（max） | 177 | 89.39% | 0 | 4 | [基线](validation/dgx-spark-gpqa-256k-20261002.md) |
| 均匀 W4A4（max） | 174 | 87.88% | 0 | 4 | [基线](validation/dgx-spark-gpqa-256k-20261002.md) |
| 均匀 W4A4（MSE） | 178 | 89.90% | 3 | 4 | [校准实验](validation/dgx-spark-w4a4-quality-20261008.md) |
| 均匀 W4A4（Local-Hessian） | 170 | 85.86% | 3 | 5 | [校准实验](validation/dgx-spark-w4a4-quality-20261008.md) |

MSE 和 Local-Hessian 的三条最终 length 均包含在解析失败数中，两列不能相加。
MSE 在这次 GPQA 比 max 多答对四题，Local-Hessian 少答对四题；泛化质量改善
和停止行为恢复尚未证实。两个实验方案保留为可选配方，默认仍是混合 W4A8，
已评测的 max 导出继续保留。

实验导出还通过完整 27B PTQ、严格导出审计以及独立 32k 文本/图像/OCR/MTP
功能检查；每份导出在普通 decode 与嵌入式 MTP 间匹配 205/205 输出 token。
这些固定用例不代表广泛视觉质量、256k MTP 质量、吞吐提升或 TP2 验证。
GPU 运行的实际源码版本、镜像摘要和耗时范围保留在日期报告中；
[v0.1.3 发布说明](releases/v0.1.3.md)说明发布包与 GPU 证据的版本关系。

### 历史测量与诊断

| 权重 | 分数 | 协议与日期 |
|---|---|---|
| 混合 / 默认 W4A8（5090 的 max + ultrachat） | 178/198；截断 0，解析失败 0 | temperature 1.0，80 GB SM120，SGLang 64 路，KV 在 GPU；2026-09-22 |
| 均匀 W4A4 | 172/198；截断 4，解析失败 6 | temperature 0，80 GB SM120，Marlin；2026-09-22 |
| Spark 混合 W4A8 的 32k 试跑 | 157/198（79.29%）；截断 30，解析失败 32，含重叠 | temperature 0，GB10 TP1，FlashInfer，16 并发，MTP 关闭；2026-10-02，[证据](validation/dgx-spark-gpqa-w4a8-20261002.json) |

这些结果的权重、后端或上下文不同，不能并入当前 Spark 成绩。32k Spark
W4A4 试跑已停止，没有完整成绩。三条 W4A8 无最终答案响应的独立诊断重试
得到两条正确、一条错误；并发为 3，不能替换正式 16 并发结果或证明逐 token
相同。停止 token 缺失，量化/kernel 原因尚未确定，见
[解析失败诊断](validation/dgx-spark-gpqa-unparsed-20261003.md)。

### 按硬件选择服务配方

Spark 完整评测使用
[`eval-gpqa-diamond.spark-gpqa.yaml`](../recipes/eval-gpqa-diamond.spark-gpqa.yaml)：
256k、16 并发、64 个 FP32 SSM slot、MTP 关闭；独立视觉/MTP 检查使用
32k、4 并发配方。完整启动、顺序切换导出及验分命令见
[Spark GPQA 操作说明](DGX_SPARK.md#full-gpqa-diamond-at-temperature-0)。
下表保留 x86 机器的配置差异。

| | 默认 | 32 GB SM120（5090） | 80 GB SM120（6000D） |
|---|---|---|---|
| 配方 | `eval-gpqa-diamond.yaml` | `eval-gpqa-diamond.5090.yaml` | `eval-gpqa-diamond.6000d.yaml` |
| 注意力 | FlashInfer | Triton | Triton |
| NVFP4 GEMM | FlashInfer（CUDA ≥ 12.9） | Marlin | Marlin |
| FP8 GEMM | FlashInfer | Triton，`SGLANG_FORCE_FP8_MARLIN=1` | CUTLASS，`SGLANG_FORCE_FP8_MARLIN=1` |
| CUDA graph | 关 | 关（32 GB 上抓 graph 会 OOM） | 开 |
| KV | HiCache，主机 12 GiB | HiCache，主机 64 GiB，按 GPU 上 KV / GDN 池的比例切 | 只在 GPU |
| 并发 | 1 | 24（bf16 GDN，96 slot） | 64（bf16 GDN，256 slot） |
| `mem-fraction-static` | 0.85 | 0.95 | 0.90 |
| chunked prefill | 2048 | 2048 | 4096 |
| 额外环境变量 | | `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` | `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1`，`SGLANG_IS_FLASHINFER_AVAILABLE=0`，`SGLANG_ENABLE_JIT_DEEPGEMM=0` |
| Radix cache | 开 | 开 | 关 |

CUDA 12.8 编不了 SM 12.x 的 FlashInfer JIT。5090 和 80 GB 配方都把 SiLU+FP4 融合关掉，因为那条路径即使 GEMM 走 Marlin，import 时仍会拉 FlashInfer。float32、64 个 mamba slot 大约占 9.3 GB HBM，GPU 上的 KV 剩不到 1 GB，16 路 HTTP 会排在 3 到 4 个 decode slot 后面。

```bash
# 80 GB SM120，CUDA 12.8
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.6000d.yaml \
  --model outputs/Qwen3.8-27B-NVFP4-W4A8
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.6000d.yaml \
  --base-url http://127.0.0.1:30000/v1

# 32 GB 5090
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.5090.yaml \
  --model outputs/Qwen3.8-27B-NVFP4-W4A8
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.5090.yaml \
  --base-url http://127.0.0.1:30000/v1
```

`megaquant serve` 先把配方里的环境变量导出，再执行 `sglang serve`。`--dry-run` 只打印 argv，不会下载 `Qwen/Qwen3.8-27B`。要和 GB300 上的 vLLM 卡片对照时，加 `--engine vllm`。

## Model facts

| | |
|---|---|
| Hugging Face id | `Qwen/Qwen3.8-27B` |
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` (`text_config.model_type`: `qwen3_5_text`) |
| Size / shape | 27B dense VLM, hidden 5120, 64 layers, FFN 17408 |
| Block layout | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| Extra modules | Vision encoder + MTP |
| Native context | 262144 |
| Public mixed ckpt | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |

Family adapter: `qwen3_5` (`src/megaquant/models/qwen3_5.py`). This is **not**
vanilla Qwen3 (`qwen3` / `qwen3_moe`).

## Recipes

| File | Scheme | Algorithm | Calib | Output |
|---|---|---|---|---|
| `recipes/qwen3.8-27b-nvfp4-w4a8.yaml` | `nvfp4_w4a8` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A8` |
| `recipes/qwen3.8-27b-nvfp4-w4a8.5090.yaml` | `nvfp4_w4a8` | `max` | ultrachat 256×1024, **batch 1** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-w4a8.public-calib.yaml` | `nvfp4_w4a8` | `max` | ultrachat 512 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-w4a8.spark.yaml` | `nvfp4_w4a8` | `max` | local `/data/calibration.jsonl`, 256 × 1024, batch 1 | `outputs/Qwen3.8-27B-NVFP4-W4A8-spark`; mixed W4A8 + vision/MTP |
| `recipes/qwen3.8-27b-nvfp4-w4a4.yaml` | `nvfp4_w4a4` | `max` | 512 | `outputs/Qwen3.8-27B-NVFP4-W4A4` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.5090.yaml` | `nvfp4_w4a4` | `max` | ultrachat 256×1024, **batch 1** | same, packed for 32 GB + 64 GB RAM |
| `recipes/qwen3.8-27b-nvfp4-w4a4.public-calib.yaml` | `nvfp4_w4a4` | `max` | ultrachat 512 (anonymous Hub) | same |
| `recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml` | `nvfp4_w4a4` | `max` | local `/data/calibration.jsonl`, 256 × 1024, batch 1 | `outputs/Qwen3.8-27B-NVFP4-W4A4-spark`; quantizes vision + MTP |
| `recipes/qwen3.8-27b-nvfp4-w4a4.spark-mse.yaml` | `nvfp4_w4a4` | `mse`（可选实验） | same Spark 256 × 1024, batch 1 | `outputs/Qwen3.8-27B-NVFP4-W4A4-spark-mse` |
| `recipes/qwen3.8-27b-nvfp4-w4a4.spark-hessian.yaml` | `nvfp4_w4a4` | `local_hessian`（可选实验） | same Spark 256 × 1024, batch 1 | `outputs/Qwen3.8-27B-NVFP4-W4A4-spark-hessian` |
| `recipes/qwen3.8-27b-nvfp4-mixed.yaml` | `nvfp4_mixed` | `local_hessian` | **2048**, `nvidia/Nemotron-Post-Training-Dataset-v3` | `outputs/Qwen3.8-27B-NVFP4-mixed` |
| `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml` | `nvfp4_mixed` | `max` | ultrachat 256×1024, **batch 1** | `outputs/Qwen3.8-27B-NVFP4-W4A8`; restores BF16 vision + MTP |
| `recipes/qwen3.8-27b-nvfp4-mixed.public-calib.yaml` | `nvfp4_mixed` | `max` | ultrachat 512 (anonymous Hub) | `outputs/Qwen3.8-27B-NVFP4-mixed` |
| `recipes/qwen3.8-27b-nvfp4-w4a16-mixed.5090.yaml` | `nvfp4_w4a16_mixed` | `max` | ultrachat 256×1024, **batch 1** | `outputs/Qwen3.8-27B-NVFP4-W4A16-mixed`; optional Marlin export, not the Spark fast path |

All default and 5090 recipes set `backend: modelopt`, `kv_cache: fp8`,
`family: qwen3_5`, `model.quantize_vision: false`, and `model.quantize_mtp: false`.
The Spark recipe sets both flags to `true` and leaves `kv_cache` unset because
ModelOpt 0.47 cannot export mixed KV-cache quantizers together with quantized
Qwen3.5 vision attention. SGLang can still select FP8 KV at serve time.

Default W4A8 and uniform W4A4 recipes (`w4a8.yaml`, `w4a4.yaml`) use
`nvidia/Nemotron-Post-Training-Dataset-v2` (gated; set `HF_TOKEN`). Mixed
**quality** (`mixed.yaml`) follows the NVIDIA card and uses
**`nvidia/Nemotron-Post-Training-Dataset-v3`**. Anonymous Hub access: the
`*.public-calib.yaml` files and all `*.5090.yaml` files use
`HuggingFaceH4/ultrachat_200k`.

## Ignore list (why these stay BF16)

`Qwen35Family.default_ignore` (unless the recipe flips the flags):

| Pattern | Why |
|---|---|
| `*visual*`, `*vision*` | Vision encoder is not the language-model PTQ target. NVIDIA mixed PTQ also left it in BF16. Set `model.quantize_vision: true` to include it. |
| `*embed_tokens*`, `*embed_positions*` | Embedding tables are poor NVFP4 candidates; keep BF16. |
| `*linear_attn.conv1d*` | Gated DeltaNet depthwise conv — not a standard Linear GEMM; ModelOpt / compressed-tensors NVFP4 paths do not treat it as `q/k/v/o` or `gate/up/down`. |
| `*linear_attn.in_proj_a*`, `*linear_attn.in_proj_b*` | GDN extras (not `in_proj_qkv` / `in_proj_z` / `out_proj`). Mixed FP8 attention still quantizes the real GDN projections; these two stay BF16. |
| `*mtp*` | Multi-Token Prediction stays **BF16 in the default export** (`mtp.safetensors`). `quantize_mtp: false` means do not quantize it, not delete it; the Spark recipe opts in to MTP quantization. See [MTP export](#mtp-export). |
| **not** `*mlp*` | MLP `gate/up/down_proj` are the main NVFP4 targets. |
| **not** `*lm_head*` | NVIDIA mixed NVFP4 quantizes `lm_head`. Default W4A8 and uniform W4A4 do too. Add it in `extra_ignore` only if you want BF16 logits. |

Norms are not `Linear` and stay unquantized automatically.

## Mixed vs SGLang W4A8 vs uniform W4A4

三列是三份配方，比特布局见 [Quantization format](#quantization-format)。`nvfp4_w4a8` 和 `nvfp4_mixed` 导出同一张混合图；质量配方把校准换成 Local-Hessian。

| | SGLang W4A8 (default) | Uniform W4A4 | Mixed quality |
|---|---|---|---|
| Scheme | `nvfp4_w4a8` | `nvfp4_w4a4` | `nvfp4_mixed` |
| ModelOpt | mixed overrides + `MIXED_PRECISION` rewrite | `mtq.NVFP4_DEFAULT_CFG` / `nvfp4` | same mixed cfg as W4A8; Local-Hessian |
| NVFP4 group / block | 16 on MLP + `lm_head` | 16 uniform | 16 on MLP + `lm_head` |
| Weights / activations | NVFP4 gs16 on MLP + `lm_head`; FP8 on self-attn + linear-attn | NVFP4 weights and activations (block 16) | same map as SGLang W4A8 |
| SGLang | `modelopt_mixed` (`MIXED_PRECISION`) | `modelopt_fp4` (`NVFP4`) | `modelopt_mixed` |
| Public HF id | — | — | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |

NVIDIA 公开卡上的其他基准（vLLM，262k，混合权重）：

| Benchmark | BF16 | NVFP4 mixed |
|---|---|---|
| GPQA Diamond | 88.92 | 88.01 |
| Terminal-Bench | 75.56 | 74.02 |
| AA-LCR | 72.63 | 73.38 |
| MMMU-Pro | 75.14 | 74.86 |
| SciCode | 47.93 | 48.41 |
| IFBench | 80.07 | 78.93 |

本仓库自己的 GPQA 在 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。

## Calibration

| | SGLang W4A8 | Uniform W4A4 | Mixed (NVIDIA quality) | 5090 packed (`*.5090.yaml`) |
|---|---|---|---|---|
| Recipe | `qwen3.8-27b-nvfp4-w4a8.yaml` | `qwen3.8-27b-nvfp4-w4a4.yaml` | `qwen3.8-27b-nvfp4-mixed.yaml` | `*.5090.yaml` for `w4a8` / `w4a4` / `mixed` |
| Algorithm | `max` | `max` | `local_hessian` (`fp8_scale_sweep: true` in ModelOpt) | `max` |
| Samples | 512 | 512 | 2048 | 256 × 1024, **batch 1** |
| Dataset | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v2` | `nvidia/Nemotron-Post-Training-Dataset-v3` | `HuggingFaceH4/ultrachat_200k` |
| Images | `with_images: false` (text-only calibration; BF16 vision weights restored after export) | same | same | same |

The Spark recipes use 256 samples and require `with_images: true`. Local JSONL
rows can contain `image`/`images` paths relative to the JSONL, plus `text` or
chat `messages`. Text-only rows may be mixed in. `AutoProcessor` prepares image
tensors and image tokens; oversize image sequences fail instead of being truncated.
The standalone MTP block runs on the target model's hidden states and next-token
embeddings during calibration. Export requires every enabled vision/MTP input
quantizer to execute and records the counts in `calibration_coverage.json`.
Embeddings and the vision patch-embedding convolution remain BF16.

Prepare a reproducible 128-image + 128-text calibration set with
`scripts/prepare_multimodal_calibration.py`; see [Spark container commands](../docker/README.md#dgx-spark--gb10).

完整 27B 的 Spark W4A8/W4A4 已于 2026-09-29 通过各 256 条图文校准和
HF 导出，视觉及 MTP 均启用量化；张量、分片索引和 scale 检查通过。
具体环境和限制见 [实机验证记录](validation/dgx-spark-pr46-20260929.md)。

Spark 推理使用独立的 `Dockerfile.sglang.spark` 和
`recipes/eval-gpqa-diamond.spark.yaml`，入口是
`docker compose --profile spark up serve-sglang-spark`。镜像固定 ARM64/CUDA 13
SGLang 0.5.20，包含量化视觉/MTP 加载兼容补丁。初始为 TP1、32k 上下文、
4 并发、0.70 内存比例、关闭 HiCache/CUDA graph；MTP 需要在文本/图像验证后
单独开启配方中的 `speculative_*` 参数。校准覆盖与服务实测是两个独立结果。
构建、检查和评测命令见 [Spark SGLang 服务](../docker/README.md#spark-sglang-服务)。
2026-09-30 的[视觉/MTP 服务验证](validation/dgx-spark-serving-20260930.md)
包含红蓝图、OCR 和嵌入式 MTP 的逐 token 对照。Spark 配方现在使用 FP32
SSM 状态，修复已复现的 W4A4 投机验证输出分歧；权重和激活仍为 W4A4。
Calibration coverage verifies execution, not model quality or speculative-decoding
acceptance rate; those still need inference evaluation.

`max` 记下激活的 amax，再 round-to-nearest。它是 SGLang W4A8、均匀 W4A4，以及 5090 混合配方的默认算法。Local-Hessian（`fp8_scale_sweep: true`，Hessian `block_size` 16）是 NVIDIA 混合卡用的质量档，2048 条在 32 GB 上放不下。5090 上跑完的是 `mixed.5090.yaml`（`max` + ultrachat 256）。`mixed.yaml` 留给更大的卡。不改 YAML 也可以临时换算法：

```bash
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml \
  --algorithm local_hessian --num-samples 2048
```

Spark 的可选 W4A4 MSE / Local-Hessian 实验使用同一份 256 条图文校准集，
只改变权重 scale 搜索与导出目录。搜索要求静态 NVFP4 权重 block scale 和
`fp8_scale_sweep=true`；输入 scale 仍为动态，激活校准仍是 max，
NVFP4 group 16 格式不变。仅更改算法名称不足以证明 ModelOpt 执行了搜索。
按 [Spark 校准实验](DGX_SPARK.md#controlled-w4a4-calibration-experiments)
先运行 GPU operator gate，再做完整 PTQ、导出审计、视觉/MTP 检查和独立 GPQA。
当前成绩见 [结果表](#当前-spark-完整结果)。

## Commands

```bash
pip install -e '.[hf,modelopt]'

# No GPU, no 27B download:
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
python -m megaquant.cli plan -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml

# PTQ (Hopper or better, multi-GPU / CPU offload for 27B BF16):
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
# 5090 production (max + ultrachat; not Hessian):
MEGAQUANT_LOW_MEMORY=1 megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
# DGX Spark GB10 (language + vision + MTP; see docker/README.md):
MEGAQUANT_LOW_MEMORY=1 megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml
MEGAQUANT_LOW_MEMORY=1 megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.spark.yaml
```

Export is a Hugging Face unified checkpoint
(`modelopt.torch.export.export_hf_checkpoint` on the ModelOpt path, then
`rewrite_sglang_mixed_export` for `nvfp4_w4a8` / `nvfp4_mixed`). Destination
is `export.output_dir`. The pipeline also writes `provenance.json` (base
model, scheme, backend, calib, git sha, timestamp).

To patch an already-exported mixed / default-W4A8 directory without
re-running PTQ:

```bash
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-W4A8
megaquant rewrite-sglang outputs/Qwen3.8-27B-NVFP4-mixed
```

ModelOpt 0.46 经常只写一个裸的 `NVFP4` 标签、没有 `quantized_layers`。这种目录在 `sglang serve` 之前先 rewrite。要加载的是 `MIXED_PRECISION` 那份导出。

27B BF16 ≈ 54 GiB of weights plus activations. Set `model.device_map` (`auto`
by default) or CUDA_VISIBLE_DEVICES; expect multiple 80 GB Hopper GPUs or
heavy CPU offload. NVFP4 **inference** still needs Blackwell SM100+.

## Single 5090 + 64 GB RAM

生产配方为
[`qwen3.8-27b-nvfp4-mixed.5090.yaml`](../recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml)：
ModelOpt 0.46.1、max、ultrachat 256×1024、batch 1。混合导出应有 401 个
`quantized_layers`（193 个 NVFP4、208 个 FP8）；跳过量化的视觉/MTP 分别
恢复为 BF16 `vision.safetensors`、`mtp.safetensors` 并加入 HF index。

The 27B BF16 does not fit in 32 GB. Mega-Quantification packs the box
instead of leaving headroom idle:

| Resource | Packing |
|---|---|
| GPU | Weights fill **VRAM − 1 GiB** (`MEGAQUANT_GPU_HEADROOM_GIB`). |
| RAM | Weights that do not fit on GPU stay in **MemTotal − 6 GiB** (~56 GiB on a 64 GB pod). Disk `offload_folder` is spill-only. |
| CPU | `nproc` threads via `OMP_NUM_THREADS` / `torch.set_num_threads`. Calib tensors are `pin_memory`'d. |
| PCIe | Idle 5090 reports **gen1 x16**. PTQ runs a pinned H2D/D2H warmup so the link trains to **gen5 x16** (~50 GiB/s DMA). CPU-resident weights are pinned; accelerate copies use `non_blocking=True`; calib prefetches the next batch on a CUDA copy stream. `CUDA_DEVICE_MAX_CONNECTIONS=16`. |
| Calib | The mixed 5090 recipe uses `batch_size: 1`; batch 4 can OOM at `lm_head` on a 32 GB instance. |

The earlier batch-4 run produced `outputs/Qwen3.8-27B-NVFP4-W4A8` on one
5090, but cards that report less free VRAM, or a desktop session holding some of
it, can run out at the calib forward through `lm_head`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.62 GiB.
...
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

Batch 1 calibrates the same 256 samples one at a time, so calib takes several
times longer, and the scales can differ from the batch-4 export in the last
bits. Compose passes the variable through:

```bash
MEGAQUANT_LOW_MEMORY=1 megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
MEGAQUANT_BATCH_SIZE=1 docker compose --profile gpu run --rm mixed
```

K8s GPU pod (already a container). Scheme argument selects the
5090-packed recipe (`w4a8` default):

```bash
bash scripts/gpu-pod.sh plan
bash scripts/gpu-pod.sh quantize
bash scripts/gpu-pod.sh plan w4a4
bash scripts/gpu-pod.sh quantize w4a4
bash scripts/gpu-pod.sh plan mixed
bash scripts/gpu-pod.sh quantize mixed
```

Usage is `scripts/gpu-pod.sh plan|quantize|publish|rewrite-sglang|serve|eval [w4a8|w4a4|mixed]`.
These commands start a job; they do not imply that W4A4 / mixed PTQ or an
OSS upload has already finished on the pod.

Compose on a machine that has Docker:

```bash
docker compose --profile gpu run --rm quantize
docker compose --profile gpu run --rm w4a4
docker compose --profile gpu run --rm mixed
```

The `mixed` Compose service uses `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`
(max, ultrachat 256×1024, batch 1). NVIDIA quality Local-Hessian:

```bash
MIXED_RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

Install GDN fused kernels for the next run (`kernels` + `flash-linear-attention`);
without them transformers falls back to a PyTorch Gated DeltaNet and SM% stays
single-digit. Do not compile `causal-conv1d` while another PTQ is on the GPU.

Caps if you need them: `MEGAQUANT_MAX_MEMORY=0:29GiB,cpu:56GiB`,
`MEGAQUANT_NUM_THREADS`, `MEGAQUANT_BATCH_SIZE`, `MEGAQUANT_GPU_HEADROOM_GIB`.

## MTP export

### Spark 量化内嵌 MTP

Spark 配方设置 `model.quantize_vision: true` 和 `model.quantize_mtp: true`。
ModelOpt 将量化 MTP 直接导出到完整 checkpoint 的 HF index，保留对应量化
张量；这条路径不恢复 BF16 MTP，也不自动生成外置 `*-draft` helper。
Spark 服务镜像包含配套的视觉/MTP loader 补丁。

[`eval-gpqa-diamond.spark-mtp.yaml`](../recipes/eval-gpqa-diamond.spark-mtp.yaml)
用 `serve.speculative_algorithm: EAGLE` 及其余 `serve.speculative_*` 字段开启
内嵌 MTP。`megaquant serve` 会将这些字段生成 SGLang 参数，不添加
`--speculative-draft-model-path`。启动、图像/OCR、draft 接受和逐 token
对照命令见 [Spark 视觉/MTP 验证](DGX_SPARK.md#vision-and-mtp-inference)。
这套功能检查使用 32k；完整 256k GPQA 的服务配方关闭 MTP。

### 默认 BF16 导出与外置 helper

Language-only PTQ skips constructing the Qwen3.5 vision tower. After export,
the backend restores all BF16 `model.visual.*` tensors from the source into
`vision.safetensors`, adds them to the HF index, copies the image/video processor
configs, and sets `language_model_only: false`. For an older export that lost
the vision tower, run:

```bash
megaquant restore-vision outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
```

`quantize_mtp: false` keeps MTP in BF16. ModelOpt `export_hf_checkpoint`
often drops CPU-pinned `mtp.*`, so the backend copies them back after
export:

1. In-memory `mtp` module if it is still on the quantized model
2. Else the original HF source — **only** shards whose index entries are
   `mtp.*` (Qwen3.8: the last BF16 shard). Never the full 27B.

The tensors land in `mtp.safetensors` and stay out of `quantized_layers`
(`exclude_modules` already lists `mtp*`). SGLang's Qwen3.5 target loader
skips `mtp`; the draft loader wants those exact top-level `mtp.*` names.

SGLang 0.5.20 does not shrink Qwen3.5 hybrid depth for the draft
`ModelConfig`, so export also writes a sibling **1-layer** directory
`<export_dir>-draft`. Point speculative decode there:

```bash
# already-exported checkpoint that lost MTP:
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
# or just the draft helper:
megaquant write-mtp-draft outputs/Qwen3.8-27B-NVFP4-W4A8

# 32 GB 5090 上的文本和图片功能冒烟测试（2048 token 上下文）:
export SGLANG_FORCE_FP8_MARLIN=1
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
sglang serve --model-path outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path outputs/Qwen3.8-27B-NVFP4-W4A8-draft \
  --speculative-draft-model-quantization unquant \
  --speculative-num-steps 1 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --quantization modelopt --context-length 2048 --mem-fraction-static 0.94 \
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 512 \
  --attention-backend triton --linear-attn-backend triton \
  --sampling-backend pytorch --fp4-gemm-backend marlin --fp8-gemm-backend triton \
  --mamba-full-memory-ratio 4.59 --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 8 \
  --max-running-requests 1 --disable-cuda-graph --disable-flashinfer-autotune
```

这条 BF16 helper 路径的 `--speculative-draft-model-path` 指向 `*-draft`，
64 层目录是 target。BF16 draft 要明确指定
`--speculative-draft-model-quantization unquant`，否则 SGLang 会继承 target
的 ModelOpt 量化设置。CLI 配方支持内嵌 MTP 的 `serve.speculative_*`，
外置 helper 的路径和量化方式按上面的 `sglang serve` 命令显式指定。
2048 token 配置用于功能冒烟测试；长上下文 GPQA 使用独立服务配方。
上传 `w4a8` / `w4a4` / `mixed` 时使用完整导出，`*-draft` 仅为旁边的 1 层助手。

## OSS publish

After a scheme finishes, upload the export as
`<prefix>/<scheme>/<content-hash>/` (`w4a8` / `w4a4` / `mixed`).
`<content-hash>` is SHA256 of sorted `{safetensors-name} {sha256}\\n` lines.
Bucket and endpoint come from `OSS_BUCKET` / `OSS_ENDPOINT` or `--bucket` /
`--endpoint` (optional gitignored `.oss.env`). Objects larger than 5 GiB use
multipart upload. Real uploads need `oss2`: `pip install oss2` or
`pip install megaquant[oss]`. `--dry-run` does not.

混合导出就是默认 W4A8。同一份目录用 `--scheme mixed` 和 `--scheme w4a8` 各发一次，content-hash 相同，不用再跑一遍 PTQ。`w4a8` 对应的是 `MIXED_PRECISION` 导出。

```bash
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A4 --scheme w4a4
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A8 --scheme mixed
python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A8 --scheme w4a8
# intended gpu-pod wrapper (same env / prefix):
bash scripts/gpu-pod.sh publish w4a4
bash scripts/gpu-pod.sh publish mixed
bash scripts/gpu-pod.sh publish w4a8
```

Local-Hessian `mixed.yaml` still writes `outputs/Qwen3.8-27B-NVFP4-mixed`. Publish that directory the same way if that is the export you ran.

To fetch a published export, use the same scheme and 64-character content hash:

```bash
python scripts/oss_fetch.py --scheme w4a4 --hash <content-hash> \
  --dest outputs/Qwen3.8-27B-NVFP4-W4A4
```

The fetcher writes `SHA256SUMS.txt` and, when present, `oss_manifest.json` beside
the downloaded files. Verify the payload before serving or republishing it:

```bash
(cd outputs/Qwen3.8-27B-NVFP4-W4A4 && sha256sum -c SHA256SUMS.txt)
```

Keep the content hash and the checksum result with the run metadata. Bucket and
endpoint are supplied by flags or the gitignored `.oss.env`; no bucket name or
credential belongs in a recipe or commit.

## Serve (SGLang / vLLM / TensorRT-LLM)

Default inference is **SGLang**. `megaquant serve --dry-run` prints the
resolved argv (cookbook flags + HiCache size). Point `--model-path` at
your export dir or at `nvidia/Qwen3.8-27B-NVFP4`.

### SGLang

Use the pinned `sglang==0.5.20` (`docker/requirements-sglang.txt`). It
routes `MIXED_PRECISION` + NVFP4 `quantized_layers` to `modelopt_mixed`
through [sgl-project/sglang#27906](https://github.com/sgl-project/sglang/pull/27906)
(Qwen3.6 ModelOpt mixed NVFP4) and
[#30443](https://github.com/sgl-project/sglang/pull/30443) (`modelopt_mixed`
with the flashinfer_cutedsl MoE runner). Both are in the `v0.5.20` tag.
#28099 and #29769 were closed without merging, so do not bisect against them.
Older builds treated non-NemotronH `MIXED_PRECISION` as `w4afp8` and will
not run this checkpoint.

```sh
# Recipe-built argv (adds HiCache so 262k fits a 32 GB card):
bash scripts/serve_sglang.sh outputs/Qwen3.8-27B-NVFP4-W4A8
# or: python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
#        --model outputs/Qwen3.8-27B-NVFP4-W4A8

sglang serve \
  --trust-remote-code \
  --model-path outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --quantization modelopt \
  --tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype float32 \
  --attention-backend flashinfer \
  --context-length 262144 \
  --disable-cuda-graph \
  --enable-hierarchical-cache \
  --hicache-size 12 \
  --host 0.0.0.0 \
  --port 30000
```

The block above is the cookbook default (`eval-gpqa-diamond.yaml`). The 32 GB and 80 GB SM120 boxes use `eval-gpqa-diamond.5090.yaml` and `eval-gpqa-diamond.6000d.yaml`. See [SGLang inference and GPQA](#sglang-inference-and-gpqa).

Docker: use `make serve-sglang` with the repository's pinned dependencies;
Spark uses its separate [ARM64 serving image](DGX_SPARK.md#what-the-images-contain).
For an old mixed / default-W4A8 export with a bare `NVFP4` tag and no
`quantized_layers`, run `megaquant rewrite-sglang <export_dir>` first.
Uniform W4A4 retains `NVFP4`; `W4A8_NVFP4_FP8` is unsupported. GPQA client:
`MEGAQUANT_SGLANG_BASE_URL` (default `http://127.0.0.1:30000/v1`).

### vLLM

Optional (`--engine vllm`). NVIDIA mixed card on GB300 used these flags:

```sh
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
  --engine vllm --model outputs/Qwen3.8-27B-NVFP4-W4A8
```

### TensorRT-LLM

NVIDIA documented vLLM + SGLang for the 27B NVFP4 checkpoint. TRT-LLM's
published Qwen3.8 guides currently target the **2.4T MoE**, not this 27B dense
VLM — treat the snippet as a starting point on Blackwell, not a certified
profile:

```sh
trtllm-serve outputs/Qwen3.8-27B-NVFP4-W4A8 \
  --backend pytorch \
  --tp_size 4 \
  --kv_cache_dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

Prefer TensorRT-LLM over vLLM when you need fused NVFP4+FP8 W4A8 kernels;
vLLM support for that combo is limited.

## GPQA Diamond (official card protocol)

比特布局和三份机器配方在 [Quantization format](#quantization-format) 和 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。这里是采样锁和启动时容易踩的地方。

每份 NVFP4 导出都在 SGLang 上评（NVIDIA Qwen3.8 cookbook 的旗标）。这份 eval 发 temperature 0，其余字段跟 Qwen thinking 卡一致。公开卡是 temperature 1.0。默认、5090、6000D 和 Spark 完整 GPQA 配方都使用 262144 的剩余窗口；Spark 32k 只保留在视觉/MTP 功能验证配方和历史试跑中，详见 [Spark GPQA 协议](DGX_SPARK.md#full-gpqa-diamond-at-temperature-0)。512 或 2048 的输出上限不是这些完整评测配方的协议。

| Knob | Value |
|---|---|
| Sampling | `temperature=0 top_p=0.95 top_k=20 min_p=0 presence_penalty=0 repetition_penalty=1.0 do_sample=true`. Published Qwen / NVIDIA cards use temperature 1.0. |
| Thinking | `enable_thinking=true preserve_thinking=true reasoning_effort=xhigh` |
| Context | Full GPQA recipes, including Spark: `context-length=262144`. Spark vision/MTP checks remain at `32768`. `max_new_tokens=0` means the remaining window of the selected recipe. |
| Truncation | Fill remaining context; `continue_on_length` continues while budget remains, up to 8 continuations. A high-confidence restarted prefix stops further appending and is recorded as `repetition` with truncation retained; when the service returns usage metadata, that rejected response's actual usage is recorded. The checker accepts this diagnostic row and marks it wrong because it is not a non-truncated `stop`. This guard protects evaluation accounting and does not establish an MSE fix. |
| KV on 32 GB | SGLang `--enable-hierarchical-cache` + `--hicache-size`. Cookbook ~58 GiB on a 64 GB box. This 5090 VM (94 GiB) pins **64 GiB** HiCache (`eval-gpqa-diamond.5090.yaml`). SGLang `_split_hicache_size` splits that host pool by the **GPU** Mamba vs KV pool sizes — a fat GPU mamba cache also steals host KV. |
| Concurrency | default recipe 1; 5090 recipe **24** (`max_running_requests: 24`, `max_mamba_cache_size: 96` bf16 GDN slots so 24×4); 80 GB recipe **64** (`max_mamba_cache_size: 256`). float32 64-slot mamba used ~9.3 GB HBM and left ~0.88 GB GPU KV, so 16 HTTP workers queued behind 3–4 decode slots. |
| Mamba / GDN | 5090 and 80 GB: `mamba_ssm_dtype: bfloat16`, `mamba_radix_cache_strategy: extra_buffer_lazy`. NVIDIA SGLang cookbook: `extra_buffer` + float32. |
| Attention / GEMM | default FlashInfer (`eval-gpqa-diamond.yaml`); both CUDA 12.8 SM120 recipes use Triton attn + Marlin NVFP4, `SGLANG_FORCE_FP8_MARLIN=1`, and `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1`. The 32 GB recipe disables CUDA graph and uses HiCache 64 GiB (`eval-gpqa-diamond.5090.yaml`). The 80 GB recipe keeps CUDA graph, leaves KV on GPU, and uses CUTLASS FP8 (`eval-gpqa-diamond.6000d.yaml`). |
| Compose eval | no GPU (`NVIDIA_VISIBLE_DEVICES=""`, no `gpus:`); client talks to `serve-sglang:30000` |
| Headline | `correct/198` once `summary.json` exists and the checker verifies exactly 198 unique dataset-matching rows. Truncated and unparsed rows count as wrong. |

分数见上面的 [SGLang inference and GPQA](#sglang-inference-and-gpqa)。32 GB 上 262k 窗口放不进 HBM，用 HiCache 把 KV 放到主机内存，保持 `max_new_tokens`。Qwen 的 GPQA 轨迹会到几万 token。宿主机上的 `docker-compose.override.yml`（不进 git）如果写死了 `sglang` argv，要和配方一致：`--hicache-size 64`、`--max-mamba-cache-size 96`、`--mamba-ssm-dtype bfloat16`，否则 YAML 到不了进程。journal 按 `item_id` 续写，保留 `gpqa_diamond.jsonl` 里已有的行。

```bash
# Terminal 1 — serve (SGLang HiCache KV → RAM)
bash scripts/gpu-pod.sh serve w4a8
# or: python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml \
#        --model outputs/Qwen3.8-27B-NVFP4-W4A8

# Terminal 2 — client (full 198; needs HF_TOKEN or GPQA_CSV for gated Hub)
MEGAQUANT_SGLANG_BASE_URL=http://127.0.0.1:30000/v1 \
  bash scripts/gpu-pod.sh eval w4a8
# same for w4a4 / mixed after those exports exist

# No GPU, no 27B, no Hub (local export, not Qwen/Qwen3.8-27B):
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.5090.yaml --dry-run
python -m megaquant.cli eval -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
python -m megaquant.cli serve -c recipes/eval-gpqa-diamond.6000d.yaml --dry-run
```

Compose: `make serve-sglang` then `make eval-gpqa`. `eval-gpqa` does not
attach a GPU; `serve-sglang` does. On CUDA 12.8 / SM 12.0 set
`EVAL_RECIPE=recipes/eval-gpqa-diamond.5090.yaml`. The client talks to
`MEGAQUANT_SGLANG_BASE_URL` (default `http://127.0.0.1:30000/v1`). Journals
land in `<model>/gpqa_diamond/` (`gpqa_diamond.jsonl` keeps the full text;
stdout is a one-line status). Optional `--engine vllm`
keeps the NVIDIA GB300 vLLM flags on port 8000.

`Idavidrein/gpqa` is gated. Set `HF_TOKEN` or point `GPQA_CSV` at a local
CSV with the Hub columns (`Question`, `Correct Answer`,
`Incorrect Answer 1/2/3`, `Record ID`).

The official public download is available through `bash scripts/fetch_gpqa.sh`;
its default output is `data/dataset/gpqa_diamond.csv`. With the repository
installed, verify a completed temperature-0 run without GPU or network access:

```bash
python scripts/check_gpqa_results.py --csv data/dataset/gpqa_diamond.csv \
  --run-dir outputs/spark-gpqa-256k-20261002/w4a8-temp0-ctx262144-c16 \
  --output-json outputs/spark-gpqa-256k-20261002/w4a8-verification.json --seed 0
```

Use the actual dated run directory for each format. This checker reads the CSV,
journal and summary without modifying them; it exits nonzero for incomplete or
inconsistent evidence. Keep repeated answer options in the official CSV.
Use a new output directory whenever weights or protocol change; resume only
the same run with unchanged inputs and settings. Do not reuse a 32k journal
for a 256k run, or publish raw questions and response traces.
The original v0.1.2 baked GPQA client predates the raw
HTTP repair, so rebuild from the current checkout or use the updated CPU client
described in the Spark runbook before starting a new run.

## Troubleshooting

- Dry-run prints `family 'qwen3_5' is not registered`: `import megaquant.models`
  (the CLI pipeline loads plugins lazily; adapters register on that import).
- Plan tries to download `config.json`: set `family: qwen3_5` in the YAML
  (already set) so resolve() skips Hub lookup.
- 这个 checkpoint 的 `family` 用 `qwen3_5`。写成 vanilla `qwen3` 时，GDN 那些额外模块不会进忽略列表。
- group size 见 [Quantization format](#quantization-format)。混合图和均匀 W4A4 都是 16。
