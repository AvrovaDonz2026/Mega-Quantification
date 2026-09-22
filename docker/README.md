# Docker / Compose — RTX 5090+ quantization box

## English

Pack PTQ (ModelOpt) and eval (SGLang + GPQA) into **two Compose images**. A
new VM only needs the driver + `install-host.sh`, then `compose build` /
`image load`. Weights, calib data, and export checkpoints stay on the host.

| Image | Role |
|---|---|
| `megaquant:nvfp4` | `plan` / `quantize` / `mixed` / `w4a4` (nvidia-modelopt) |
| `megaquant:sglang` | `serve-sglang` / `eval-gpqa` / `fetch-export` / `fetch-gpqa` |

SGLang and ModelOpt cannot share a torch, so they are not one venv.

### Production PTQ vs NVIDIA quality (5090)

Compose `mixed` defaults to [`recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`](../recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml).
That **is** the 32 GB 5090 production run: ModelOpt **0.46.1**, algorithm
**`max`**, `ultrachat_200k` 256×1024 batch 4, mixed NVFP4 gs16 MLP +
`lm_head` / FP8 attention, `MIXED_PRECISION` export. Default W4A8 is the
same encoding — publish one mixed export as both `mixed` and `w4a8`.

NVIDIA's public `nvidia/Qwen3.8-27B-NVFP4` uses the **same layer map** with
**Local-Hessian** + Nemotron v3 2048 on `nvidia-modelopt` **0.48.0**. That
recipe is `qwen3.8-27b-nvfp4-mixed.yaml`. A 32 GB 5090 cannot hold Hessian
2048.

```bash
docker compose --profile gpu run --rm mixed   # 5090 max + ultrachat (production)
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

5090 GPQA (`recipes/eval-gpqa-diamond.5090.yaml`): **24-way**, HiCache
**64 GiB**, bf16 GDN / 96 mamba slots, Triton + Marlin, CUDA graph off. Host
`docker-compose.override.yml` (never git) that hardcodes `sglang` argv must
use the same `--hicache-size 64 --max-mamba-cache-size 96 --mamba-ssm-dtype
bfloat16`. An 80 GB SM120 uses `recipes/eval-gpqa-diamond.6000d.yaml`
(64-way, KV on GPU, CUDA graph on, SiLU+FP4 fusion off). Do not wipe
`gpqa_diamond.jsonl`; the client resumes by `item_id`.

The checkpoint those recipes serve is mixed: MLP + `lm_head` are NVFP4
group 16 weights **and** NVFP4 activations, attention projections are FP8,
KV is fp8_e4m3, and `hf_quant_config.json` is `MIXED_PRECISION`. SGLang is
the eval runtime (`megaquant serve` then `megaquant eval` on `:30000`).

| Box | `EVAL_RECIPE` | Serve |
|---|---|---|
| CUDA ≥ 12.9 | `recipes/eval-gpqa-diamond.yaml` | FlashInfer, HiCache 12 GiB |
| 32 GB / CUDA 12.8 | `recipes/eval-gpqa-diamond.5090.yaml` | Triton + Marlin, HiCache 64 GiB, 24-way, CUDA graph off |
| 80 GB / CUDA 12.8 | `recipes/eval-gpqa-diamond.6000d.yaml` | Triton + Marlin + CUTLASS, KV on GPU, 64-way, CUDA graph on |

---

## 中文

# Docker / Compose — RTX 5090+ 量化箱

把量化（ModelOpt PTQ）和评测（SGLang + GPQA）打进 **两套 Compose 镜像**，换虚拟机只装驱动 + `install-host.sh`，然后 `compose build` / `image load` 就能得到同一套环境。权重、校准数据和导出 checkpoint **不进镜像**。

### 生产 PTQ vs NVIDIA 质量（5090）

Compose `mixed` **默认就是** `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`：这是
32 GB 5090 上实际跑完的生产 PTQ（ModelOpt **0.46.1**、算法 **`max`**、
ultrachat 256×1024 batch 4）。NVIDIA 公开 `nvidia/Qwen3.8-27B-NVFP4` 层图相同，
校准是 Local-Hessian + Nemotron v3 2048（modelopt **0.48.0**），配方
`qwen3.8-27b-nvfp4-mixed.yaml`，单卡 32 GB 放不下。

5090 GPQA：`eval-gpqa-diamond.5090.yaml`，**24 路** / HiCache **64 GiB** /
bf16 GDN 96 slot / Triton + Marlin / CUDA graph 关。宿主机 override 写死
argv 时必须同步 hicache / mamba。80 GB SM120 用
`eval-gpqa-diamond.6000d.yaml`（64 路、KV 在 GPU、CUDA graph 开、关掉
SiLU+FP4 融合）。不要清空 `gpqa_diamond.jsonl`。

这些配方加载的权重是混合格式：MLP + `lm_head` 为 NVFP4 group 16 权重和
NVFP4 激活，注意力投影为 FP8，KV 为 fp8_e4m3，`hf_quant_config.json` 为
`MIXED_PRECISION`。评测运行时是 SGLang（先 `megaquant serve`，再
`megaquant eval`，端口 `:30000`）。

| 机器 | `EVAL_RECIPE` | 推理 |
|---|---|---|
| CUDA ≥ 12.9 | `recipes/eval-gpqa-diamond.yaml` | FlashInfer，HiCache 12 GiB |
| 32 GB / CUDA 12.8 | `recipes/eval-gpqa-diamond.5090.yaml` | Triton + Marlin，HiCache 64 GiB，24 路，CUDA graph 关 |
| 80 GB / CUDA 12.8 | `recipes/eval-gpqa-diamond.6000d.yaml` | Triton + Marlin + CUTLASS，KV 在 GPU，64 路，CUDA graph 开 |

| 镜像 | 用途 |
|---|---|
| `megaquant:nvfp4` | `plan` / `quantize` / `mixed` / `w4a4`（nvidia-modelopt） |
| `megaquant:sglang` | `serve-sglang` / `eval-gpqa` / `fetch-export` / `fetch-gpqa` |

SGLang 与 ModelOpt 不能共用一个 torch，所以不要把它们装进同一个 venv。

如果你的 5090 已经在 **K8s GPU 容器**里（没有 Docker），改用 `bash scripts/gpu-pod.sh`。新虚拟机：

```bash
# 0) 驱动 570+ 已装好后
bash docker/install-host.sh
# 可选：云厂商加速 DNS 只写在这台宿主机（systemd-resolved + Docker daemon）。
# 部分加速器不改写 registry-1.docker.io / CloudFront，install-host 会用
# canary（默认 pypi.org）查到的内网 IP 把 Hub 主机名写进 /etc/hosts。
# 不要把具体 IP 提交进 git / 镜像；换机时再 export 一次。
# MEGAQUANT_DOCKER_DNS="<ip> <ip>" bash docker/install-host.sh
# MEGAQUANT_DOCKER_DNS="<ip> <ip>" bash docker/install-host.sh --dns-only
bash docker/host-check.sh

# 1) 可选 token / OSS
cp .env.example .env

# 2) 构建两套镜像（或 docker image load -i megaquant-stack.tar）
make build

# 3) 干跑：不下载 27B
make plan

# 4) PTQ
docker compose --profile gpu run --rm quantize   # SGLang mixed W4A8
docker compose --profile gpu run --rm w4a4
docker compose --profile gpu run --rm mixed

# 5) 评测：先拉起 serve，再 eval（eval 不占 GPU：无 gpus/deploy devices，
#    NVIDIA_VISIBLE_DEVICES=""。serve-sglang 才挂 GPU）
make fetch-gpqa
docker compose --profile gpu up serve-sglang
docker compose --profile gpu run --rm eval-gpqa
# 5090 / CUDA 12.8：Triton，不要用默认 FlashInfer 配方
# EVAL_RECIPE=recipes/eval-gpqa-diamond.5090.yaml docker compose --profile gpu up serve-sglang
# EVAL_RECIPE=recipes/eval-gpqa-diamond.5090.yaml docker compose --profile gpu run --rm eval-gpqa
```

`docker compose up` 默认只跑 `plan`，不会误触发 27B 校准。

`mixed` Compose 服务使用 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`
（`max`，ultrachat 256×1024，batch 4）。**这就是 5090 生产 PTQ**，不是
占位配方。NVIDIA 公开权重用 Local-Hessian（`local_hessian`，2048，
`Nemotron-Post-Training-Dataset-v3`，modelopt 0.48.0）；32 GB 5090 放不下
Hessian 2048：

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

等价 Make 入口：`make host-check`、`make build`、`make plan`、`make quantize`、`make mixed`、`make serve-sglang`、`make eval-gpqa`。默认推理是 SGLang `:30000`（HiCache），不是 `make serve-vllm`。

GPQA Diamond 走官方 Qwen thinking 采样，推理用 SGLang（NVIDIA Qwen3.8
cookbook）。两个终端：先 `serve-sglang` 再 `eval-gpqa`。32 GB 卡上保持
`context-length 262144`，KV 放不下就 HiCache offload 进主机内存，不要截断
生成。5090 配方钉死 **64 GiB** HiCache、**24 路**、bf16 GDN / 96 slot、
Triton + Marlin。宿主机 `docker-compose.override.yml`（不要进 git）如果写死了
`sglang` argv，必须同步 `--hicache-size 64` / `--max-mamba-cache-size 96` /
`--mamba-ssm-dtype bfloat16`。eval journal 按 `item_id` 续跑，不要清空
`gpqa_diamond.jsonl`。Mixed / 默认 W4A8 导出后，若
`hf_quant_config.json` 仍是没有 `quantized_layers` 的裸 `NVFP4`，先
`megaquant rewrite-sglang <export_dir>` 再 serve。均匀 `w4a8_nvfp4_fp8`
只给 TensorRT-LLM，不要当 SGLang 权重加载。

PTQ 完成后上传（桶和端点用 `OSS_BUCKET` / `OSS_ENDPOINT`，可选
`.oss.env`）。默认 W4A8 就是 mixed 编码：一份 mixed 导出可按
`--scheme mixed` 和 `--scheme w4a8` 各发一次（同一 content-hash），不必
再跑一遍 PTQ：

```bash
python scripts/oss_publish.py <export> --scheme w4a4|mixed|w4a8
bash scripts/gpu-pod.sh publish [w4a8|w4a4|mixed]
```

## 把已经建好的镜像拷到另一台 5090

在有网络的机器上：

```bash
make image-tar          # 写出 megaquant-stack.tar（nvfp4 + sglang）
```

拷贝 `megaquant-stack.tar` + 本仓库（至少 `docker-compose.yml`、`recipes/`、`.env`）到目标机器：

```bash
docker image load -i megaquant-stack.tar
docker compose run --rm megaquant plan
docker compose --profile gpu up serve-sglang
```

权重建议单独 rsync `./.cache/huggingface` 或把 `Qwen/Qwen3.8-27B` 快照放到 `./models`，避免每台机器重新下 50GB+ BF16。

## 显存

| 机器 | 建议 |
|---|---|
| 单卡 5090 32 GB + 64 GB RAM | `MEGAQUANT_LOW_MEMORY=1` + `recipes/qwen3.8-27b-nvfp4-w4a8.5090.yaml` / `w4a4.5090.yaml` / `mixed.5090.yaml`。默认打满 VRAM−1GiB、内存 MemTotal−6GiB、全部 CPU 线程、`batch_size=4`。BF16 拷到本地盘，不要打 UPFS `/model`。 |
| 双卡 5090 / 96 GB 级 Blackwell | `CUDA_VISIBLE_DEVICES=0,1`，正常 PTQ |
| B200 / GB300 | 直接跑；混合质量配方可把 Local-Hessian 开到 2048（`mixed.yaml`） |

Qwen3.8-27B BF16 权重大约 54 GB，单卡 32 GB 放不下整模，必须 offload 或多卡。

## NGC 底包（5090 / sm_120 更稳）

公共 `nvidia/cuda` + pip torch 在大多数 5090 上可用。若遇到 `sm_120 is not compatible with the current PyTorch`，改用 NGC PyTorch（需要 `docker login nvcr.io`）：

```bash
docker compose -f docker-compose.yml -f docker-compose.ngc.yml build
docker compose -f docker-compose.yml -f docker-compose.ngc.yml run --rm megaquant plan
```

## RTX 5090 / CUDA 12.8 serve image

默认 `megaquant:sglang` 底包是 `nvidia/cuda:12.8.1-devel-ubuntu24.04`（`SGLANG_BASE_IMAGE`），不要改这个默认。FlashInfer JIT 在 nvcc 12.8 上看不见 SM 12.0（`SM 12.x requires CUDA >= 12.9`），DeepGEMM `set_pdl` 也要求 nvcc 12.9+，所以 Compose 默认 `SGLANG_ENABLE_JIT_DEEPGEMM=0`。

这台 5090 上先用 `recipes/eval-gpqa-diamond.5090.yaml`（不要改 Compose 里的默认 `EVAL_RECIPE`）：Triton 注意力/GDN、PyTorch sampling、Triton FP8 GEMM、Marlin NVFP4，以及 `SGLANG_FORCE_FP8_MARLIN=1`。只改 `--attention-backend triton` 不够——mixed 的 FP8 `linear_attn` 投影仍会走 FlashInfer BMM。94 GiB 内存机器把 `--hicache-size` 钉在 **64 GiB**。HiCache 按 GPU 池比例切 host RAM，所以 5090 用 **bf16** GDN（`max_mamba_cache_size: 96`）把 HBM 还给注意力 KV，GPQA 开 **24** 路。float32 64-slot mamba 大约占 9.3 GB HBM，GPU KV 只剩不到 1 GB，16 路会排队。Host 上的 `docker-compose.override.yml` 如果写死了 `sglang` argv，也要把 `--hicache-size` / `--max-mamba-cache-size` / `--mamba-ssm-dtype` 改成同样的数，否则 recipe 不会生效。

```bash
EVAL_RECIPE=recipes/eval-gpqa-diamond.5090.yaml docker compose --profile gpu up serve-sglang
```

默认 `EVAL_RECIPE` 仍是 `recipes/eval-gpqa-diamond.yaml`（flashinfer）。

长期、磁盘够了再可选把 serve 镜像重建到 CUDA 12.9+ devel（FlashInfer JIT 才能编 SM 12.0）。候选 pin：`nvidia/cuda:12.9.1-devel-ubuntu24.04`（请在 NGC / Docker Hub 核对；CUDA 13.0 devel 也可以）。新镜像大约 28GB，磁盘紧张时不要现在重建。

```bash
SGLANG_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04 docker compose build serve-sglang
```

12.9+ 重建后把 `SGLANG_ENABLE_JIT_DEEPGEMM=1` 写进 `.env`，就可以走默认 flashinfer 配方。不要在 Python 里探测 nvcc。

## 常用覆盖

```bash
# 均匀 W4A4
docker compose --profile gpu run --rm w4a4
RECIPE=recipes/qwen3.8-27b-nvfp4-w4a4.5090.yaml docker compose --profile gpu run --rm w4a4

# 混合：5090 打包（服务默认）vs NVIDIA 质量 Hessian
docker compose --profile gpu run --rm mixed
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed

# GPQA：先 serve（SGLang 服务名 `serve-sglang:30000`），再 eval
make fetch-gpqa
docker compose --profile gpu up serve-sglang
docker compose --profile gpu run --rm eval-gpqa
# 可选：CLI `--engine vllm`（GB300 旗标，端口 8000），不是默认路径

# 只用第 0 张卡
CUDA_VISIBLE_DEVICES=0 docker compose --profile gpu run --rm quantize

# 已经下好的本地权重
# 把 snapshot 放到 ./models/Qwen3.8-27B，并改 recipe 的 model.source 为 /models/Qwen3.8-27B
```

挂载约定：

| 宿主机 | 容器 |
|---|---|
| `./.cache/huggingface` | `/cache/huggingface` |
| `./models` | `/models` |
| `./outputs` | `/opt/megaquant/outputs` |
| `./offload_folder` | `/opt/megaquant/offload` |
| `./recipes` | `/opt/megaquant/recipes` |
| `./src` | `/opt/megaquant/src` |

`compose up` bind-mounts git `./src` over `/opt/megaquant/src` so the baked `megaquant:sglang` image picks up latest Python without a rebuild.

## 镜像里有什么

- CUDA 12.8 devel（Triton / Local-Hessian NVFP4 扫描需要）
- PyTorch CUDA 12.8 轮子，`TORCH_CUDA_ARCH_LIST=9.0;10.0;12.0`
- `nvidia-modelopt[hf]`：默认 W4A8 / mixed = NVFP4 group_size **16**（MLP + `lm_head`）+ 注意力 FP8，导出 `MIXED_PRECISION`；均匀 W4A4 = `NVFP4_DEFAULT_CFG`（block **16**）；TensorRT-LLM 均匀 W4A8 = `W4A8_NVFP4_FP8_CFG`（weight block **32**）。镜像钉 **0.46.1**（PyPI 没有 0.48）；5090 生产算法是 **`max`**，不是 Local-Hessian。
- llm-compressor / compressed-tensors（vLLM 路径）
- 本仓库 `megaquant` CLI

不包含 27B 权重、校准数据集和导出 checkpoint。

`nvidia/Qwen3.8-27B-NVFP4` 是混合 NVFP4/FP8（group_size 16 + FP8），校准是
Local-Hessian + Nemotron v3。默认 `nvfp4_w4a8` 对齐这套**层图**给 SGLang
用；5090 生产校准是 `max` + ultrachat 256。`W4A8_NVFP4_FP8` block 32 是
`w4a8_nvfp4_fp8`，SGLang 不认。
