# Docker / Compose — RTX 5090+ 量化箱

把量化（ModelOpt PTQ）和评测（SGLang + GPQA）打进 **两套 Compose 镜像**，换虚拟机只装驱动 + `install-host.sh`，然后 `compose build` / `image load` 就能得到同一套环境。权重、校准数据和导出 checkpoint **不进镜像**。

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

# 5) 评测：先拉起 serve，再 eval（eval 不占 GPU）
make fetch-gpqa
docker compose --profile gpu up serve-sglang
docker compose --profile gpu run --rm eval-gpqa
```

`docker compose up` 默认只跑 `plan`，不会误触发 27B 校准。

`mixed` Compose 服务使用 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`
（`max`，ultrachat 256×1024，batch 4）。NVIDIA 质量 Local-Hessian
（`local_hessian`，2048，`Nemotron-Post-Training-Dataset-v3`）：

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

等价 Make 入口：`make host-check`、`make build`、`make plan`、`make quantize`、`make mixed`、`make serve-sglang`、`make eval-gpqa`。默认推理是 SGLang `:30000`（HiCache），不是 `make serve-vllm`。

GPQA Diamond 走官方 Qwen thinking 采样，推理用 SGLang（NVIDIA Qwen3.8
cookbook）。两个终端：先 `serve-sglang` 再 `eval-gpqa`。32 GB 卡上保持
`context-length 262144`，KV 放不下就 HiCache offload 进主机内存
（MemTotal − 6 GiB），不要截断生成。Mixed / 默认 W4A8 导出后，若
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

## 镜像里有什么

- CUDA 12.8 devel（Triton / Local-Hessian NVFP4 扫描需要）
- PyTorch CUDA 12.8 轮子，`TORCH_CUDA_ARCH_LIST=9.0;10.0;12.0`
- `nvidia-modelopt[hf]`：默认 W4A8 / mixed = NVFP4 group_size **16**（MLP + `lm_head`）+ 注意力 FP8，导出 `MIXED_PRECISION`；均匀 W4A4 = `NVFP4_DEFAULT_CFG`（block **16**）；TensorRT-LLM 均匀 W4A8 = `W4A8_NVFP4_FP8_CFG`（weight block **32**）
- llm-compressor / compressed-tensors（vLLM 路径）
- 本仓库 `megaquant` CLI

不包含 27B 权重、校准数据集和导出 checkpoint。

`nvidia/Qwen3.8-27B-NVFP4` 是混合 NVFP4/FP8（group_size 16 + FP8）。默认
`nvfp4_w4a8` 对齐这套图给 SGLang 用。`W4A8_NVFP4_FP8` block 32 是
`w4a8_nvfp4_fp8`，SGLang 不认。
