# Docker / Compose — RTX 5090+ 量化箱

把 Mega-Quantification 的 CUDA、PyTorch（sm_90 / sm_120）、NVIDIA ModelOpt、llm-compressor 和 CLI 打进一个镜像。权重不进镜像：Hugging Face 缓存和导出目录挂到宿主机，方便在 5090 / 5090 D / Blackwell 工作站之间搬。

如果你的 5090 已经在 **K8s GPU 容器**里（没有 Docker、系统盘只有几十 GB、权重在只读盘），不要硬套 Compose，改用：

```bash
bash scripts/gpu-pod.sh plan|quantize|serve|eval [w4a8|w4a4|mixed]
```

省略 scheme 时默认 `w4a8`。这些命令会启动对应配方的 plan / PTQ，不表示 W4A4 或 mixed 已经在 pod 上跑完。

## 在 5090 机器上只要这几步

```bash
# 0) 驱动 570+ 与 NVIDIA Container Toolkit
bash docker/host-check.sh

# 1) 可选：Hugging Face token
cp .env.example .env
# 编辑 .env 填 HF_TOKEN

# 2) 构建（第一次会拉 CUDA / torch，大约 15–25 GB）
docker compose build

# 3) 干跑：不下载 27B，只打印方案
docker compose run --rm megaquant plan

# 4) 真正做 Qwen3.8-27B BF16 PTQ
docker compose --profile gpu run --rm quantize   # 均匀 W4A8
docker compose --profile gpu run --rm w4a4       # 均匀 W4A4
docker compose --profile gpu run --rm mixed      # 混合；默认 mixed.5090.yaml
```

`docker compose up` 默认只跑 `plan`，不会误触发 27B 校准。

`mixed` Compose 服务使用 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`
（`max`，ultrachat 256×1024，batch 4）。NVIDIA 质量 Local-Hessian
（`local_hessian`，2048，`Nemotron-Post-Training-Dataset-v3`）：

```bash
RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

等价 Make 入口：`make host-check`、`make build`、`make plan`、`make quantize`、`make mixed`、`make serve-vllm`、`make eval-gpqa`。

GPQA Diamond 走官方 Qwen thinking 采样和 NVIDIA vLLM 卡参数。32 GB 卡上
保持 `max-model-len 262144`，KV 放不下就 native offload 进主机内存
（MemTotal − 6 GiB），不要截断生成。先 `serve-vllm` 再 `eval-gpqa`。

## 把已经建好的镜像拷到另一台 5090

在有网络的机器上：

```bash
make image-tar          # 写出 megaquant-nvfp4.tar
```

拷贝 `megaquant-nvfp4.tar` + 本仓库（至少 `docker-compose.yml`、`recipes/`、`.env`）到目标机器：

```bash
docker image load -i megaquant-nvfp4.tar
docker compose run --rm megaquant plan
docker compose --profile gpu run --rm quantize
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

# GPQA：先 serve（KV → RAM），再 eval
docker compose --profile gpu run --rm serve-vllm
MEGAQUANT_VLLM_BASE_URL=http://127.0.0.1:8000/v1 docker compose --profile gpu run --rm eval-gpqa

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
- `nvidia-modelopt[hf]`：均匀 W4A8 = `W4A8_NVFP4_FP8_CFG`（weight block **32**）；均匀 W4A4 = `NVFP4_DEFAULT_CFG`（block **16**，权重和激活）；NVIDIA 混合 NVFP4 层也是 group_size **16**（MLP + `lm_head`），注意力为 FP8
- llm-compressor / compressed-tensors（vLLM 路径）
- 本仓库 `megaquant` CLI

不包含 27B 权重、校准数据集和导出 checkpoint。

`nvidia/Qwen3.8-27B-NVFP4` 是混合 NVFP4/FP8（group_size 16 + FP8），不是均匀 W4A8，也不是 NVFP4 block 32。
