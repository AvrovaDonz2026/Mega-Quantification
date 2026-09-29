# Docker

量化和推理使用独立镜像，换机器时先装驱动，再跑 `install-host.sh`，然后 `compose build` 或 `image load`。权重、校准数据和导出留在宿主机，不进镜像。

| 镜像 | 用途 |
|---|---|
| `megaquant:nvfp4` | `plan` / `quantize` / `mixed` / `w4a4`（nvidia-modelopt） |
| `megaquant:sglang` | `serve-sglang` / `eval-gpqa` / `fetch-export` / `fetch-gpqa` |
| `megaquant:gb10` | DGX Spark ARM64 / CUDA 13 量化 |
| `megaquant:sglang-spark` | DGX Spark 的 `serve-sglang-spark` / `eval-gpqa-spark` |

SGLang 和 ModelOpt 各用各的 torch。`mixed` 服务默认是 `recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml`（32 GB 5090 上跑过的那次）。Local-Hessian 把 `RECIPE` 指到 `recipes/qwen3.8-27b-nvfp4-mixed.yaml`。机器配方和分数在 [Qwen3.8 手册](../docs/qwen3.8-27b.md)，代理读 [SKILL.md](../SKILL.md)。

已经在 K8s GPU 容器里、没有 Docker 的机器用 `bash scripts/gpu-pod.sh`。新虚拟机：

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

## DGX Spark / GB10

DGX Spark 是 ARM64、CUDA 13、SM 12.1 的统一内存机器。用
`Dockerfile.spark` 保留仓库的 `megaquant quantize` 流程，避免默认
CUDA 12.8/cu128 镜像把 ARM64 的 PyTorch 替换掉：

```bash
# Spark 上已有 quant-env:gb10 时可离线构建
DOCKER_BUILDKIT=0 docker build \
  --build-arg BASE_IMAGE=quant-env:gb10 \
  --build-arg INSTALL_SPARK_DEPS=0 \
  -f Dockerfile.spark -t megaquant:gb10 .

# 纯净环境可使用多架构 NGC PyTorch（需要能访问 NGC / PyPI）
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3 \
  --build-arg INSTALL_SPARK_DEPS=1 \
  -f Dockerfile.spark -t megaquant:gb10 .
```

权重和校准集挂载到容器，统一内存下建议先给 Accelerate 一个明确预算，
避免把同一块 UMA 同时当成完整 GPU 池和完整 CPU 池：

```bash
docker run --rm --gpus all --ipc=host --shm-size=64g \
  -e MEGAQUANT_LOW_MEMORY=1 \
  -e MEGAQUANT_MAX_MEMORY=0:96GiB,cpu:24GiB \
  -e MEGAQUANT_OFFLOAD_DIR=/opt/megaquant/offload \
  -v /path/to/models:/models:ro \
  -v /path/to/calibration:/data:ro \
  -v "$PWD/outputs:/opt/megaquant/outputs" \
  -v "$PWD/offload:/opt/megaquant/offload" \
  megaquant:gb10 quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml \
  --model /models/Qwen3.8-27B
```

混合 NVFP4/FP8 使用 `recipes/qwen3.8-27b-nvfp4-w4a8.spark.yaml`。
两个 Spark 配方都要求图像校准：把 `calibration.jsonl` 和图片一起挂载到
`/data`，例如一行 `{"image":"images/example.png","text":"Describe this image."}`。
允许混入文本行。MTP 使用真实 target hidden states 和下一 token embedding
执行校准；`calibration_coverage.json` 记录视觉/MTP 各激活量化器的调用次数。
视觉位置嵌入与 patch embedding 保持 BF16。

可从固定版本的 [COCO-Caption2017](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017/tree/3bdd5827e243cc3084ac69a1111e69c3ab9193ff)
生成 128 图像 + 128 文本的数据目录。先下载 `data/val-00000-of-00002.parquet`，
再执行（需要 `pyarrow` 和 `Pillow`）：

```bash
python scripts/prepare_multimodal_calibration.py \
  --parquet /path/to/coco-val-00000.parquet \
  --output-dir /path/to/calibration \
  --source-dataset lmms-lab/COCO-Caption2017 \
  --source-revision 3bdd5827e243cc3084ac69a1111e69c3ab9193ff \
  --expected-parquet-sha256 c60673a81babec10030027aafe5369d7c89955efa925f9af43501b52c51994a3 \
  --text-jsonl /path/to/text-calibration.jsonl
```

固定种子为 42，图像最长边 448，文本最多 600 字符；manifest 包含源文件和
每张图片的 SHA256。该数据用于验证量化流程，模型质量和 MTP 接受率需另做评测。

`TORCH_CUDA_ARCH_LIST=12.1` 已写入 Spark 镜像。ModelOpt 0.47 支持仓库的
NVFP4 presets；Spark 镜像单独使用 `docker/requirements-gpu-spark.txt`，
不会改变 5090 镜像的 ModelOpt 0.46.1 pin。FLA/causal-conv1d 没有 ARM64
预编译包时，PTQ 会使用 Transformers 的 PyTorch fallback；这只影响速度。

### Spark SGLang 服务

`Dockerfile.sglang.spark` 使用官方 `lmsysorg/sglang:v0.5.20-cu130`，固定
多架构镜像 digest，保留其 ARM64 PyTorch、SGLang 和 CUDA 13 内核。
镜像内对该版本的视觉/MTP ModelOpt 加载器应用兼容补丁，构建时验证源文件；
更换 `SGLANG_SPARK_BASE_IMAGE` 后若源码不匹配，构建会停止。
[NVIDIA Spark SGLang 指南](https://build.nvidia.com/spark/sglang/instructions)
也使用 CUDA 13 镜像。

```bash
docker compose --profile spark build serve-sglang-spark
docker compose --profile spark run --rm --entrypoint python serve-sglang-spark \
  scripts/check_sglang_spark.py
docker compose --profile spark up -d serve-sglang-spark
docker compose --profile spark logs -f serve-sglang-spark
# 服务就绪后再评测；此客户端不挂 GPU
docker compose --profile spark run --rm eval-gpqa-spark
```

默认加载 `outputs/Qwen3.8-27B-NVFP4-W4A8-spark`。通过 `OUTPUTS_DIR`
指定宿主机的导出目录，通过 `SPARK_EVAL_MODEL` 选择目录内的 W4A4 或 W4A8
产物。`SPARK_EVAL_RECIPE` 和 `SGLANG_SPARK_PORT` 分别覆盖配方和端口。
也可使用 `make serve-sglang-spark` / `make eval-gpqa-spark`。

Spark 配方启用视觉，初始上下文 32768、并发 4、静态内存比例 0.70，关闭
CUDA graph 与 HiCache。CPU/GPU 共享物理内存，不能照搬独立显存机器的
自动主机缓存预算。该配方的上下文限制与 262144 的完整 GPQA 配方不同，
分数不能直接视为同一评测设置。环境检查通过只表示 CUDA 和基础依赖可用；
必须继续验证实际权重加载、文本和图片请求。

单台 Spark 有一颗 GPU，配方默认 TP1。跨两台 Spark 的 TP2 还需要两端
一致的镜像/权重、分布式初始化地址和网络配置，不能只把本配方的 TP 改成 2。
MTP 默认为关闭；文本/图片通过后，可在配方中启用注释列出的 EAGLE 参数，
使用完整导出里的 `mtp.*`。量化 MTP 不应指向丢失量化元数据的 BF16 draft。

`mixed` 默认就是 5090 那份 `max` + ultrachat。更大的卡上改走 Local-Hessian：

```bash
MIXED_RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed
```

等价的 Make 入口：`make host-check`、`make build`、`make plan`、`make quantize`、`make mixed`、`make serve-sglang`、`make eval-gpqa`。默认推理是 SGLang `:30000`。

先起 `serve-sglang`，再跑 `eval-gpqa`。32 GB 上保持 `context-length 262144`，KV 放不下就用 HiCache 放到主机内存。5090 配方是 64 GiB HiCache、24 路、bf16 GDN / 96 slot、Triton + Marlin。宿主机的 `docker-compose.override.yml`（不进 git）如果写死了 `sglang` argv，把 `--hicache-size 64`、`--max-mamba-cache-size 96`、`--mamba-ssm-dtype bfloat16` 一起写上。journal 按 `item_id` 续写，已有的 `gpqa_diamond.jsonl` 留着。导出如果还是没有 `quantized_layers` 的裸 `NVFP4`，先 `megaquant rewrite-sglang <export_dir>` 再 serve。

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
| 单卡 5090 32 GB + 64 GB RAM | `MEGAQUANT_LOW_MEMORY=1` + `recipes/qwen3.8-27b-nvfp4-w4a8.5090.yaml` / `w4a4.5090.yaml` / `mixed.5090.yaml`。默认打满 VRAM−1GiB、内存 MemTotal−6GiB、全部 CPU 线程、`batch_size=1`。BF16 拷到本地盘，不要打 UPFS `/model`。 |
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
W4A4_RECIPE=recipes/qwen3.8-27b-nvfp4-w4a4.yaml docker compose --profile gpu run --rm w4a4

# 混合：5090 打包（服务默认）vs NVIDIA 质量 Hessian
docker compose --profile gpu run --rm mixed
MIXED_RECIPE=recipes/qwen3.8-27b-nvfp4-mixed.yaml docker compose --profile gpu run --rm mixed

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
- `nvidia-modelopt[hf]`：默认 W4A8 / mixed = NVFP4 group_size **16**（MLP + `lm_head`）+ 注意力 FP8，导出 `MIXED_PRECISION`；均匀 W4A4 = `NVFP4_DEFAULT_CFG`（block **16**）。镜像钉 **0.46.1**（PyPI 没有 0.48）；5090 生产算法是 **`max`**，不是 Local-Hessian。不实现 SGLang 拒收的均匀 `W4A8_NVFP4_FP8`。
- llm-compressor / compressed-tensors（vLLM 路径）
- 本仓库 `megaquant` CLI

不包含 27B 权重、校准数据集和导出 checkpoint。

层图和校准差异见 [Qwen3.8 手册](../docs/qwen3.8-27b.md)。
