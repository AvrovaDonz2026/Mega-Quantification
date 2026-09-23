# Mega-Quantification

把 Hugging Face 上的 BF16 模型做训练后量化，导出 SGLang 能直接加载的 checkpoint。目前跑通的模型是 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。

默认方案叫 `nvfp4_w4a8`。MLP 和 `lm_head` 是 NVFP4（group 16），注意力投影是 FP8。导出文件里的 `quant_algo` 写成 `MIXED_PRECISION`。均匀的 W4A4 是另一份配方，整层都是 NVFP4 group 16。配方名叫 W4A8，是因为层的分法和 NVIDIA 公开的混合权重一致；MLP 的激活也是 NVFP4，不是 FP8。

仓库里是代码和 YAML。27B 权重和量化结果不进 git。再跑一遍量化会重新估计 scale，文件不会和旧导出逐字节相同。机器差异、测过的分数、MTP 和上传方式写在 [Qwen3.8 手册](docs/qwen3.8-27b.md)。给代理用的约定在 [SKILL.md](SKILL.md)。

协议是 GNU AGPL v3，或以后的版本（`AGPL-3.0-or-later`）。Copyright 2026 Donz。全文在 `LICENSE`。

## 先看计划

不需要 GPU，也不会去拉 27B。

```bash
pip install -e '.[dev]'
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
```

## 在卡上量化

```bash
pip install -e '.[hf,modelopt]'

# 32 GB 5090 上实际用过的配方：算法 max，ultrachat 256×1024
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml

# 均匀 W4A4
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a4.yaml

# 更大的卡、想贴近 NVIDIA 公开卡的校准时再用 Local-Hessian
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.yaml
```

有 Docker 的机器看 [docker/README.md](docker/README.md)。镜像把 ModelOpt 钉在 0.46.1，和这次 5090 量化用的版本相同。

已经导出、但目录里没有 MTP 的 checkpoint，不必重跑量化：

```bash
megaquant restore-mtp outputs/Qwen3.8-27B-NVFP4-W4A8 --source Qwen/Qwen3.8-27B
```

## 目录

```
src/megaquant/    库和命令行
recipes/          量化和 GPQA 的 YAML，索引在 recipes/README.md
docs/             架构说明和 Qwen3.8 手册
docker/           镜像脚本；Dockerfile 在仓库根目录
scripts/          OSS、gpu-pod、serve / eval 包装
SKILL.md          给代理的入口
```

密钥放在不会进 git 的 `.env` 和 `.oss.env`。`docker-compose.override.yml` 也不要提交。

## 接一个新模型

在 `src/megaquant/models/` 里加一个 family，实现 `default_ignore` 和 `load_kwargs`，从 `models/__init__.py` 导入，让它在 import 时注册。再写一份 `recipes/*.yaml`，然后 `megaquant plan -c` 看忽略列表。embedding、视觉塔、GDN 的 conv 留在 BF16。不要把整个 `*mlp*` 放进忽略列表，那是要量化的 GEMM。

## English

Mega-Quantification post-trains a BF16 Hugging Face model into a checkpoint SGLang can load. The model this tree actually runs is Qwen/Qwen3.8-27B.

The default recipe, `nvfp4_w4a8`, puts NVFP4 group 16 on the MLP and `lm_head`, and FP8 on attention. The export tag is `MIXED_PRECISION`. Uniform W4A4 (`nvfp4_w4a4`) is NVFP4 group 16 on every targeted linear. The recipe is named W4A8 because the layer split matches NVIDIA's public mixed checkpoint; the MLP activations are NVFP4 as well.

Weights are not in git. A second PTQ pass estimates scales again, so the files will differ from an older export. The runbook is [docs/qwen3.8-27b.md](docs/qwen3.8-27b.md). Agents should start at [SKILL.md](SKILL.md).

License: GNU AGPL version 3 or later (`AGPL-3.0-or-later`). Copyright 2026 Donz.

```bash
pip install -e '.[dev]'
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-w4a8.yaml --dry-run
pip install -e '.[hf,modelopt]'
megaquant quantize -c recipes/qwen3.8-27b-nvfp4-mixed.5090.yaml
```

Dry-run does not download the 27B. The 5090 recipe above is what fit in 32 GB (`max`, ultrachat). `recipes/qwen3.8-27b-nvfp4-mixed.yaml` is the Local-Hessian run for a larger GPU. Docker notes are in [docker/README.md](docker/README.md).
