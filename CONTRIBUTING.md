# Contributing

Mega-Quantification is AGPL-3.0-or-later. Copyright 2026 Donz. See `LICENSE`.

协议是 GNU AGPL v3，或以后的版本。

Start with the [documentation index](docs/README.md) for operation guides and
the [architecture guide](docs/ARCHITECTURE.md) for extension points.
操作指南见 [文档导航](docs/README.md)，实机结果见 [验证索引](docs/validation/README.md)。

## English

- Install CPU test dependencies in a Python 3.12 virtual environment. On Linux,
  select the CPU PyTorch wheel before installing the test extra:

  ```bash
  python3 -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
  python3 -m pip install -e '.[dev,test]'
  ```

  On macOS, install `.[dev,test]` directly. This extra includes image/parquet
  dependencies and pins Transformers 5.14.1 for the CPU MTP forwards. The
  optional ModelOpt integration test still needs the ModelOpt extra.
- Run `make test` or `python3 -m pytest` before opening a pull request. Tests
  are CPU-only. `make test` uses `python3` (`PYTHON ?= python3`; override if
  needed). `megaquant plan` and eval `--dry-run` must not download the 27B
  checkpoint.
- Recipe paths stay `recipes/<name>.yaml`. Add a row to `recipes/README.md`
  when you add a YAML file.
- Keep secrets out of git: `.env`, `.oss.env`, `docker-compose.override.yml`,
  SSH material, bucket names, and tenant DNS addresses.
- Weights, Hub caches, and GPQA journals belong in `outputs/`, `.cache/`,
  `models/`, or `data/`, which are gitignored.
- Human guides are indexed in `docs/README.md`; model details live in
  `docs/qwen3.8-27b.md` and Spark execution in `docs/DGX_SPARK.md`. Agent
  constraints live in `SKILL.md`. Update the relevant guides and agent
  constraints when the public protocol changes; preserve dated evidence.

## 中文

- Python 3.12 虚拟环境中安装 `.[dev,test]`；Linux 先按上面的命令安装 CPU
  PyTorch。测试 extra 包含图像/parquet 依赖，以及 CPU MTP 前向使用的
  Transformers 5.14.1；可选 ModelOpt 集成测试另外需要 ModelOpt extra。
- 提交前跑 `make test` 或 `python3 -m pytest`。测试不需要 GPU。`make test`
  默认走 `python3`。`plan` 和评测 `--dry-run` 不能去下载 27B 权重。
- 配方路径保持 `recipes/<name>.yaml`。新增 YAML 时在 `recipes/README.md` 补一行。
- 不要把密钥提交进 git：`.env`、`.oss.env`、`docker-compose.override.yml`、
  SSH、桶名、租户 DNS。
- 权重、Hub 缓存和 GPQA journal 放在 `outputs/`、`.cache/`、`models/` 或
  `data/`，这些目录已被忽略。
- 操作说明以 `docs/README.md` 为入口，模型细节在 `docs/qwen3.8-27b.md`，
  Spark 流程在 `docs/DGX_SPARK.md`，代理约定在 `SKILL.md`。公开协议变动时，
  同步相关指南与代理约定，保留日期命名的历史证据。
