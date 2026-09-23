# Contributing

Mega-Quantification is AGPL-3.0-or-later. Copyright 2026 Donz. See `LICENSE`.

协议是 GNU AGPL v3，或以后的版本。

## English

- Run `python -m pytest` before opening a pull request. Tests are CPU-only.
  `megaquant plan` and eval `--dry-run` must not download the 27B checkpoint.
- Recipe paths stay `recipes/<name>.yaml`. Add a row to `recipes/README.md`
  when you add a YAML file.
- Keep secrets out of git: `.env`, `.oss.env`, `docker-compose.override.yml`,
  SSH material, bucket names, and tenant DNS addresses.
- Weights, Hub caches, and GPQA journals belong in `outputs/`, `.cache/`,
  `models/`, or `data/`, which are gitignored.
- Human notes live in `docs/qwen3.8-27b.md`. Agent constraints live in
  `SKILL.md`. Update both when the public protocol changes.

## 中文

- 提交前跑 `python -m pytest`。测试不需要 GPU。`plan` 和评测 `--dry-run`
  不能去下载 27B 权重。
- 配方路径保持 `recipes/<name>.yaml`。新增 YAML 时在 `recipes/README.md` 补一行。
- 不要把密钥提交进 git：`.env`、`.oss.env`、`docker-compose.override.yml`、
  SSH、桶名、租户 DNS。
- 权重、Hub 缓存和 GPQA journal 放在 `outputs/`、`.cache/`、`models/` 或
  `data/`，这些目录已被忽略。
- 给人读的说明在 `docs/qwen3.8-27b.md`，给代理的约定在 `SKILL.md`。公开协议变了，两处一起改。
