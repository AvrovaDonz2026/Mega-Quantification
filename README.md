# Mega-Quantification

Generic Hugging Face PTQ pipeline. First target: **Qwen/Qwen3.8-27B BF16 → NVFP4 W4A8**.

Run it from Docker Compose on an RTX 5090+ (Blackwell) box — do not fight CUDA wheels on the host.

```bash
bash docker/host-check.sh
cp .env.example .env          # optional HF_TOKEN
docker compose build
docker compose run --rm megaquant plan
docker compose --profile gpu run --rm quantize
```

Full 5090 runbook: [docker/README.md](docker/README.md). Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
