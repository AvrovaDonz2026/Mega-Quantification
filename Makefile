.PHONY: host-check build plan quantize mixed w4a4 shell image-tar

host-check:
	bash docker/host-check.sh

build:
	docker compose build

plan:
	docker compose run --rm megaquant plan -c $${RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml}

quantize:
	docker compose --profile gpu run --rm quantize

mixed:
	docker compose --profile gpu run --rm mixed

w4a4:
	docker compose --profile gpu run --rm w4a4

shell:
	docker compose --profile dev run --rm shell

image-tar:
	bash docker/export-image.sh megaquant-nvfp4.tar
