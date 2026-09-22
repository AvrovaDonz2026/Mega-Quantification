.PHONY: host-check install-host build plan quantize mixed w4a4 serve-sglang serve-vllm eval-gpqa fetch-export fetch-gpqa shell image-tar test

test:
	python -m pytest

host-check:
	bash docker/host-check.sh

install-host:
	bash docker/install-host.sh

build:
	docker compose --profile gpu --profile dev build

plan:
	docker compose run --rm megaquant plan -c $${RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml}

quantize:
	docker compose --profile gpu run --rm quantize

mixed:
	docker compose --profile gpu run --rm mixed

w4a4:
	docker compose --profile gpu run --rm w4a4

serve-sglang:
	docker compose --profile gpu up serve-sglang

serve-vllm: serve-sglang

eval-gpqa:
	docker compose --profile gpu run --rm eval-gpqa

fetch-export:
	docker compose --profile gpu run --rm fetch-export

fetch-gpqa:
	docker compose --profile gpu run --rm fetch-gpqa

shell:
	docker compose --profile dev run --rm shell

image-tar:
	bash docker/export-image.sh megaquant-stack.tar
