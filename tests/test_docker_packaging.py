"""Docker packaging invariants. CPU-only; does not run Docker."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_dockerfile_does_not_copy_weights(repo_root: Path) -> None:
    dockerfile = (repo_root / "Dockerfile").read_text()
    copy_lines = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().startswith("COPY ")
    ]
    assert copy_lines
    for line in copy_lines:
        lowered = line.lower()
        assert "weight" not in lowered
        assert ".safetensors" not in lowered
        assert "huggingface" not in lowered
        # Do not bake a local 27B checkout into the image.
        assert " models" not in f" {line} ".lower()
        assert not line.lower().startswith("copy models")


def test_compose_services_profiles_and_volumes(repo_root: Path) -> None:
    compose_path = repo_root / "docker-compose.yml"
    text = compose_path.read_text()
    data = yaml.safe_load(text)
    services = data["services"]
    for name in ("megaquant", "quantize", "mixed", "w4a4", "serve-vllm", "eval-gpqa", "shell"):
        assert name in services, f"missing compose service {name}"

    profiles = services["quantize"].get("profiles") or []
    assert "gpu" in profiles
    assert "gpu" in (services["w4a4"].get("profiles") or [])
    assert "gpu" in (services["eval-gpqa"].get("profiles") or [])
    assert "gpu" in (services["serve-vllm"].get("profiles") or [])
    mixed_cmd = " ".join(str(x) for x in (services["mixed"].get("command") or []))
    assert "mixed.5090.yaml" in mixed_cmd
    w4a4_cmd = " ".join(str(x) for x in (services["w4a4"].get("command") or []))
    assert "w4a4.5090.yaml" in w4a4_cmd
    eval_cmd = " ".join(str(x) for x in (services["eval-gpqa"].get("command") or []))
    assert "eval-gpqa-diamond.yaml" in eval_cmd
    serve_cmd = " ".join(str(x) for x in (services["serve-vllm"].get("command") or []))
    assert "serve" in serve_cmd
    assert "eval-gpqa-diamond.yaml" in serve_cmd

    volume_blob = text
    megaquant_vols = services["megaquant"].get("volumes") or []
    if megaquant_vols:
        volume_blob = "\n".join(str(item) for item in megaquant_vols)
    assert "/cache/huggingface" in volume_blob
    assert "/models" in volume_blob
    assert "/opt/megaquant/outputs" in volume_blob
    assert "/opt/megaquant/offload" in volume_blob
    assert "offload" in text


def test_gpu_pod_packs_host_ram_threads_and_batch(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "gpu-pod.sh").read_text()
    assert "MEGAQUANT_GPU_HEADROOM_GIB" in script
    assert "MEGAQUANT_CPU_RESERVE_GIB" in script
    assert "MEGAQUANT_BATCH_SIZE" in script
    assert "OMP_NUM_THREADS" in script
    assert "nproc" in script
    assert "0:26GiB,cpu:40GiB" not in script
    assert "batch_size: 4" in script
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in script
    assert "w4a4" in script
    assert "mixed" in script
    assert "qwen3.8-27b-nvfp4-${SCHEME}.5090.yaml" in script
    packed = "qwen3.8-27b-nvfp4-${SCHEME}.pod.yaml"
    assert packed in script or "qwen3.8-27b-nvfp4-w4a4.pod.yaml" in script
    assert "nvfp4_mixed" in script
    assert "nvfp4_w4a4" in script
    assert 'MEGAQUANT_GPU_HEADROOM_GIB:-1' in script or 'MEGAQUANT_GPU_HEADROOM_GIB:-"1"' in script
    assert "eval-gpqa-diamond.yaml" in script
    assert "serve|eval" in script
    assert "KV CPU offload" in script
    assert "gpqa_diamond-${SCHEME}" in script


def test_ngc_compose_covers_eval_and_serve(repo_root: Path) -> None:
    data = yaml.safe_load((repo_root / "docker-compose.ngc.yml").read_text())
    services = data["services"]
    for name in ("eval-gpqa", "serve-vllm", "w4a4", "mixed"):
        assert name in services, name


def test_dockerfile_copies_scripts(repo_root: Path) -> None:
    dockerfile = (repo_root / "Dockerfile").read_text()
    assert "COPY scripts ./scripts" in dockerfile


def test_entrypoint_bare_plan_uses_recipe_default(repo_root: Path) -> None:
    script = (repo_root / "docker" / "entrypoint.sh").read_text()
    assert "RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml" in script
    assert '== "plan"' in script or "== 'plan'" in script
    assert "-c" in script
    assert "eval-gpqa-diamond.yaml" in script
    assert '"serve"' in script


def test_serve_and_eval_scripts_are_executable_helpers(repo_root: Path) -> None:
    serve = (repo_root / "scripts" / "serve_vllm.sh").read_text()
    eval_sh = (repo_root / "scripts" / "eval_gpqa.sh").read_text()
    assert "megaquant.cli serve" in serve
    assert "KV" in serve
    assert "megaquant.cli eval" in eval_sh
    assert "eval-gpqa-diamond.yaml" in eval_sh
