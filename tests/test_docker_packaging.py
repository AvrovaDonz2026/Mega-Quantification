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
    for name in ("megaquant", "quantize", "mixed", "shell"):
        assert name in services, f"missing compose service {name}"

    profiles = services["quantize"].get("profiles") or []
    assert "gpu" in profiles

    volume_blob = text
    megaquant_vols = services["megaquant"].get("volumes") or []
    if megaquant_vols:
        volume_blob = "\n".join(str(item) for item in megaquant_vols)
    assert "/cache/huggingface" in volume_blob
    assert "/models" in volume_blob
    assert "/opt/megaquant/outputs" in volume_blob
    assert "/opt/megaquant/offload" in volume_blob
    assert "offload" in text


def test_entrypoint_bare_plan_uses_recipe_default(repo_root: Path) -> None:
    script = (repo_root / "docker" / "entrypoint.sh").read_text()
    assert "RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml" in script
    assert '== "plan"' in script or "== 'plan'" in script
    assert "-c" in script
