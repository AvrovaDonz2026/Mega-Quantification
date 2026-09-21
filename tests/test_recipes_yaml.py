"""Recipe YAML shape — no pydantic / GPU / Hub required."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_KEYS = (
    "name",
    "model",
    "backend",
    "scheme",
    "algorithm",
    "kv_cache",
    "family",
    "extra_ignore",
    "calibration",
    "export",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES = REPO_ROOT / "recipes"
W4A8 = RECIPES / "qwen3.8-27b-nvfp4-w4a8.yaml"
MIXED = RECIPES / "qwen3.8-27b-nvfp4-mixed.yaml"
W4A4 = RECIPES / "qwen3.8-27b-nvfp4-w4a4.yaml"


def _load(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict), f"{path} must be a mapping"
    return payload


def _assert_schema(data: dict, path: Path) -> None:
    missing = [key for key in REQUIRED_TOP_KEYS if key not in data]
    assert not missing, f"{path.name} missing keys: {missing}"
    assert "source" in data["model"]
    assert "output_dir" in data["export"]
    assert "num_samples" in data["calibration"]
    assert "dataset" in data["calibration"]


def test_w4a8_recipe_yaml_shape_and_source() -> None:
    data = _load(W4A8)
    _assert_schema(data, W4A8)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_w4a8"
    assert data["backend"] == "modelopt"
    assert data["algorithm"] == "max"
    assert data["kv_cache"] == "fp8"
    assert data["family"] == "qwen3_5"
    assert data["calibration"]["num_samples"] == 512
    assert data["calibration"]["dataset"] == "nvidia/Nemotron-Post-Training-Dataset-v2"
    assert data["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-W4A8"
    assert data["model"]["quantize_vision"] is False
    assert data["model"]["quantize_mtp"] is False


def test_w4a8_recipe_documents_block_size_32() -> None:
    text = W4A8.read_text()
    lowered = text.lower()
    assert "32" in text
    assert "block" in lowered or "nvfp4_bs32" in lowered or "group" in lowered
    assert "w4a8" in lowered


def test_mixed_recipe_matches_nvidia_intent() -> None:
    data = _load(MIXED)
    _assert_schema(data, MIXED)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_mixed"
    assert data["algorithm"] == "local_hessian"
    assert data["calibration"]["num_samples"] == 2048
    assert data["export"]["output_dir"] == "outputs/Qwen3.8-27B-NVFP4-mixed"
    assert data["family"] == "qwen3_5"


def test_w4a4_comparison_recipe_if_present() -> None:
    if not W4A4.is_file():
        return
    data = _load(W4A4)
    _assert_schema(data, W4A4)
    assert data["model"]["source"] == "Qwen/Qwen3.8-27B"
    assert data["scheme"] == "nvfp4_w4a4"
    text = W4A4.read_text().lower()
    assert "16" in text
    assert "block" in text or "group" in text
