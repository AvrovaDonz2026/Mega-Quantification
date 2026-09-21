"""Recipe YAML ↔ pydantic schema (skip if Agent Core config is absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
W4A8 = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a8.yaml"
MIXED = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-mixed.yaml"
W4A4 = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a4.yaml"


def _load_recipe_fn():
    try:
        from megaquant import config as config_mod
    except ImportError:
        return None
    fn = getattr(config_mod, "load_recipe", None)
    return fn if callable(fn) else None


def test_load_w4a8_and_mixed_yaml() -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")

    w4a8 = load_recipe(W4A8)
    mixed = load_recipe(MIXED)

    assert w4a8.model.source == "Qwen/Qwen3.8-27B"
    assert mixed.model.source == "Qwen/Qwen3.8-27B"
    assert w4a8.scheme == "nvfp4_w4a8"
    assert mixed.scheme == "nvfp4_mixed"
    assert w4a8.algorithm == "max"
    assert mixed.algorithm == "local_hessian"
    assert w4a8.calibration.num_samples == 512
    assert mixed.calibration.num_samples == 2048
    assert w4a8.family == "qwen3_5"
    assert mixed.family == "qwen3_5"
    assert w4a8.backend == "modelopt"
    assert w4a8.model.quantize_vision is False
    assert w4a8.model.quantize_mtp is False
    assert w4a8.export.output_dir == "outputs/Qwen3.8-27B-NVFP4-W4A8"
    assert mixed.export.output_dir == "outputs/Qwen3.8-27B-NVFP4-mixed"


def test_load_w4a4_comparison_if_present() -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")
    if not W4A4.is_file():
        pytest.skip("optional W4A4 comparison recipe not shipped")
    recipe = load_recipe(W4A4)
    assert recipe.scheme == "nvfp4_w4a4"
    assert recipe.model.source == "Qwen/Qwen3.8-27B"
