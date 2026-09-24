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
    assert mixed.calibration.dataset == "nvidia/Nemotron-Post-Training-Dataset-v3"


def test_load_w4a4_comparison_if_present() -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")
    if not W4A4.is_file():
        pytest.skip("optional W4A4 comparison recipe not shipped")
    recipe = load_recipe(W4A4)
    assert recipe.scheme == "nvfp4_w4a4"
    assert recipe.model.source == "Qwen/Qwen3.8-27B"


def test_load_5090_packed_recipes() -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")
    w4a4 = load_recipe(REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a4.5090.yaml")
    mixed = load_recipe(REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-mixed.5090.yaml")
    assert w4a4.scheme == "nvfp4_w4a4"
    assert mixed.scheme == "nvfp4_mixed"
    assert w4a4.algorithm == "max"
    assert mixed.algorithm == "max"
    assert w4a4.calibration.batch_size == 4
    assert mixed.calibration.batch_size == 4
    assert w4a4.calibration.dataset == "HuggingFaceH4/ultrachat_200k"
    assert mixed.calibration.dataset == "HuggingFaceH4/ultrachat_200k"


def test_load_public_calib_recipes_if_present() -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")
    paths = (
        REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a4.public-calib.yaml",
        REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-mixed.public-calib.yaml",
        REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a8.public-calib.yaml",
    )
    loaded = 0
    for path in paths:
        if not path.is_file():
            continue
        recipe = load_recipe(path)
        assert recipe.model.source == "Qwen/Qwen3.8-27B"
        assert recipe.family == "qwen3_5"
        assert recipe.algorithm == "max"
        assert recipe.calibration.dataset == "HuggingFaceH4/ultrachat_200k"
        loaded += 1
    if loaded == 0:
        pytest.skip("public-calib recipes not shipped")


def _minimal_recipe_payload(**overrides) -> dict:
    payload: dict = {
        "name": "generic-lm-head",
        "scheme": "nvfp4_w4a8",
        "family": "generic",
        "model": {"source": "dummy/model"},
        "calibration": {},
        "export": {"output_dir": "outputs/x"},
    }
    payload.update(overrides)
    return payload


def test_ignore_lm_head_accepted_on_recipe_and_model_spec() -> None:
    from pydantic import ValidationError

    from megaquant.config import ModelSpec, Recipe

    recipe = Recipe.model_validate(_minimal_recipe_payload(ignore_lm_head=False))
    assert recipe.ignore_lm_head is False
    assert recipe.model.ignore_lm_head is None

    nested = _minimal_recipe_payload()
    nested["model"] = {"source": "dummy/model", "ignore_lm_head": False}
    recipe = Recipe.model_validate(nested)
    assert recipe.ignore_lm_head is None
    assert recipe.model.ignore_lm_head is False

    spec = ModelSpec.model_validate({"source": "dummy/model", "ignore_lm_head": True})
    assert spec.ignore_lm_head is True

    default = Recipe.model_validate(_minimal_recipe_payload())
    assert default.ignore_lm_head is None
    assert default.model.ignore_lm_head is None

    with pytest.raises(ValidationError):
        Recipe.model_validate(_minimal_recipe_payload(not_a_real_field=True))
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({"source": "x", "not_a_real_field": True})


def test_load_recipe_yaml_honors_ignore_lm_head(tmp_path: Path) -> None:
    load_recipe = _load_recipe_fn()
    if load_recipe is None:
        pytest.skip("megaquant.config.load_recipe not available")
    import yaml

    top_level = tmp_path / "ignore-lm-head-top.yaml"
    top_level.write_text(
        yaml.safe_dump(_minimal_recipe_payload(ignore_lm_head=False)),
        encoding="utf-8",
    )
    recipe = load_recipe(top_level)
    assert recipe.ignore_lm_head is False

    nested = _minimal_recipe_payload()
    nested["model"] = {"source": "dummy/model", "ignore_lm_head": False}
    nested_path = tmp_path / "ignore-lm-head-model.yaml"
    nested_path.write_text(yaml.safe_dump(nested), encoding="utf-8")
    recipe = load_recipe(nested_path)
    assert recipe.model.ignore_lm_head is False
