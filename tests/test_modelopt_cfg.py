"""ModelOpt quant_cfg construction (no nvidia-modelopt required)."""

from __future__ import annotations

from pathlib import Path

from megaquant.backends.modelopt import ModelOptBackend, _call_export_hf, describe_cfg
from megaquant.config import load_recipe
from megaquant.pipeline import QuantPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
W4A8 = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a8.yaml"
MIXED = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-mixed.yaml"


def _entries(cfg: dict) -> list[dict]:
    quant_cfg = cfg.get("quant_cfg") or []
    if isinstance(quant_cfg, list):
        return [e for e in quant_cfg if isinstance(e, dict)]
    if isinstance(quant_cfg, dict):
        out = []
        for key, value in quant_cfg.items():
            if isinstance(value, dict):
                out.append({"quantizer_name": key, **value})
            else:
                out.append({"quantizer_name": key, "cfg": value})
        return out
    return []


def test_describe_w4a8_group_size_32() -> None:
    summary = describe_cfg("nvfp4_w4a8")
    assert summary["supported"] is True
    assert summary["group_size"] == 32
    assert summary["qformat"] == "w4a8_nvfp4_fp8"


def test_w4a8_cfg_reenables_lm_head_and_ignores_vision() -> None:
    recipe = load_recipe(W4A8)
    plan = QuantPipeline(recipe).resolve()
    assert plan.family_name == "qwen3_5"
    assert not any("lm_head" in p for p in plan.ignore)
    assert any("visual" in p for p in plan.ignore)
    assert any("conv1d" in p for p in plan.ignore)
    assert any("in_proj_a" in p for p in plan.ignore)

    cfg = ModelOptBackend().build_quant_cfg(plan)
    assert cfg["algorithm"] == "max"
    names = [e.get("quantizer_name", "") for e in _entries(cfg)]
    assert any("lm_head" in n and "weight_quantizer" in n for n in names)
    lm_weight = [e for e in _entries(cfg) if "lm_head" in e.get("quantizer_name", "") and "weight" in e.get("quantizer_name", "")]
    assert lm_weight
    last = lm_weight[-1]
    assert last.get("enable", True) is True
    assert last.get("cfg")


def test_mixed_local_hessian_uses_block_size_32() -> None:
    recipe = load_recipe(MIXED)
    plan = QuantPipeline(recipe).resolve()
    cfg = ModelOptBackend().build_quant_cfg(plan)
    algo = cfg["algorithm"]
    assert isinstance(algo, dict)
    assert algo["method"] == "local_hessian"
    assert algo["fp8_scale_sweep"] is True
    assert algo["block_size"] == 32


def test_export_hf_uses_export_dir_keyword(tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_export(model, dtype=None, export_dir="/tmp"):  # noqa: ANN001
        seen["dtype"] = dtype
        seen["export_dir"] = export_dir
        seen["model"] = model

    _call_export_hf(fake_export, object(), tmp_path / "out")
    assert seen["export_dir"] == str(tmp_path / "out")
    assert seen["dtype"] is None
