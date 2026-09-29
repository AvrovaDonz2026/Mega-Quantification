"""Auxiliary branches must execute during activation calibration."""

from fnmatch import fnmatchcase
from types import SimpleNamespace

import pytest

from megaquant.backends.modelopt import ModelOptBackend, _auxiliary_calibration_coverage
from megaquant.config import load_recipe
from megaquant.exceptions import BackendError
from megaquant.pipeline import QuantPipeline


def test_opt_in_preserves_embedding_and_kv_exclusions(recipes_dir):
    recipe = load_recipe(recipes_dir / "qwen3.8-27b-nvfp4-w4a8.spark.yaml")
    cfg = ModelOptBackend().build_quant_cfg(QuantPipeline(recipe).resolve())
    entries = cfg["quant_cfg"]
    if isinstance(entries, dict):
        entries = [{"quantizer_name": key, **value} for key, value in entries.items()]

    def enabled(name):
        active = False
        for entry in entries:
            if entry.get("parent_class") or not fnmatchcase(name, entry["quantizer_name"]):
                continue
            if "enable" in entry:
                active = entry["enable"]
            elif entry.get("cfg"):
                active = True
        return active

    for name in ("model.visual.blocks.0.attn.qkv", "mtp.fc", "mtp.layers.0.mlp.up_proj"):
        assert enabled(name + ".weight_quantizer")
        assert enabled(name + ".input_quantizer")
        assert not enabled(name + ".output_quantizer")
    for name in ("model.visual.pos_embed", "model.visual.patch_embed.proj"):
        assert not enabled(name + ".weight_quantizer")
        assert not enabled(name + ".input_quantizer")
    assert not enabled("mtp.layers.0.self_attn.k_bmm_quantizer")


def test_coverage_rejects_unvisited_branch_and_removes_hooks():
    torch = pytest.importorskip("torch")

    class Quantizer(torch.nn.Identity):
        is_enabled = True

    model = torch.nn.Module()
    model.visual = torch.nn.Module()
    model.visual.input_quantizer = Quantizer()
    model.mtp = torch.nn.Module()
    model.mtp.input_quantizer = Quantizer()
    recipe = SimpleNamespace(model=SimpleNamespace(quantize_vision=True, quantize_mtp=True))
    with pytest.raises(BackendError, match="mtp activation calibration did not run"):
        with _auxiliary_calibration_coverage(model, recipe):
            model.visual.input_quantizer(torch.ones(1))
    assert not model.visual.input_quantizer._forward_hooks
    assert not model.mtp.input_quantizer._forward_hooks
    with _auxiliary_calibration_coverage(model, recipe):
        model.visual.input_quantizer(torch.ones(1))
        model.mtp.input_quantizer(torch.ones(1))
    assert model._megaquant_calibration_coverage == {
        "vision": {"visual.input_quantizer": 1},
        "mtp": {"mtp.input_quantizer": 1},
    }
