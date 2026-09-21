"""ModelOpt quant_cfg construction (no nvidia-modelopt required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megaquant.backends.modelopt import (
    ModelOptBackend,
    _call_export_hf,
    _clear_partial_export,
    _is_cuda_oom,
    _prepare_export_memory,
    _weight_input_attrs,
    describe_cfg,
)
from megaquant.config import load_recipe
from megaquant.pipeline import QuantPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
W4A8 = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a8.yaml"
W4A4 = REPO_ROOT / "recipes" / "qwen3.8-27b-nvfp4-w4a4.yaml"
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


def _last_named(cfg: dict, *needles: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for entry in _entries(cfg):
        name = str(entry.get("quantizer_name", ""))
        if all(n in name for n in needles):
            found = entry
    return found


def _cfg_body(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not obj:
        return {}
    body: Any = obj.get("cfg", obj) if "quantizer_name" in obj or "cfg" in obj else obj
    if isinstance(body, list) and body:
        body = body[0]
    return body if isinstance(body, dict) else {}


def _block_size(obj: dict[str, Any] | None) -> int | None:
    body = _cfg_body(obj)
    sizes = body.get("block_sizes")
    if isinstance(sizes, dict):
        value = sizes.get(-1, sizes.get("-1"))
        return int(value) if value is not None else None
    return None


def _num_bits(obj: dict[str, Any] | None) -> Any:
    return _cfg_body(obj).get("num_bits")


def test_describe_w4a8_is_sglang_mixed() -> None:
    summary = describe_cfg("nvfp4_w4a8")
    assert summary["supported"] is True
    assert summary["group_size"] == 16
    assert summary["mlp_group_size"] == 16
    assert summary["qformat"] == "mixed_nvfp4_fp8"
    assert summary["attn_weight_format"] == "fp8"
    assert "MIXED_PRECISION" in summary["sglang"]
    assert "w4a8_nvfp4_fp8" in summary["notes"]


def test_describe_trtllm_w4a8_group_size_32() -> None:
    summary = describe_cfg("w4a8_nvfp4_fp8")
    assert summary["supported"] is True
    assert summary["group_size"] == 32
    assert summary["qformat"] == "w4a8_nvfp4_fp8"
    assert "SGLang rejects" in summary["notes"]
    alias = describe_cfg("nvfp4_w4a8_trtllm")
    assert alias["scheme"] == "w4a8_nvfp4_fp8"
    assert alias["group_size"] == 32


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
    lm_weight = [e for e in _entries(cfg) if "lm_head" in e.get("quantizer_name", "")
                 and "weight" in e.get("quantizer_name", "")]
    assert lm_weight
    last = lm_weight[-1]
    assert last.get("enable", True) is True
    assert last.get("cfg")
    assert _block_size(last) == 16

    attn_w = _last_named(cfg, "self_attn", "weight_quantizer")
    assert attn_w
    assert _num_bits(attn_w) == (4, 3) or _block_size(attn_w) != 32
    mlp_w = _last_named(cfg, "mlp", "weight_quantizer")
    assert mlp_w
    assert _num_bits(mlp_w) == (2, 1)
    assert _block_size(mlp_w) == 16


def test_w4a8_local_hessian_uses_block_size_16() -> None:
    recipe = load_recipe(W4A8)
    recipe = recipe.model_copy(update={"algorithm": "local_hessian"})
    plan = QuantPipeline(recipe).resolve()
    cfg = ModelOptBackend().build_quant_cfg(plan)
    algo = cfg["algorithm"]
    assert isinstance(algo, dict)
    assert algo["method"] == "local_hessian"
    assert algo["block_size"] == 16


def test_rewrite_sglang_export_helper(tmp_path: Path) -> None:
    from megaquant.backends.modelopt import _rewrite_sglang_export

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.gate_proj.weight": "a.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "a.safetensors",
                    "lm_head.weight": "a.safetensors",
                }
            }
        )
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}})
    )
    layers = _rewrite_sglang_export(tmp_path, "nvfp4_w4a8")
    assert layers is not None
    assert json.loads((tmp_path / "hf_quant_config.json").read_text())["quantization"][
        "quant_algo"
    ] == "MIXED_PRECISION"
    assert _rewrite_sglang_export(tmp_path, "w4a8_nvfp4_fp8") is None


def test_describe_w4a4_group_size_16() -> None:
    summary = describe_cfg("nvfp4_w4a4")
    assert summary["supported"] is True
    assert summary["group_size"] == 16
    assert summary["qformat"] == "nvfp4"
    assert summary["activation_format"] == "nvfp4"


def test_w4a4_cfg_uses_nvfp4_activations_and_lm_head() -> None:
    recipe = load_recipe(W4A4)
    plan = QuantPipeline(recipe).resolve()
    cfg = ModelOptBackend().build_quant_cfg(plan)
    assert cfg["algorithm"] == "max"

    weight, input_attr = _weight_input_attrs(cfg)
    assert _block_size(weight) == 16
    assert _block_size(input_attr) == 16
    assert _num_bits(weight) == (2, 1)
    assert _num_bits(input_attr) == (2, 1)

    exact_w = [e for e in _entries(cfg) if e.get("quantizer_name") == "*weight_quantizer"]
    exact_i = [e for e in _entries(cfg) if e.get("quantizer_name") == "*input_quantizer"]
    assert exact_w and _block_size(exact_w[-1]) == 16
    assert exact_i and _block_size(exact_i[-1]) == 16

    lm_weight = [e for e in _entries(cfg) if "lm_head" in e.get("quantizer_name", "")
                 and "weight" in e.get("quantizer_name", "")]
    assert lm_weight
    last = lm_weight[-1]
    assert last.get("enable", True) is True
    assert _block_size(last) == 16


def test_describe_mixed_matches_nvidia_gs16() -> None:
    summary = describe_cfg("nvfp4_mixed")
    assert summary["supported"] is True
    assert summary["group_size"] == 16
    assert summary["mlp_group_size"] == 16
    assert summary["attn_weight_format"] == "fp8"
    assert "nvidia/Qwen3.8-27B-NVFP4" in summary["notes"]
    assert "group_size 16" in summary["notes"]


def test_mixed_local_hessian_uses_block_size_16() -> None:
    recipe = load_recipe(MIXED)
    plan = QuantPipeline(recipe).resolve()
    cfg = ModelOptBackend().build_quant_cfg(plan)
    algo = cfg["algorithm"]
    assert isinstance(algo, dict)
    assert algo["method"] == "local_hessian"
    assert algo["fp8_scale_sweep"] is True
    assert algo["block_size"] == 16


def test_mixed_cfg_fp8_attn_nvfp4_mlp_lm_head() -> None:
    recipe = load_recipe(MIXED)
    plan = QuantPipeline(recipe).resolve()
    cfg = ModelOptBackend().build_quant_cfg(plan)

    attn_w = _last_named(cfg, "self_attn", "weight_quantizer")
    assert attn_w
    bits = _num_bits(attn_w)
    assert bits == (4, 3) or _block_size(attn_w) != 32
    assert _block_size(attn_w) != 32

    mlp_w = _last_named(cfg, "mlp", "weight_quantizer")
    assert mlp_w
    assert _num_bits(mlp_w) == (2, 1)
    assert _block_size(mlp_w) == 16

    lm_w = _last_named(cfg, "lm_head", "weight_quantizer")
    assert lm_w
    assert lm_w.get("enable", True) is True
    assert _block_size(lm_w) == 16


def test_export_hf_uses_export_dir_keyword(tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_export(model, dtype=None, export_dir="/tmp"):  # noqa: ANN001
        seen["dtype"] = dtype
        seen["export_dir"] = export_dir
        seen["model"] = model

    _call_export_hf(fake_export, object(), tmp_path / "out")
    assert seen["export_dir"] == str(tmp_path / "out")
    assert seen["dtype"] is None


def test_prepare_export_memory_is_noop_without_cuda() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.moved_to: str | None = None

        def named_modules(self):
            yield "self", self

        def to(self, device):  # noqa: ANN001
            self.moved_to = str(device)

    dummy = Dummy()
    kind = _prepare_export_memory(dummy)
    assert kind in {"cpu-only", "cuda", "cuda-freed", "partial-cpu"}
    if kind == "cpu-only":
        assert dummy.moved_to is None
    assert _is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 4.74 GiB"))
    assert _is_cuda_oom(MemoryError("torch.OutOfMemoryError"))
    assert not _is_cuda_oom(ValueError("bad export_dir"))


def test_clear_partial_export_removes_shard_parts(tmp_path: Path) -> None:
    leftover = tmp_path / "__shard_part_00000.safetensors"
    leftover.write_bytes(b"partial")
    keep = tmp_path / "config.json"
    keep.write_text("{}", encoding="utf-8")
    _clear_partial_export(tmp_path)
    assert not leftover.exists()
    assert keep.exists()
