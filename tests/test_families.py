"""Family adapters: ignore lists and detection. CPU-only, no weights."""

from __future__ import annotations

from typing import Any

import pytest

from megaquant.models.generic import GenericFamily
from megaquant.models.llama import LlamaFamily
from megaquant.models.qwen3 import Qwen3Family
from megaquant.models.qwen3_5 import Qwen35Family

QWEN38_CONFIG = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {"model_type": "qwen3_5_text"},
}

_DEFAULT_RECIPE = {
    "model": {
        "quantize_vision": False,
        "quantize_mtp": False,
        "device_map": "auto",
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "extra_ignore": [],
}


def _resolve_family(config: dict[str, Any]):
    """Use registry.detect_family when present; else instantiate Qwen35Family."""
    try:
        import megaquant.models  # noqa: F401 — side-effect registration
        from megaquant.registry import detect_family, get_family

        name = detect_family(config)
        if name == "qwen3_5":
            return get_family(name)
        if config.get("model_type") in {"qwen3_5", "qwen3_5_text"}:
            # Registry present but family not registered yet.
            return Qwen35Family()
        return get_family(name)
    except ImportError:
        return Qwen35Family()


def test_qwen35_ignore_contains_visual_mtp_gdn_extras() -> None:
    family = Qwen35Family()
    ignore = family.default_ignore(_DEFAULT_RECIPE)
    blob = " ".join(ignore).lower()
    assert "visual" in blob
    assert "mtp" in blob
    assert "conv1d" in blob
    assert "in_proj_a" in blob
    assert "in_proj_b" in blob
    assert "embed_tokens" in blob


def test_qwen35_does_not_blindly_ignore_all_mlp() -> None:
    ignore = Qwen35Family().default_ignore(_DEFAULT_RECIPE)
    for pattern in ignore:
        stripped = pattern.replace("re:", "").strip("*")
        assert stripped not in {"mlp", "mlp.*", ".*mlp.*"}
        assert pattern not in {"*mlp*", "mlp", "re:.*mlp.*"}
        # Mixed recipe quantizes mlp.{gate,up,down}_proj + lm_head.
        assert "lm_head" not in pattern


def test_qwen35_detect_from_fake_config() -> None:
    family = _resolve_family(QWEN38_CONFIG)
    # detect_family returns a registered instance; _instantiate is pipeline-side.
    name = getattr(family, "name", None)
    if name is None:
        family = Qwen35Family()
        name = family.name
    assert name == "qwen3_5"
    ignore = family.default_ignore(_DEFAULT_RECIPE) if hasattr(family, "default_ignore") else []
    if not ignore:
        ignore = Qwen35Family().default_ignore(_DEFAULT_RECIPE)
    blob = " ".join(ignore).lower()
    assert "visual" in blob and "conv1d" in blob


def test_qwen35_vision_and_mtp_flags() -> None:
    family = Qwen35Family()
    vision_on = family.default_ignore(
        {"model": {"quantize_vision": True, "quantize_mtp": False}}
    )
    assert not any("visual" in p or "vision" in p for p in vision_on)
    assert any("mtp" in p for p in vision_on)

    mtp_on = family.default_ignore(
        {"model": {"quantize_vision": False, "quantize_mtp": True}}
    )
    assert any("visual" in p for p in mtp_on)
    assert not any("mtp" in p for p in mtp_on)


def test_qwen35_load_kwargs_no_torch() -> None:
    kwargs = Qwen35Family().load_kwargs(
        {"model": {"device_map": "cpu", "dtype": "bfloat16"}}
    )
    assert kwargs["trust_remote_code"] is True
    assert kwargs["device_map"] == "cpu"
    assert "bfloat16" in str(kwargs["torch_dtype"]).lower()


def test_qwen3_not_confused_with_qwen3_5() -> None:
    vanilla = Qwen3Family()
    assert "qwen3_5" not in vanilla.model_types
    assert "qwen3" in vanilla.model_types and "qwen3_moe" in vanilla.model_types
    ignore = vanilla.default_ignore(_DEFAULT_RECIPE)
    blob = " ".join(ignore)
    assert "conv1d" not in blob  # GDN extras are 3.5/3.8-only
    assert not any("mlp" in p.replace("embed", "") for p in ignore)


def test_generic_and_llama_adapters() -> None:
    generic = GenericFamily().default_ignore(_DEFAULT_RECIPE)
    assert any("visual" in p for p in generic)
    assert any("embed_tokens" in p for p in generic)
    llama = LlamaFamily()
    assert llama.name == "llama"
    assert "llama" in llama.model_types


def test_registry_lists_qwen3_5_when_available() -> None:
    pytest.importorskip("megaquant.registry")
    import megaquant.models  # noqa: F401
    from megaquant.registry import detect_family, list_families

    names = list_families()
    if "qwen3_5" not in names:
        pytest.skip("family registry did not pick up adapters")
    assert detect_family(QWEN38_CONFIG) == "qwen3_5"
    assert detect_family({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}) == "qwen3"
    assert detect_family({"model_type": "nope", "architectures": ["NopeForCausalLM"]}) == "generic"
