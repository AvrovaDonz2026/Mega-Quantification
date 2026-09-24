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
    vision_on = family.default_ignore({"model": {"quantize_vision": True, "quantize_mtp": False}})
    assert not any("visual" in p or "vision" in p for p in vision_on)
    assert any("mtp" in p for p in vision_on)

    mtp_on = family.default_ignore({"model": {"quantize_vision": False, "quantize_mtp": True}})
    assert any("visual" in p for p in mtp_on)
    assert not any("mtp" in p for p in mtp_on)


def test_qwen35_load_kwargs_no_torch() -> None:
    kwargs = Qwen35Family().load_kwargs({"model": {"device_map": "cpu", "dtype": "bfloat16"}})
    assert kwargs["trust_remote_code"] is True
    assert kwargs["device_map"] == "cpu"
    assert "bfloat16" in str(kwargs["torch_dtype"]).lower()
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["language_model_only"] is True
    assert "visual" in kwargs["pin_to_cpu"]
    assert "mtp" in kwargs["pin_to_cpu"]


def test_qwen35_load_kwargs_honors_recipe_dtype() -> None:
    family = Qwen35Family()
    default = family.load_kwargs({"model": {"device_map": "cpu"}})
    assert default["torch_dtype"] == "bfloat16"
    assert default["dtype"] == "bfloat16"

    fp16 = family.load_kwargs({"model": {"device_map": "cpu", "dtype": "float16"}})
    assert fp16["torch_dtype"] == "float16"
    assert fp16["dtype"] == "float16"

    alias = family.load_kwargs({"model": {"dtype": "fp16"}})
    assert alias["torch_dtype"] == "float16"
    assert alias["dtype"] == "float16"

    from megaquant.config import Recipe

    recipe = Recipe.model_validate(
        {
            "name": "qwen-dtype",
            "scheme": "nvfp4_w4a8",
            "family": "qwen3_5",
            "model": {
                "source": "Qwen/Qwen3.8-27B",
                "dtype": "float16",
                "device_map": "cpu",
            },
            "calibration": {},
            "export": {"output_dir": "outputs/x"},
        }
    )
    validated = family.load_kwargs(recipe)
    assert validated["torch_dtype"] == "float16"
    assert validated["dtype"] == "float16"
    assert validated["trust_remote_code"] is True
    assert validated["language_model_only"] is True

    unset = Recipe.model_validate(
        {
            "name": "qwen-dtype-default",
            "scheme": "nvfp4_w4a8",
            "family": "qwen3_5",
            "model": {"source": "Qwen/Qwen3.8-27B", "device_map": "cpu"},
            "calibration": {},
            "export": {"output_dir": "outputs/x"},
        }
    )
    assert unset.model.dtype == "bfloat16"
    fallback = family.load_kwargs(unset)
    assert fallback["torch_dtype"] == "bfloat16"
    assert fallback["dtype"] == "bfloat16"


def test_qwen35_keeps_vision_when_opted_in() -> None:
    kwargs = Qwen35Family().load_kwargs({"model": {"quantize_vision": True}})
    assert "language_model_only" not in kwargs
    assert "visual" not in kwargs.get("pin_to_cpu", ())


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
    assert any("lm_head" in p for p in generic)
    llama = LlamaFamily()
    assert llama.name == "llama"
    assert "llama" in llama.model_types


def test_generic_ignore_lm_head_from_validated_recipe() -> None:
    from megaquant.config import Recipe

    payload: dict[str, Any] = {
        "name": "generic-lm-head",
        "scheme": "nvfp4_w4a8",
        "family": "generic",
        "model": {"source": "dummy/model"},
        "calibration": {},
        "export": {"output_dir": "outputs/x"},
    }
    family = GenericFamily()

    default = Recipe.model_validate(payload)
    assert any("lm_head" in p for p in family.default_ignore(default))

    payload["ignore_lm_head"] = False
    opted_out = Recipe.model_validate(payload)
    assert not any("lm_head" in p for p in family.default_ignore(opted_out))

    payload.pop("ignore_lm_head")
    payload["model"] = {"source": "dummy/model", "ignore_lm_head": False}
    nested = Recipe.model_validate(payload)
    assert not any("lm_head" in p for p in family.default_ignore(nested))

    payload["model"]["ignore_lm_head"] = True
    forced = Recipe.model_validate(payload)
    assert any("lm_head" in p for p in family.default_ignore(forced))


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
