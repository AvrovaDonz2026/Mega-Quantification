"""Qwen3.5 / 3.6 / 3.8 dense VLMs (Qwen3_5ForConditionalGeneration).

Qwen3.8-27B uses ``model_type: qwen3_5``. Hybrid layout:

    16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))

Keep vision, MTP, embeddings, and GDN extras (conv1d / in_proj_a / in_proj_b)
in BF16 unless the recipe opts in.
"""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, get_field, glob_to_ignore, recipe_model


class Qwen35Family(BaseFamily):
    name = "qwen3_5"
    model_types = ("qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_8")
    architectures = (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
    )

    def default_ignore(self, recipe: Any) -> list[str]:
        ignore = glob_to_ignore(
            "*visual*",
            "*vision*",
            "*embed_tokens*",
            "*embed_positions*",
            "*linear_attn.conv1d*",
            "*linear_attn.in_proj_a*",
            "*linear_attn.in_proj_b*",
        )
        model = recipe_model(recipe)
        if get_field(model, "quantize_vision", False):
            ignore = [p for p in ignore if "visual" not in p and "vision" not in p]
        if not get_field(model, "quantize_mtp", False):
            ignore.append("*mtp*")
        return ignore

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        model = recipe_model(recipe)
        dtype = get_field(model, "dtype", "bfloat16")
        return {
            "trust_remote_code": get_field(model, "trust_remote_code", True),
            "device_map": get_field(model, "device_map", "auto"),
            "torch_dtype": dtype,
            "model_cls": "AutoModelForImageTextToText",
        }
