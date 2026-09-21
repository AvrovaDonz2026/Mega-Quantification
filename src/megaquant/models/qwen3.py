"""Vanilla Qwen3 dense / MoE (not the Qwen3.5/3.8 Gated-DeltaNet hybrid).

``model_type`` values ``qwen3`` and ``qwen3_moe`` must not steal Qwen3.8, which
is ``qwen3_5`` / ``Qwen3_5ForConditionalGeneration``.
"""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, glob_to_ignore, recipe_flag

__all__ = ["Qwen3Family"]


class Qwen3Family(BaseFamily):
    name = "qwen3"
    model_types = ("qwen3", "qwen3_moe")
    architectures = (
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3ForConditionalGeneration",
        "Qwen3MoeForConditionalGeneration",
    )

    def default_ignore(self, recipe: Any) -> list[str]:
        patterns = ["*embed_tokens*", "*embed_positions*"]
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            patterns.extend(["*visual*", "*vision*"])
        # Do not ignore mlp / experts — MoE routed experts are the main GEMMs.
        return glob_to_ignore(patterns)


Qwen3Family().register()
