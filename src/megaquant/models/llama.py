"""Thin Llama adapter — proves the family registry is not Qwen-only."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, glob_to_ignore, recipe_flag

__all__ = ["LlamaFamily"]


class LlamaFamily(BaseFamily):
    name = "llama"
    model_types = ("llama", "mllama")
    architectures = ("LlamaForCausalLM", "MllamaForConditionalGeneration")

    def default_ignore(self, recipe: Any) -> list[str]:
        patterns = ["*embed_tokens*"]
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            patterns.append("*visual*")
        return glob_to_ignore(patterns)


LlamaFamily().register()
