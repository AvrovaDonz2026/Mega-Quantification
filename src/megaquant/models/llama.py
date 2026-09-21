"""Thin Llama adapter — proves the family registry is model-agnostic."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, glob_to_ignore


class LlamaFamily(BaseFamily):
    name = "llama"
    model_types = ("llama", "mllama")
    architectures = ("LlamaForCausalLM", "MllamaForConditionalGeneration")

    def default_ignore(self, recipe: Any) -> list[str]:
        return glob_to_ignore("*embed_tokens*", "*visual*")
