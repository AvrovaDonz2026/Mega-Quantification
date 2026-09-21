"""Thin Llama adapter — proves the family registry is model-agnostic."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import glob_to_ignore


class LlamaFamily:
    name = "llama"
    model_types = ("llama", "mllama")
    architectures = ("LlamaForCausalLM", "MllamaForConditionalGeneration")

    def default_ignore(self, recipe: Any) -> list[str]:
        return glob_to_ignore("*embed_tokens*", "*visual*")

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        model = getattr(recipe, "model", None)
        return {
            "trust_remote_code": getattr(model, "trust_remote_code", True),
            "device_map": getattr(model, "device_map", "auto"),
        }
