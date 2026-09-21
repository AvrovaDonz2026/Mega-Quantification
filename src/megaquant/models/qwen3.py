"""Vanilla Qwen3 / Qwen3-MoE (not the 3.5/3.8 hybrid Gated-DeltaNet stack)."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, glob_to_ignore


class Qwen3Family(BaseFamily):
    name = "qwen3"
    model_types = ("qwen3", "qwen3_moe")
    architectures = ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM")

    def default_ignore(self, recipe: Any) -> list[str]:
        return glob_to_ignore("*visual*", "*vision*", "*embed_tokens*")
