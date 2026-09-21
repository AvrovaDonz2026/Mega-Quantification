"""Fallback family used when model_type is unknown."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import glob_to_ignore


class GenericFamily:
    name = "generic"
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    def default_ignore(self, recipe: Any) -> list[str]:
        ignore = glob_to_ignore("*visual*", "*vision*", "*embed_tokens*")
        model = getattr(recipe, "model", None)
        if not getattr(model, "quantize_mtp", False):
            ignore.append("*mtp*")
        return ignore

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        model = getattr(recipe, "model", None)
        return {
            "trust_remote_code": getattr(model, "trust_remote_code", True),
            "device_map": getattr(model, "device_map", "auto"),
        }
