"""Fallback family used when model_type is unknown."""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, get_field, glob_to_ignore, recipe_model


class GenericFamily(BaseFamily):
    name = "generic"
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    def default_ignore(self, recipe: Any) -> list[str]:
        ignore = glob_to_ignore("*visual*", "*vision*", "*embed_tokens*")
        model = recipe_model(recipe)
        if not get_field(model, "quantize_mtp", False):
            ignore.append("*mtp*")
        return ignore
