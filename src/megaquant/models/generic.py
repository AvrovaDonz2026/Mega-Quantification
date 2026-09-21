"""Fallback family used when ``model_type`` / ``architectures`` match nothing else.

Keep the ignore list conservative: vision towers and token embeddings stay BF16.
``lm_head`` is ignored by default (typical PTQ) but can be kept quantizable by
setting ``ignore_lm_head: false`` on the recipe or ``model.ignore_lm_head: false``.
"""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, get_field, glob_to_ignore, recipe_model

__all__ = ["GenericFamily"]


class GenericFamily(BaseFamily):
    name = "generic"
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    ignore_lm_head: bool = True

    def default_ignore(self, recipe: Any) -> list[str]:
        patterns = ["*visual*", "*vision*", "*embed_tokens*"]
        model = recipe_model(recipe)
        if not get_field(model, "quantize_mtp", False):
            patterns.append("*mtp*")
        if self._should_ignore_lm_head(recipe):
            patterns.append("*lm_head*")
        return glob_to_ignore(patterns)

    def _should_ignore_lm_head(self, recipe: Any) -> bool:
        value = get_field(recipe, "ignore_lm_head", None)
        if value is None:
            value = get_field(recipe_model(recipe), "ignore_lm_head", None)
        if value is None:
            return bool(self.ignore_lm_head)
        return bool(value)


GenericFamily().register()
