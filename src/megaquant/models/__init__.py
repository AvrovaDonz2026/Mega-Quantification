"""Register built-in model families."""

from __future__ import annotations

from megaquant.models.generic import GenericFamily
from megaquant.models.llama import LlamaFamily
from megaquant.models.qwen3 import Qwen3Family
from megaquant.models.qwen3_5 import Qwen35Family

__all__ = ["GenericFamily", "LlamaFamily", "Qwen3Family", "Qwen35Family", "_register"]


def _register() -> None:
    try:
        from megaquant.registry import register_family
    except ImportError:
        return
    for family in (Qwen35Family(), Qwen3Family(), LlamaFamily(), GenericFamily()):
        try:
            register_family(family.name, family)
        except Exception:
            continue


_register()
