"""Model-family adapters (ignore lists, load hints)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelFamily(Protocol):
    name: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]

    def default_ignore(self, recipe: Any) -> list[str]: ...

    def load_kwargs(self, recipe: Any) -> dict[str, Any]: ...


def glob_to_ignore(*patterns: str) -> list[str]:
    return list(patterns)


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def recipe_model(recipe: Any) -> Any:
    return get_field(recipe, "model", None)


class BaseFamily:
    """Concrete family with ``register()``. Adapters subclass this."""

    name: str = "generic"
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    def default_ignore(self, recipe: Any) -> list[str]:
        return []

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        model = recipe_model(recipe)
        dtype = get_field(model, "dtype", "bfloat16")
        return {
            "trust_remote_code": get_field(model, "trust_remote_code", True),
            "device_map": get_field(model, "device_map", "auto"),
            "torch_dtype": dtype,
        }

    def register(self) -> None:
        from megaquant.registry import register_family

        register_family(self.name, self)
