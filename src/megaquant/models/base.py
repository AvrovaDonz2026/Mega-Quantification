"""Model-family adapters (ignore lists, load hints)."""

from __future__ import annotations

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
