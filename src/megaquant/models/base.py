"""Model-family protocol and ignore-list helpers.

A family adapter maps a Hugging Face ``config.json`` (``model_type`` /
``architectures``) onto a default ignore list (modules that stay BF16) and
``from_pretrained`` kwargs.

Concrete adapters register themselves on import via
:func:`megaquant.registry.register_family` when the registry is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """One-level attribute or mapping lookup."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def recipe_model(recipe: Any) -> Any:
    return get_field(recipe, "model", None)


def recipe_get(recipe: Any, *path: str, default: Any = None) -> Any:
    """Walk a recipe as a Pydantic object or a mapping. Missing nodes yield ``default``."""
    cur: Any = recipe
    for i, key in enumerate(path):
        if cur is None:
            return default
        last = i == len(path) - 1
        if isinstance(cur, Mapping):
            cur = cur.get(key, default if last else None)
            continue
        cur = getattr(cur, key, default if last else None)
    if cur is None:
        return default
    return cur


def recipe_flag(recipe: Any, *path: str, default: bool = False) -> bool:
    value = recipe_get(recipe, *path, default=default)
    if value is None:
        return default
    return bool(value)


def glob_to_ignore(*parts: Any) -> list[str]:
    """Flatten glob / module-name patterns into a de-duplicated ignore list.

    Bare names without glob metacharacters or an ``re:`` prefix are wrapped as
    ``*{name}*``. Order of first appearance is preserved.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(pattern: str) -> None:
        pattern = pattern.strip()
        if not pattern:
            return
        if pattern.startswith("re:") or any(ch in pattern for ch in "*?["):
            normalized = pattern
        else:
            normalized = f"*{pattern}*"
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)

    def walk(item: Any) -> None:
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            add(item)
            return
        if isinstance(item, bytes):
            add(item.decode())
            return
        if isinstance(item, Mapping):
            return
        if isinstance(item, Iterable):
            for sub in item:
                walk(sub)

    for part in parts:
        walk(part)
    return out


def register_this_family(family: Any) -> None:
    """Register ``family`` on :mod:`megaquant.registry` when that module exists."""
    try:
        from megaquant.registry import register_family
    except ImportError:
        return
    name = getattr(family, "name", None)
    if not name:
        return
    try:
        register_family(name, family)
    except TypeError:
        try:
            register_family(family)
        except Exception:
            return


@runtime_checkable
class ModelFamily(Protocol):
    """Adapter for one Hugging Face model family."""

    name: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]

    def default_ignore(self, recipe: Any) -> list[str]: ...

    def load_kwargs(self, recipe: Any) -> dict[str, Any]: ...


class BaseFamily:
    """Shared ``load_kwargs`` / registration. Subclasses override ``default_ignore``."""

    name: str = "generic"
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()

    def default_ignore(self, recipe: Any) -> list[str]:
        return glob_to_ignore("*visual*", "*embed_tokens*")

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        """``from_pretrained`` kwargs. Never imports torch (string dtypes)."""
        model = recipe_model(recipe)
        trust = get_field(model, "trust_remote_code", True)
        if trust is None:
            trust = True
        device_map = get_field(model, "device_map", "auto")
        if device_map is None:
            device_map = "auto"
        raw_dtype = get_field(model, "dtype", "bfloat16") or "bfloat16"
        dtype_key = str(raw_dtype).lower().replace("torch.", "")
        torch_dtype = _DTYPE_ALIASES.get(dtype_key, str(raw_dtype))
        return {
            "trust_remote_code": bool(trust),
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }

    def register(self) -> None:
        register_this_family(self)
