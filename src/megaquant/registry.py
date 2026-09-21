"""Plugin registries for backends, model families, and named schemes.

Optional sibling packages are imported lazily so `import megaquant` still works
when backends / schemes / families have not been written yet (or their extra
dependencies are missing).
"""

from __future__ import annotations

import importlib
from typing import Any

from megaquant.exceptions import BackendError, FamilyError, RecipeError

try:
    from megaquant.backends.base import QuantBackend
except ImportError:
    QuantBackend = object  # type: ignore

try:
    from megaquant.models.base import ModelFamily
except ImportError:
    ModelFamily = object  # type: ignore

_BACKENDS: dict[str, Any] = {}
_FAMILIES: dict[str, Any] = {}
_SCHEMES: dict[str, Any] = {}
_PLUGINS_LOADED = False

_LAZY_MODULES = (
    "megaquant.backends.modelopt",
    "megaquant.backends.llmcompressor",
    "megaquant.schemes.catalog",
    "megaquant.models",
)


def _ensure_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for mod in _LAZY_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            continue


def register_backend(name: str, backend: QuantBackend) -> None:
    if not name:
        raise BackendError("Backend name must be a non-empty string")
    _BACKENDS[name] = backend


def get_backend(name: str) -> QuantBackend:
    _ensure_plugins()
    if name not in _BACKENDS:
        known = ", ".join(list_backends()) or "(none)"
        raise BackendError(f"Unknown backend '{name}'. Registered: {known}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    _ensure_plugins()
    return sorted(_BACKENDS)


def register_family(name: str, family: ModelFamily) -> None:
    if not name:
        raise FamilyError("Family name must be a non-empty string")
    _FAMILIES[name] = family


def get_family(name: str) -> ModelFamily:
    _ensure_plugins()
    if name not in _FAMILIES:
        known = ", ".join(list_families()) or "(none)"
        raise FamilyError(f"Unknown family '{name}'. Registered: {known}")
    return _FAMILIES[name]


def list_families() -> list[str]:
    _ensure_plugins()
    return sorted(_FAMILIES)


def detect_family(model_source_or_config: dict) -> str:
    """Match a Hugging Face ``config.json`` mapping to a registered family.

    Returns ``generic`` when nothing matches or no families are registered.
    """
    _ensure_plugins()
    if not isinstance(model_source_or_config, dict):
        raise FamilyError("detect_family expects a Hugging Face config dict")
    model_type = model_source_or_config.get("model_type")
    architectures = model_source_or_config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]

    arch_hit: str | None = None
    for name, family in _FAMILIES.items():
        types = tuple(getattr(family, "model_types", ()) or ())
        archs = tuple(getattr(family, "architectures", ()) or ())
        if model_type and model_type in types:
            return name
        if arch_hit is None and any(a in archs for a in architectures):
            arch_hit = name
    return arch_hit or "generic"


def register_scheme(name: str, scheme: object) -> None:
    if not name:
        raise RecipeError("Scheme name must be a non-empty string")
    _SCHEMES[name] = scheme


def get_scheme(name: str) -> object:
    _ensure_plugins()
    if name not in _SCHEMES:
        known = ", ".join(list_schemes()) or "(none)"
        raise RecipeError(f"Unknown scheme '{name}'. Registered: {known}")
    return _SCHEMES[name]


def list_schemes() -> list[str]:
    _ensure_plugins()
    return sorted(_SCHEMES)
