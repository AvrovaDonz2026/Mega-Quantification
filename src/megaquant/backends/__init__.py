"""Quantization backends.

``QuantBackend`` is always exported. ``ModelOptBackend`` is imported when the
module is available. llm-compressor is owned by another agent and is loaded
only through :func:`load_all`.
"""

from __future__ import annotations

from megaquant.backends.base import QuantBackend, forward_loop_from_iter

try:
    from megaquant.backends.modelopt import ModelOptBackend, describe_cfg
except ImportError:  # pragma: no cover - optional sibling may be absent
    ModelOptBackend = None  # type: ignore[misc, assignment]
    describe_cfg = None  # type: ignore[misc, assignment]


def load_all() -> None:
    """Import backend implementations so they can self-register.

    ``modelopt`` and ``llmcompressor`` are imported independently. Missing
    sibling modules are skipped.
    """
    import importlib

    for mod in ("megaquant.backends.modelopt", "megaquant.backends.llmcompressor"):
        try:
            importlib.import_module(mod)
        except ImportError:
            continue


__all__ = [
    "ModelOptBackend",
    "QuantBackend",
    "describe_cfg",
    "forward_loop_from_iter",
    "load_all",
]
