"""Model-family adapters. Importing this package registers every family.

``megaquant.registry`` does ``from megaquant.models.base import ModelFamily``
while it is still executing. If this package imported adapters at that moment,
``register_family`` would not exist yet and a half-initialized ``megaquant.models``
would sit in ``sys.modules``, so the later lazy plugin import would be a no-op.

Raising ``ImportError`` while the registry is incomplete makes that ``try/except``
fall back to ``ModelFamily = object``. ``_ensure_plugins()`` then re-imports this
package once ``register_family`` exists.
"""

from __future__ import annotations

import sys

_registry = sys.modules.get("megaquant.registry")
if _registry is not None and not callable(getattr(_registry, "register_family", None)):
    sys.modules.pop("megaquant.models", None)
    raise ImportError("megaquant.registry is still loading; defer family adapters")

from megaquant.models.base import BaseFamily, ModelFamily, glob_to_ignore  # noqa: E402
from megaquant.models.generic import GenericFamily  # noqa: E402
from megaquant.models.llama import LlamaFamily  # noqa: E402
from megaquant.models.qwen3 import Qwen3Family  # noqa: E402
from megaquant.models.qwen3_5 import Qwen35Family  # noqa: E402

__all__ = [
    "BaseFamily",
    "GenericFamily",
    "LlamaFamily",
    "ModelFamily",
    "Qwen35Family",
    "Qwen3Family",
    "glob_to_ignore",
]


def _register() -> None:
    for family in (Qwen35Family(), Qwen3Family(), LlamaFamily(), GenericFamily()):
        try:
            family.register()
        except Exception:
            continue


_register()
