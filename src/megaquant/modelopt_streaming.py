"""Scoped repair for resident QuantLinear packing in ModelOpt 0.47 offload exports."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import textwrap
import threading
from collections.abc import Iterator
from typing import Any

from megaquant.exceptions import BackendError

_VERSION = "0.47.0"
_SOURCE_SHA256 = "c38d1d81851990287474a3533e44f0f3ce904a1afabf5eb40b2034d90852df3f"
_MODULE = "modelopt.torch.export.unified_export_hf_streaming"
_NEEDLE = (
    "        if not requires_weight_materialization(module, model, name_to_module):\n"
    "            continue\n"
)
_ANCHOR = "    seen_keys: set[str] = set()\n"
_DISPATCH = "_dispatch_export_handler(full_name, sub_mod, ctx)"
_LOCK = threading.RLock()

# Defer serialization of resident modules to the vendor's final fallback loop.
# Streaming a resident parent's state_dict here would mark a child's BF16 weight
# as seen before that child has had a chance to run its packing handler.
_TRACK_DISPATCH = """    seen_keys: set[str] = set()
    dispatched_module_ids: set[int] = set()

    def _megaquant_dispatch_once(name, module, context):
        if id(module) in dispatched_module_ids:
            return
        _dispatch_export_handler(name, module, context)
        dispatched_module_ids.add(id(module))

    def _megaquant_dispatch_resident(name, module, context):
        if id(module) in dispatched_module_ids:
            return
        # Match the vendor QuantLinear eligibility, then consult its registry.
        # Do not invoke a generic MoE parent handler on offloaded descendants.
        class_name = type(module).__name__
        if not any(tag in class_name for tag in (
            "QuantLinear", "QuantCompressedLinear", "QuantFP8Linear"
        )):
            return
        if getattr(module, "weight", None) is None:
            return
        values = tuple(module.parameters(recurse=False)) + tuple(module.buffers(recurse=False))
        if not values or any(getattr(value, "is_meta", False) for value in values):
            return
        from modelopt.torch.export.registry import ExportModuleRegistry
        if ExportModuleRegistry.match(module) is not None:
            _megaquant_dispatch_once(name, module, context)
"""


def _modelopt_version() -> str | None:
    try:
        return importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError:
        return None


def _patched_exporter(original: Any) -> Any:
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise BackendError("Cannot inspect ModelOpt streaming exporter; refusing patch") from exc
    if hashlib.sha256(source.encode()).hexdigest() != _SOURCE_SHA256:
        raise BackendError("Unrecognized ModelOpt 0.47 streaming exporter source; refusing patch")
    if source.count(_NEEDLE) != 1 or source.count(_ANCHOR) != 1 or source.count(_DISPATCH) != 2:
        raise BackendError(
            "Unrecognized ModelOpt 0.47 streaming exporter structure; refusing patch"
        )
    source = source.replace(_ANCHOR, _TRACK_DISPATCH, 1)
    source = source.replace(_DISPATCH, "_megaquant_dispatch_once(full_name, sub_mod, ctx)")
    source = source.replace(
        _NEEDLE,
        "        if not requires_weight_materialization(module, model, name_to_module):\n"
        "            _megaquant_dispatch_resident(name, module, ctx)\n"
        "            continue\n",
        1,
    )
    namespace: dict[str, Any] = {}
    try:
        # Keep the actual module globals: diagnostic dispatch aliases and vendor
        # helpers can change during an export. No helper globals are injected.
        exec(
            compile(textwrap.dedent(source), "<megaquant-modelopt-streaming>", "exec"),
            original.__globals__,
            namespace,
        )
        patched = namespace[original.__name__]
    except Exception as exc:
        raise BackendError("Cannot build ModelOpt streaming compatibility patch") from exc
    patched.__megaquant_streaming_compat__ = True
    return patched


@contextlib.contextmanager
def streaming_export_compat(*, offloaded: bool) -> Iterator[None]:
    """Repair the pinned 0.47 exporter for the duration of an offloaded export.

    Other versions retain their vendor implementation and the normal export
    precision guard. A modified 0.47 source is rejected before export begins.
    """
    if not offloaded:
        yield
        return
    with _LOCK:
        if _modelopt_version() != _VERSION:
            yield
            return
        try:
            streaming = importlib.import_module(_MODULE)
            original = streaming._export_transformers_checkpoint_streaming
        except (ImportError, AttributeError) as exc:
            raise BackendError("ModelOpt 0.47 streaming exporter is unavailable") from exc
        if getattr(original, "__megaquant_streaming_compat__", False):
            # RLock permits nested export wrappers in the same thread.
            yield
            return
        patched = _patched_exporter(original)
        streaming._export_transformers_checkpoint_streaming = patched
        try:
            yield
        finally:
            streaming._export_transformers_checkpoint_streaming = original
