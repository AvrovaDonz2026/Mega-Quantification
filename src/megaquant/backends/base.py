"""Quantization backend protocol and shared calibration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def _first_param_device(model: Any) -> Any | None:
    """Best-effort device for ``model`` parameters (skips meta tensors)."""
    try:
        parameters = model.parameters()
    except Exception:
        return None
    try:
        for param in parameters:
            device = getattr(param, "device", None)
            if device is None:
                continue
            if getattr(device, "type", None) == "meta":
                continue
            return device
    except (StopIteration, TypeError, RuntimeError):
        return None
    return None


def _move_to_device(obj: Any, device: Any) -> Any:
    """Move tensors (or nested dict/list/tuple batches) onto ``device`` if possible."""
    if device is None or obj is None:
        return obj
    to_fn = getattr(obj, "to", None)
    if callable(to_fn) and not isinstance(obj, (str, bytes)):
        try:
            return to_fn(device)
        except (TypeError, RuntimeError, AttributeError, ValueError):
            pass
    if isinstance(obj, Mapping):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_move_to_device(value, device) for value in obj)
    if isinstance(obj, list):
        return [_move_to_device(value, device) for value in obj]
    return obj


def _run_forward(model: Any, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, list):
        return model(*batch)
    return model(batch)


def forward_loop_from_iter(calib_iter: Iterable[Any]) -> Callable[[Any], None]:
    """Return the ``forward_loop(model)`` callable that ModelOpt ``mtq.quantize`` expects.

    Batches may be mappings of tensors (Hugging Face ``**kwargs``), raw tensors, or
    tuple/list positional args. Tensors are moved onto the model's parameter device
    when that device can be inferred.
    """

    def forward_loop(model: Any) -> None:
        device = _first_param_device(model)
        for batch in calib_iter:
            _run_forward(model, _move_to_device(batch, device))

    return forward_loop


@runtime_checkable
class QuantBackend(Protocol):
    """PTQ backend: config, calibrate/quantize, export."""

    name: str

    def available(self) -> bool: ...

    def supports(self, scheme_name: str) -> bool: ...

    def build_quant_cfg(self, plan: Any) -> Any: ...

    def quantize(self, model: Any, plan: Any, calib_iter: Any) -> Any: ...

    def export(self, model: Any, recipe: Any, tokenizer: Any = None) -> Path: ...
