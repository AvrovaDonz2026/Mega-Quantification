"""Quantization backend protocol and shared calibration helpers."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def _is_accelerator_placement(value: Any) -> bool:
    if isinstance(value, int):
        return True
    text = str(value).strip().lower()
    if not text or text in {"cpu", "disk", "meta"} or text.startswith("disk"):
        return False
    return text.isdigit() or any(tag in text for tag in ("cuda", "xpu", "npu", "mps", "hpu"))


def _placement_to_device(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, int):
        return torch.device(f"cuda:{value}" if torch.cuda.is_available() else "cpu")
    text = str(value).strip()
    if text.isdigit():
        return _placement_to_device(int(text))
    try:
        return torch.device(text)
    except (RuntimeError, TypeError, ValueError):
        return value


def _first_non_meta_param_device(model: Any, *, prefer_accelerator: bool) -> Any | None:
    try:
        parameters = model.parameters()
    except Exception:
        return None
    fallback = None
    try:
        for param in parameters:
            device = getattr(param, "device", None)
            if device is None:
                continue
            kind = getattr(device, "type", None)
            if kind == "meta":
                continue
            if prefer_accelerator and kind not in {None, "cpu", "meta"}:
                return device
            if fallback is None:
                fallback = device
                if not prefer_accelerator:
                    return device
    except (StopIteration, TypeError, RuntimeError):
        return fallback
    return fallback


def preferred_forward_device(model: Any) -> Any | None:
    """Device for calibration ``input_ids``.

    ``parameters()`` follows constructor order, so a CPU-pinned ViT would make
    the first parameter CPU even when the language model is on CUDA. Prefer
    input embeddings, then ``hf_device_map`` accelerator entries, then CUDA
    parameters.
    """
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        try:
            emb = get_emb()
            device = _first_non_meta_param_device(emb, prefer_accelerator=True)
            if device is not None and getattr(device, "type", None) not in {None, "cpu"}:
                return device
        except Exception:
            pass

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for value in device_map.values():
            if _is_accelerator_placement(value):
                return _placement_to_device(value)

    accelerator = _first_non_meta_param_device(model, prefer_accelerator=True)
    if accelerator is not None:
        return accelerator
    return _first_non_meta_param_device(model, prefer_accelerator=False)


def _first_param_device(model: Any) -> Any | None:
    """Best-effort device for ``model`` parameters (skips meta tensors)."""
    return preferred_forward_device(model)


def _move_to_device(obj: Any, device: Any) -> Any:
    """Move tensors (or nested dict/list/tuple batches) onto ``device`` if possible."""
    if device is None or obj is None:
        return obj
    to_fn = getattr(obj, "to", None)
    if callable(to_fn) and not isinstance(obj, (str, bytes)):
        try:
            return to_fn(device, non_blocking=True)
        except TypeError:
            try:
                return to_fn(device)
            except (TypeError, RuntimeError, AttributeError, ValueError):
                pass
        except (RuntimeError, AttributeError, ValueError):
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


def _pin_cpu_tensors(obj: Any) -> Any:
    """Pin host tensors so H2D copies overlap with compute."""
    pin = getattr(obj, "pin_memory", None)
    if callable(pin) and not isinstance(obj, (str, bytes)):
        device = getattr(obj, "device", None)
        kind = getattr(device, "type", "cpu") if device is not None else "cpu"
        if kind == "cpu":
            try:
                return pin()
            except Exception:
                return obj
    if isinstance(obj, Mapping):
        return {key: _pin_cpu_tensors(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_pin_cpu_tensors(value) for value in obj)
    if isinstance(obj, list):
        return [_pin_cpu_tensors(value) for value in obj]
    return obj


_FORWARD_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "pixel_values",
        "image_grid_thw",
        "image_sizes",
        "cross_attention_mask",
        "labels",
    }
)


def _run_forward(model: Any, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        payload = {key: value for key, value in batch.items() if key in _FORWARD_KEYS}
        if not payload:
            payload = dict(batch)
        payload.setdefault("use_cache", False)
        return model(**payload)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, list):
        return model(*batch)
    return model(batch)


def _materialize_batches(calib_iter: Iterable[Any]) -> list[Any] | Iterable[Any]:
    if calib_iter is None:
        return []
    if isinstance(calib_iter, (list, tuple)):
        return list(calib_iter)
    # Sequence-like (has len + getitem): re-iterable; still copy so a generator
    # subclass cannot be exhausted on a second Local-Hessian pass.
    if hasattr(calib_iter, "__len__") and hasattr(calib_iter, "__getitem__"):
        try:
            return [calib_iter[i] for i in range(len(calib_iter))]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            pass
    return list(calib_iter)


def forward_loop_from_iter(calib_iter: Iterable[Any]) -> Callable[[Any], None]:
    """Return the ``forward_loop(model)`` callable that ModelOpt ``mtq.quantize`` expects.

    Batches may be mappings of tensors (Hugging Face ``**kwargs``), raw tensors, or
    tuple/list positional args. Tensors are moved onto the model's parameter device
    when that device can be inferred. The iterator is materialized so Local-Hessian
    can run more than one forward pass.
    """
    batches = _materialize_batches(calib_iter)
    try:
        from megaquant.runtime import env_flag

        if env_flag("MEGAQUANT_PIN_MEMORY", default=True):
            batches = [_pin_cpu_tensors(batch) for batch in batches]
    except Exception:
        pass

    def forward_loop(model: Any) -> None:
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            try:
                config.use_cache = False
            except (TypeError, AttributeError):
                pass
        device = preferred_forward_device(model)
        try:
            import torch

            ctx = torch.no_grad()
        except ImportError:
            ctx = nullcontext()
        items = list(batches) if not isinstance(batches, list) else batches
        total = len(items)
        started = time.monotonic()
        copy_stream = None
        try:
            import torch as _torch

            kind = getattr(device, "type", None)
            if kind == "cuda" or (kind is None and device is not None and "cuda" in str(device)):
                copy_stream = _torch.cuda.Stream()
        except Exception:
            copy_stream = None

        def _prefetch(item: Any) -> Any:
            if copy_stream is None:
                return _move_to_device(item, device)
            import torch as _torch

            with _torch.cuda.stream(copy_stream):
                return _move_to_device(item, device)

        with ctx:
            pending = _prefetch(items[0]) if items else None
            for index, batch in enumerate(items, start=1):
                if copy_stream is not None:
                    copy_stream.synchronize()
                payload = pending if pending is not None else _move_to_device(batch, device)
                if index < total:
                    pending = _prefetch(items[index])
                else:
                    pending = None
                _run_forward(model, payload)
                step = max(1, total // 20)
                if index == 1 or index == total or index % step == 0:
                    elapsed = time.monotonic() - started
                    per = elapsed / index
                    eta = per * (total - index)
                    print(
                        f"[megaquant] calib {index}/{total}  {elapsed:.1f}s elapsed  "
                        f"{per:.2f}s/step  eta {eta:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )

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
