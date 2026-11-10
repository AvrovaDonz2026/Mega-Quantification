"""Opt-in ModelOpt 0.47 NVFP4 live export codec.

This module only provides a scoped experimental hook. Production export does not
enter it unless a future backend explicitly opts in after validation.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib
import importlib.metadata
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

VERSION = "0.47.0"
MODULES = {
    "nvfp4": "modelopt.torch.quantization.qtensor.nvfp4_tensor",
    "resident": "modelopt.torch.export.unified_export_hf",
    "streaming": "modelopt.torch.export.unified_export_hf_streaming",
    "quant_utils": "modelopt.torch.export.quant_utils",
}
SOURCE_SHA256 = {
    "nvfp4": "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e",
    "resident": "7790006c1279f7a8749c8a97118c6a5f825054c89abd43fccd692af18e979c39",
    "streaming": "ed981fc57ede6910be139cc854f45c10a9d1ba26603ede28507f688863951b7a",
    "quant_utils": "9045bccd1ad27bd7c757bb1000ce0ddc05fe4fc1ff140a42f171f5dbc8009b07",
}
FUNCTION_SHA256 = {
    "quantize": "194f4ee2182370c3d77f8bd72f0e908236bfe10d4605bbe66115019bb8c187cf",
    "_cast_fp4": "e89b24eaedcbdda3a6ce256ca4d6d34c05ebb2181d57df89e6b0d5e6a8a67022",
    "_is_static_quantizer": "f8ebe9391955a42f0e973645c0d3cafb970132e4578fab5a5d42b8c044179423",
    "_get_static_global_amax": "0f6b27ad86d6bc04f4cf0c39330405c8a794edfb8fe4dbe5870009387413f6e9",
    "to_quantized_weight": "bc8dc784a253021607d1d8839bb346a85c5460acb845a9314f883c6ba6ae9e49",
    "get_quantization_format": "5d8f22e0a9b350b814ef98a4fd7e06aaaccaa40b59f95fae349824491cadda9c",
    "_dispatch_export_handler": "0d6362a04d9f502195cdc65f67cc2bff2768d555eaf3a1bfc3b16aeb38576d37",
}
_ACTIVE = contextvars.ContextVar("megaquant_nvfp4_codec", default=False)
_LOCK = threading.RLock()
_DEPTH = 0


@dataclass
class CodecAudit:
    candidate_attempts: int = 0
    candidate_calls: int = 0
    delegated_calls: int = 0
    failures: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def delegate(self, reason: str) -> None:
        self.delegated_calls += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _module_sha(module: Any) -> str:
    source = inspect.getsourcefile(module)
    return hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else ""


def _verify_function(fn: Any, module: Any, name: str, parameters: tuple[str, ...]) -> None:
    if not inspect.isfunction(fn) or hasattr(fn, "__wrapped__") or fn.__module__ != module.__name__:
        raise RuntimeError(f"{name} live binding mismatch")
    source_file = inspect.getsourcefile(fn)
    if not source_file or Path(source_file).resolve() != Path(module.__file__).resolve():
        raise RuntimeError(f"{name} source file mismatch")
    if fn.__qualname__.split(".")[-1] != name:
        raise RuntimeError(f"{name} qualname mismatch")
    if tuple(inspect.signature(fn, follow_wrapped=False).parameters) != parameters:
        raise RuntimeError(f"{name} full signature mismatch")
    digest = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if digest != FUNCTION_SHA256[name]:
        raise RuntimeError(f"{name} function fingerprint mismatch")


def _gate(modules: dict[str, Any]) -> None:
    if importlib.metadata.version("nvidia-modelopt") != VERSION:
        raise RuntimeError("requires nvidia-modelopt==0.47.0")
    nv, uq = modules["nvfp4"], modules["quant_utils"]
    qcls = nv.NVFP4QTensor
    qdesc = inspect.getattr_static(qcls, "quantize")
    if not isinstance(qdesc, classmethod):
        raise RuntimeError("NVFP4QTensor.quantize descriptor mismatch")
    names = tuple(inspect.signature(qdesc.__func__).parameters)
    if names[:5] != (
        "cls",
        "input",
        "block_size",
        "weights_scaling_factor",
        "weights_scaling_factor_2",
    ):
        raise RuntimeError("NVFP4QTensor.quantize signature mismatch")
    cast = inspect.getattr_static(qcls, "_cast_fp4")
    if not isinstance(cast, classmethod) or tuple(inspect.signature(cast.__func__).parameters) != (
        "cls",
        "weight",
    ):
        raise RuntimeError("NVFP4QTensor._cast_fp4 descriptor mismatch")
    if (
        modules["resident"]._dispatch_export_handler
        is not modules["streaming"]._dispatch_export_handler
    ):
        raise RuntimeError("resident/streaming dispatcher aliases differ")
    for key, module in modules.items():
        if _module_sha(module) != SOURCE_SHA256[key]:
            raise RuntimeError(f"{key} source fingerprint mismatch")
    _verify_function(
        qdesc.__func__,
        nv,
        "quantize",
        (
            "cls",
            "input",
            "block_size",
            "weights_scaling_factor",
            "weights_scaling_factor_2",
            "keep_high_precision",
            "try_tensorrt",
        ),
    )
    _verify_function(cast.__func__, nv, "_cast_fp4", ("cls", "weight"))
    _verify_function(
        qcls._is_static_quantizer.__func__, nv, "_is_static_quantizer", ("cls", "weight_quantizer")
    )
    _verify_function(
        qcls._get_static_global_amax.__func__,
        nv,
        "_get_static_global_amax",
        ("cls", "weight_quantizer"),
    )
    _verify_function(
        uq.to_quantized_weight,
        uq,
        "to_quantized_weight",
        (
            "weight",
            "weights_scaling_factor",
            "quantization",
            "weights_scaling_factor2",
            "block_size",
        ),
    )
    _verify_function(uq.get_quantization_format, uq, "get_quantization_format", ("module",))
    _verify_function(
        modules["resident"]._dispatch_export_handler,
        modules["resident"],
        "_dispatch_export_handler",
        ("name", "sub_module", "ctx"),
    )
    if (
        getattr(modules["resident"], "to_quantized_weight", uq.to_quantized_weight)
        is not uq.to_quantized_weight
    ):
        raise RuntimeError("resident quantization alias mismatch")
    for key in (
        "QUANTIZATION_NVFP4",
        "QUANTIZATION_NVFP4_AWQ",
        "QUANTIZATION_NVFP4_SVDQUANT",
        "QUANTIZATION_W4A16_NVFP4",
        "QUANTIZATION_W4A8_NVFP4_FP8",
    ):
        if not hasattr(uq, key):
            raise RuntimeError(f"missing pinned quantization constant: {key}")


def _codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if (
        weight.device.type != "cuda"
        or weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[1] == 0
        or weight.shape[1] % 16
        or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("unsupported NVFP4 input")
    if (
        scale.device != weight.device
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0], weight.shape[1] // 16)
    ):
        raise ValueError("invalid effective scale")
    if (
        not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(scale).all())
        or not bool((scale > 0).all())
    ):
        raise ValueError("nonfinite or nonpositive codec input")
    ratio = weight.float().abs() / scale.repeat_interleave(16, -1)
    mag = torch.where(
        ratio <= 0.25,
        0,
        torch.where(
            ratio < 0.75,
            1,
            torch.where(
                ratio <= 1.25,
                2,
                torch.where(
                    ratio < 1.75,
                    3,
                    torch.where(
                        ratio <= 2.5,
                        4,
                        torch.where(ratio < 3.5, 5, torch.where(ratio <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(torch.uint8)
    return mag | ((weight < 0).to(torch.uint8) << 3)


def _validate_inputs(weight: torch.Tensor, scale: torch.Tensor) -> None:
    if (
        weight.device.type != "cuda"
        or weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[1] == 0
        or weight.shape[1] % 16
    ):
        raise ValueError("unsupported NVFP4 input")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not bool(
        torch.isfinite(weight).all()
    ):
        raise ValueError("unsupported NVFP4 weight dtype or values")
    if (
        scale.device != weight.device
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0], weight.shape[1] // 16)
        or not bool(torch.isfinite(scale).all())
        or not bool((scale > 0).all())
    ):
        raise ValueError("invalid effective scale")


def _static_nvfp4(modules: dict[str, Any], module: Any) -> bool:
    quantizer = getattr(module, "weight_quantizer", None)
    if quantizer is None:
        return False
    uq = modules["quant_utils"]
    fmt = uq.get_quantization_format(module)
    return fmt in {
        getattr(uq, key)
        for key in (
            "QUANTIZATION_NVFP4",
            "QUANTIZATION_NVFP4_AWQ",
            "QUANTIZATION_NVFP4_SVDQUANT",
            "QUANTIZATION_W4A16_NVFP4",
            "QUANTIZATION_W4A8_NVFP4_FP8",
        )
    } and bool(modules["nvfp4"].NVFP4QTensor._is_static_quantizer(quantizer))


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack even-index codes into the low nibble and odd into the high nibble."""
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("codes must be rank-2 uint8 with an even width")
    pairs = codes.reshape(codes.shape[0], -1, 2).to(torch.int16)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8).contiguous()


@contextlib.contextmanager
def install(*, codec: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None):
    """Install the strict opt-in hook and yield :class:`CodecAudit`."""
    global _DEPTH
    with _LOCK:
        if _DEPTH:
            raise RuntimeError("NVFP4 codec context cannot be nested")
        modules = {key: importlib.import_module(name) for key, name in MODULES.items()}
        _gate(modules)
        nv, resident, streaming = modules["nvfp4"], modules["resident"], modules["streaming"]
        qcls = nv.NVFP4QTensor
        original_dispatch = resident._dispatch_export_handler
        original_quant = inspect.getattr_static(qcls, "quantize")
        original_fn = original_quant.__func__
        audit = CodecAudit()
        active_codec = codec

        def dispatch(name, module, ctx):
            token = _ACTIVE.set(_static_nvfp4(modules, module))
            try:
                return original_dispatch(name, module, ctx)
            finally:
                _ACTIVE.reset(token)

        def quantize(
            cls,
            input,
            block_size,
            weights_scaling_factor=None,
            weights_scaling_factor_2=None,
            keep_high_precision=False,
            try_tensorrt=False,
        ):
            if (
                not _ACTIVE.get()
                or block_size != 16
                or weights_scaling_factor is None
                or weights_scaling_factor_2 is None
                or keep_high_precision
                or try_tensorrt
            ):
                audit.delegate("vendor_fallback")
                return original_fn(
                    cls,
                    input,
                    block_size,
                    weights_scaling_factor,
                    weights_scaling_factor_2,
                    keep_high_precision,
                    try_tensorrt,
                )
            try:
                effective = weights_scaling_factor.float() * weights_scaling_factor_2
                _validate_inputs(input, effective)
            except (ValueError, RuntimeError, TypeError) as exc:
                audit.delegate(type(exc).__name__)
                return original_fn(
                    cls,
                    input,
                    block_size,
                    weights_scaling_factor,
                    weights_scaling_factor_2,
                    keep_high_precision,
                    try_tensorrt,
                )
            audit.candidate_attempts += 1
            try:
                packed = (
                    active_codec(input, effective)
                    if active_codec
                    else _pack(_codes(input, effective))
                )
                if (
                    not isinstance(packed, torch.Tensor)
                    or packed.dtype != torch.uint8
                    or packed.device != input.device
                    or not packed.is_contiguous()
                    or packed.shape != (input.shape[0], input.shape[1] // 2)
                ):
                    raise ValueError("codec returned invalid packed tensor")
                audit.candidate_calls += 1
                return (
                    cls(input.shape, input.dtype, packed),
                    weights_scaling_factor,
                    weights_scaling_factor_2,
                )
            except Exception:
                audit.failures += 1
                raise

        mutations = []
        try:
            for target, name, value in (
                (resident, "_dispatch_export_handler", dispatch),
                (streaming, "_dispatch_export_handler", dispatch),
                (qcls, "quantize", classmethod(quantize)),
            ):
                old = inspect.getattr_static(target, name)
                mutations.append((target, name, old))
                setattr(target, name, value)
            _DEPTH = 1
            yield audit
        finally:
            restore_errors = []
            for target, name, old in reversed(mutations):
                try:
                    setattr(target, name, old)
                except BaseException as exc:
                    restore_errors.append(exc)
            _DEPTH = 0
            if restore_errors:
                raise RuntimeError("failed to restore NVFP4 codec bindings") from restore_errors[0]
