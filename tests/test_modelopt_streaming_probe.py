"""CPU contract checks for the public ModelOpt streaming probe."""

from __future__ import annotations

import struct

import pytest
from scripts.check_modelopt_streaming_export import (
    _assert_aux_ready,
    _key_hash,
    validate_export_report,
)


def _report(*, dtype: str = "U8", shape: list[int] | None = None,
            include_scale: bool = True, include_mtp: bool = True) -> dict:
    rows = []
    shape = shape or [32, 16]
    for name in ("visual_aux", "mtp_aux"):
        if name == "mtp_aux" and not include_mtp:
            continue
        rows.append({"key_hash": _key_hash(f"{name}.weight"), "dtype": dtype, "shape": shape})
        if include_scale:
            rows.extend([
                {"key_hash": _key_hash(f"{name}.weight_scale"),
                 "dtype": "F8_E4M3", "shape": [32, 2]},
                {"key_hash": _key_hash(f"{name}.weight_scale_2"),
                 "dtype": "F32", "shape": []},
            ])
    return {"status": "ok", "headers": {"aux_rows": rows}}


def test_rejects_wrong_packed_dtype() -> None:
    with pytest.raises(AssertionError, match="dtype"):
        validate_export_report(_report(dtype="BF16"), compat=True)


def test_rejects_wrong_packed_shape() -> None:
    with pytest.raises(AssertionError, match="shape"):
        validate_export_report(_report(shape=[32, 32]), compat=True)


def test_rejects_missing_scale() -> None:
    with pytest.raises(AssertionError, match="scale"):
        validate_export_report(_report(include_scale=False), compat=True)


def test_rejects_missing_auxiliary_weight() -> None:
    with pytest.raises(AssertionError, match="dtype"):
        validate_export_report(_report(include_mtp=False), compat=True)


def test_rejects_non_scalar_second_scale() -> None:
    report = _report()
    report["headers"]["aux_rows"][2]["shape"] = [1]
    with pytest.raises(AssertionError, match="scalar scale shape"):
        validate_export_report(report, compat=True)


def test_rejects_duplicate_auxiliary_keys() -> None:
    report = _report()
    report["headers"]["aux_rows"].append(dict(report["headers"]["aux_rows"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_export_report(report, compat=True)


def test_truncated_header_is_rejected(tmp_path) -> None:
    from scripts.check_modelopt_streaming_export import summarize_headers

    path = tmp_path / "broken.safetensors"
    path.write_bytes(struct.pack("<Q", 20) + b"{}")
    with pytest.raises(ValueError, match="truncated"):
        summarize_headers(tmp_path)


def test_aux_ready_requires_modelopt_scale_bits_and_group() -> None:
    class Tensor:
        dtype = "bf16"
        device = type("Device", (), {"type": "cuda"})()

    class Quantizer:
        is_enabled = True
        num_bits = (2, 1)
        block_sizes = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}

    class Module:
        weight = Tensor()
        weight_quantizer = Quantizer()

    class Model:
        visual_aux = Module()
        mtp_aux = Module()

    class Torch:
        bfloat16 = "bf16"

    _assert_aux_ready(Model(), Torch())
    Model.visual_aux.weight_quantizer.block_sizes["scale_bits"] = (8, 0)
    with pytest.raises(AssertionError, match="NVFP4"):
        _assert_aux_ready(Model(), Torch())
