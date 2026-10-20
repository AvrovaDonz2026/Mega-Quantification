"""Offline regression tests for the MSE-prefix request metadata gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CONTROLLER = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnostics/gdn_parity_20261019/mse-prefix/controller.py"
)
SPEC = importlib.util.spec_from_file_location("mse_prefix_controller", CONTROLLER)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER_MODULE)


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"num_retractions": 0}, (0, None)),
        ({"num_retractions": 0, "spec_verify_ct": 0}, (0, 0)),
        ({"num_retractions": 0, "retraction_count": 0}, (0, None)),
    ],
)
def test_validate_request_counters_accepts_exact_zero(
    meta: dict, expected: tuple[int, int | None]
) -> None:
    assert CONTROLLER_MODULE.validate_request_counters(meta) == expected


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"num_retractions": None},
        {"num_retractions": True},
        {"num_retractions": 0.0},
        {"num_retractions": "0"},
        {"num_retractions": -1},
        {"num_retractions": 1},
        {"retraction_count": 1},
        {"num_retractions": 0, "retraction_count": 1},
        {"num_retractions": 0, "retraction_count": False},
        {"num_retractions": 0, "spec_verify_ct": None},
        {"num_retractions": 0, "spec_verify_ct": True},
        {"num_retractions": 0, "spec_verify_ct": 0.0},
        {"num_retractions": 0, "spec_verify_ct": -1},
        {"num_retractions": 0, "spec_verify_ct": 1},
    ],
)
def test_validate_request_counters_rejects_missing_or_invalid_metadata(meta: dict) -> None:
    with pytest.raises(RuntimeError):
        CONTROLLER_MODULE.validate_request_counters(meta)
