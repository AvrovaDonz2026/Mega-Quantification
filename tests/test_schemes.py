"""Scheme catalog facts (skip if sibling schemes package is absent)."""

from __future__ import annotations

from typing import Any

import pytest


def _load_catalog() -> dict[str, Any] | None:
    try:
        from megaquant.schemes.catalog import SCHEME_CATALOG
    except ImportError:
        return None
    if not SCHEME_CATALOG:
        return None
    return SCHEME_CATALOG


def _groups(entry: Any) -> list[Any]:
    if entry is None:
        return []
    if hasattr(entry, "groups"):
        return list(entry.groups or [])
    if isinstance(entry, dict):
        if "groups" in entry:
            return list(entry["groups"] or [])
        return [entry]
    return []


def _as_map(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, dict):
            return dumped
    out: dict[str, Any] = {}
    for key in ("format", "bits", "group_size", "weights", "activations"):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def _weight_act(group: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _as_map(group)
    weights = data.get("weights", group if "group_size" in data else {})
    activations = data.get("activations") or {}
    return _as_map(weights), _as_map(activations)


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    loaded = _load_catalog()
    if loaded is None:
        pytest.skip("megaquant.schemes.catalog.SCHEME_CATALOG not importable")
    return loaded


def test_nvfp4_w4a8_group_size_32_act_bits_8(catalog: dict[str, Any]) -> None:
    if "nvfp4_w4a8" not in catalog:
        pytest.skip("nvfp4_w4a8 not in scheme catalog")
    groups = _groups(catalog["nvfp4_w4a8"])
    assert groups, "nvfp4_w4a8 has no layer groups"
    weights, activations = _weight_act(groups[0])
    assert weights.get("group_size") == 32
    assert activations.get("bits") == 8


def test_nvfp4_w4a4_group_size_16(catalog: dict[str, Any]) -> None:
    if "nvfp4_w4a4" not in catalog:
        pytest.skip("nvfp4_w4a4 not in scheme catalog")
    groups = _groups(catalog["nvfp4_w4a4"])
    assert groups, "nvfp4_w4a4 has no layer groups"
    weights, _activations = _weight_act(groups[0])
    assert weights.get("group_size") == 16
