"""Runtime env overrides (device_map / low-memory). CPU-only, no weights."""

from __future__ import annotations

import pytest

from megaquant.config import load_recipe
from megaquant.models.qwen3_5 import Qwen35Family
from megaquant.pipeline import QuantPipeline, format_plan
from megaquant.runtime import (
    env_truthy,
    from_pretrained_env_kwargs,
    parse_max_memory,
)

_RUNTIME_KEYS = (
    "MEGAQUANT_DEVICE_MAP",
    "MEGAQUANT_LOW_MEMORY",
    "MEGAQUANT_OFFLOAD_DIR",
    "MEGAQUANT_MAX_MEMORY",
    "MEGAQUANT_FAST",
    "MEGAQUANT_NUM_SAMPLES",
    "MEGAQUANT_MAX_SEQ_LENGTH",
)


@pytest.fixture
def clean_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _RUNTIME_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        (None, False),
    ],
)
def test_env_truthy(value: str | None, expected: bool) -> None:
    assert env_truthy(value) is expected


def test_parse_max_memory_spec() -> None:
    assert parse_max_memory("0:20GiB,cpu:128GiB") == {0: "20GiB", "cpu": "128GiB"}


def test_device_map_env_override_on_plan(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
    w4a8_recipe_path,
) -> None:
    monkeypatch.setenv("MEGAQUANT_DEVICE_MAP", "cpu")
    recipe = load_recipe(w4a8_recipe_path)
    assert recipe.model.device_map == "auto"
    plan = QuantPipeline(recipe).resolve()
    assert plan.recipe.model.device_map == "cpu"
    dumped = format_plan(plan)
    assert "device_map" in dumped
    assert "cpu" in dumped


def test_low_memory_note_on_plan(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
    w4a8_recipe_path,
) -> None:
    monkeypatch.setenv("MEGAQUANT_LOW_MEMORY", "1")
    recipe = load_recipe(w4a8_recipe_path)
    plan = QuantPipeline(recipe).resolve()
    assert any("low-memory" in note.lower() for note in plan.notes)
    dumped = format_plan(plan)
    assert "low-memory" in dumped.lower()
    assert "device_map" in dumped


def test_low_memory_from_pretrained_kwargs(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGAQUANT_LOW_MEMORY", "yes")
    monkeypatch.setenv("MEGAQUANT_OFFLOAD_DIR", "/tmp/mq-offload")
    monkeypatch.setenv("MEGAQUANT_MAX_MEMORY", "0:20GiB,cpu:128GiB")
    kwargs = from_pretrained_env_kwargs()
    assert kwargs["offload_folder"] == "/tmp/mq-offload"
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["max_memory"] == {0: "20GiB", "cpu": "128GiB"}


def test_env_device_map_wins_after_family_kwargs(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGAQUANT_DEVICE_MAP", "sequential")
    family_kwargs = Qwen35Family().load_kwargs(
        {"model": {"device_map": "cpu", "dtype": "bfloat16"}}
    )
    merged = {**family_kwargs, **from_pretrained_env_kwargs()}
    assert family_kwargs["device_map"] == "cpu"
    assert merged["device_map"] == "sequential"


def test_low_memory_skips_max_memory_without_torch_or_spec(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGAQUANT_LOW_MEMORY", "on")
    kwargs = from_pretrained_env_kwargs()
    assert kwargs["low_cpu_mem_usage"] is True
    assert "offload_folder" in kwargs
    try:
        import torch
    except ImportError:
        assert "max_memory" not in kwargs
        return
    if not torch.cuda.is_available():
        assert "max_memory" not in kwargs


def test_fast_env_shrinks_calib(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
    w4a8_recipe_path,
) -> None:
    monkeypatch.setenv("MEGAQUANT_FAST", "1")
    recipe = load_recipe(w4a8_recipe_path)
    plan = QuantPipeline(recipe).resolve()
    assert plan.recipe.calibration.num_samples == 128
    assert plan.recipe.calibration.max_seq_length == 1024
    assert any("fast calib" in note.lower() for note in plan.notes)
