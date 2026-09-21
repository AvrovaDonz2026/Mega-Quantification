"""Runtime env overrides (device_map / low-memory). CPU-only, no weights."""

from __future__ import annotations

import os

import pytest

from megaquant.config import load_recipe
from megaquant.models.qwen3_5 import Qwen35Family
from megaquant.pipeline import QuantPipeline, format_plan
from megaquant.runtime import (
    configure_host_parallelism,
    cpu_budget_str_from_meminfo,
    env_truthy,
    from_pretrained_env_kwargs,
    highest_pin_names,
    parse_max_memory,
    parse_meminfo_kib,
    pin_keys_to_cpu,
)

_RUNTIME_KEYS = (
    "MEGAQUANT_DEVICE_MAP",
    "MEGAQUANT_LOW_MEMORY",
    "MEGAQUANT_OFFLOAD_DIR",
    "MEGAQUANT_MAX_MEMORY",
    "MEGAQUANT_FAST",
    "MEGAQUANT_NUM_SAMPLES",
    "MEGAQUANT_MAX_SEQ_LENGTH",
    "MEGAQUANT_BATCH_SIZE",
    "MEGAQUANT_NUM_THREADS",
    "MEGAQUANT_GPU_HEADROOM_GIB",
    "MEGAQUANT_CPU_RESERVE_GIB",
    "MEGAQUANT_PIN_MEMORY",
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
    assert kwargs["offload_state_dict"] is True
    assert kwargs["offload_buffers"] is True
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


def test_pin_keys_to_cpu_matches_nested_visual() -> None:
    mapping = {
        "model.visual": 0,
        "model.visual.blocks.0": 0,
        "model.language_model.layers.0": 0,
        "lm_head": 0,
    }
    pinned = pin_keys_to_cpu(mapping, ("visual", "vision", "mtp"))
    assert pinned["model.visual"] == "cpu"
    assert pinned["model.visual.blocks.0"] == "cpu"
    assert pinned["model.language_model.layers.0"] == 0
    assert pinned["lm_head"] == 0


def test_highest_pin_names_keeps_parent_visual() -> None:
    names = [
        "model",
        "model.visual",
        "model.visual.blocks.0",
        "model.visual.blocks.0.attn",
        "model.language_model",
        "model.language_model.layers.0",
        "lm_head",
    ]
    assert highest_pin_names(names, ("visual", "vision")) == ["model.visual"]
    assert highest_pin_names(names, ("mtp",)) == []


def test_qwen35_plan_notes_mention_vision_pin(
    clean_runtime_env: None,
    w4a8_recipe_path,
) -> None:
    recipe = load_recipe(w4a8_recipe_path)
    plan = QuantPipeline(recipe).resolve()
    blob = " ".join(plan.notes).lower()
    assert "visual" in blob or "language-only" in blob
    dumped = format_plan(plan)
    assert "device_map" in dumped


def test_cpu_budget_uses_memtotal_minus_reserve(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGAQUANT_CPU_RESERVE_GIB", "8")
    text = "MemTotal:       67108864 kB\nMemAvailable:      1234 kB\n"
    total, available = parse_meminfo_kib(text)
    assert total == 67108864
    assert available == 1234
    # 64 GiB − 8 GiB reserve = 56 GiB, ignoring the tiny MemAvailable figure.
    assert cpu_budget_str_from_meminfo(text) == "56GiB"


def test_batch_size_env_override(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
    w4a8_recipe_path,
) -> None:
    monkeypatch.setenv("MEGAQUANT_BATCH_SIZE", "4")
    recipe = load_recipe(w4a8_recipe_path)
    plan = QuantPipeline(recipe).resolve()
    assert plan.recipe.calibration.batch_size == 4
    blob = " ".join(plan.notes).lower()
    assert "host pack" in blob
    assert "threads=" in blob


def test_configure_host_parallelism_sets_omp(
    clean_runtime_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEGAQUANT_NUM_THREADS", "12")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    n = configure_host_parallelism()
    assert n == 12
    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert os.environ["MKL_NUM_THREADS"] == "12"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
