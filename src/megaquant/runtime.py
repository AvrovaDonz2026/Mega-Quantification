"""Docker / host environment overrides for model loading.

Read by the pipeline so Compose flags such as ``MEGAQUANT_DEVICE_MAP`` and
``MEGAQUANT_LOW_MEMORY`` actually change ``from_pretrained`` behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from megaquant.config import Recipe

_TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_OFFLOAD_DIR = "/opt/megaquant/offload"
LOW_MEMORY_NOTE = (
    "low-memory mode enabled: from_pretrained will use offload_folder, "
    "low_cpu_mem_usage, and max_memory when CUDA is available"
)


def env_truthy(value: str | None) -> bool:
    """Return True for ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive)."""
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return env_truthy(raw)


def low_memory_enabled() -> bool:
    return env_flag("MEGAQUANT_LOW_MEMORY")


def offload_dir() -> str:
    raw = os.environ.get("MEGAQUANT_OFFLOAD_DIR", "").strip()
    return raw or DEFAULT_OFFLOAD_DIR


def parse_max_memory(spec: str) -> dict[int | str, str]:
    """Parse ``0:20GiB,cpu:128GiB`` into a transformers ``max_memory`` mapping."""
    parsed: dict[int | str, str] = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid MEGAQUANT_MAX_MEMORY fragment '{item}'")
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid MEGAQUANT_MAX_MEMORY fragment '{item}'")
        parsed[int(key) if key.isdigit() else key] = value
    return parsed


def _bytes_to_gib_str(num_bytes: int) -> str:
    gib = max(1, int(num_bytes // (1024**3)))
    return f"{gib}GiB"


def _host_cpu_memory_str() -> str:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return "128GiB"
    available_kib: int | None = None
    total_kib: int | None = None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
        elif line.startswith("MemTotal:"):
            total_kib = int(line.split()[1])
    kib = available_kib if available_kib is not None else total_kib
    if kib is None:
        return "128GiB"
    return _bytes_to_gib_str(kib * 1024)


def default_max_memory() -> dict[int | str, str] | None:
    """~70% of each visible GPU plus host CPU RAM. None if torch/CUDA is missing."""
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    mapping: dict[int | str, str] = {}
    for index in range(cuda.device_count()):
        total = cuda.get_device_properties(index).total_memory
        mapping[index] = _bytes_to_gib_str(int(total * 0.7))
    mapping["cpu"] = _host_cpu_memory_str()
    return mapping


def max_memory_from_env() -> dict[int | str, str] | None:
    raw = os.environ.get("MEGAQUANT_MAX_MEMORY", "").strip()
    if raw:
        return parse_max_memory(raw)
    return default_max_memory()


def from_pretrained_env_kwargs() -> dict[str, Any]:
    """Kwargs merged onto ``from_pretrained`` after family ``load_kwargs``.

    ``MEGAQUANT_DEVICE_MAP`` always wins when set. Low-memory mode adds
    ``offload_folder``, ``low_cpu_mem_usage=True``, and ``max_memory`` when it
    can be resolved (explicit env, or ~70% of each GPU). If torch is missing
    at dry-run time, ``max_memory`` is omitted.
    """
    kwargs: dict[str, Any] = {}
    device_map = os.environ.get("MEGAQUANT_DEVICE_MAP", "").strip()
    if device_map:
        kwargs["device_map"] = device_map
    if not low_memory_enabled():
        return kwargs
    kwargs["offload_folder"] = offload_dir()
    kwargs["low_cpu_mem_usage"] = True
    max_memory = max_memory_from_env()
    if max_memory:
        kwargs["max_memory"] = max_memory
    return kwargs


def apply_env_to_recipe(recipe: Recipe) -> Recipe:
    """Mutate ``recipe.model.device_map`` when ``MEGAQUANT_DEVICE_MAP`` is set."""
    device_map = os.environ.get("MEGAQUANT_DEVICE_MAP", "").strip()
    if device_map:
        recipe.model.device_map = device_map
    return recipe


def low_memory_plan_note() -> str | None:
    if low_memory_enabled():
        return LOW_MEMORY_NOTE
    return None
