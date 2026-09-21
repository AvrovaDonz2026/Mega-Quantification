"""Docker / host environment overrides for model loading.

Read by the pipeline so Compose flags such as ``MEGAQUANT_DEVICE_MAP`` and
``MEGAQUANT_LOW_MEMORY`` actually change ``from_pretrained`` behaviour.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from megaquant.config import Recipe

_TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_OFFLOAD_DIR = "/opt/megaquant/offload"
LOW_MEMORY_NOTE = (
    "low-memory mode: pack GPU to (VRAM − headroom), keep the rest in RAM "
    "(MemTotal − reserve), spill to offload_folder only if RAM is exhausted"
)
_AUTO_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "sequential"})


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


def skip_vision_init_enabled() -> bool:
    """Skip constructing VLM vision towers when the family asks. Default on."""
    return env_flag("MEGAQUANT_SKIP_VISION_INIT", default=True)


def offload_dir() -> str:
    raw = os.environ.get("MEGAQUANT_OFFLOAD_DIR", "").strip()
    return raw or DEFAULT_OFFLOAD_DIR


def is_auto_device_map(device_map: Any) -> bool:
    if device_map is None:
        return True
    if isinstance(device_map, str):
        return device_map.strip().lower() in _AUTO_DEVICE_MAPS
    return False


def pin_keys_to_cpu(device_map: dict[Any, Any], needles: tuple[str, ...]) -> dict[Any, Any]:
    """Force device_map entries whose names contain ``needles`` onto CPU.

    ``device_map='auto'`` walks modules in constructor order. Qwen3.8 puts the
    ViT (``model.visual``) *before* the 27B language model, so the GPU fills
    with the tower we are not even quantizing.
    """
    out = dict(device_map)
    lowered_needles = tuple(n.lower() for n in needles if n)
    if not lowered_needles:
        return out
    for key in out:
        blob = str(key).lower()
        if any(needle in blob for needle in lowered_needles):
            out[key] = "cpu"
    return out


def highest_pin_names(module_names: list[str], needles: tuple[str, ...]) -> list[str]:
    """Keep the shallowest module names that match ``needles``.

    ``model.visual`` wins over ``model.visual.blocks.0`` so a single Identity
    swap covers the whole tower.
    """
    lowered = tuple(n.lower() for n in needles if n)
    if not lowered:
        return []
    matched = [name for name in module_names if any(n in name.lower() for n in lowered)]
    matched.sort(key=lambda item: (item.count("."), len(item)))
    kept: list[str] = []
    for name in matched:
        if any(name.startswith(prefix + ".") for prefix in kept):
            continue
        kept.append(name)
    return kept


def highest_matching_module_names(model: Any, needles: tuple[str, ...]) -> list[str]:
    named = getattr(model, "named_modules", None)
    if callable(named):
        names = [name for name, _ in named() if name]
        return highest_pin_names(names, needles)
    children = getattr(model, "named_children", None)
    if not callable(children):
        return []
    return highest_pin_names([name for name, _ in children()], needles)


def _parent_and_attr(root: Any, dotted: str) -> tuple[Any, str]:
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]


@contextmanager
def swapped_pinned_children(model: Any, needles: tuple[str, ...]) -> Iterator[list[str]]:
    """Temporarily replace matching submodules with ``nn.Identity`` (size ~0).

    ``infer_auto_device_map`` then spends the GPU budget on the language model
    instead of the vision tower. Modules are restored afterwards.
    """
    import torch.nn as nn

    names = highest_matching_module_names(model, needles)
    saved: dict[str, Any] = {}
    for name in names:
        parent, attr = _parent_and_attr(model, name)
        saved[name] = getattr(parent, attr)
        setattr(parent, attr, nn.Identity())
    try:
        yield names
    finally:
        for name, module in saved.items():
            parent, attr = _parent_and_attr(model, name)
            setattr(parent, attr, module)


def infer_device_map_with_cpu_pins(
    model: Any,
    *,
    max_memory: dict[Any, Any] | None = None,
    needles: tuple[str, ...] = (),
    dtype: Any = None,
    no_split_module_classes: list[str] | None = None,
) -> dict[Any, Any]:
    """Pack unpinned modules onto GPU first; force matching subtrees onto CPU."""
    from accelerate import infer_auto_device_map

    infer_kwargs: dict[str, Any] = {}
    if max_memory is not None:
        infer_kwargs["max_memory"] = max_memory
    if dtype is not None:
        infer_kwargs["dtype"] = dtype
    if no_split_module_classes is None:
        no_split_module_classes = list(getattr(model, "_no_split_modules", None) or [])
    if no_split_module_classes:
        infer_kwargs["no_split_module_classes"] = no_split_module_classes

    with swapped_pinned_children(model, needles) as names:
        device_map = dict(infer_auto_device_map(model, **infer_kwargs))
    for name in names:
        device_map[name] = "cpu"
    if needles:
        device_map = pin_keys_to_cpu(device_map, needles)
    return device_map


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def gpu_headroom_bytes() -> int:
    """Activation / allocator slack left on each GPU. Default 1 GiB (was 2)."""
    return max(1, _env_int("MEGAQUANT_GPU_HEADROOM_GIB", 1)) * 1024**3


def cpu_reserve_bytes() -> int:
    """RAM kept for OS / Python / dataset. Default 6 GiB of a 64 GiB box."""
    return max(1, _env_int("MEGAQUANT_CPU_RESERVE_GIB", 6)) * 1024**3


def parse_meminfo_kib(text: str) -> tuple[int | None, int | None]:
    """Return ``(MemTotal KiB, MemAvailable KiB)`` from a ``/proc/meminfo`` blob."""
    total_kib: int | None = None
    available_kib: int | None = None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total_kib = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
    return total_kib, available_kib


def cpu_budget_str_from_meminfo(text: str) -> str:
    """Spend MemTotal − reserve, not the (often tiny) MemAvailable snapshot."""
    total_kib, available_kib = parse_meminfo_kib(text)
    reserve = cpu_reserve_bytes()
    if total_kib is not None:
        usable = total_kib * 1024 - reserve
        return _bytes_to_gib_str(max(8 * 1024**3, usable))
    if available_kib is not None:
        usable = available_kib * 1024 - reserve
        return _bytes_to_gib_str(max(8 * 1024**3, usable))
    return "56GiB"


def _host_cpu_memory_str() -> str:
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return "56GiB"
    return cpu_budget_str_from_meminfo(text)


FAST_NUM_SAMPLES = 128
FAST_MAX_SEQ_LENGTH = 1024
FAST_MAX_BATCH_SIZE = 4


def default_max_memory() -> dict[int | str, str] | None:
    """Fill each GPU up to (VRAM − headroom) and put the rest in host RAM."""
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    headroom = gpu_headroom_bytes()
    mapping: dict[int | str, str] = {}
    for index in range(cuda.device_count()):
        total = cuda.get_device_properties(index).total_memory
        usable = max(4 * 1024**3, total - headroom)
        mapping[index] = _bytes_to_gib_str(int(usable))
    mapping["cpu"] = _host_cpu_memory_str()
    return mapping


def max_memory_from_env() -> dict[int | str, str] | None:
    raw = os.environ.get("MEGAQUANT_MAX_MEMORY", "").strip()
    if raw:
        return parse_max_memory(raw)
    return default_max_memory()


def cpu_thread_count() -> int:
    raw = os.environ.get("MEGAQUANT_NUM_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, os.cpu_count() or 1)


def configure_host_parallelism() -> int:
    """Use every CPU core for intra-op math and tokenizer parallelism.

    Must run before ``import torch`` for ``OMP_NUM_THREADS`` to stick; also
    calls ``torch.set_num_threads`` when torch is already imported.
    """
    n = cpu_thread_count()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Concurrent copy engines + kernels so CPU-offload H2D can overlap compute
    # on a PCIe 5.0 x16 5090 (idle link otherwise drops to gen1).
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "16")
    try:
        import torch

        torch.set_num_threads(n)
        interop = 1 if n == 1 else max(2, min(8, n // 2))
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            pass
    except Exception:
        pass
    return n


def from_pretrained_env_kwargs() -> dict[str, Any]:
    """Kwargs merged onto ``from_pretrained`` after family ``load_kwargs``.

    ``MEGAQUANT_DEVICE_MAP`` always wins when set. Low-memory mode adds
    ``offload_folder`` (disk spill only), ``low_cpu_mem_usage=True``,
    ``offload_state_dict``, ``offload_buffers``, and ``max_memory`` packed from
    VRAM/RAM. If torch is missing at dry-run time, ``max_memory`` is omitted.
    """
    kwargs: dict[str, Any] = {}
    device_map = os.environ.get("MEGAQUANT_DEVICE_MAP", "").strip()
    if device_map:
        kwargs["device_map"] = device_map
    if not low_memory_enabled():
        return kwargs
    kwargs["offload_folder"] = offload_dir()
    kwargs["low_cpu_mem_usage"] = True
    kwargs["offload_state_dict"] = True
    kwargs["offload_buffers"] = True
    max_memory = max_memory_from_env()
    if max_memory:
        kwargs["max_memory"] = max_memory
    return kwargs


def apply_env_to_recipe(recipe: Recipe) -> Recipe:
    """Mutate device_map / calibration when Compose or gpu-pod env vars are set."""
    device_map = os.environ.get("MEGAQUANT_DEVICE_MAP", "").strip()
    if device_map:
        recipe.model.device_map = device_map
    if env_flag("MEGAQUANT_FAST"):
        recipe.calibration.num_samples = min(recipe.calibration.num_samples, FAST_NUM_SAMPLES)
        recipe.calibration.max_seq_length = min(
            recipe.calibration.max_seq_length, FAST_MAX_SEQ_LENGTH
        )
        recipe.calibration.batch_size = max(recipe.calibration.batch_size, FAST_MAX_BATCH_SIZE)
    raw_samples = os.environ.get("MEGAQUANT_NUM_SAMPLES", "").strip()
    if raw_samples:
        recipe.calibration.num_samples = int(raw_samples)
    raw_seq = os.environ.get("MEGAQUANT_MAX_SEQ_LENGTH", "").strip()
    if raw_seq:
        recipe.calibration.max_seq_length = int(raw_seq)
    raw_batch = os.environ.get("MEGAQUANT_BATCH_SIZE", "").strip()
    if raw_batch:
        recipe.calibration.batch_size = max(1, int(raw_batch))
    return recipe


def low_memory_plan_note() -> str | None:
    if low_memory_enabled():
        return LOW_MEMORY_NOTE
    return None


def fast_plan_note() -> str | None:
    if env_flag("MEGAQUANT_FAST"):
        return (
            f"fast calib: num_samples≤{FAST_NUM_SAMPLES}, "
            f"max_seq_length≤{FAST_MAX_SEQ_LENGTH} (max/absmax still real PTQ)"
        )
    return None


def parse_pcie_link_csv(text: str) -> dict[str, int] | None:
    """Parse ``nvidia-smi --query-gpu=pcie.link.gen.current,...`` csv."""
    line = (text or "").strip().splitlines()
    if not line:
        return None
    payload = line[-1]
    if payload.lower().startswith("pcie"):
        return None
    parts = [item.strip() for item in payload.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "gen_current": int(parts[0]),
            "gen_max": int(parts[1]),
            "width_current": int(parts[2]),
            "width_max": int(parts[3]),
        }
    except ValueError:
        return None


def query_pcie_link() -> dict[str, int] | None:
    """Live GPU PCIe gen/width. Idle 5090s often report gen1 until CUDA warmup."""
    try:
        import subprocess

        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
                "pcie.link.width.current,pcie.link.width.max",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    return parse_pcie_link_csv(raw)


def pcie_plan_note() -> str | None:
    info = query_pcie_link()
    if info is None:
        return (
            "PCIe: CUDA warmup trains the link; idle 5090s report gen1 even "
            "when the slot is gen5 x16 (~50+ GiB/s after warmup)"
        )
    return (
        f"PCIe gen {info['gen_current']}/{info['gen_max']} "
        f"x{info['width_current']}/{info['width_max']} "
        "(idle often gen1; warmup before PTQ to lock gen5 x16 DMA)"
    )


def warmup_pcie_link(nbytes: int = 512 * 1024 * 1024, rounds: int = 4) -> str | None:
    """Create a CUDA context and DMA a pinned buffer so the link trains to gen5.

    Compshare / idle driver power-saving leaves RTX 5090 at PCIe **gen1 x16**
    (~4 GB/s). After a few GiB of pinned H2D/D2H the same card reports
    **gen5 x16** and ~50 GiB/s. Run this before ``from_pretrained``.
    """
    try:
        import time

        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    before = query_pcie_link()
    try:
        cuda.init()
        n = max(1, int(nbytes) // 4)
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
        device = torch.empty(n, dtype=torch.float32, device="cuda")
        stream = cuda.Stream()
        cuda.synchronize()
        started = time.perf_counter()
        with cuda.stream(stream):
            for _ in range(max(1, rounds)):
                device.copy_(host, non_blocking=True)
                host.copy_(device, non_blocking=True)
        stream.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-6)
        gib = max(1, rounds) * 2 * (n * 4) / 1024**3
        rate = gib / elapsed
    except Exception as exc:
        return f"PCIe warmup skipped ({type(exc).__name__}: {exc})"
    after = query_pcie_link()
    before_s = (
        f"gen{before['gen_current']}x{before['width_current']}" if before else "?"
    )
    after_s = f"gen{after['gen_current']}x{after['width_current']}" if after else "?"
    return (
        f"PCIe warmup {before_s}→{after_s}  {gib:.1f} GiB in {elapsed:.2f}s "
        f"({rate:.1f} GiB/s pinned DMA)"
    )


def pin_cpu_parameters(model: Any) -> int:
    """Pin CPU-resident weights so accelerate H2D uses PCIe DMA, not pageable copies."""
    pinned = 0

    def _pin(tensor: Any) -> Any:
        nonlocal pinned
        device = getattr(tensor, "device", None)
        kind = getattr(device, "type", None)
        if kind != "cpu":
            return tensor
        is_pinned = getattr(tensor, "is_pinned", None)
        try:
            if callable(is_pinned) and is_pinned():
                return tensor
        except Exception:
            return tensor
        pin = getattr(tensor, "pin_memory", None)
        if not callable(pin):
            return tensor
        try:
            out = pin()
            pinned += 1
            return out
        except Exception:
            return tensor

    named_params = getattr(model, "named_parameters", None)
    if callable(named_params):
        for _name, param in named_params():
            data = getattr(param, "data", None)
            if data is None:
                continue
            pinned_data = _pin(data)
            if pinned_data is not data:
                try:
                    param.data = pinned_data
                except Exception:
                    pass
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        for name, buf in named_buffers():
            pinned_buf = _pin(buf)
            if pinned_buf is not buf:
                try:
                    parts = name.split(".")
                    obj = model
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], pinned_buf)
                except Exception:
                    pass
    return pinned


def enable_accelerate_non_blocking() -> str | None:
    """Make accelerate's CPU→GPU copies async so they can overlap compute."""
    try:
        import accelerate.utils.operations as ops
    except ImportError:
        try:
            import accelerate.utils as ops
        except ImportError:
            return None
    orig = getattr(ops, "send_to_device", None)
    if not callable(orig) or getattr(ops, "_megaquant_non_blocking", False):
        return "accelerate send_to_device non_blocking" if callable(orig) else None

    def send_to_device(tensor: Any, device: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("non_blocking", True)
        return orig(tensor, device, *args, **kwargs)

    ops.send_to_device = send_to_device  # type: ignore[method-assign]
    ops._megaquant_non_blocking = True
    return "accelerate send_to_device non_blocking=True"


def host_pack_plan_note() -> str | None:
    n = cpu_thread_count()
    headroom = max(1, _env_int("MEGAQUANT_GPU_HEADROOM_GIB", 1))
    reserve = max(1, _env_int("MEGAQUANT_CPU_RESERVE_GIB", 6))
    cpu_budget = _host_cpu_memory_str()
    conns = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "16")
    return (
        f"host pack: threads={n}, GPU headroom={headroom}GiB, "
        f"RAM reserve={reserve}GiB, CPU weight budget={cpu_budget}, "
        f"CUDA_DEVICE_MAX_CONNECTIONS={conns}"
    )


def gdn_kernel_plan_note() -> str | None:
    """Warn when Qwen3.8 Gated DeltaNet will run the slow PyTorch fallback."""
    missing: list[str] = []
    for name in ("fla", "causal_conv1d"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    try:
        __import__("kernels")
    except ImportError:
        missing.append("kernels")
    if not missing:
        return None
    return (
        "GDN fast path missing ("
        + ", ".join(missing)
        + "); calib uses the PyTorch fallback (low SM%, ~5s/step on 5090). "
        "pip install kernels flash-linear-attention, and causal-conv1d when a "
        "sm_120 wheel or compile is available"
    )
