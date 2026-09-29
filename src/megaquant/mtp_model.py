"""Attach Qwen3.5's standalone MTP block before ModelOpt quantization.

Qwen3.5 checkpoints store ``mtp.*`` tensors at the repository root, while the
Transformers conditional-generation class does not construct that optional
block.  ModelOpt can therefore never see MTP unless the pipeline attaches a
small state-dict compatible module first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from megaquant.exceptions import BackendError
from megaquant.mtp_export import _open_source_shard, _source_index, is_mtp_tensor_name


def _load_source_tensors(source: str | Path) -> dict[str, Any]:
    """Load only the source shards that contain standalone MTP tensors."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - pipeline dependency
        raise BackendError("safetensors is required to attach Qwen3.5 MTP") from exc

    index, root = _source_index(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return {}
    by_shard: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if is_mtp_tensor_name(str(key)):
            by_shard.setdefault(str(filename), []).append(str(key))

    tensors: dict[str, Any] = {}
    for filename, expected in by_shard.items():
        shard = _open_source_shard(root, filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in expected:
                if key not in handle.keys():
                    raise BackendError(f"MTP tensor {key} missing from {filename}")
                tensors[key] = handle.get_tensor(key)
    return tensors


def _linear(torch: Any, tensor: Any) -> Any:
    import torch.nn as nn

    if tensor.ndim != 2:
        raise BackendError(f"MTP projection must be rank 2, got shape {tuple(tensor.shape)}")
    layer = nn.Linear(
        int(tensor.shape[1]), int(tensor.shape[0]), bias=False, dtype=tensor.dtype
    )
    with torch.no_grad():
        layer.weight.copy_(tensor)
    return layer


def _norm(torch: Any, tensor: Any) -> Any:
    import torch.nn as nn

    layer = nn.Module()
    layer.register_parameter("weight", nn.Parameter(tensor.detach().clone()))
    return layer


def attach_mtp(model: Any, source: str | Path) -> int:
    """Attach source MTP weights under ``model.mtp`` and return tensor count.

    The standalone block is intentionally state-dict compatible.  Its Linear
    projections are visible to ModelOpt's existing ``*mtp*`` opt-in patterns;
    the target model does not call the block during language calibration.
    """
    if getattr(model, "mtp", None) is not None:
        return sum(1 for name, _ in model.named_parameters() if is_mtp_tensor_name(name))

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - pipeline dependency
        raise BackendError("torch is required to attach Qwen3.5 MTP") from exc

    tensors = _load_source_tensors(source)
    if not tensors:
        raise BackendError(f"source checkpoint has no standalone mtp.* tensors: {source}")

    mtp = nn.Module()
    layer = nn.Module()
    self_attn = nn.Module()
    mlp = nn.Module()
    layer.add_module("self_attn", self_attn)
    layer.add_module("mlp", mlp)
    mtp.add_module("layers", nn.ModuleList([layer]))

    projections = {
        "fc",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    for name, tensor in sorted(tensors.items()):
        relative = name.removeprefix("mtp.")
        parts = relative.split(".")
        if parts[-1] != "weight":
            continue
        parent = parts[-2]
        if parent in projections:
            module: Any = _linear(torch, tensor)
        else:
            module = _norm(torch, tensor)
        if relative.startswith("layers.0.self_attn."):
            setattr(self_attn, parent, module)
        elif relative.startswith("layers.0.mlp."):
            setattr(mlp, parent, module)
        elif relative.startswith("layers.0."):
            setattr(layer, parent, module)
        else:
            setattr(mtp, parent, module)

    model.add_module("mtp", mtp)
    return len(tensors)
