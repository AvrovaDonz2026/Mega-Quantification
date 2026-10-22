"""Fail-closed precision checks for opted-in vision and MTP modules.

The snapshot is deliberately taken from the live ModelOpt modules before export;
the validator then checks only safetensors headers and never materializes payloads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from megaquant.exceptions import BackendError


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enabled(q: Any) -> bool:
    value = _field(q, "is_enabled", False)
    return bool(value() if callable(value) else value)


def _bits(q: Any) -> tuple[int, int] | None:
    value = _field(q, "num_bits")
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _block_size(q: Any) -> int | None:
    blocks = _field(q, "block_sizes", {})
    if isinstance(blocks, Mapping):
        value = blocks.get(-1, blocks.get("-1"))
        if value is None:
            value = blocks.get("group_size")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _format(q: Any, *, input_quantizer: bool = False) -> tuple[str, int | None]:
    bits = _bits(q)
    if bits == (2, 1):
        group = _block_size(q)
        if group not in (16, 32):
            where = "input" if input_quantizer else "weight"
            raise BackendError(f"Unsupported auxiliary {where} NVFP4 group size: {group!r}")
        return "NVFP4", group
    if bits == (4, 3):
        return "FP8", _block_size(q)
    where = "input" if input_quantizer else "weight"
    raise BackendError(f"Unsupported auxiliary {where} quantizer precision: num_bits={bits!r}")


def _recipe_flag(recipe: Any, name: str) -> bool:
    model = _field(recipe, "model", {})
    return bool(_field(model, name, False))


def _aux_kind(name: str) -> str | None:
    parts = name.replace("\\", ".").split(".")
    if "mtp" in parts:
        return "mtp"
    if "visual" in parts or "vision" in parts:
        return "vision"
    return None


def snapshot_auxiliary_quantizers(model: Any, recipe: Any) -> dict[str, dict[str, Any]]:
    """Capture enabled auxiliary quantizers before ModelOpt mutates/export modules."""
    wanted = {
        "vision": _recipe_flag(recipe, "quantize_vision"),
        "mtp": _recipe_flag(recipe, "quantize_mtp"),
    }
    expected: dict[str, dict[str, Any]] = {}
    modules = model.named_modules() if hasattr(model, "named_modules") else ()
    for name, parent in modules:
        kind = _aux_kind(str(name))
        if kind is None or not wanted[kind] or not hasattr(parent, "weight_quantizer"):
            continue
        weight_q = parent.weight_quantizer
        if not _enabled(weight_q):
            continue
        shape = tuple(int(x) for x in getattr(getattr(parent, "weight", None), "shape", ()))
        if len(shape) != 2:
            raise BackendError(f"Auxiliary module {name} weight must be 2D, got {shape}")
        fmt, group = _format(weight_q)
        input_q = getattr(parent, "input_quantizer", None)
        input_enabled = input_q is not None and _enabled(input_q)
        input_fmt = input_group = None
        if input_enabled:
            input_fmt, input_group = _format(input_q, input_quantizer=True)
            if input_fmt != fmt or (fmt == "NVFP4" and input_group != group):
                raise BackendError(
                    f"Unsupported auxiliary weight/input quantizer combination for {name}"
                )
        expected[str(name)] = {
            "kind": kind,
            "format": fmt,
            "group_size": group,
            "weight_shape": list(shape),
            "input_enabled": input_enabled,
            "input_format": input_fmt,
            "input_group_size": input_group,
        }
    for kind, enabled in wanted.items():
        if enabled and not any(item["kind"] == kind for item in expected.values()):
            raise BackendError(
                f"Recipe opted into quantized {kind}, but no enabled auxiliary "
                "weight quantizer was found"
            )
    return expected


def _headers(root: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        except Exception as exc:
            raise BackendError(f"Unable to read safetensors index: {exc}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise BackendError("safetensors index has no weight_map")
    metadata: dict[str, dict[str, Any]] = {}
    actual_map: dict[str, str] = {}
    by_file: dict[str, list[str]] = {}
    indexed_owner: dict[str, str] = {}
    if index_path.is_file():
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str):
                raise BackendError("Invalid safetensors index key or shard")
            previous = indexed_owner.setdefault(key, shard)
            if previous != shard:
                raise BackendError(f"Safetensors key mapped to multiple shards: {key}")
            by_file.setdefault(shard, []).append(key)
    else:
        by_file["model.safetensors"] = []
    for shard, indexed in by_file.items():
        path = root / shard
        if (
            Path(shard).is_absolute()
            or ".." in Path(shard).parts
            or path.suffix != ".safetensors"
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise BackendError("Invalid safetensors shard path")
        if not path.is_file():
            raise BackendError(f"Indexed safetensors shard missing: {shard}")
        try:
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated header prefix")
                size = int.from_bytes(prefix, "little")
                if size > 100_000_000 or size > path.stat().st_size - 8:
                    raise ValueError("invalid header size")
                header = json.loads(handle.read(size))
        except Exception as exc:
            raise BackendError(f"Unable to inspect safetensors shard {shard}: {exc}") from exc
        if not isinstance(header, dict):
            raise BackendError(f"Invalid safetensors header: {shard}")
        available = {
            str(k)
            for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict)
        }
        missing = set(indexed) - available
        if missing:
            raise BackendError(
                f"Indexed safetensors shard {shard} missing key {sorted(missing)[0]}"
            )
        # An indexed checkpoint is defined by the index, not by the union of
        # shard headers.  Unindexed copies must never satisfy this contract or
        # overwrite the metadata of the shard the loader will actually use.
        for key in indexed if index_path.is_file() else available:
            if index_path.is_file() and actual_map.get(key) not in (None, shard):
                raise BackendError(f"Safetensors key appears in multiple shards: {key}")
            info = header[key]
            metadata[key] = info
            actual_map[key] = shard
    if index_path.is_file():
        for key in weight_map:
            if str(key) not in actual_map:
                raise BackendError(f"Indexed safetensors key missing: {key}")
    return actual_map, metadata


def _aliases(name: str) -> tuple[str, ...]:
    aliases = [name]
    if name.startswith("model.visual."):
        aliases.append(name[len("model."):])
    elif name.startswith("visual."):
        aliases.append("model." + name)
    elif name.startswith("model.vision."):
        aliases.append(name[len("model."):])
    elif name.startswith("vision."):
        aliases.append("model." + name)
    elif name.startswith("model.mtp."):
        aliases.append(name[len("model."):])
    elif name.startswith("mtp."):
        aliases.append("model." + name)
    return tuple(dict.fromkeys(aliases))


def _lookup(name: str, actual: Mapping[str, Any], *, tensors: bool = False) -> str:
    found = [
        candidate
        for candidate in _aliases(name)
        if (candidate + ".weight" if tensors else candidate) in actual
    ]
    if not found:
        raise BackendError(f"Auxiliary module missing from exported safetensors: {name}")
    if len(found) > 1:
        raise BackendError(f"Ambiguous auxiliary module aliases for {name}: {found}")
    return found[0]


def _tensor_key(
    actual: Mapping[str, str], exported: str, suffix: str, module: str
) -> str:
    candidate = exported + suffix
    if candidate not in actual:
        raise BackendError(f"Auxiliary tensor missing for {module}: {candidate}")
    return candidate


def _shape(info: Mapping[str, Any], module: str) -> list[int]:
    shape = info.get("shape")
    if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
        raise BackendError(f"Invalid auxiliary tensor shape for {module}")
    return shape


def validate_auxiliary_precision(
    root: str | Path,
    expected: Mapping[str, Mapping[str, Any]],
    layers: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Validate expected auxiliary precision using safetensors headers only."""
    counts = {"checked": 0, "nvfp4": 0, "fp8": 0, "input_scaled": 0}
    if not expected:
        return counts
    actual, meta = _headers(Path(root))
    layer_map = layers if isinstance(layers, Mapping) else None
    for module, spec in expected.items():
        exported = _lookup(str(module), actual, tensors=True)
        weight_key = _tensor_key(actual, exported, ".weight", str(module))
        info = meta[weight_key]
        if info.get("dtype") in {"BF16", "F16", "F32", "F64"}:
            raise BackendError(
                f"Auxiliary module {module} exported unquantized as {info.get('dtype')}"
            )
        shape = _shape(info, str(module))
        original = [int(x) for x in spec.get("weight_shape", [])]
        if len(original) != 2 or any(x <= 0 for x in original):
            raise BackendError(f"Invalid expected auxiliary weight shape for {module}")
        if shape != original and spec.get("format") == "FP8":
            raise BackendError(f"FP8 weight shape mismatch for {module}: {shape} != {original}")
        fmt = str(spec.get("format"))
        if fmt == "NVFP4":
            if (
                info.get("dtype") != "U8"
                or len(shape) != 2
                or shape[:-1] != original[:-1]
                or shape[-1] * 2 != original[-1]
            ):
                raise BackendError(f"NVFP4 packed weight shape/dtype mismatch for {module}")
            group_size = int(spec.get("group_size") or 0)
            if group_size not in (16, 32):
                raise BackendError(f"NVFP4 group size missing or invalid for {module}")
            scale = meta[_tensor_key(actual, exported, ".weight_scale", str(module))]
            scale_shape = _shape(scale, str(module))
            if (
                scale.get("dtype")
                not in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
                or len(scale_shape) != 2
                or original[-1] % group_size
                or scale_shape[:-1] != original[:-1]
                or scale_shape[-1] != original[-1] // group_size
            ):
                raise BackendError(f"NVFP4 weight_scale shape/dtype mismatch for {module}")
            scale2 = meta[_tensor_key(actual, exported, ".weight_scale_2", str(module))]
            if (
                scale2.get("dtype") not in {"F32", "F16", "BF16"}
                or _shape(scale2, str(module)) not in ([], [1])
            ):
                raise BackendError(f"NVFP4 weight_scale_2 must be a float scalar for {module}")
            counts["nvfp4"] += 1
        elif fmt == "FP8":
            if info.get("dtype") not in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}:
                raise BackendError(f"FP8 weight dtype mismatch for {module}")
            scale = meta[_tensor_key(actual, exported, ".weight_scale", str(module))]
            if any(x == 0 for x in _shape(scale, str(module))):
                raise BackendError(f"Empty FP8 weight_scale for {module}")
            if scale.get("dtype") not in {
                "F32",
                "F16",
                "BF16",
                "F8_E4M3",
                "F8_E4M3FN",
                "F8_E4M3FNUZ",
            }:
                raise BackendError(f"FP8 weight_scale dtype mismatch for {module}")
            counts["fp8"] += 1
        else:
            raise BackendError(f"Unsupported expected auxiliary format for {module}: {fmt}")
        if bool(spec.get("input_enabled")):
            input_key = next(
                (
                    exported + suffix
                    for suffix in (".input_scale", ".input_global_scale")
                    if exported + suffix in actual
                ),
                None,
            )
            if input_key is None:
                raise BackendError(f"Auxiliary input scale missing for {module}")
            if any(x == 0 for x in _shape(meta[input_key], str(module))):
                raise BackendError(f"Empty auxiliary input scale for {module}")
            if meta[input_key].get("dtype") not in {
                "F32",
                "F16",
                "BF16",
                "F8_E4M3",
                "F8_E4M3FN",
                "F8_E4M3FNUZ",
            }:
                raise BackendError(f"Auxiliary input scale dtype invalid for {module}")
            counts["input_scaled"] += 1
        if layer_map is not None:
            layer_key = _lookup(str(module), layer_map)
            entry = layer_map[layer_key]
            if not isinstance(entry, Mapping):
                raise BackendError(f"Invalid quantized layer metadata for {module}")
            algo = str(entry.get("quant_algo", "")).upper()
            wanted_algos = {fmt}
            if fmt == "NVFP4" and not spec.get("input_enabled"):
                wanted_algos.add("W4A16_NVFP4")
            if algo not in wanted_algos or (
                fmt == "NVFP4" and entry.get("group_size") != spec.get("group_size")
            ):
                raise BackendError(f"Quantized layer metadata mismatch for {module}")
        counts["checked"] += 1
    return counts
