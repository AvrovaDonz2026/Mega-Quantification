#!/usr/bin/env python3
"""Create independent BF16 precision ablations from a ModelOpt HF export.

This is an offline diagnostic tool, not a new PTQ recipe. It copies raw tensor
bytes without loading Torch, changing scales, or writing either input. Only
selected packed language projections are replaced with original BF16 payloads.
SGLang's modelopt_mixed loader uses unquantized linear methods for modules
absent from quantized_layers; the map is rebuilt from actual serialized tensors,
including quantized vision and MTP. No synthetic lm_head aliases are added.

The default is a header-only plan. --write creates a new independent directory
through a private staging directory, verifies every tensor payload hash, then
renames it into place. Original provenance is retained; precision_ablation.json
records the derivative instead of claiming a fresh quantization. Output contains
no input paths. A GPU serving check is still required before any quality claim.
MLP projection and layer-band selectors are diagnostic controls only; they do not
establish GPU serving behavior or a quantization quality result.

For example, --variant mlp --mlp-projection down_proj --layer-start 16
--layer-end 32 selects down_proj in layers 16 through 31. Repeat
--mlp-projection to select multiple families; omit the selectors for all MLPs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, NamedTuple

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4,
    "U64": 8, "I64": 8, "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E4M3FNUZ": 1, "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1, "F8_E8M0": 1,
}
FP8_WEIGHTS = {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
SCALE_SUFFIXES = {
    "weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale",
    "input_global_scale", "output_scale", "amax",
}
FUSED_PARTS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}
INDEX_NAME = "model.safetensors.index.json"
CHUNK_SIZE = 8 * 1024 * 1024
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class Tensor(NamedTuple):
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shard(root: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise ValueError(f"Unsafe shard filename: {filename}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping shard: {filename}")
    return path


def read_header(path: Path) -> dict[str, Tensor]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        size = path.stat().st_size
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError(f"Invalid safetensors header: {path.name}")
        header = json.loads(handle.read(length), object_pairs_hook=_unique_pairs)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors object: {path.name}")
    tensors = {}
    ranges = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or info.get("dtype") not in DTYPE_BYTES:
            raise ValueError(f"Unsupported tensor dtype: {name}")
        shape, offsets = info.get("shape"), info.get("data_offsets")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid shape: {name}")
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise ValueError(f"Invalid offsets: {name}")
        start, end = offsets
        nbytes = math.prod(shape) * DTYPE_BYTES[info["dtype"]]
        if start < 0 or end - start != nbytes or end > size - 8 - length:
            raise ValueError(f"Tensor byte count or bounds mismatch: {name}")
        tensors[name] = Tensor(path, info["dtype"], tuple(shape), 8 + length + start, nbytes)
        if nbytes:
            ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f"Overlapping or gapped tensor payloads: {path.name}")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError(f"Unindexed trailing safetensors data: {path.name}")
    return tensors


def read_checkpoint(root: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    index_path = root / INDEX_NAME
    if index_path.is_file():
        index = _json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Missing or empty HF weight_map")
    else:
        tensors = read_header(_shard(root, "model.safetensors"))
        index = {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
        weight_map = index["weight_map"]
    if any(not isinstance(name, str) or not isinstance(file, str)
           for name, file in weight_map.items()):
        raise ValueError("Invalid HF weight_map names")
    tensors: dict[str, Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        header = read_header(_shard(root, filename))
        expected = {name for name, file in weight_map.items() if file == filename}
        if set(header) != expected:
            raise ValueError(f"HF index/header disagreement: {filename}")
        tensors.update(header)
    return tensors, index


def canonical(name: str) -> str:
    # The branches must be checked before removing language prefix aliases.
    if any(part in name.split(".") for part in ("mtp", "visual", "vision")):
        return name
    for prefix in ("model.language_model.", "language_model.model.", "language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _mlp_selection(
    variant: str,
    mlp_projections: list[str] | tuple[str, ...] | None,
    layer_start: int | None,
    layer_end: int | None,
) -> tuple[tuple[str, ...], int | None, int | None]:
    if any(value is not None for value in (mlp_projections, layer_start, layer_end)):
        if variant != "mlp":
            raise ValueError("MLP projection and layer selectors require --variant mlp")
    if (layer_start is None) != (layer_end is None):
        raise ValueError("--layer-start and --layer-end must be supplied together")
    if layer_start is not None and (layer_start < 0 or layer_end <= layer_start):
        raise ValueError("MLP layer interval must satisfy 0 <= start < end")
    requested = MLP_PROJECTIONS if mlp_projections is None else tuple(mlp_projections)
    if not requested:
        raise ValueError("MLP projection selector cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("MLP projection selector contains duplicates")
    unknown = set(requested) - set(MLP_PROJECTIONS)
    if unknown:
        raise ValueError(f"Unknown MLP projection: {sorted(unknown)[0]}")
    projections = tuple(projection for projection in MLP_PROJECTIONS if projection in requested)
    return projections, layer_start, layer_end


def selected(
    module: str,
    variant: str,
    mlp_projections: list[str] | tuple[str, ...] | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> bool:
    _mlp_selection(variant, mlp_projections, layer_start, layer_end)
    name = canonical(module)
    return (
        variant in {"lm-head", "both"} and name == "lm_head"
    ) or (
        variant in {"attention", "both"}
        and re.fullmatch(r"layers\.\d+\.(?:self_attn|linear_attn)\.[^.]+", name) is not None
    ) or (
        variant == "mlp"
        and _selected_mlp(name, mlp_projections, layer_start, layer_end)
    )


def _selected_mlp(
    name: str,
    mlp_projections: list[str] | tuple[str, ...] | None,
    layer_start: int | None,
    layer_end: int | None,
) -> bool:
    match = re.fullmatch(r"layers\.(\d+)\.mlp\.([^.]+)", name)
    if match is None:
        return False
    projections, start, end = _mlp_selection("mlp", mlp_projections, layer_start, layer_end)
    layer, projection = int(match.group(1)), match.group(2)
    return projection in projections and (start is None or start <= layer < end)


def quantized_layers(tensors: dict[str, Tensor]) -> dict[str, dict[str, Any]]:
    layers = {}
    for name, weight in tensors.items():
        if not name.endswith(".weight") or weight.dtype not in ({"U8"} | FP8_WEIGHTS):
            continue
        module = name.removesuffix(".weight")
        scale = tensors.get(module + ".weight_scale")
        if scale is None:
            raise ValueError(f"Packed weight lacks weight_scale: {module}")
        if not any(module + "." + suffix in tensors
                   for suffix in ("input_scale", "input_global_scale")):
            raise ValueError(f"Packed weight lacks activation scale: {module}")
        if weight.dtype in FP8_WEIGHTS:
            layers[module] = {"quant_algo": "FP8"}
            continue
        if module + ".weight_scale_2" not in tensors:
            raise ValueError(f"NVFP4 weight lacks weight_scale_2: {module}")
        if (
            len(weight.shape) < 2 or len(scale.shape) != len(weight.shape)
            or weight.shape[:-1] != scale.shape[:-1]
            or weight.shape[-1] * 2 != scale.shape[-1] * 16
        ):
            raise ValueError(f"NVFP4 weight is not packed group 16: {module}")
        layers[module] = {"quant_algo": "NVFP4", "group_size": 16}
    if not layers:
        raise ValueError("No supported packed weights remain")
    return dict(sorted(layers.items()))


def _bf16_parts(
    name: str, expected_shape: tuple[int, ...], source: dict[str, Tensor]
) -> tuple[Tensor, ...]:
    key = canonical(name)
    if key in source:
        parts = (source[key],)
    else:
        module, suffix = key.rsplit(".", 1)
        base, projection = module.rsplit(".", 1)
        names = FUSED_PARTS.get(projection, ())
        keys = [f"{base}.{part}.{suffix}" for part in names]
        if not keys or any(part not in source for part in keys):
            raise ValueError(f"No unambiguous original BF16 tensor for: {name}")
        parts = tuple(source[part] for part in keys)
    if any(part.dtype != "BF16" for part in parts):
        raise ValueError(f"Original tensor must be BF16: {name}")
    shape = parts[0].shape
    if len(parts) > 1:
        if not shape or any(part.shape[1:] != shape[1:] for part in parts):
            raise ValueError(f"Fused source shape mismatch: {name}")
        shape = (sum(part.shape[0] for part in parts), *shape[1:])
    if shape != expected_shape:
        raise ValueError(f"BF16 logical shape mismatch: {name}")
    return parts


def prepare_plan(
    export: Path,
    bf16: Path,
    variant: str,
    *,
    mlp_projections: list[str] | tuple[str, ...] | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict[str, Any]:
    projections, layer_start, layer_end = _mlp_selection(
        variant, mlp_projections, layer_start, layer_end
    )
    original, index = read_checkpoint(export)
    source_tensors, _ = read_checkpoint(bf16)
    source: dict[str, Tensor] = {}
    for name, tensor in source_tensors.items():
        key = canonical(name)
        if key in source:
            raise ValueError(f"Ambiguous BF16 tensor alias: {key}")
        source[key] = tensor
    original_layers = quantized_layers(original)
    modules = sorted(
        module for module in original_layers
        if selected(module, variant, projections if variant == "mlp" else None,
                    layer_start, layer_end)
    )
    if not modules and variant != "control":
        if variant == "mlp" and (mlp_projections is not None or layer_start is not None):
            raise ValueError("MLP selector matched no packed language modules")
        raise ValueError(f"No packed language modules selected by variant: {variant}")
    if variant in {"lm-head", "both"}:
        heads = [module for module in modules if canonical(module) == "lm_head"]
        if len(heads) != 1:
            raise ValueError("Expected exactly one main language lm_head")
    replacements = {}
    removed = set()
    shapes = {}
    for module in modules:
        name = module + ".weight"
        packed = original[name]
        shape = (*packed.shape[:-1], packed.shape[-1] * (2 if packed.dtype == "U8" else 1))
        replacements[name] = _bf16_parts(name, shape, source)
        shapes[name] = shape
        if module + ".bias" in original:
            name = module + ".bias"
            shape = original[name].shape
            replacements[name] = _bf16_parts(name, shape, source)
            shapes[name] = shape
        for name in original:
            if name.startswith(module + "."):
                suffix = name.removeprefix(module + ".")
                if suffix in SCALE_SUFFIXES:
                    removed.add(name)
                elif suffix not in {"weight", "bias"}:
                    raise ValueError(f"Unknown selected-module state; refusing ghost keys: {name}")
    result = {name: tensor for name, tensor in original.items() if name not in removed}
    for name, parts in replacements.items():
        result[name] = Tensor(parts[0].path, "BF16", shapes[name], parts[0].offset,
                              sum(part.nbytes for part in parts))
    layers = quantized_layers(result)
    summary = {
        "variant": variant, "restored_modules": len(modules),
        "restored_tensors": len(replacements), "removed_scale_tensors": len(removed),
        "preserved_tensors": len(original) - len(removed) - len(replacements),
        "quantized_layer_counts": dict(Counter(entry["quant_algo"] for entry in layers.values())),
        "output_tensor_bytes": sum(tensor.nbytes for tensor in result.values()),
        "requires_quantization": "modelopt_mixed", "gpu_serving_verified": False,
        "selection": {
            "mlp_projections": list(projections) if variant == "mlp" else None,
            "layer_start": layer_start if variant == "mlp" else None,
            "layer_end": layer_end if variant == "mlp" else None,
        },
    }
    return {
        "original": original, "index": index, "result": result, "replacements": replacements,
        "removed": removed, "layers": layers, "summary": summary,
    }


def _stream(parts: tuple[Tensor, ...], output: Any = None) -> str:
    digest = hashlib.sha256()
    for tensor in parts:
        with tensor.path.open("rb") as handle:
            handle.seek(tensor.offset)
            remaining = tensor.nbytes
            while remaining:
                block = handle.read(min(CHUNK_SIZE, remaining))
                if not block:
                    raise ValueError("Tensor source changed or was truncated while copying")
                digest.update(block)
                if output is not None:
                    output.write(block)
                remaining -= len(block)
    return digest.hexdigest()


def _write_shard(
    path: Path, entries: dict[str, tuple[Tensor, tuple[Tensor, ...]]]
) -> dict[str, str]:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (tensor, _) in entries.items():
        header[name] = {
            "dtype": tensor.dtype, "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    hashes = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name, (_, parts) in entries.items():
            hashes[name] = _stream(parts, handle)
    return hashes


def _metadata(export: Path, stage: Path, layers: dict[str, Any]) -> None:
    config = _json(export / "config.json")
    hf = _json(export / "hf_quant_config.json")
    old = hf.get("quantization", hf)
    config_quant = config.get("quantization_config", {})
    if not isinstance(old, dict) or not isinstance(config_quant, dict):
        raise ValueError("Invalid ModelOpt quantization metadata")
    if config_quant.get("quant_method") in {"compressed-tensors", "compressed_tensors"}:
        raise ValueError("Compressed-tensors checkpoints are unsupported")
    if old.get("quant_algo") not in {"NVFP4", "MIXED_PRECISION"}:
        raise ValueError("Only serialized ModelOpt NVFP4/MIXED_PRECISION inputs are supported")
    quant = dict(old)
    quant.update({"quant_algo": "MIXED_PRECISION", "quantized_layers": layers})
    # Uniform-specific fields must not masquerade as the derivative precision.
    quant.pop("group_size", None)
    for field in ("ignore", "exclude_modules"):
        patterns = config_quant.get(field, old.get(field, []))
        if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
            raise ValueError(f"Invalid quantization {field}")
        quant[field] = [p for p in patterns if not any(
            fnmatchcase(module, p) or module.startswith(p + ".") for module in layers
        )]
    config_quant = {**config_quant, **quant, "quant_method": "modelopt"}
    config_quant.pop("group_size", None)
    config["quantization_config"] = config_quant
    _write_json(stage / "config.json", config)
    _write_json(stage / "hf_quant_config.json", {
        "producer": hf.get("producer", {"name": "modelopt", "version": "unknown"}),
        "quantization": quant,
    })


def write_ablation(
    export: Path,
    bf16: Path,
    output: Path,
    variant: str,
    *,
    mlp_projections: list[str] | tuple[str, ...] | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
) -> dict[str, Any]:
    export, bf16, output = export.resolve(), bf16.resolve(), output.resolve()
    for source in (export, bf16):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be disjoint from both input checkpoints")
    if output.exists():
        raise FileExistsError("Ablation output already exists; refusing to overwrite")
    plan = prepare_plan(
        export, bf16, variant, mlp_projections=mlp_projections,
        layer_start=layer_start, layer_end=layer_end,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        original, result = plan["original"], plan["result"]
        weight_map = plan["index"]["weight_map"]
        hashes = {}
        for filename in sorted(set(weight_map.values())):
            names = sorted(name for name, file in weight_map.items() if file == filename)
            target = stage / filename
            affected = any(
                name in plan["removed"] or name in plan["replacements"] for name in names
            )
            if not affected:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_shard(export, filename), target)
                hashes.update({name: _stream((original[name],)) for name in names})
                continue
            entries = {
                name: (result[name], plan["replacements"].get(name, (original[name],)))
                for name in names if name in result
            }
            if entries:
                hashes.update(_write_shard(target, entries))
        # Copy only root-level ancillary files, excluding unindexed weight/state blobs.
        for path in export.iterdir():
            if path.is_file() and path.suffix in {".json", ".jinja", ".txt", ".model", ".md"}:
                if path.name not in {INDEX_NAME, "config.json", "hf_quant_config.json",
                                     "precision_ablation.json"}:
                    shutil.copyfile(path, stage / path.name)
        index = dict(plan["index"])
        index["weight_map"] = {name: weight_map[name] for name in sorted(result)}
        index["metadata"] = {**index.get("metadata", {}), "total_size":
                             plan["summary"]["output_tensor_bytes"]}
        _write_json(stage / INDEX_NAME, index)
        _metadata(export, stage, plan["layers"])
        actual, _ = read_checkpoint(stage)
        if set(actual) != set(result):
            raise ValueError("Output index contains missing or ghost tensors")
        for name, tensor in actual.items():
            expected = result[name]
            if tensor.dtype != expected.dtype or tensor.shape != expected.shape:
                raise ValueError(f"Output tensor metadata mismatch: {name}")
            if _stream((tensor,)) != hashes[name]:
                raise ValueError(f"Output tensor payload hash mismatch: {name}")
        inventory = "".join(f"{name} {hashes[name]}\n" for name in sorted(hashes))
        original_hashes = {
            name: (hashes[name] if name not in plan["replacements"] and name not in plan["removed"]
                   else _stream((tensor,)))
            for name, tensor in original.items()
        }
        original_inventory = "".join(
            f"{name} {original_hashes[name]}\n" for name in sorted(original_hashes)
        )
        summary = {**plan["summary"], "payloads_verified": True,
                   "tensor_inventory_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
                   "input_tensor_inventory_sha256":
                   hashlib.sha256(original_inventory.encode()).hexdigest(),
                   "input_metadata_sha256": {
                       name: hashlib.sha256((export / name).read_bytes()).hexdigest()
                       for name in ("config.json", "hf_quant_config.json", INDEX_NAME)
                       if (export / name).is_file()
                   }}
        _write_json(stage / "precision_ablation.json", summary)
        if output.exists():
            raise FileExistsError("Output appeared during export; refusing to overwrite")
        os.rename(stage, output)
        return summary
    except BaseException:
        shutil.rmtree(stage)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True,
                        help="Existing scored ModelOpt export")
    parser.add_argument("--bf16", type=Path, required=True,
                        help="Original local BF16 HF checkpoint")
    parser.add_argument("--variant", choices=("control", "lm-head", "attention", "mlp", "both"),
                        required=True)
    parser.add_argument("--mlp-projection", dest="mlp_projections", action="append",
                        choices=MLP_PROJECTIONS,
                        help="Diagnostic MLP projection to restore; repeat to select multiple")
    parser.add_argument("--layer-start", type=int, help="Diagnostic inclusive MLP layer start")
    parser.add_argument("--layer-end", type=int, help="Diagnostic exclusive MLP layer end")
    parser.add_argument("--output", type=Path, help="New independent checkpoint directory")
    parser.add_argument("--write", action="store_true", help="Create and verify the derived export")
    args = parser.parse_args(argv)
    if args.write and args.output is None:
        parser.error("--write requires --output")
    try:
        if args.write:
            summary = write_ablation(
                args.export, args.bf16, args.output, args.variant,
                mlp_projections=args.mlp_projections,
                layer_start=args.layer_start, layer_end=args.layer_end,
            )
        else:
            summary = prepare_plan(
                args.export, args.bf16, args.variant,
                mlp_projections=args.mlp_projections,
                layer_start=args.layer_start, layer_end=args.layer_end,
            )["summary"]
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Precision ablation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
