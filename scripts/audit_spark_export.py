#!/usr/bin/env python3
"""Audit Spark exports without loading weight payloads (stdlib only).

Reads safetensors headers and streams scale payloads in bounded chunks.
Exit 0 means structural/calibration checks passed, not inference correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}
FP8_WEIGHTS = {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
FLOAT_WEIGHTS = {"BF16", "F16", "F32", "F64"}


def canonical(name):
    """Known HF/ModelOpt/SGLang prefix and vision projection aliases."""
    if "mtp." in name:
        name = name[name.index("mtp.") :]
    elif "visual." in name:
        name = "vision." + name.split("visual.", 1)[1]
    else:
        for prefix in ("model.language_model.", "language_model.", "model."):
            if name.startswith(prefix):
                name = "lm." + name[len(prefix) :]
                break
    if name.startswith("lm.lm_head."):
        name = name.removeprefix("lm.")
    name = name.replace(".attn.qkv.", ".attn.qkv_proj.")
    if name.endswith(".attn.qkv"):
        name = name + "_proj"
    return name


def branch(name):
    name = canonical(name)
    return (
        "vision" if name.startswith("vision.") else "mtp" if name.startswith("mtp.") else "language"
    )


def projection(name, info):
    return (
        name.endswith(".weight")
        and len(info["shape"]) == 2
        and not any(part in name for part in ("embed", "conv1d", "in_proj_a", "in_proj_b"))
    )


def equivalents(name, available):
    key = canonical(name)
    if key in available:
        return [key]
    # A source fused projection can be serialized as three/two projections.
    for fused, splits in (
        ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
        ("gate_up_proj", ("gate_proj", "up_proj")),
    ):
        marker = "." + fused + "."
        if marker in key:
            candidates = [key.replace(marker, "." + split + ".") for split in splits]
            if all(candidate in available for candidate in candidates):
                return candidates
        for split in splits:
            marker = "." + split + "."
            candidate = key.replace(marker, "." + fused + ".")
            if marker in key and candidate in available:
                return [candidate]
    return []


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def read_json(self, path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            return value
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path.name}: {exc}")
            return {}

    def checkpoint(self, root, label):
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            index = self.read_json(index_path)
            weight_map = index.get("weight_map", {})
        elif (root / "model.safetensors").exists():
            weight_map = None
        else:
            self.errors.append(f"{label}: no HF safetensors index or single shard")
            return {}, {}
        if weight_map is not None and not isinstance(weight_map, dict):
            self.errors.append(f"{label}: weight_map is not an object")
            return {}, {}
        files = (
            sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
        )
        tensors = {}
        for filename in files:
            path = root / filename
            try:
                if Path(filename).is_absolute() or ".." in Path(filename).parts:
                    raise ValueError("unsafe shard filename in index")
                size = path.stat().st_size
                with path.open("rb") as handle:
                    prefix = handle.read(8)
                    if len(prefix) != 8:
                        raise ValueError("missing safetensors header length")
                    header_size = int.from_bytes(prefix, "little")
                    if not 2 <= header_size <= min(100_000_000, size - 8):
                        raise ValueError(f"invalid header size {header_size}")
                    header = json.loads(handle.read(header_size))
                spans = []
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    shape, dtype = info["shape"], info["dtype"]
                    start, end = info["data_offsets"]
                    if not isinstance(shape, list) or not all(
                        isinstance(d, int) and d >= 0 for d in shape
                    ):
                        raise ValueError(f"invalid shape: {name}")
                    if not (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start <= end <= size - 8 - header_size
                    ):
                        raise ValueError(f"invalid data offsets: {name}")
                    if dtype not in DTYPE_BYTES:
                        raise ValueError(f"unsupported tensor dtype {dtype}: {name}")
                    expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
                    if expected_bytes != end - start:
                        raise ValueError(f"tensor size mismatch: {name}")
                    spans.append((start, end, name))
                    if name in tensors:
                        raise ValueError(f"duplicate tensor across shards: {name}")
                    tensors[name] = dict(
                        info, filename=filename, path=str(path), offset=8 + header_size + start
                    )
                spans.sort()
                cursor = 0
                for start, end, name in spans:
                    if start != cursor:
                        raise ValueError(f"noncontiguous/overlapping data: {name}")
                    cursor = end
                if cursor != size - 8 - header_size:
                    raise ValueError("trailing data outside tensors")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.errors.append(f"{label} shard {filename}: {exc}")
        if weight_map is None:
            weight_map = {name: info["filename"] for name, info in tensors.items()}
        self.check(bool(weight_map), f"{label}: empty weight map")
        for name, filename in weight_map.items():
            self.check(
                name in tensors and tensors[name]["filename"] == filename,
                f"{label}: missing/misdirected index reference {name} -> {filename}",
            )
        for name in tensors:
            self.check(name in weight_map, f"{label}: unindexed tensor {name}")
        return weight_map, tensors

    def finite_scales(self, tensors):
        checked, nbytes = 0, 0
        for name, info in tensors.items():
            if not any(marker in name for marker in ("_scale", ".amax")):
                continue
            dtype = info["dtype"]
            size = info["data_offsets"][1] - info["data_offsets"][0]
            if not self.check(size > 0, f"empty scale: {name}"):
                continue
            finite = True
            with Path(info["path"]).open("rb") as handle:
                handle.seek(info["offset"])
                remaining = size
                while remaining:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        finite = False
                        break
                    remaining -= len(chunk)
                    nbytes += len(chunk)
                    if dtype in ("F16", "F32", "F64"):
                        fmt = {"F16": "<e", "F32": "<f", "F64": "<d"}[dtype]
                        valid = all(
                            math.isfinite(value[0]) for value in struct.iter_unpack(fmt, chunk)
                        )
                    elif dtype == "BF16":
                        valid = all(
                            (value[0] & 0x7F80) != 0x7F80
                            for value in struct.iter_unpack("<H", chunk)
                        )
                    elif dtype in {"F8_E4M3", "F8_E4M3FN"}:
                        valid = all((value & 0x7F) != 0x7F for value in chunk)
                    elif dtype in {"F8_E4M3FNUZ", "F8_E5M2FNUZ"}:
                        valid = 0x80 not in chunk
                    elif dtype == "F8_E5M2":
                        valid = all((value & 0x7C) != 0x7C for value in chunk)
                    elif dtype == "F8_E8M0":
                        valid = 0xFF not in chunk
                    else:
                        self.errors.append(f"unsupported scale dtype {dtype}: {name}")
                        valid = False
                    finite = finite and valid
            self.check(finite, f"nonfinite/invalid scale tensor: {name}")
            checked += 1
        self.check(checked > 0, "no scale payloads found")
        return {"tensors_checked": checked, "bytes_streamed": nbytes}


def run(args):
    audit = Audit()
    root, source = args.export_dir.resolve(), args.source_dir.resolve()
    export_map, tensors = audit.checkpoint(root, "export")
    _, source_tensors = audit.checkpoint(source, "source")
    available = {}
    for name, info in tensors.items():
        key = canonical(name)
        audit.check(key not in available, f"ambiguous canonical tensor alias: {name}")
        available[key] = (name, info)
    config = audit.read_json(root / "config.json")
    hf = audit.read_json(root / "hf_quant_config.json")
    quant = hf.get("quantization", hf)
    config_quant = config.get("quantization_config", {})
    expected_algo = "NVFP4" if args.scheme == "nvfp4_w4a4" else "MIXED_PRECISION"
    audit.check(quant.get("quant_algo") == expected_algo, f"hf quant_algo must be {expected_algo}")
    if config_quant:
        audit.check(
            config_quant.get("quant_algo") == expected_algo,
            "config quant_algo disagrees with scheme",
        )
    audit.check(
        isinstance(config_quant, dict) and bool(config_quant), "missing config quantization_config"
    )
    for field in ("quantized_layers", "kv_cache_quant_algo"):
        audit.check(
            config_quant.get(field) == quant.get(field),
            f"config and hf quantization disagree on {field}",
        )
    audit.check(config.get("model_type") == "qwen3_5", "export is not a qwen3_5 model")
    audit.check(isinstance(config.get("vision_config"), dict), "missing vision_config")
    audit.check(config.get("language_model_only") is not True, "vision disabled in config")
    audit.check(
        quant.get("kv_cache_quant_algo") in (None, "NONE", "None"),
        "Spark quantized vision recipe expects no exported KV-cache quantization",
    )

    layers = quant.get("quantized_layers", {})
    canonical_layers = {canonical(name): value for name, value in layers.items()}
    if args.scheme == "nvfp4_w4a8":
        audit.check(bool(layers), "mixed export has no quantized_layers")
    source_groups = Counter()
    all_source_groups = Counter()
    projected = defaultdict(set)
    source_to_export = []
    # Group fused/split aliases before checking logical element counts.
    source_matches = defaultdict(set)
    for name, info in source_tensors.items():
        group = branch(name)
        all_source_groups[group] += 1
        if group in {"vision", "mtp"}:
            source_groups[group] += 1
        matches = equivalents(name, available)
        if not audit.check(bool(matches), f"source {group} tensor absent from export: {name}"):
            continue
        source_matches[tuple(matches)].add(name)
        if group in {"vision", "mtp"}:
            source_to_export.append(
                {"source": name, "export": [available[key][0] for key in matches]}
            )
        if group in {"vision", "mtp"} and projection(name, info):
            projected[group].update(matches)
    audit.check(all_source_groups["language"] > 0, "source has no language tensors")
    for group in ("vision", "mtp"):
        audit.check(source_groups[group] > 0, f"source has no {group} tensors")
        audit.check(bool(projected[group]), f"no {group} projections checked")
    for matches, source_names in source_matches.items():
        expected = sum(math.prod(source_tensors[name]["shape"]) for name in source_names)
        actual = sum(
            math.prod(available[key][1]["shape"]) * (2 if available[key][1]["dtype"] == "U8" else 1)
            for key in matches
        )
        audit.check(
            expected == actual,
            f"logical tensor element count mismatch: {sorted(source_names)} -> "
            f"{list(matches)} ({expected} vs {actual})",
        )

    counts = {group: Counter() for group in ("language", "vision", "mtp")}
    for name, info in tensors.items():
        if not name.endswith(".weight") or info["dtype"] not in ({"U8"} | FP8_WEIGHTS):
            continue
        module = name.removesuffix(".weight")
        group = branch(name)
        algo = "NVFP4" if info["dtype"] == "U8" else "FP8"
        counts[group][algo] += 1
        for field in ("ignore", "exclude_modules"):
            for pattern in config_quant.get(field, []):
                audit.check(
                    not fnmatchcase(module, pattern),
                    f"quantized layer excluded in config {field}: {module} matches {pattern}",
                )
        if args.scheme == "nvfp4_w4a4":
            audit.check(algo == "NVFP4", f"W4A4 projection is {algo}: {module}")
        else:
            expected = (
                "FP8" if any(part in module for part in ("self_attn", "linear_attn")) else "NVFP4"
            )
            audit.check(
                algo == expected, f"mixed precision differs from recipe ({expected}): {module}"
            )
            entry = canonical_layers.get(canonical(module), {})
            audit.check(
                entry.get("quant_algo") == algo, f"missing/wrong mixed layer metadata: {module}"
            )
            for pattern in quant.get("exclude_modules", []):
                audit.check(
                    not fnmatchcase(module, pattern),
                    f"quantized layer excluded in metadata: {module} matches {pattern}",
                )
        audit.check(module + ".weight_scale" in tensors, f"weight scale missing: {module}")
        audit.check(
            any(module + suffix in tensors for suffix in (".input_scale", ".input_global_scale")),
            f"activation scale missing: {module}",
        )
        if algo == "NVFP4":
            audit.check(
                module + ".weight_scale_2" in tensors,
                f"NVFP4 global weight scale missing: {module}",
            )
            scale = tensors.get(module + ".weight_scale", {})
            shape = scale.get("shape", [])
            audit.check(
                bool(shape)
                and len(info["shape"]) == len(shape)
                and info["shape"][:-1] == shape[:-1]
                and info["shape"][-1] * 2 == shape[-1] * 16,
                f"NVFP4 block size/scale shape is not group 16: {module}",
            )
    for group, keys in projected.items():
        for key in keys:
            audit.check(
                available[key][1]["dtype"] in ({"U8"} | FP8_WEIGHTS),
                f"source {group} projection remained unquantized: {available[key][0]}",
            )
    for group in counts:
        audit.check(sum(counts[group].values()) > 0, f"no quantized {group} projections")
    for module in layers:
        key = canonical(module + ".weight")
        audit.check(key in available, f"quantized_layers refers to absent weight: {module}")
        if key in available:
            audit.check(
                available[key][1]["dtype"] not in FLOAT_WEIGHTS,
                f"quantized_layers incorrectly labels BF16/FP weight: {module}",
            )

    coverage = audit.read_json(root / "calibration_coverage.json")
    coverage_summary = {}
    for group in ("vision", "mtp"):
        entries = coverage.get(group, {})
        audit.check(
            isinstance(entries, dict) and bool(entries), f"empty {group} calibration coverage"
        )
        if isinstance(entries, dict) and entries:
            valid = all(type(value) is int and value > 0 for value in entries.values())
            audit.check(valid, f"unvisited/invalid {group} input quantizers")
            coverage_summary[group] = {
                "quantizers": len(entries),
                "min_calls": min(entries.values()),
                "max_calls": max(entries.values()),
            }
            covered = {canonical(name.removesuffix(".input_quantizer")) for name in entries}
            for key in projected[group]:
                module = key.removesuffix(".weight")
                aliases = equivalents(
                    module + ".weight", {key + ".weight": None for key in covered}
                )
                audit.check(
                    bool(aliases),
                    f"quantized {group} projection lacks calibration evidence: {module}",
                )
    vision_note = audit.read_json(root / "vision_export.json")
    mtp_note = audit.read_json(root / "mtp_export.json")
    audit.check(
        vision_note.get("method") == "modelopt-export" and vision_note.get("restored_tensors") == 0,
        "vision export was restored or not quantized",
    )
    audit.check(
        mtp_note.get("method") == "modelopt-export" and mtp_note.get("mtp_quantized") is True,
        "MTP export missing or not quantized",
    )
    for filename in ("tokenizer_config.json", "preprocessor_config.json"):
        audit.check((root / filename).is_file(), f"missing processor/tokenizer file: {filename}")
    provenance = audit.read_json(root / "provenance.json")
    audit.check(
        provenance.get("git_sha") == args.expected_revision,
        f"provenance revision mismatch: expected {args.expected_revision}, "
        f"found {provenance.get('git_sha')!r}",
    )
    audit.check(provenance.get("scheme") == args.scheme, "provenance scheme mismatch")
    calibration = provenance.get("calibration", {})
    audit.check(calibration.get("with_images") is True, "provenance does not enable images")
    scales = audit.finite_scales(tensors)
    return {
        "passed": not audit.errors,
        "scheme": args.scheme,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "export_tensor_count": len(tensors),
        "export_shard_count": len(set(export_map.values())),
        "source_auxiliary_tensor_counts": dict(source_groups),
        "source_tensor_counts": dict(all_source_groups),
        "quantized_projection_counts": {key: dict(value) for key, value in counts.items()},
        "calibration_coverage": coverage_summary,
        "scale_payload_scan": scales,
        "source_auxiliary_mapping": source_to_export,
        "git_sha": provenance.get("git_sha"),
        "expected_revision": args.expected_revision,
        "calibration_provenance": calibration,
        "verified": [
            "provenance git_sha exactly matches the supplied expected source revision",
            "safetensors header layout, offsets, sizes and HF index references",
            "all source language/vision/MTP tensor presence and logical element counts "
            "including known aliases",
            "vision/MTP projection quantization and calibration execution evidence",
            "config/hf quantization layer and KV metadata agreement, "
            "with no quantized module excluded by config",
            "scheme metadata, packed NVFP4 group 16 shapes, weight and activation scale presence",
            "scale payloads contain no NaN or infinity",
        ],
        "not_verified": [
            "weight payload values/checksums or numerical quality",
            "actual calibration sample content and exact sample count "
            "(coverage proves execution only)",
            "SGLang loading, text/image generation, speculative MTP acceptance",
            "GPU OOM recovery behavior",
        ],
    }


def revision(value):
    """Require the complete expected source commit, rather than a prefix."""
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("expected a full lowercase 40-character git SHA")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("scheme", choices=("nvfp4_w4a4", "nvfp4_w4a8"))
    parser.add_argument("--expected-revision", required=True, type=revision)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "passed": False,
            "errors": [f"audit could not complete: {type(exc).__name__}: {exc}"],
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"source_auxiliary_mapping"}
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
