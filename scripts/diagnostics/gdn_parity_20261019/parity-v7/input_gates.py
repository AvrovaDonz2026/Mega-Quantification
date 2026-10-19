#!/usr/bin/env python3
"""Stdlib source/checkpoint identity and quantized auxiliary metadata gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

SGLANG_COMMIT = "4a667614d810f7b5570be23eb75956c603a85c3a"
ARCHIVE_SHA = "ac27ebc4ada7c0e895ad3366501bd8213ca7d0c92abed98f77057ac9ab24ef9e"
FRESH_MEGA_COMMIT = "11407cdc7bdd79e28e57c82a43f0f519260f0ff2"
_SOURCE_CACHE = {}
_MODEL_CACHE = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object")
    return value


def safe_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe relative path")
    result = root.joinpath(*relative.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Path escapes input root")
    return result


def fingerprint(root):
    return tuple(
        (
            str(p.relative_to(root)),
            p.lstat().st_size,
            p.lstat().st_mtime_ns,
            p.lstat().st_ctime_ns,
            p.readlink().as_posix() if p.is_symlink() else None,
        )
        for p in sorted(root.rglob("*"))
        if p.is_file() or p.is_symlink()
    )


def verify_source(source, archive, manifest):
    if archive is None or manifest is None:
        raise RuntimeError("Exact source archive and manifest required")
    if sha256(archive) != ARCHIVE_SHA:
        raise RuntimeError("Source archive hash differs from pinned 4a667614 archive")
    info = read_json(manifest)
    item = next((x for x in info.get("revisions", []) if x.get("name") == "vision-mtp"), None)
    if not item or item.get("commit") != SGLANG_COMMIT or item.get("archive_sha256") != ARCHIVE_SHA:
        raise RuntimeError("Source manifest does not identify pinned vision-MTP HEAD")
    key = (str(source.resolve()), fingerprint(source))
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    source_files = set()
    archive_files = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != item["archive_prefix"] or ".." in parts:
                raise RuntimeError("Unsafe or unexpected source archive path")
            relative = "/".join(parts[1:])
            if not relative or member.isdir():
                continue
            path = safe_file(source, relative)
            archive_files.add(relative)
            if member.issym():
                if not path.is_symlink() or path.readlink().as_posix() != member.linkname:
                    raise RuntimeError(f"Source symlink mismatch: {relative}")
            elif member.isfile():
                incoming = handle.extractfile(member)
                h = hashlib.sha256()
                for block in iter(lambda stream=incoming: stream.read(1024 * 1024), b""):
                    h.update(block)
                if not path.is_file() or path.is_symlink() or sha256(path) != h.hexdigest():
                    raise RuntimeError(f"Source file mismatch: {relative}")
            else:
                raise RuntimeError("Unsupported source archive member type")
    for path in source.rglob("*"):
        if path.is_file() or path.is_symlink():
            source_files.add(str(path.relative_to(source)))
    if source_files != archive_files:
        raise RuntimeError("Extracted source has missing or additional files")
    proof = {
        "commit": SGLANG_COMMIT,
        "archive_sha256": ARCHIVE_SHA,
        "manifest_sha256": sha256(manifest),
        "exact_archive_tree": True,
        "verified_files": len(archive_files),
    }
    _SOURCE_CACHE[key] = proof
    return proof


def tensor_headers(root):
    index = read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Missing HF safetensors index")
    tensors = {}
    for name in sorted(set(weight_map.values())):
        path = safe_file(root, name)
        with path.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
                raise RuntimeError("Invalid safetensors header size")
            header = json.loads(handle.read(size))
        for tensor, data in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in tensors or weight_map.get(tensor) != name:
                raise RuntimeError("Tensor index or duplicate header mismatch")
            start, end = data["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - size - 8:
                raise RuntimeError("Invalid tensor offsets")
            tensors[tensor] = data
    if set(tensors) != set(weight_map):
        raise RuntimeError("Index and serialized tensor sets differ")
    return tensors


def quantized_metadata_gate(root, scheme):
    config = read_json(root / "config.json")
    hf = read_json(root / "hf_quant_config.json")
    quant = hf.get("quantization", hf)
    config_quant = config.get("quantization_config")
    expected_algo = {"w4a4": "NVFP4", "w4a8": "MIXED_PRECISION"}[scheme]
    if (
        not isinstance(config_quant, dict)
        or quant.get("quant_algo") != expected_algo
        or config_quant.get("quant_algo") != expected_algo
    ):
        raise RuntimeError("Scheme quantization metadata mismatch")
    if config.get("language_model_only") is True or not isinstance(
        config.get("vision_config"), dict
    ):
        raise RuntimeError("Vision is disabled or absent")
    layers = quant.get("quantized_layers")
    if not isinstance(layers, dict) or not layers or layers != config_quant.get("quantized_layers"):
        raise RuntimeError("Config and HF quantized_layers must be equal and nonempty")
    tensors = tensor_headers(root)
    expected = {"visual": [], "mtp": []}
    counts = {"visual": {}, "mtp": {}}
    for name, entry in layers.items():
        branch = (
            "visual"
            if name.startswith(("visual.", "model.visual."))
            else "mtp"
            if name.startswith("mtp.")
            else None
        )
        if branch is None:
            continue
        weight = tensors.get(name + ".weight")
        algo = entry.get("quant_algo")
        if weight is None:
            raise RuntimeError(f"Auxiliary metadata references absent weight: {name}")
        dtype = weight.get("dtype")
        wanted = (
            "NVFP4"
            if scheme == "w4a4"
            or branch == "visual"
            or not any(x in name for x in ("self_attn", "linear_attn"))
            else "FP8"
        )
        if algo != wanted:
            raise RuntimeError(f"Auxiliary layer precision differs from scheme: {name}")
        scales = [name + ".weight_scale", name + ".input_scale"]
        if algo == "NVFP4":
            scales.append(name + ".weight_scale_2")
            if dtype != "U8" or entry.get("group_size") != 16:
                raise RuntimeError(f"Auxiliary NVFP4 dtype/group mismatch: {name}")
            shape = weight.get("shape", [])
            scale_shape = tensors.get(name + ".weight_scale", {}).get("shape", [])
            if not shape or not scale_shape or shape[-1] * 2 != scale_shape[-1] * 16:
                raise RuntimeError(f"Auxiliary NVFP4 packed group shape mismatch: {name}")
        elif dtype not in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}:
            raise RuntimeError(f"Auxiliary FP8 dtype mismatch: {name}")
        if any(scale not in tensors for scale in scales):
            raise RuntimeError(f"Auxiliary scale missing: {name}")
        if any(fnmatchcase(name, pattern) for pattern in config_quant.get("exclude_modules", [])):
            raise RuntimeError(f"Auxiliary layer excluded by metadata: {name}")
        expected[branch].append(name)
        counts[branch][algo] = counts[branch].get(algo, 0) + 1
    for name, weight in tensors.items():
        module = name.removesuffix(".weight")
        auxiliary = module.startswith(("visual.", "model.visual.", "mtp."))
        if (
            name.endswith(".weight")
            and auxiliary
            and weight.get("dtype") in {"U8", "F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
            and module not in layers
        ):
            raise RuntimeError(
                f"Serialized quantized auxiliary projection lacks metadata: {module}"
            )
    if len(expected["visual"]) != 110 or len(expected["mtp"]) != 8:
        raise RuntimeError(
            "Expected 110 vision and 8 logical MTP projections for full 27B checkpoint"
        )
    return {
        "passed": True,
        "quant_algo": expected_algo,
        "projection_counts": counts,
        "tensor_count": len(tensors),
        "metadata_layer_count": len(layers),
        "auxiliary_projections": expected,
        "config_sha256": sha256(root / "config.json"),
        "hf_quant_config_sha256": sha256(root / "hf_quant_config.json"),
        "scope": (
            "Metadata and serialized tensor headers; "
            "scale numerical values and serving remain separate gates"
        ),
    }


def verify_model(root, scheme, provenance_path):
    record = read_json(provenance_path).get("models", {}).get(scheme)
    if not isinstance(record, dict):
        raise RuntimeError("Model identity record missing selected scheme")
    generation = record.get("generation")
    if generation not in {"historical", "historical-metadata-view", "robo-fresh-20261013"}:
        raise RuntimeError("Historical versus fresh generation must be explicit")
    actual_provenance = read_json(root / "provenance.json")
    if actual_provenance.get("scheme") != "nvfp4_" + scheme:
        raise RuntimeError("On-disk model provenance scheme differs")
    if not re.fullmatch("[0-9a-f]{40}", str(actual_provenance.get("git_sha", ""))):
        raise RuntimeError("On-disk quantization revision is missing")
    if actual_provenance.get("calibration", {}).get("with_images") is not True:
        raise RuntimeError("On-disk calibration does not record images")
    if sha256(root / "provenance.json") != record.get("quantization_provenance_sha256"):
        raise RuntimeError("On-disk quantization provenance changed")
    if generation == "robo-fresh-20261013":
        if actual_provenance.get("git_sha") != FRESH_MEGA_COMMIT:
            raise RuntimeError(
                "Fresh Robo classification requires actual 10/13 quantization revision"
            )
        audit_path = Path(record.get("audit_path", ""))
        audit = read_json(audit_path)
        if (
            sha256(audit_path) != record.get("audit_sha256")
            or audit.get("passed") is not True
            or audit.get("expected_revision") != FRESH_MEGA_COMMIT
            or audit.get("scheme") != "nvfp4_" + scheme
        ):
            raise RuntimeError("Fresh checkpoint requires passing exact fresh quantization audit")
    if generation == "historical-metadata-view":
        for key in ("scored_original_config_sha256", "scored_original_hf_quant_config_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", str(record.get(key, ""))):
                raise RuntimeError(
                    "Historical metadata view must identify original scored JSON hashes"
                )
        if not re.fullmatch("[0-9a-f]{40}", str(record.get("metadata_patch_revision", ""))):
            raise RuntimeError(
                "Historical metadata view must identify exact metadata patch revision"
            )
        original = Path(record.get("scored_original_model", "")).resolve()
        if original == root.resolve() or not original.is_dir():
            raise RuntimeError(
                "Historical metadata view requires distinct original scored directory"
            )
        for filename, key in [
            ("config.json", "scored_original_config_sha256"),
            ("hf_quant_config.json", "scored_original_hf_quant_config_sha256"),
        ]:
            if sha256(original / filename) != record[key]:
                raise RuntimeError("Scored original metadata changed")
        current_names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        original_names = {str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()}
        if current_names != original_names:
            raise RuntimeError("Historical view inventory differs from original checkpoint")
        for name in current_names - {"config.json", "hf_quant_config.json"}:
            current_file = safe_file(root, name)
            original_file = safe_file(original, name)
            if not current_file.samefile(original_file) and sha256(current_file) != sha256(
                original_file
            ):
                raise RuntimeError(
                    "Historical view changed bytes outside its two JSON metadata files"
                )
    key = (str(root.resolve()), sha256(provenance_path), fingerprint(root))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Model manifest must contain complete file hashes")
    names = [entry.get("path") for entry in files]
    if len(set(names)) != len(names):
        raise RuntimeError("Model manifest contains duplicate paths")
    if set(names) != {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}:
        raise RuntimeError("Model file inventory changed")
    for entry in files:
        path = safe_file(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Model file content identity mismatch: {entry['path']}")
    gate = quantized_metadata_gate(root, scheme)
    proof = {
        "passed": True,
        "generation": generation,
        "model_manifest_sha256": sha256(provenance_path),
        "quantization_revision": actual_provenance["git_sha"],
        "quantization_provenance_sha256": record["quantization_provenance_sha256"],
        "metadata_gate": gate,
        "file_count": len(files),
        "gpqa_claim": (
            "No new GPQA score; "
            "historical artifact status does not transfer accuracy to current source"
        ),
    }
    _MODEL_CACHE[key] = proof
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check-source", "check-model"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scheme", choices=["w4a4", "w4a8"])
    parser.add_argument("--model-provenance", type=Path)
    args = parser.parse_args()
    if args.action == "check-source":
        result = verify_source(args.source.resolve(), args.archive, args.source_manifest)
    else:
        result = verify_model(args.model.resolve(), args.scheme, args.model_provenance)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
