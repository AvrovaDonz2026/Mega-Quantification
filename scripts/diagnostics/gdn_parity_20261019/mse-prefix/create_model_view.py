#!/usr/bin/env python3
"""Build a single-scheme historical MSE metadata view.

The view hard-links model weights to the scored export and rewrites only the
two JSON metadata files through the pinned SGLang export helper.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path


SCHEME = "w4a4"
EXPECTED_SCHEME = "nvfp4_w4a4"
JSON_FILES = ("config.json", "hf_quant_config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def files_manifest(root: Path) -> list[dict]:
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Symlink is forbidden in model view: {path}")
        if path.is_file():
            manifest.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not manifest:
        raise RuntimeError("Model view contains no files")
    return manifest


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_mse_export_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rewrite = getattr(helper, "rewrite_sglang_nvfp4_export", None)
    if not callable(rewrite):
        raise RuntimeError("Helper lacks rewrite_sglang_nvfp4_export")
    return rewrite


def failed_marker(output: Path, error: BaseException) -> None:
    marker = output.with_name(output.name + ".failed")
    try:
        marker.write_text(
            json.dumps(
                {
                    "state": "failed",
                    "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                    "output": str(output),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def build(args: argparse.Namespace) -> dict:
    original = args.original_model.resolve()
    output = args.output.resolve()
    helper_path = args.helper.resolve()
    gates_path = args.input_gates.resolve()
    if output.exists() or output.with_name(output.name + ".failed").exists():
        raise RuntimeError(f"Refusing to overwrite existing destination: {output}")
    if not original.is_dir() or original.is_symlink():
        raise RuntimeError("Original model must be a real directory")
    if not helper_path.is_file() or helper_path.is_symlink():
        raise RuntimeError("Helper must be a real file")
    if not re.fullmatch(r"[0-9a-f]{64}", args.helper_sha256):
        raise RuntimeError("helper-sha256 must be 64 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{40}", args.helper_revision):
        raise RuntimeError("helper-revision must be a 40-hex revision")
    if sha256(helper_path) != args.helper_sha256:
        raise RuntimeError("Helper SHA-256 does not match --helper-sha256")
    if any(path.is_symlink() for path in original.rglob("*")):
        raise RuntimeError("Original model contains symlinks")
    provenance_path = original / "provenance.json"
    provenance = read_object(provenance_path)
    if provenance.get("algorithm") != "mse" or provenance.get("backend") != "modelopt":
        raise RuntimeError("Original provenance is not the MSE ModelOpt export")
    if provenance.get("scheme") != EXPECTED_SCHEME:
        raise RuntimeError("Original provenance scheme is not nvfp4_w4a4")
    original_revision = provenance.get("git_sha")
    if not re.fullmatch(r"[0-9a-f]{40}", str(original_revision)):
        raise RuntimeError("Original provenance git_sha is missing or invalid")
    if provenance.get("calibration", {}).get("with_images") is not True:
        raise RuntimeError("Original provenance must record image calibration")
    original_hashes = {name: sha256(original / name) for name in JSON_FILES}

    output.parent.mkdir(parents=True, exist_ok=True)
    view = output / "models" / SCHEME
    view.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original, view, copy_function=os.link)
    for name in JSON_FILES:
        source = original / name
        view_file = view / name
        if not view_file.samefile(source):
            raise RuntimeError(f"Expected hard-link before metadata rewrite: {name}")
        contents = view_file.read_bytes()
        view_file.unlink()
        view_file.write_bytes(contents)
        if view_file.samefile(source):
            raise RuntimeError(f"Metadata inode was not separated: {name}")

    rewrite = load_helper(helper_path)
    layers = rewrite(view)
    if not isinstance(layers, dict):
        raise RuntimeError("Export helper returned a non-dictionary layer map")
    visual = sum(name.startswith("model.visual.") for name in layers)
    mtp = sum(name.startswith("mtp.") for name in layers)
    if (visual, mtp) != (110, 8):
        raise RuntimeError(f"Expected 110 vision and 8 MTP layers, got {(visual, mtp)}")
    if any(sha256(original / name) != value for name, value in original_hashes.items()):
        raise RuntimeError("Original metadata changed during view creation")
    manifest = files_manifest(view)

    inputs = {
        "schema_version": 1,
        "models": {
            SCHEME: {
                "path": str(view),
                "generation": "historical-metadata-view",
                "scored_original_model": str(original),
                "metadata_patch_revision": args.helper_revision,
                "mse_algorithm": "mse",
                "mse_backend": "modelopt",
            }
        },
    }
    inputs_path = output / "model-inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    identity = {
        "schema_version": 1,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_inputs_sha256": sha256(inputs_path),
        "models": {
            SCHEME: {
                **inputs["models"][SCHEME],
                "quantization_provenance_sha256": sha256(view / "provenance.json"),
                "scored_original_config_sha256": original_hashes["config.json"],
                "scored_original_hf_quant_config_sha256": original_hashes["hf_quant_config.json"],
                "files": manifest,
            }
        },
    }
    identity_path = output / "model-identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("mse_input_gates", gates_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load input gates: {gates_path}")
    gates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gates)
    gate = gates.verify_model(view, SCHEME, identity_path)
    report = {
        "state": "created",
        "scheme": SCHEME,
        "original_model": str(original),
        "original_git_revision": original_revision,
        "original_provenance_sha256": sha256(provenance_path),
        "original_metadata_sha256": original_hashes,
        "view_metadata_sha256": {name: sha256(view / name) for name in JSON_FILES},
        "metadata_patch_revision": args.helper_revision,
        "helper_sha256": args.helper_sha256,
        "visual_layers": visual,
        "mtp_layers": mtp,
        "total_explicit_layers": len(layers),
        "weights_same_inodes": all(
            (view / entry["path"]).samefile(original / entry["path"])
            for entry in manifest
            if entry["path"] not in JSON_FILES
        ),
        "file_count": len(manifest),
        "input_gate": gate,
    }
    (output / "metadata-view-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--helper-revision", required=True)
    parser.add_argument("--input-gates", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failed_marker(args.output, error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
