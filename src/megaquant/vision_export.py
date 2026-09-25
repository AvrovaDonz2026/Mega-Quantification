"""Restore unquantized vision weights after language-only Qwen3.5 PTQ."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from megaquant.exceptions import BackendError
from megaquant.mtp_export import _open_source_shard, _source_index, update_export_index
from megaquant.sglang_export import load_weight_map

VISION_SHARD_NAME = "vision.safetensors"
_PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def _restore_processor_files(root: Path, source_root: Path | str) -> None:
    for filename in _PROCESSOR_FILES:
        dest = root / filename
        if dest.is_file():
            continue
        if isinstance(source_root, Path):
            src = source_root / filename
            if not src.is_file():
                continue
        else:
            from huggingface_hub.errors import EntryNotFoundError

            try:
                src = _open_source_shard(source_root, filename)
            except EntryNotFoundError:
                continue
        shutil.copyfile(src, dest)

    if not (root / "preprocessor_config.json").is_file():
        raise BackendError("Vision export requires preprocessor_config.json in model.source")


def restore_vision(export_dir: str | Path, source: str | Path) -> dict[str, Any] | None:
    """Copy source vision tensors missing from a language-only HF export."""
    root = Path(export_dir)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5" or not isinstance(config.get("vision_config"), dict):
        return None

    if not source:
        raise BackendError("Vision export requires model.source to restore source weights")
    source_index, source_root = _source_index(source)
    source_map = source_index.get("weight_map", {})
    if not isinstance(source_map, dict):
        raise BackendError(f"Source checkpoint has no weight map: {source}")
    expected = {key: shard for key, shard in source_map.items() if key.startswith("model.visual.")}
    if not expected:
        raise BackendError(f"Source checkpoint has no model.visual.* tensors: {source}")
    _restore_processor_files(root, source_root)

    exported = load_weight_map(root)
    missing = {
        key: shard
        for key, shard in expected.items()
        if key not in exported or not (root / str(exported[key])).is_file()
    }
    if missing:
        if (root / VISION_SHARD_NAME).exists():
            missing.update(
                {
                    key: shard
                    for key, shard in expected.items()
                    if exported.get(key) == VISION_SHARD_NAME
                }
            )
        from safetensors import safe_open
        from safetensors.torch import save_file

        by_shard: dict[str, list[str]] = {}
        for key, shard in missing.items():
            by_shard.setdefault(str(shard), []).append(key)
        tensors = {}
        for shard, keys in by_shard.items():
            path = _open_source_shard(source_root, shard)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key in keys:
                    if key not in available:
                        raise BackendError(f"Vision tensor {key} missing from {shard}")
                    tensors[key] = handle.get_tensor(key)

        shard_path = root / VISION_SHARD_NAME
        save_file(tensors, str(shard_path))
        update_export_index(
            root,
            VISION_SHARD_NAME,
            {key: tensor.numel() * tensor.element_size() for key, tensor in tensors.items()},
        )

    # The PTQ loader intentionally creates Qwen3.5 with the vision tower
    # disabled. Once the original vision tensors are restored, the exported
    # config must opt back into constructing that tower at inference time.
    if config.get("language_model_only") is True:
        config["language_model_only"] = False
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    note = {
        "vision_tensors": len(expected),
        "restored_tensors": len(missing),
        "method": "hf-source" if missing else "already-present",
    }
    (root / "vision_export.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    return note


def restore_vision_from_recipe(export_dir: Path, recipe: Any) -> dict[str, Any] | None:
    from megaquant.models.base import recipe_flag, recipe_get

    if recipe_flag(recipe, "model", "quantize_vision", default=False):
        return None
    source = recipe_get(recipe, "model", "source", default=None)
    return restore_vision(export_dir, source)
