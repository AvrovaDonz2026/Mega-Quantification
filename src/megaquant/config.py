"""Pydantic v2 recipe schema and YAML loaders."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from megaquant.exceptions import RecipeError

CLI_OVERRIDE_MAP = {
    "model": "model.source",
    "output": "export.output_dir",
    "backend": "backend",
    "num_samples": "calibration.num_samples",
    "max_seq_length": "calibration.max_seq_length",
    "batch_size": "calibration.batch_size",
    "algorithm": "algorithm",
    "scheme": "scheme",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrecisionSpec(StrictModel):
    format: Literal["nvfp4", "fp8", "mxfp4", "bf16", "int4"]
    bits: int
    group_size: int | None = None
    scale_dtype: str = "float8_e4m3fn"
    dynamic: bool | str = False
    strategy: str | None = None
    observer: str | None = None


class LayerGroup(StrictModel):
    name: str
    targets: list[str]
    weights: PrecisionSpec
    activations: PrecisionSpec | None = None
    ignore: list[str] = Field(default_factory=list)


class CalibrationSpec(StrictModel):
    dataset: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    num_samples: int = 512
    max_seq_length: int = 2048
    batch_size: int = 1
    seed: int = 42
    with_images: bool = False
    text_field: str | None = None


class ExportSpec(StrictModel):
    output_dir: str
    format: Literal["hf", "compressed-tensors"] = "hf"
    pack: bool = True


class ModelSpec(StrictModel):
    source: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"
    quantize_vision: bool = False
    quantize_mtp: bool = False
    ignore_lm_head: bool | None = None


class Recipe(StrictModel):
    name: str
    model: ModelSpec
    backend: Literal["auto", "modelopt", "llmcompressor"] = "auto"
    scheme: str
    groups: list[LayerGroup] | None = None
    algorithm: str = "max"
    kv_cache: str | None = "fp8"
    family: str | None = None
    extra_ignore: list[str] = Field(default_factory=list)
    ignore_lm_head: bool | None = None
    calibration: CalibrationSpec
    export: ExportSpec


def _deep_override(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if "." not in key and isinstance(value, dict):
            node = out.get(key)
            if isinstance(node, dict):
                out[key] = _deep_override(node, value)
            else:
                out[key] = copy.deepcopy(value)
            continue
        cursor: dict[str, Any] = out
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out


def recipe_from_mapping(data: dict[str, Any]) -> Recipe:
    if not isinstance(data, dict):
        raise RecipeError("Recipe mapping must be a dict")
    try:
        return Recipe.model_validate(data)
    except ValidationError as exc:
        raise RecipeError(str(exc)) from exc


def load_recipe(path: str | Path, overrides: dict[str, Any] | None = None) -> Recipe:
    recipe_path = Path(path)
    if not recipe_path.exists():
        raise RecipeError(f"Recipe not found: {recipe_path}")
    try:
        loaded = yaml.safe_load(recipe_path.read_text())
    except yaml.YAMLError as exc:
        raise RecipeError(f"Invalid recipe YAML ({recipe_path}): {exc}") from exc
    except OSError as exc:
        raise RecipeError(f"Could not read recipe {recipe_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RecipeError(f"Recipe YAML must be a mapping: {recipe_path}")
    if overrides:
        loaded = _deep_override(loaded, overrides)
    return recipe_from_mapping(loaded)
