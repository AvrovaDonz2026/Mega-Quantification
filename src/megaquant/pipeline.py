"""Resolve a recipe into a plan and optionally run PTQ."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from megaquant.calibration import build_calibration_iter
from megaquant.config import LayerGroup, PrecisionSpec, Recipe
from megaquant.exceptions import BackendError, FamilyError, MegaQuantError, RecipeError
from megaquant.registry import (
    detect_family,
    get_backend,
    get_family,
    get_scheme,
    list_backends,
)
from megaquant.runtime import apply_env_to_recipe, from_pretrained_env_kwargs, low_memory_plan_note


@dataclass
class ResolvedPlan:
    recipe: Recipe
    backend_name: str
    family_name: str
    ignore: list[str]
    groups: list[LayerGroup]
    notes: list[str] = field(default_factory=list)


def format_plan(plan: ResolvedPlan) -> str:
    """YAML-ish human-readable dry-run plan."""
    recipe = plan.recipe
    groups: list[dict[str, Any]] = []
    for group in plan.groups:
        item: dict[str, Any] = {
            "name": group.name,
            "targets": group.targets,
            "weights": group.weights.model_dump(),
        }
        if group.activations is not None:
            item["activations"] = group.activations.model_dump()
        if group.ignore:
            item["ignore"] = group.ignore
        groups.append(item)
    payload: dict[str, Any] = {
        "model": recipe.model.source,
        "device_map": recipe.model.device_map,
        "scheme": recipe.scheme,
        "backend": plan.backend_name,
        "algorithm": recipe.algorithm,
        "kv_cache": recipe.kv_cache,
        "family": plan.family_name,
        "groups": groups,
        "ignore": plan.ignore,
        "calibration": {
            "dataset": recipe.calibration.dataset,
            "num_samples": recipe.calibration.num_samples,
            "max_seq_length": recipe.calibration.max_seq_length,
        },
        "export": {"output_dir": recipe.export.output_dir},
    }
    if plan.notes:
        payload["notes"] = plan.notes
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip()


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _placeholder_group(scheme: str) -> LayerGroup:
    key = scheme.lower()
    if "w4a4" in key:
        weights = PrecisionSpec(format="nvfp4", bits=4, group_size=16)
        activations: PrecisionSpec | None = PrecisionSpec(
            format="nvfp4", bits=4, group_size=16, dynamic=True
        )
    elif "w4a16" in key:
        weights = PrecisionSpec(format="nvfp4", bits=4, group_size=32)
        activations = None
    elif key.startswith("fp8") or "w8a8" in key:
        weights = PrecisionSpec(format="fp8", bits=8)
        activations = PrecisionSpec(format="fp8", bits=8, dynamic=True, strategy="token")
    elif "int4" in key:
        weights = PrecisionSpec(format="int4", bits=4)
        activations = None
    else:
        weights = PrecisionSpec(format="nvfp4", bits=4, group_size=32)
        activations = PrecisionSpec(format="fp8", bits=8, dynamic=True, strategy="token")
    return LayerGroup(name=scheme, targets=["*"], weights=weights, activations=activations)


def _sanitize_group_dict(item: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "format",
        "bits",
        "group_size",
        "scale_dtype",
        "dynamic",
        "strategy",
        "observer",
    }
    payload = dict(item)
    for key in ("weights", "activations"):
        value = payload.get(key)
        if isinstance(value, dict):
            payload[key] = {k: v for k, v in value.items() if k in allowed}
    return payload


def _coerce_groups(raw: object, scheme: str) -> list[LayerGroup]:
    if isinstance(raw, LayerGroup):
        return [raw]
    if raw is not None and hasattr(raw, "groups") and not isinstance(raw, dict):
        raw = raw.groups
    if isinstance(raw, dict):
        if "groups" in raw:
            raw = raw["groups"]
        elif {"name", "targets", "weights"} <= raw.keys():
            raw = [raw]
        else:
            groups: list[LayerGroup] = []
            for key, value in raw.items():
                if isinstance(value, LayerGroup):
                    groups.append(value)
                    continue
                if not isinstance(value, dict):
                    raise RecipeError(
                        f"Scheme '{scheme}' catalog entry '{key}' is not a layer group"
                    )
                payload = _sanitize_group_dict(value)
                payload.setdefault("name", key)
                try:
                    groups.append(LayerGroup.model_validate(payload))
                except ValidationError as exc:
                    raise RecipeError(
                        f"Invalid layer group '{key}' in scheme '{scheme}': {exc}"
                    ) from exc
            return groups
    if not isinstance(raw, list):
        raise RecipeError(f"Scheme '{scheme}' catalog entry is not a list of layer groups")
    groups = []
    for item in raw:
        if isinstance(item, LayerGroup):
            groups.append(item)
            continue
        if not isinstance(item, dict):
            raise RecipeError(f"Invalid layer group in scheme '{scheme}': {type(item).__name__}")
        try:
            groups.append(LayerGroup.model_validate(_sanitize_group_dict(item)))
        except ValidationError as exc:
            raise RecipeError(f"Invalid layer group in scheme '{scheme}': {exc}") from exc
    return groups


def _load_hf_config(source: str) -> dict[str, Any]:
    path = Path(source)
    local: Path | None = None
    if path.is_dir() and (path / "config.json").is_file():
        local = path / "config.json"
    elif path.is_file() and path.name == "config.json":
        local = path
    if local is not None:
        try:
            payload = json.loads(local.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FamilyError(f"Could not read {local}: {exc}") from exc
        if not isinstance(payload, dict):
            raise FamilyError(f"{local} is not a JSON object")
        return payload
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FamilyError("huggingface_hub is required to inspect remote model configs") from exc
    try:
        cfg_path = hf_hub_download(repo_id=source, filename="config.json")
        payload = json.loads(Path(cfg_path).read_text())
    except Exception as exc:
        raise FamilyError(f"Could not load config.json for '{source}': {exc}") from exc
    if not isinstance(payload, dict):
        raise FamilyError(f"config.json for '{source}' is not a JSON object")
    return payload


def _backend_supports(backend: object, scheme: str) -> bool:
    fn = getattr(backend, "supports", None)
    if not callable(fn):
        return True
    try:
        return bool(fn(scheme))
    except TypeError:
        if isinstance(backend, type):
            return bool(backend().supports(scheme))
        return False


def _instantiate(obj: object) -> object:
    if isinstance(obj, type):
        return obj()
    return obj


def _pick_backend(recipe: Recipe, notes: list[str]) -> str:
    registered = list_backends()

    def supports(name: str) -> bool:
        try:
            return _backend_supports(get_backend(name), recipe.scheme)
        except BackendError:
            return False

    requested = recipe.backend
    if requested != "auto":
        if requested not in registered:
            notes.append(
                f"backend '{requested}' is not registered; dry-run continues without installing it"
            )
        return requested

    if "modelopt" in registered and supports("modelopt"):
        return "modelopt"
    if "llmcompressor" in registered and supports("llmcompressor"):
        return "llmcompressor"
    if "modelopt" in registered:
        return "modelopt"
    if "llmcompressor" in registered:
        return "llmcompressor"

    planned = "auto/modelopt" if recipe.scheme.startswith("nvfp4") else "auto/llmcompressor"
    notes.append(
        f"no backends registered; planned default is '{planned}' "
        "(pip install megaquant[modelopt] or megaquant[llmcompressor] to run)"
    )
    return planned


def _family_ignore(family_name: str, recipe: Recipe, notes: list[str]) -> list[str]:
    try:
        family = _instantiate(get_family(family_name))
    except FamilyError:
        if family_name != "generic":
            notes.append(f"family '{family_name}' is not registered; using extra_ignore only")
        return []
    fn = getattr(family, "default_ignore", None)
    if not callable(fn):
        return []
    ignore = fn(recipe)
    return list(ignore) if ignore else []


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _torch_dtype(name: str) -> Any:
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name in aliases:
        return aliases[name]
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise BackendError(f"Unknown torch dtype '{name}'")
    return dtype


def _load_model_and_tokenizer(recipe: Recipe, family_name: str) -> tuple[Any, Any]:
    missing_torch = importlib.util.find_spec("torch") is None
    missing_hf = importlib.util.find_spec("transformers") is None
    if missing_torch or missing_hf:
        raise BackendError(
            "transformers and torch are required to load weights. "
            "Install with: pip install megaquant[hf]"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kwargs: dict[str, Any] = {"trust_remote_code": recipe.model.trust_remote_code}
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": recipe.model.trust_remote_code,
        "torch_dtype": _torch_dtype(recipe.model.dtype),
        "device_map": recipe.model.device_map,
    }
    model_cls: Any = AutoModelForCausalLM
    try:
        family = _instantiate(get_family(family_name))
    except FamilyError:
        family = None
    if family is not None:
        extra_fn = getattr(family, "load_kwargs", None)
        if callable(extra_fn):
            extra = dict(extra_fn(recipe) or {})
            model_cls = extra.pop("model_cls", extra.pop("auto_model_class", model_cls))
            load_kwargs.update(extra)
    # Env overrides win over family.load_kwargs (device_map, CPU offload, max_memory).
    load_kwargs.update(from_pretrained_env_kwargs())
    offload = load_kwargs.get("offload_folder")
    if offload:
        Path(offload).mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(recipe.model.source, **tok_kwargs)
        if isinstance(model_cls, str):
            import transformers

            model_cls = getattr(transformers, model_cls)
        model = model_cls.from_pretrained(recipe.model.source, **load_kwargs)
    except MegaQuantError:
        raise
    except Exception as exc:
        raise BackendError(f"Failed to load model '{recipe.model.source}': {exc}") from exc
    return model, tokenizer


def _require_backend(plan: ResolvedPlan) -> Any:
    name = plan.backend_name
    if name.startswith("auto/"):
        name = name.split("/", 1)[1]
    try:
        backend = _instantiate(get_backend(name))
    except BackendError as exc:
        raise BackendError(
            f"Backend '{name}' is not registered. "
            f"Install with: pip install megaquant[{name}] "
            "(NVFP4 W4A8 prefers megaquant[modelopt])"
        ) from exc
    available = getattr(backend, "available", None)
    if callable(available) and not available():
        raise BackendError(
            f"Backend '{name}' is registered but not available. "
            f"Install with: pip install megaquant[{name}]"
        )
    return backend


def _write_provenance(plan: ResolvedPlan, export_path: Path) -> None:
    prov_dir = export_path.parent if export_path.suffix else export_path
    if export_path.exists() and export_path.is_file():
        prov_dir = export_path.parent
    prov_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": plan.recipe.name,
        "base_model": plan.recipe.model.source,
        "scheme": plan.recipe.scheme,
        "backend": plan.backend_name,
        "algorithm": plan.recipe.algorithm,
        "kv_cache": plan.recipe.kv_cache,
        "family": plan.family_name,
        "calibration": plan.recipe.calibration.model_dump(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (prov_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


class QuantPipeline:
    def __init__(self, recipe: Recipe):
        self.recipe = recipe

    def resolve(self) -> ResolvedPlan:
        recipe = apply_env_to_recipe(self.recipe)
        self.recipe = recipe
        notes: list[str] = []
        low_mem_note = low_memory_plan_note()
        if low_mem_note:
            notes.append(low_mem_note)

        if recipe.family:
            family_name = recipe.family
        else:
            try:
                cfg = _load_hf_config(recipe.model.source)
                family_name = detect_family(cfg)
            except FamilyError as exc:
                notes.append(f"{exc}; falling back to family 'generic'")
                family_name = "generic"

        if recipe.groups:
            groups = list(recipe.groups)
        else:
            try:
                catalog_entry = get_scheme(recipe.scheme)
            except RecipeError:
                groups = [_placeholder_group(recipe.scheme)]
                notes.append(
                    f"scheme catalog has no '{recipe.scheme}'; synthesized a placeholder LayerGroup"
                )
            else:
                try:
                    groups = _coerce_groups(catalog_entry, recipe.scheme)
                except RecipeError as exc:
                    groups = [_placeholder_group(recipe.scheme)]
                    notes.append(
                        f"scheme '{recipe.scheme}' is registered but could not be "
                        f"coerced into layer groups ({exc}); using a placeholder"
                    )

        ignore = _unique(_family_ignore(family_name, recipe, notes) + list(recipe.extra_ignore))
        backend_name = _pick_backend(recipe, notes)
        return ResolvedPlan(
            recipe=recipe,
            backend_name=backend_name,
            family_name=family_name,
            ignore=ignore,
            groups=groups,
            notes=notes,
        )

    def dry_run(self) -> ResolvedPlan:
        return self.resolve()

    def run(self, dry_run: bool = False) -> Path | ResolvedPlan:
        plan = self.resolve()
        if dry_run:
            print(format_plan(plan))
            return plan

        backend = _require_backend(plan)
        model, tokenizer = _load_model_and_tokenizer(self.recipe, plan.family_name)
        calib_iter = build_calibration_iter(self.recipe, tokenizer)
        backend.quantize(model, plan, calib_iter)
        exported = backend.export(model, self.recipe, tokenizer=tokenizer)
        out = Path(exported) if exported is not None else Path(self.recipe.export.output_dir)
        _write_provenance(plan, out)
        return out
