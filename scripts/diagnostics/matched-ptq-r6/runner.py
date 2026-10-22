"""Matched static NVFP4 pair runner. Dry-run is the default; no model is loaded."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _repository_root(start: Path) -> Path:
    """Find the repository containing a runner, including the runner directory itself."""
    candidates = (start, *start.parents)
    for candidate in candidates:
        if all((candidate / marker).exists() for marker in ("pyproject.toml", "src", "recipes")):
            return candidate
    raise RuntimeError(
        "could not find repository root from runner directory; expected pyproject.toml, src, "
        "and recipes"
    )


ROOT = _repository_root(HERE)
SOURCE_ROOT = Path(os.environ.get("MEGAQUANT_SOURCE_ROOT", str(ROOT / "src"))).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

try:  # Keep response-only evidence attachment usable without HF quant deps.
    from megaquant.backends.modelopt import ModelOptBackend  # noqa: E402
    from megaquant.config import load_recipe  # noqa: E402
    from megaquant.pipeline import QuantPipeline, format_plan  # noqa: E402
    from megaquant.registry import get_backend, register_backend  # noqa: E402

    _DEPENDENCY_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised by minimal offline environments
    ModelOptBackend = object  # type: ignore[assignment,misc]
    QuantPipeline = None  # type: ignore[assignment]
    format_plan = None  # type: ignore[assignment]
    load_recipe = None  # type: ignore[assignment]
    get_backend = None  # type: ignore[assignment]
    register_backend = None  # type: ignore[assignment]
    _DEPENDENCY_ERROR = exc

MODEL_OPT_VERSION = "0.47.0"
SEED = 42


class RunnerError(RuntimeError):
    pass


def _require_quant_dependencies() -> None:
    if _DEPENDENCY_ERROR is not None:
        raise RunnerError(
            "PTQ execution requires the repository quantization dependencies"
        ) from _DEPENDENCY_ERROR


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            entries = []
            for key, part in item.items():
                if isinstance(key, bool):
                    typed_key = ["bool", key]
                elif isinstance(key, int):
                    typed_key = ["int", key]
                elif isinstance(key, float):
                    typed_key = ["float", key]
                elif key is None:
                    typed_key = ["null", None]
                else:
                    typed_key = ["str", str(key)]
                entries.append([typed_key, normalize(part)])
            entries.sort(key=lambda pair: json.dumps(pair[0], separators=(",", ":")))
            return {"__megaquant_mapping__": entries}
        if isinstance(item, tuple):
            return {"__megaquant_tuple__": [normalize(part) for part in item]}
        if isinstance(item, list):
            return [normalize(part) for part in item]
        return item

    return _digest_bytes(
        json.dumps(normalize(value), sort_keys=True, separators=(",", ":")).encode()
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _file_hash(path) if path.is_file() else None,
        }
    digest = hashlib.sha256()
    count = 0
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode() + b"\0")
        digest.update(bytes.fromhex(_file_hash(item)))
        count += 1
    return {"path": str(path), "file_count": count, "sha256": digest.hexdigest()}


def _source_fingerprint() -> dict[str, Any]:
    files = sorted(p for p in (SOURCE_ROOT / "megaquant").rglob("*.py") if p.is_file())
    files.extend((Path(__file__).resolve(), SOURCE_ROOT / "megaquant/backends/modelopt.py"))
    files = sorted(set(files))
    digest = hashlib.sha256()
    for item in files:
        try:
            label = item.relative_to(ROOT).as_posix()
        except ValueError:
            label = item.relative_to(SOURCE_ROOT).as_posix()
        digest.update(label.encode() + b"\0")
        digest.update(bytes.fromhex(_file_hash(item)))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def _protocol(recipe: Any) -> dict[str, Any]:
    value = recipe.model_dump(mode="json")
    value.pop("name", None)
    value.pop("algorithm", None)
    export = value.get("export")
    if isinstance(export, dict):
        export.pop("output_dir", None)
    return value


def _quantizer_inventory(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: item.get(key)
            for key in ("name", "scope", "calibrator", "calibrator_leaves", "amax", "global_amax")
        }
        for item in report.get("quantizers", [])
    ]


def compare_arm_evidence(
    max_evidence: dict[str, Any], mse_evidence: dict[str, Any]
) -> dict[str, Any]:
    """Compare paired evidence using the offline fail-closed comparator."""
    comparator = Path(__file__).with_name("pair_compare.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("r6_pair_compare", comparator)
    if spec is None or spec.loader is None:
        raise RunnerError(f"could not load comparator: {comparator}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compare(max_evidence, mse_evidence)
    checks = dict(result["checks"])
    # Preserve the runner's historical check name for downstream evidence readers.
    checks["recipe_protocol_equal"] = checks.get("matched_protocol", False)
    return {
        "checks": checks,
        "mse_search": result["mse_search"],
        "all_required": bool(result["comparable"]),
        "comparable": bool(result["comparable"]),
    }


def _protocol_from_evidence(evidence: dict[str, Any]) -> Any:
    return evidence.get("recipe_protocol")


def _provenance_protocol(evidence: dict[str, Any]) -> Any:
    value = dict(evidence.get("provenance", {}))
    value.pop("algorithm", None)
    value.pop("recipe_selector", None)
    value.pop("runtime_algorithm", None)
    return value


def _activation_hashes(report: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        (item.get("name", ""), (item.get("amax") or {}).get("sha256"))
        for item in report.get("activation_scales", [])
    ]


_RESPONSE_ID_KEYS = ("output_ids", "completion_ids", "token_ids")


def _integer_ids(value: Any, name: str, *, allow_empty: bool = False) -> list[int]:
    """Validate token IDs while keeping the sequence private to the caller."""
    if not isinstance(value, list) or (not allow_empty and not value):
        raise RunnerError(f"generated response {name} must be a non-empty list of token IDs")
    if any(type(item) is not int or item < 0 for item in value):
        raise RunnerError(f"generated response {name} must contain non-negative integer token IDs")
    return list(value)


def _response_ids(record: dict[str, Any], name: str) -> tuple[list[int], list[int]]:
    """Extract prompt and completion IDs from a native response record.

    Native servers commonly wrap the response under ``response``. Text fields
    are deliberately ignored; callers must provide token IDs so replay evidence
    cannot be fabricated from decoded text.
    """
    response = record.get("response")
    if isinstance(response, dict):
        record = response
    prompt = record.get("prompt_token_ids", record.get("prompt_ids"))
    output = next((record.get(key) for key in _RESPONSE_ID_KEYS if key in record), None)
    return _integer_ids(prompt, f"{name}.prompt_token_ids"), _integer_ids(
        output, f"{name}.output_ids"
    )


def _ids_sha256(ids: list[int]) -> str:
    return _digest_bytes(json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def _first_divergence(left: list[int], right: list[int]) -> dict[str, int | None]:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {"index": index, "left_length": len(left), "right_length": len(right)}
    return {
        "index": limit if len(left) != len(right) else None,
        "left_length": len(left),
        "right_length": len(right),
    }


def _cycle_metrics(
    ids: list[int], *, max_period: int = 8192, min_full_repeats: int = 4
) -> dict[str, Any]:
    """Return bounded exact-cycle statistics without retaining token IDs."""
    n = len(ids)
    upper = min(max_period, (n - 1) // min_full_repeats)
    best: tuple[int, int, int] | None = None
    for period in range(1, upper + 1):
        run = run_start = 0
        local: tuple[int, int] | None = None
        for index in range(period, n):
            if ids[index] == ids[index - period]:
                if run == 0:
                    run_start = index - period
                run += 1
            else:
                if run >= min_full_repeats * period and (
                    local is None or run > local[0] or (run == local[0] and run_start < local[1])
                ):
                    local = (run, run_start)
                run = 0
        if run >= min_full_repeats * period and (
            local is None or run > local[0] or (run == local[0] and run_start < local[1])
        ):
            local = (run, run_start)
        if local is None:
            continue
        matching, onset = local
        candidate = (matching, period, onset)
        if (
            best is None
            or matching > best[0]
            or (
                matching == best[0]
                and (period < best[1] or (period == best[1] and onset < best[2]))
            )
        ):
            best = candidate
    if best is None:
        return {
            "found": False,
            "period": None,
            "onset": None,
            "matching_positions": 0,
            "coverage_tokens": 0,
            "full_repeats": 0,
            "output_tokens": n,
            "output_ids_sha256": _ids_sha256(ids),
        }
    matching, period, onset = best
    return {
        "found": True,
        "period": period,
        "onset": onset,
        "matching_positions": matching,
        "coverage_tokens": matching + period,
        "full_repeats": matching // period,
        "output_tokens": n,
        "output_ids_sha256": _ids_sha256(ids),
    }


def _text_only_protocol(
    response_bundle: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate and sanitize the generation protocol from native responses."""
    protocol = response_bundle.get("text_only_protocol", response_bundle.get("protocol"))
    if not isinstance(protocol, dict):
        raise RunnerError("generated responses are missing text-only protocol metadata")
    if protocol.get("text_only") is not True or protocol.get("modalities") != ["text"]:
        raise RunnerError("generated responses must declare text_only=true and modalities=['text']")
    raw_temperature = protocol.get("temperature")
    if isinstance(raw_temperature, bool):
        raise RunnerError("generated response temperature is missing or invalid")
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError) as exc:
        raise RunnerError("generated response temperature is missing or invalid") from exc
    if temperature != 0.0:
        raise RunnerError("generated responses must use temperature=0")
    if not records:
        raise RunnerError("generated response bundle is empty")
    prompt_hashes: list[str] = []
    for index, record in enumerate(records):
        prompt_ids, _ = _response_ids(record, str(index))
        prompt_hashes.append(_ids_sha256(prompt_ids))
    if len(set(prompt_hashes)) != 1:
        raise RunnerError("generated responses do not use one stable prompt token sequence")
    # Copy only scalar protocol controls and aggregate hashes/counts. Never persist
    # decoded prompts, decoded completions, or raw token ID arrays.
    sanitized = {
        "schema": "r6-text-only-v1",
        "text_only": True,
        "modalities": ["text"],
        "temperature": 0.0,
    }
    for key in ("top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens", "seed"):
        if key in protocol and isinstance(protocol[key], (bool, int, float, str)):
            sanitized[key] = protocol[key]
    return sanitized


def _response_observation(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_hashes: list[str] = []
    output_lengths: list[int] = []
    output_hashes: list[str] = []
    for index, record in enumerate(records):
        prompt_ids, output_ids = _response_ids(record, str(index))
        prompt_hashes.append(_ids_sha256(prompt_ids))
        output_lengths.append(len(output_ids))
        output_hashes.append(_ids_sha256(output_ids))
    if len(set(prompt_hashes)) != 1:
        raise RunnerError("generated responses do not use one stable prompt token sequence")
    return {
        "schema": "r6-response-observation-v1",
        "response_count": len(records),
        "prompt_ids_sha256": prompt_hashes[0],
        "completion_token_counts": output_lengths,
        "output_ids_sha256": output_hashes,
    }


def attach_response_evidence(evidence_path: Path, response_path: Path) -> Path:
    """Attach privacy-preserving replay metrics from responses bound to an export."""
    try:
        evidence = json.loads(evidence_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("could not read completed arm evidence") from exc
    if not isinstance(evidence, dict) or evidence.get("execution") != "complete":
        raise RunnerError("response metrics require completed arm evidence")
    export = evidence.get("export")
    if not isinstance(export, dict) or not isinstance(export.get("sha256"), str):
        raise RunnerError("response metrics require an exported tree fingerprint")
    index = evidence.get("export_index_verification")
    if not isinstance(index, dict) or index.get("keys_validated") is not True:
        raise RunnerError("response metrics require validated exported safetensors keys")
    bundle, records = _load_generated_responses(
        response_path, export["sha256"], str(evidence.get("arm"))
    )
    protocol = _text_only_protocol(bundle, records)
    observation = _response_observation(records)
    trajectory = _trajectory_metrics(records)
    evidence["text_only_protocol"] = protocol
    evidence["response_observation"] = observation
    evidence["trajectory_metrics"] = trajectory
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    return evidence_path


def _trajectory_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual output token trajectories without persisting raw IDs."""
    if not records:
        raise RunnerError("cannot summarize an empty generated response set")
    outputs: list[list[int]] = []
    for index, record in enumerate(records):
        _, output_ids = _response_ids(record, str(index))
        outputs.append(output_ids)
    reference = outputs[0]
    divergences = [_first_divergence(reference, output) for output in outputs[1:]]
    return {
        "schema": "r6-trajectory-v1",
        "response_count": len(outputs),
        "first_divergence": divergences[0]
        if divergences
        else _first_divergence(reference, reference),
        "divergence_count": sum(item["index"] is not None for item in divergences),
        "cycle": _cycle_metrics(reference),
        "output_ids_sha256": _ids_sha256(reference),
        "output_tokens": len(reference),
    }


def _load_generated_responses(
    path: Path, exported_hash: str, arm: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a response bundle and enforce binding to the completed export."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"could not read generated response bundle: {path}") from exc
    if not isinstance(payload, dict):
        raise RunnerError("generated response bundle must be a JSON object")
    if payload.get("arm") != arm:
        raise RunnerError("generated response bundle arm does not match the exported arm")
    if payload.get("export_sha256") != exported_hash:
        raise RunnerError("generated response bundle is not bound to this exported arm")
    records = payload.get("responses")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise RunnerError("generated response bundle must contain response objects")
    return payload, records


def _calibration_fingerprint(recipe: Any) -> dict[str, Any]:
    value = str(recipe.calibration.dataset)
    path = Path(value).expanduser()
    if path.is_file():
        refs = _referenced_images(path)
        return {"kind": "local-file", **_tree_hash(path), "referenced_images": refs}
    if path.is_dir():
        manifests = [
            ref
            for item in sorted(path.rglob("*.json*"))
            if item.is_file()
            for ref in _referenced_images(item)
        ]
        unique = {item["path"]: item for item in manifests}
        return {
            "kind": "local-directory",
            **_tree_hash(path),
            "referenced_images": [unique[key] for key in sorted(unique)],
        }
    return {
        "kind": "reference-only",
        "reference_sha256": _digest_bytes(value.encode()),
        "reference": value,
    }


def _referenced_images(path: Path) -> list[dict[str, Any]]:
    refs: set[Path] = set()
    missing: set[Path] = set()
    try:
        lines = path.read_text().splitlines() if path.suffix != ".json" else [path.read_text()]
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key.lower() in {
                            "image",
                            "images",
                            "path",
                            "image_url",
                            "url",
                        } and isinstance(child, str):
                            candidate = (path.parent / child).resolve()
                            if candidate.suffix.lower() in {
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".webp",
                                ".bmp",
                            }:
                                (refs if candidate.is_file() else missing).add(candidate)
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(payload)
    except OSError:
        return []
    if missing:
        raise RunnerError(f"calibration JSON references missing images: {len(missing)}")
    return [{"path": str(item), "sha256": _file_hash(item)} for item in sorted(refs)]


def _model_fingerprint(source: str) -> dict[str, Any]:
    path = Path(source).expanduser()
    if path.exists():
        config = path / "config.json" if path.is_dir() else path
        if not path.is_dir():
            return {"kind": "local-file", "source": str(path), **_tree_hash(path)}
        if not config.is_file():
            return {"kind": "local-no-config", "source": str(path), "config_exists": False}
        files = [p for p in path.rglob("*") if p.is_file()]
        has_weights = any(p.suffix in {".safetensors", ".bin", ".pt", ".pth"} for p in files)
        has_index = any(p.name.endswith(".index.json") for p in files)
        if not (has_weights or has_index):
            return {"kind": "local-no-weights", "source": str(path), **_tree_hash(path)}
        return {"kind": "local", "source": str(path), **_tree_hash(path)}
    return {
        "kind": "remote-reference",
        "reference_sha256": _digest_bytes(source.encode()),
        "reference": source,
    }


def _modelopt_version() -> str:
    try:
        return importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RunnerError("--execute requires nvidia-modelopt==0.47.0") from exc


def _modelopt_fingerprint() -> dict[str, Any]:
    dist = importlib.metadata.distribution("nvidia-modelopt")
    files = sorted((item for item in (dist.files or [])), key=str)
    digest = hashlib.sha256()
    count = 0
    for item in files:
        resolved = Path(dist.locate_file(item))
        if not resolved.is_file():
            continue
        digest.update(str(item).encode() + b"\0")
        digest.update(bytes.fromhex(_file_hash(resolved)))
        count += 1
    try:
        import modelopt
        import modelopt.torch.quantization as mtq

        modelopt_file = Path(modelopt.__file__).resolve()
        mtq_file = Path(mtq.__file__).resolve()
    except (ImportError, AttributeError) as exc:
        raise RunnerError("could not resolve imported modelopt.torch.quantization source") from exc
    if not any(Path(dist.locate_file(item)).resolve() == mtq_file for item in (dist.files or [])):
        raise RunnerError(
            f"imported mtq source is outside nvidia-modelopt distribution: {mtq_file}"
        )
    if not any(
        Path(dist.locate_file(item)).resolve() == modelopt_file for item in (dist.files or [])
    ):
        raise RunnerError(
            f"imported modelopt root is outside nvidia-modelopt distribution: {modelopt_file}"
        )
    return {
        "name": "nvidia-modelopt",
        "version": dist.version,
        "files_count": count,
        "sha256": digest.hexdigest(),
        "mtq_file": str(mtq_file),
        "mtq_sha256": _file_hash(mtq_file),
        "modelopt_file": str(modelopt_file),
        "modelopt_sha256": _file_hash(modelopt_file),
        "distribution_root": str(Path(dist.locate_file(".")).resolve()),
    }


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        import torch

        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.contiguous().cpu()
        count = tensor.numel()
        finite_count = int(torch.isfinite(tensor).sum().item())
        minimum = float(tensor.min().item()) if count else None
        maximum = float(tensor.max().item()) if count else None
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "count": count,
            "finite_count": finite_count,
            "finite": count > 0 and finite_count == count,
            "positive": count > 0 and minimum is not None and minimum > 0,
            "min": minimum,
            "max": maximum,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": _digest_bytes(
                str(tensor.dtype).encode()
                + b"\0"
                + repr(tuple(tensor.shape)).encode()
                + b"\0"
                + raw_bytes
            ),
        }
    except Exception as exc:
        return {"count": None, "finite": False, "inspection_error": type(exc).__name__}


def _inspect_scales(model: Any, arm: str, recipe: Any | None = None) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    activation_scales: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not getattr(module, "is_enabled", False):
            continue
        if name.endswith("input_quantizer"):
            scale = _tensor_summary(getattr(module, "_amax", getattr(module, "amax", None)))
            activation_scales.append({"name": name, "amax": scale})
            continue
        if not name.endswith("weight_quantizer"):
            continue
        num_bits = getattr(module, "num_bits", None)
        try:
            bits = tuple(int(x) for x in num_bits)
        except (TypeError, ValueError):
            bits = ()
        if bits != (2, 1) and str(num_bits).lower() != "e2m1":
            continue
        block_sizes = getattr(module, "block_sizes", getattr(module, "_block_sizes", {}))
        if not isinstance(block_sizes, dict):
            block_sizes = {}
        if block_sizes.get(-1, block_sizes.get("-1")) != 16:
            raise RunnerError(f"{arm}: enabled NVFP4 weight is not group 16: {name}")
        if block_sizes.get("type") != "static":
            raise RunnerError(f"{arm}: enabled NVFP4 weight remains dynamic: {name}")
        amax = getattr(module, "_amax", None)
        global_amax = getattr(module, "global_amax", getattr(module, "_global_amax", None))
        amax_info = _tensor_summary(amax)
        global_info = _tensor_summary(global_amax)
        calibrator = getattr(module, "_calibrator", None)
        calibrator_names = _leaf_calibrator_names(calibrator)
        calibrator_name = type(calibrator).__name__ if calibrator is not None else None
        scope = (
            "mtp"
            if "mtp" in name.lower()
            else "vision"
            if any(part in name.lower() for part in ("vision", "visual"))
            else "core"
        )
        item = {
            "name": name,
            "scope": scope,
            "calibrator": calibrator_name,
            "calibrator_leaves": sorted(calibrator_names),
            "amax": amax_info,
            "global_amax": global_info,
        }
        found.append(item)
        if not amax_info or not amax_info.get("finite"):
            raise RunnerError(f"{arm}: static weight scale missing or non-finite for {name}")
        if not amax_info.get("positive"):
            raise RunnerError(f"{arm}: static weight scale must be positive for {name}")
        if not global_info or not global_info.get("finite"):
            raise RunnerError(f"{arm}: global static scale missing or non-finite for {name}")
        if not global_info.get("positive"):
            raise RunnerError(f"{arm}: global static scale must be positive for {name}")
        if not calibrator_name:
            raise RunnerError(f"{arm}: calibration state missing for {name}")
        expected_calibrator = "NVFP4MSECalibrator" if arm == "static-MSE" else "MaxCalibrator"
        if calibrator_names != {expected_calibrator}:
            raise RunnerError(
                f"{arm}: expected {expected_calibrator} for {name}, found {calibrator_names}"
            )
    if not found:
        raise RunnerError(
            f"{arm}: no enabled static NVFP4 weight quantizers found after calibration"
        )
    report = {
        "quantizer_count": len(found),
        "scope_counts": {
            scope: sum(item["scope"] == scope for item in found)
            for scope in ("core", "vision", "mtp")
        },
        "quantizers": found,
        "activation_quantizer_count": len(activation_scales),
        "activation_scales": activation_scales,
    }
    if recipe is not None:
        expected = {
            "core": True,
            "vision": bool(recipe.model.quantize_vision),
            "mtp": bool(recipe.model.quantize_mtp),
        }
        for scope, enabled in expected.items():
            if enabled and report["scope_counts"][scope] == 0:
                raise RunnerError(
                    f"{arm}: recipe enables {scope} but no calibrated weight quantizers were found"
                )
    return report


def _leaf_calibrator_names(value: Any) -> set[str]:
    return _leaf_calibrator_names_inner(value, set())


def _leaf_calibrator_names_inner(value: Any, seen: set[int]) -> set[str]:
    if value is None:
        return set()
    if id(value) in seen:
        return set()
    seen.add(id(value))
    leaves: set[str] = set()
    children = []
    if isinstance(value, dict):
        children = list(value.values())
    else:
        children = list(getattr(value, "children", lambda: ())())
        attributes = vars(value) if hasattr(value, "__dict__") else {}
        for name, child in attributes.items():
            if "calibrator" in name.lower() and child is not value:
                children.append(child)
    if not children:
        leaves.add(type(value).__name__)
    else:
        for child in children:
            leaves.update(_leaf_calibrator_names_inner(child, seen))
    return leaves


def _execution_environment_gate() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RunnerError("--execute requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    try:
        import torch
    except ImportError as exc:
        raise RunnerError("--execute requires torch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RunnerError("--execute requires CUDA availability")


class StaticMaxBackend(ModelOptBackend):
    """Observation-only backend; native recipe selector owns configuration."""

    def quantize(self, model: Any, plan: Any, calib_iter: Any) -> Any:
        wrapped = _HashingCalibrationIter(calib_iter)
        result = super().quantize(model, plan, wrapped)
        self.calibration_fingerprint = wrapped.report
        self.runtime_coverage = _inspect_scales(result, "static-max", plan.recipe)
        return result


class StaticMSEBackend(ModelOptBackend):
    def quantize(self, model: Any, plan: Any, calib_iter: Any) -> Any:
        wrapped = _HashingCalibrationIter(calib_iter)
        result = super().quantize(model, plan, wrapped)
        self.calibration_fingerprint = wrapped.report
        self.runtime_coverage = _inspect_scales(result, "static-MSE", plan.recipe)
        return result


def _arm_name(arm: str) -> str:
    return "static_max" if arm == "static-max" else "static_mse"


def _backend_for_arm(arm: str) -> ModelOptBackend:
    return StaticMaxBackend() if arm == "static-max" else StaticMSEBackend()


def _actual_backend_config(recipe: Any, arm: str) -> tuple[Any, dict[str, Any]]:
    _require_quant_dependencies()
    get_backend("modelopt")
    plan = QuantPipeline(recipe).resolve()
    backend = _backend_for_arm(arm)
    return plan, backend.build_quant_cfg(plan)


def _environment_fingerprint() -> dict[str, Any]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "MEGAQUANT_DEVICE_MAP",
        "MEGAQUANT_MAX_MEMORY",
        "MEGAQUANT_GPU_HEADROOM_GIB",
        "MEGAQUANT_CPU_RESERVE_GIB",
        "MEGAQUANT_OFFLOAD_DIR",
        "MEGAQUANT_PIN_MEMORY",
        "MEGAQUANT_LOW_MEMORY",
        "MEGAQUANT_SKIP_VISION_INIT",
        "MEGAQUANT_FAST",
        "MEGAQUANT_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MODELOPT_NVFP4_TRITON_SWEEP",
        "TORCH_ALLOW_TF32",
        "NVIDIA_TF32_OVERRIDE",
        "PYTORCH_CUDA_ALLOC_CONF",
    )
    env = {key: os.environ[key] for key in keys if key in os.environ}
    return {
        "python": sys.version,
        "platform": sys.platform,
        "environment": env,
        "sha256": _digest_json(env),
    }


def _seed_process(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _HashingCalibrationIter:
    def __init__(self, source: Any):
        self.source = source
        self.report = {"batch_count": 0, "batches": [], "pass_count": 0}

    def __len__(self):
        return len(self.source)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def __iter__(self):
        self.report["pass_count"] += 1
        for batch in self.source:
            entries = {}
            for key, value in sorted(batch.items()):
                summary = _tensor_summary(value)
                if summary is not None and (
                    summary.get("inspection_error")
                    or summary.get("count") is None
                    or not summary.get("finite", False)
                ):
                    raise RunnerError(f"calibration batch fingerprint failed for key {key!r}")
                entries[str(key)] = summary or {
                    "type": type(value).__name__,
                    "sha256": _digest_json(value),
                }
            entries["modality"] = sorted(str(key) for key in batch)
            entries["sha256"] = _digest_json(entries)
            self.report["batches"].append(entries)
            self.report["batch_count"] += 1
            yield batch


def prepare(recipe_path: Path, arm: str, evidence_dir: Path) -> tuple[Any, dict[str, Any], Path]:
    _require_quant_dependencies()
    recipe = load_recipe(recipe_path)
    recipe.backend = "modelopt"
    resolved_recipe = recipe.model_copy(deep=True)
    resolved_recipe.model.dtype = "bfloat16"
    resolved_recipe.calibration.seed = SEED
    resolved_recipe.algorithm = "static_max" if arm == "static-max" else "mse"
    output = Path(resolved_recipe.export.output_dir).expanduser()
    resolved_recipe.export.output_dir = str(output.with_name(output.name + f"-{_arm_name(arm)}"))
    plan, actual_cfg = _actual_backend_config(resolved_recipe, arm)
    plan_text = format_plan(plan)
    summary = {
        "arm": arm,
        "model_dtype": "bfloat16",
        "seed": SEED,
        "recipe": str(recipe_path.resolve()),
        "recipe_sha256": _file_hash(recipe_path.resolve()),
        "resolved_recipe_sha256": _digest_json(resolved_recipe.model_dump(mode="json")),
        "resolved_plan": plan_text,
        "recipe_protocol": _protocol(resolved_recipe),
        "modelopt_config": actual_cfg,
        "modelopt_config_sha256": _digest_json(actual_cfg),
        "modelopt_config_targets": {"native_selector": resolved_recipe.algorithm},
        "package": {"name": "nvidia-modelopt", "required_version": MODEL_OPT_VERSION},
        "source": {**_source_fingerprint(), "root": str(SOURCE_ROOT)},
        "environment": _environment_fingerprint(),
        "model": _model_fingerprint(recipe.model.source),
        "calibration": _calibration_fingerprint(recipe),
        "output_dir": str(Path(resolved_recipe.export.output_dir).resolve()),
        "execution": "not-started",
        "export_reload": "not validated",
    }
    summary["runtime_fingerprint"] = {
        "source": summary["source"],
        "environment": summary["environment"],
        "package": summary["package"],
    }
    return resolved_recipe, summary, evidence_dir / f"{_arm_name(arm)}-plan.json"


def run(recipe_path: Path, arm: str, evidence_dir: Path, execute: bool) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    recipe, evidence, evidence_path = prepare(recipe_path, arm, evidence_dir)
    output_dir = Path(recipe.export.output_dir).expanduser()
    if output_dir.exists():
        raise RunnerError(f"Refusing to overwrite existing output: {output_dir}")
    evidence_text = json.dumps(evidence, indent=2, default=str) + "\n"
    evidence_path.write_text(evidence_text)
    print(evidence_text, end="")
    if not execute:
        return evidence_path

    if evidence["model"].get("kind") != "local":
        raise RunnerError(
            "--execute requires a local model directory so config and all checkpoint "
            "shards can be hashed"
        )
    _execution_environment_gate()

    version = _modelopt_version()
    if version != MODEL_OPT_VERSION:
        raise RunnerError(f"Expected nvidia-modelopt=={MODEL_OPT_VERSION}, found {version}")
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401
    except ImportError as exc:
        raise RunnerError(
            "ModelOpt import failed: expected modelopt.torch.quantization (mtq)"
        ) from exc
    if evidence["calibration"].get("kind") not in ("local-file", "local-directory"):
        raise RunnerError(
            "--execute requires a local calibration file/directory so its content can be hashed"
        )

    previous_backend = get_backend("modelopt")
    backend: ModelOptBackend = _backend_for_arm(arm)
    register_backend("modelopt", backend)
    evidence["package"]["installed_version"] = version
    evidence["package"].update(_modelopt_fingerprint())
    evidence["runtime_fingerprint"] = {
        "source": evidence["source"],
        "environment": evidence["environment"],
        "package": evidence["package"],
    }
    evidence["execution"] = "started"
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    # QuantPipeline.run is the regular load -> calibration -> quantize -> export path.
    _seed_process(SEED)
    try:
        exported = QuantPipeline(recipe).run(dry_run=False)
    finally:
        register_backend("modelopt", previous_backend)
    model_after = _model_fingerprint(recipe.model.source)
    calibration_after = _calibration_fingerprint(recipe)
    if model_after != evidence["model"]:
        raise RunnerError("model directory changed while the paired arm was running")
    if calibration_after != evidence["calibration"]:
        raise RunnerError(
            "calibration input or referenced images changed while the arm was running"
        )
    batch_report = backend.calibration_fingerprint
    if batch_report["batch_count"] == 0 or not batch_report["batches"]:
        raise RunnerError("no tokenized calibration batches were consumed")
    batch_hashes = [batch["sha256"] for batch in batch_report["batches"]]
    batch_report["ordered_sha256"] = _digest_json(batch_hashes)
    batch_report["unique_batch_count"] = len(set(batch_hashes))
    exported_path = Path(exported)
    if not exported_path.exists():
        raise RunnerError(f"Pipeline returned missing export path: {exported_path}")
    evidence["execution"] = "complete"
    evidence["export"] = _tree_hash(exported_path)
    index_report = _export_index_report(exported_path)
    evidence["export_index_verification"] = index_report
    evidence["export_index_keys_validated"] = bool(index_report["keys_validated"])
    provenance_path = (
        exported_path / "provenance.json"
        if exported_path.is_dir()
        else exported_path.parent / "provenance.json"
    )
    if provenance_path.is_file():
        evidence["provenance"] = json.loads(provenance_path.read_text())
    evidence["export_reload"] = "not validated"
    evidence["calibration_gate"] = backend.runtime_coverage
    evidence["calibration_batches"] = backend.calibration_fingerprint
    evidence["model_after"] = model_after
    evidence["calibration_after"] = calibration_after
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    return evidence_path


def _validate_export_index(exported_path: Path) -> bool:
    return bool(_export_index_report(exported_path)["keys_validated"])


def _safetensors_header(path: Path) -> tuple[set[str], str]:
    """Read and validate only a safetensors header, never tensor payload bytes."""
    if path.suffix.lower() != ".safetensors":
        raise ValueError("unsupported shard format")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        header_length = int.from_bytes(prefix, "little", signed=False)
        file_size = path.stat().st_size
        if header_length <= 0 or header_length > file_size - 8:
            raise ValueError("invalid safetensors header length")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ValueError("truncated safetensors header")
    payload = json.loads(header)
    if not isinstance(payload, dict):
        raise ValueError("safetensors header is not a mapping")
    keys: set[str] = set()
    for key, metadata in payload.items():
        if key == "__metadata__":
            if not isinstance(metadata, dict):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(key, str) or not isinstance(metadata, dict):
            raise ValueError("invalid safetensors tensor entry")
        keys.add(key)
    if not keys:
        raise ValueError("safetensors header contains no tensor keys")
    return keys, _digest_bytes(header)


def _safe_shard_path(root: Path, shard: str) -> Path:
    candidate = (root / shard).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("shard path escapes export directory") from exc
    return candidate


def _export_index_report(exported_path: Path) -> dict[str, Any]:
    root = exported_path if exported_path.is_dir() else exported_path.parent
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return {"keys_validated": False, "index_path": str(index), "reason": "missing"}
    try:
        index_bytes = index.read_bytes()
        payload = json.loads(index_bytes)
        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            return {
                "keys_validated": False,
                "index_path": str(index),
                "index_sha256": _digest_bytes(index_bytes),
                "reason": "invalid-weight-map",
                "validation": "safetensors-header-keys",
            }
        malformed = sorted(
            str(key)
            for key, shard in weight_map.items()
            if not isinstance(key, str) or not key or not isinstance(shard, str) or not shard
        )
        shards = {str(item) for item in weight_map.values() if isinstance(item, str)}
        escaped = []
        shard_paths: dict[str, Path] = {}
        for shard in sorted(shards):
            try:
                shard_paths[shard] = _safe_shard_path(root, shard)
            except ValueError:
                escaped.append(shard)
        missing = sorted(shard for shard, path in shard_paths.items() if not path.is_file())
        unsupported = sorted(
            shard
            for shard, path in shard_paths.items()
            if path.is_file() and path.suffix.lower() != ".safetensors"
        )
        invalid_headers: list[str] = []
        header_hashes: dict[str, str] = {}
        shard_keys: dict[str, set[str]] = {}
        if not malformed and not escaped and not missing and not unsupported:
            for shard in sorted(shards):
                try:
                    keys, header_sha256 = _safetensors_header(shard_paths[shard])
                except (OSError, ValueError, json.JSONDecodeError):
                    invalid_headers.append(shard)
                    continue
                shard_keys[shard] = keys
                header_hashes[shard] = header_sha256
        missing_keys = sorted(
            key
            for key, shard in weight_map.items()
            if isinstance(key, str)
            and isinstance(shard, str)
            and key not in shard_keys.get(shard, set())
        )
        indexed_by_shard = {
            shard: {key for key, assigned in weight_map.items() if assigned == shard}
            for shard in shards
        }
        unindexed_keys = sorted(
            f"{shard}:{key}"
            for shard, keys in shard_keys.items()
            for key in keys - indexed_by_shard.get(shard, set())
        )
        indexed_key_count = len(weight_map)
        validated_key_count = indexed_key_count - len(missing_keys) if not malformed else 0
        return {
            "keys_validated": bool(weight_map)
            and not malformed
            and not escaped
            and not missing
            and not unsupported
            and not invalid_headers
            and not missing_keys
            and not unindexed_keys,
            "index_path": str(index),
            "index_sha256": _digest_bytes(index_bytes),
            "validation": "safetensors-header-keys",
            "header_keys_validated": bool(shard_keys)
            and not malformed
            and not escaped
            and not missing
            and not unsupported
            and not invalid_headers,
            "key_count": indexed_key_count,
            "validated_key_count": validated_key_count,
            "shard_count": len(shards),
            "missing_shards": missing,
            "escaped_shards": escaped,
            "missing_keys": missing_keys,
            "unindexed_keys": unindexed_keys,
            "malformed_assignments": malformed,
            "unsupported_shards": unsupported,
            "invalid_headers": invalid_headers,
            "validated_shards": sorted(shard_keys),
            "header_sha256": header_hashes,
        }
    except (OSError, ValueError, TypeError):
        return {
            "keys_validated": False,
            "index_path": str(index),
            "reason": "invalid",
            "validation": "safetensors-header-keys",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--arm", choices=("static-max", "static-MSE"))
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--attach-responses", type=Path, help="Attach sanitized metrics from a response bundle"
    )
    parser.add_argument(
        "--evidence-file",
        type=Path,
        help="Completed arm evidence to update with --attach-responses",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Load a fresh BF16 model and run PTQ"
    )
    args = parser.parse_args()
    try:
        if args.attach_responses is not None:
            if (
                args.evidence_file is None
                or args.recipe
                or args.arm
                or args.evidence_dir
                or args.execute
            ):
                parser.error(
                    "--attach-responses requires --evidence-file and cannot be combined with "
                    "PTQ options"
                )
            result = attach_response_evidence(
                args.evidence_file.resolve(), args.attach_responses.resolve()
            )
        else:
            if (
                args.evidence_file is not None
                or args.recipe is None
                or args.arm is None
                or args.evidence_dir is None
            ):
                parser.error("PTQ mode requires --recipe, --arm, and --evidence-dir")
            result = run(args.recipe.resolve(), args.arm, args.evidence_dir.resolve(), args.execute)
    except Exception as exc:
        print(f"runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"evidence: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
