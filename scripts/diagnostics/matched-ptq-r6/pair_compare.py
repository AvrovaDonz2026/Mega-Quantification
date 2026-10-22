"""Compare two completed R6 arm evidence records without loading a model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _load(root: Path, arm: str) -> dict[str, Any]:
    slug = "static_max" if arm == "static-max" else "static_mse"
    candidates = [
        root / arm / f"{slug}-plan.json",
        root / arm / f"{arm}-plan.json",
        root / arm / "evidence.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    matches = sorted((root / arm).glob("*.json"))
    if len(matches) == 1:
        return json.loads(matches[0].read_text())
    raise RuntimeError(f"missing unambiguous evidence for {arm}")


def _protocol(value: dict[str, Any]) -> Any:
    protocol = value.get("recipe_protocol")
    if protocol is not None:
        return protocol
    recipe = dict(value.get("resolved_recipe", {}))
    recipe.pop("name", None)
    recipe.pop("algorithm", None)
    export = recipe.get("export")
    if isinstance(export, dict):
        export.pop("output_dir", None)
    return recipe


def _inventory(value: dict[str, Any]) -> list[Any]:
    gate = value.get("calibration_gate")
    if not isinstance(gate, dict):
        return []
    inventory = gate.get("quantizers")
    if not isinstance(inventory, list):
        return []
    normalized = []
    for item in inventory:
        if not isinstance(item, dict):
            return []
        normalized.append(
            {
                key: item.get(key)
                for key in (
                    "name",
                    "scope",
                    "calibrator",
                    "calibrator_leaves",
                    "amax",
                    "global_amax",
                )
            }
        )
    return sorted(normalized, key=lambda item: str(item.get("name")))


def _inventory_present(value: dict[str, Any]) -> bool:
    gate = _required_mapping(value, "calibration_gate")
    return (
        isinstance(gate, dict)
        and isinstance(gate.get("quantizers"), list)
        and bool(gate["quantizers"])
    )


def _inventory_structure(value: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        (str(item.get("name")), str(item.get("scope")))
        for item in _inventory(value)
        if item.get("name") and item.get("scope")
    )


def _inventory_valid(value: dict[str, Any], expected_calibrator: str | None = None) -> bool:
    gate = _required_mapping(value, "calibration_gate")
    inventory = gate.get("quantizers") if gate else None
    if not isinstance(inventory, list) or not inventory:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and bool(item["name"])
        and isinstance(item.get("scope"), str)
        and item["scope"] in {"core", "vision", "mtp"}
        and (expected_calibrator is None or item.get("calibrator") == expected_calibrator)
        and (expected_calibrator is None or item.get("calibrator_leaves") == [expected_calibrator])
        and _scale_evidence_valid(item.get("amax"))
        and _scale_evidence_valid(item.get("global_amax"))
        for item in inventory
    )


def _scale_evidence_valid(value: Any) -> bool:
    """Require the complete finite, positive tensor summary emitted by the runner."""
    if not isinstance(value, dict) or not _sha256(value.get("sha256")):
        return False
    count = value.get("count")
    finite_count = value.get("finite_count")
    return (
        type(count) is int
        and count > 0
        and type(finite_count) is int
        and finite_count == count
        and value.get("finite") is True
        and value.get("positive") is True
        and _finite_number(value.get("min"))
        and _finite_number(value.get("max"))
        and 0 < value["min"] <= value["max"]
    )


def _activation_evidence_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not _sha256(value.get("sha256")):
        return False
    count = value.get("count")
    finite_count = value.get("finite_count")
    return (
        type(count) is int
        and count > 0
        and type(finite_count) is int
        and finite_count == count
        and value.get("finite") is True
        and _finite_number(value.get("min"))
        and _finite_number(value.get("max"))
        and value["min"] <= value["max"]
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _arm_algorithm_valid(value: dict[str, Any], arm: str) -> bool:
    expected_selector = "static_max" if arm == "static-max" else "mse"
    expected_runtime = "max" if arm == "static-max" else "mse"
    targets = value.get("modelopt_config_targets")
    provenance = value.get("provenance")
    if not isinstance(targets, dict) or targets.get("native_selector") != expected_selector:
        return False
    if not isinstance(provenance, dict) or provenance.get("algorithm") != expected_selector:
        return False
    runtime = provenance.get("runtime_algorithm")
    if arm == "static-max":
        return runtime == expected_runtime
    return isinstance(runtime, dict) and runtime.get("method") == expected_runtime


def _arm_complete(value: dict[str, Any], arm: str) -> bool:
    return value.get("arm") == arm and value.get("execution") == "complete"


def _quant_cfg(value: dict[str, Any]) -> Any:
    config = value.get("modelopt_config")
    if not isinstance(config, dict):
        return None
    entries = config.get("quant_cfg")
    return entries if isinstance(entries, (dict, list)) and entries else None


def _required_mapping(value: dict[str, Any], key: str) -> dict[str, Any] | None:
    item = value.get(key)
    return item if isinstance(item, dict) and item else None


def _identity(value: dict[str, Any], key: str) -> Any:
    """Return a stable identity section, rejecting absent or incomplete evidence."""
    item = _required_mapping(value, key)
    if item is None:
        return None
    if key == "source" and not _sha256(item.get("sha256")):
        return None
    if key in {"model", "calibration"}:
        if not any(_sha256(item.get(name)) for name in ("sha256", "reference_sha256")):
            return None
    return item


def _runtime_identity(value: dict[str, Any]) -> Any:
    item = _required_mapping(value, "runtime_fingerprint")
    if item is not None:
        if any(
            not isinstance(item.get(name), dict) or not _sha256(item[name].get("sha256"))
            for name in ("source", "environment", "package")
        ):
            return None
        return item
    # Older records can be normalized only when every component is present.
    environment = _required_mapping(value, "environment")
    package = _required_mapping(value, "package")
    if environment is None or package is None:
        return None
    if not _sha256(environment.get("sha256")) or not _sha256(package.get("sha256")):
        return None
    source = _identity(value, "source")
    if source is None:
        return None
    return {
        "environment": environment,
        "package": package,
        "source": source,
    }


def _text_protocol(value: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("text_only_protocol", "evaluation_protocol"):
        protocol = value.get(key)
        if isinstance(protocol, dict):
            break
    else:
        return None
    modalities = protocol.get("modalities")
    if isinstance(protocol.get("temperature"), bool):
        return None
    try:
        temperature = float(protocol.get("temperature"))
    except (TypeError, ValueError):
        return None
    if protocol.get("text_only") is not True or modalities != ["text"] or temperature != 0.0:
        return None
    return protocol


def _ordered_calibration(value: dict[str, Any]) -> Any:
    batches = _required_mapping(value, "calibration_batches")
    if batches is None or not _sha256(batches.get("ordered_sha256")):
        return None
    if not isinstance(batches.get("batches"), list) or not batches["batches"]:
        return None
    if type(batches.get("batch_count")) is not int or batches["batch_count"] != len(
        batches["batches"]
    ):
        return None
    for batch in batches["batches"]:
        if not isinstance(batch, dict) or not _sha256(batch.get("sha256")):
            return None
    return batches


def _activation_hashes(value: dict[str, Any]) -> list[tuple[str, str]] | None:
    gate = _required_mapping(value, "calibration_gate")
    scales = gate.get("activation_scales") if gate else None
    if not isinstance(scales, list) or not scales:
        return None
    result: list[tuple[str, str]] = []
    for item in scales:
        if not isinstance(item, dict) or not item.get("name"):
            return None
        amax = item.get("amax")
        if not _activation_evidence_valid(amax):
            return None
        result.append((str(item["name"]), str(amax["sha256"])))
    return sorted(result)


def _scope_inventory(value: dict[str, Any]) -> dict[str, int] | None:
    gate = _required_mapping(value, "calibration_gate")
    counts = gate.get("scope_counts") if gate else None
    if not isinstance(counts, dict):
        return None
    required = ("core", "vision", "mtp")
    if any(not isinstance(counts.get(key), int) or counts[key] < 0 for key in required):
        return None
    return {key: counts[key] for key in required}


def _trajectory_metrics(value: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("trajectory_metrics", "repetition_metrics", "metrics"):
        metrics = value.get(key)
        if isinstance(metrics, dict):
            break
    else:
        return None
    first = metrics.get("first_divergence", metrics.get("first_divergence_0based", _MISSING))
    cycle = metrics.get("cycle", metrics.get("cycle_metrics", _MISSING))
    if first is _MISSING or cycle is _MISSING or not isinstance(cycle, dict):
        return None
    return {"first_divergence": first, "cycle": cycle}


_MISSING = object()

_PTQ_CHECKS = (
    "arms_completed",
    "static_max_algorithm_valid",
    "static_mse_algorithm_valid",
    "static_max_scale_evidence_valid",
    "static_mse_scale_evidence_valid",
    "matched_protocol",
    "quant_cfg_equal",
    "source_fingerprint_equal",
    "model_fingerprint_equal",
    "calibration_fingerprint_equal",
    "runtime_fingerprint_equal",
    "calibration_order_equal",
    "quantizer_inventory_equal",
    "scope_inventory_equal",
    "activation_amax_hashes_equal",
    "export_index_keys_validated",
)


def _export_index(value: dict[str, Any]) -> dict[str, Any] | None:
    report = value.get("export_index_verification")
    if isinstance(report, dict):
        if (
            report.get("keys_validated") is not True
            or report.get("header_keys_validated") is not True
            or report.get("validation") != "safetensors-header-keys"
            or not _sha256(report.get("index_sha256"))
        ):
            return None
        if type(report.get("key_count")) is not int or report["key_count"] <= 0:
            return None
        if (
            type(report.get("validated_key_count")) is not int
            or report["validated_key_count"] != report["key_count"]
        ):
            return None
        if type(report.get("shard_count")) is not int or report["shard_count"] <= 0:
            return None
        errors = (
            "missing_shards",
            "escaped_shards",
            "missing_keys",
            "unindexed_keys",
            "malformed_assignments",
            "unsupported_shards",
            "invalid_headers",
        )
        if any(report.get(key) != [] for key in errors):
            return None
        shards = report.get("validated_shards")
        hashes = report.get("header_sha256")
        if (
            not isinstance(shards, list)
            or len(shards) != report["shard_count"]
            or any(not isinstance(shard, str) or not shard for shard in shards)
            or len(set(shards)) != len(shards)
            or not isinstance(hashes, dict)
            or set(hashes) != set(shards)
            or any(not _sha256(sha) for sha in hashes.values())
        ):
            return None
        return report
    if value.get("export_index_keys_validated") is True:
        # Boolean-only records are not sufficient evidence for a matched run.
        return None
    return None


_NATIVE_RESPONSE_FIELDS = (
    "response_protocol_sha256",
    "response_thinking_enabled",
    "rendered_prompt_ids_sha256",
    "response_server_model_path_sha256",
    "response_loaded_weights_verified",
)


def _response_prompt(value: dict[str, Any]) -> str | None:
    observation = _required_mapping(value, "response_observation")
    if observation is None or not _sha256(observation.get("prompt_ids_sha256")):
        return None
    count = observation.get("response_count")
    lengths = observation.get("completion_token_counts")
    hashes = observation.get("output_ids_sha256")
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(lengths, list)
        or len(lengths) != count
        or any(type(length) is not int or length < 0 for length in lengths)
        or not isinstance(hashes, list)
        or len(hashes) != count
        or any(not _sha256(sha) for sha in hashes)
    ):
        return None
    return observation["prompt_ids_sha256"]


def _native_response_protocol(value: dict[str, Any]) -> tuple[str, bool] | None:
    """Validate native-client provenance without attesting loaded weight bytes."""
    if (
        not _sha256(value.get("response_protocol_sha256"))
        or type(value.get("response_thinking_enabled")) is not bool
        or not _sha256(value.get("rendered_prompt_ids_sha256"))
        or value["rendered_prompt_ids_sha256"] != _response_prompt(value)
        or not _sha256(value.get("response_server_model_path_sha256"))
        or type(value.get("response_loaded_weights_verified")) is not bool
    ):
        return None
    return value["response_protocol_sha256"], value["response_thinking_enabled"]


def compare(max_evidence: dict[str, Any], mse_evidence: dict[str, Any]) -> dict[str, Any]:
    max_protocol = _text_protocol(max_evidence)
    mse_protocol = _text_protocol(mse_evidence)
    max_trajectory = _trajectory_metrics(max_evidence)
    mse_trajectory = _trajectory_metrics(mse_evidence)
    max_index = _export_index(max_evidence)
    mse_index = _export_index(mse_evidence)
    max_scope = _scope_inventory(max_evidence)
    mse_scope = _scope_inventory(mse_evidence)
    max_activation = _activation_hashes(max_evidence)
    mse_activation = _activation_hashes(mse_evidence)
    max_calibration = _ordered_calibration(max_evidence)
    mse_calibration = _ordered_calibration(mse_evidence)
    max_prompt = _response_prompt(max_evidence)
    mse_prompt = _response_prompt(mse_evidence)
    native_fields_present = any(
        key in evidence
        for evidence in (max_evidence, mse_evidence)
        for key in _NATIVE_RESPONSE_FIELDS
    )
    max_native_protocol = _native_response_protocol(max_evidence)
    mse_native_protocol = _native_response_protocol(mse_evidence)
    checks = {
        "arms_completed": _arm_complete(max_evidence, "static-max")
        and _arm_complete(mse_evidence, "static-MSE"),
        "static_max_algorithm_valid": _arm_algorithm_valid(max_evidence, "static-max"),
        "static_mse_algorithm_valid": _arm_algorithm_valid(mse_evidence, "static-MSE"),
        "static_max_scale_evidence_valid": _inventory_valid(max_evidence, "MaxCalibrator"),
        "static_mse_scale_evidence_valid": _inventory_valid(mse_evidence, "NVFP4MSECalibrator"),
        "matched_protocol": _protocol(max_evidence) == _protocol(mse_evidence),
        "quant_cfg_equal": _quant_cfg(max_evidence) is not None
        and _quant_cfg(max_evidence) == _quant_cfg(mse_evidence),
        "source_fingerprint_equal": _identity(max_evidence, "source") is not None
        and _identity(max_evidence, "source") == _identity(mse_evidence, "source"),
        "model_fingerprint_equal": _identity(max_evidence, "model") is not None
        and _identity(max_evidence, "model") == _identity(mse_evidence, "model"),
        "calibration_fingerprint_equal": _identity(max_evidence, "calibration") is not None
        and _identity(max_evidence, "calibration") == _identity(mse_evidence, "calibration"),
        "runtime_fingerprint_equal": _runtime_identity(max_evidence) is not None
        and _runtime_identity(max_evidence) == _runtime_identity(mse_evidence),
        "text_only_protocol_equal": max_protocol is not None and max_protocol == mse_protocol,
        "calibration_order_equal": max_calibration is not None
        and max_calibration == mse_calibration,
        "quantizer_inventory_equal": _inventory_valid(max_evidence)
        and _inventory_valid(mse_evidence)
        and bool(_inventory_structure(max_evidence))
        and _inventory_structure(max_evidence) == _inventory_structure(mse_evidence),
        "scope_inventory_equal": max_scope is not None and max_scope == mse_scope,
        "activation_amax_hashes_equal": max_activation is not None
        and max_activation == mse_activation,
        "export_index_keys_validated": max_index is not None and mse_index is not None,
        "trajectory_metrics_present": max_trajectory is not None and mse_trajectory is not None,
        "response_prompt_hashes_equal": max_prompt is not None and max_prompt == mse_prompt,
        "native_response_protocol_equal": max_native_protocol is not None
        and max_native_protocol == mse_native_protocol,
    }
    activation_confounder = not checks["activation_amax_hashes_equal"]
    ptq_comparable = all(checks[key] for key in _PTQ_CHECKS) and not activation_confounder
    # Older offline attachments carry prompt observations but no native-client
    # controls. Keep that format usable, explicitly identified as such. Any
    # native field opts the pair into complete native provenance validation.
    response_controls_matched = (
        not native_fields_present or checks["native_response_protocol_equal"]
    )
    replay_ready = (
        checks["text_only_protocol_equal"]
        and checks["trajectory_metrics_present"]
        and checks["response_prompt_hashes_equal"]
        and response_controls_matched
    )
    return {
        "schema": "matched-ptq-r6-1022",
        "checks": checks,
        "activation_confounder": activation_confounder,
        "mse_search": "metadata-observed-only",
        "export_reload": "not validated",
        "cross_arm_token_comparison": "not_performed",
        "trajectory_metrics": {"static-max": max_trajectory, "static-MSE": mse_trajectory},
        "serving_ready": False,
        "response_loaded_weights_verified": False,
        "replay_provenance": (
            "native-client"
            if checks["native_response_protocol_equal"]
            else "incomplete-or-mixed-native"
            if native_fields_present
            else "offline-attachment"
        ),
        "matched_protocol": checks["matched_protocol"]
        and checks["quant_cfg_equal"]
        and checks["calibration_order_equal"],
        "ptq_comparable": ptq_comparable,
        "replay_ready": replay_ready,
        "comparable": ptq_comparable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(
        _load(args.evidence_root, "static-max"), _load(args.evidence_root, "static-MSE")
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["ptq_comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
