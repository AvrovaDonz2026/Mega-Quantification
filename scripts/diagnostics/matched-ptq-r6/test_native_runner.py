import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r6_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _evidence(algorithm):
    h = "a" * 64
    is_mse = isinstance(algorithm, dict) and algorithm.get("method") == "mse"
    arm = "static-MSE" if is_mse else "static-max"
    calibrator = "NVFP4MSECalibrator" if is_mse else "MaxCalibrator"
    selector = "mse" if is_mse else "static_max"
    runtime_algorithm = algorithm if is_mse else "max"
    evidence = {
        "arm": arm,
        "execution": "complete",
        "recipe_protocol": {"scheme": "nvfp4_w4a8", "model": {"source": "/model"}},
        "modelopt_config": {"quant_cfg": [{"quantizer_name": "*weight_quantizer"}]},
        "calibration_batches": {"ordered_sha256": h, "batch_count": 1, "batches": [{"sha256": h}]},
        "calibration_gate": {
            "quantizers": [
                {
                    "name": "layer.weight_quantizer",
                    "scope": "core",
                    "calibrator": calibrator,
                    "calibrator_leaves": [calibrator],
                    "amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                    "global_amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                },
                {
                    "name": "vision.weight_quantizer",
                    "scope": "vision",
                    "calibrator": calibrator,
                    "calibrator_leaves": [calibrator],
                    "amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                    "global_amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                },
                {
                    "name": "mtp.weight_quantizer",
                    "scope": "mtp",
                    "calibrator": calibrator,
                    "calibrator_leaves": [calibrator],
                    "amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                    "global_amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "positive": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                },
            ],
            "scope_counts": {"core": 1, "vision": 1, "mtp": 1},
            "activation_scales": [
                {
                    "name": "layer.input_quantizer",
                    "amax": {
                        "sha256": h,
                        "count": 1,
                        "finite_count": 1,
                        "finite": True,
                        "min": 1.0,
                        "max": 1.0,
                    },
                }
            ],
        },
        "modelopt_config_targets": {"native_selector": selector},
        "provenance": {"algorithm": selector, "runtime_algorithm": runtime_algorithm},
        "source": {"sha256": h},
        "model": {"kind": "local", "sha256": h},
        "calibration": {"kind": "local-file", "sha256": h},
        "environment": {"sha256": h, "environment": {"HF_HUB_OFFLINE": "1"}},
        "package": {"name": "nvidia-modelopt", "version": "0.47.0", "sha256": h},
        "runtime_fingerprint": {
            "source": {"sha256": h},
            "environment": {"sha256": h, "environment": {"HF_HUB_OFFLINE": "1"}},
            "package": {"name": "nvidia-modelopt", "version": "0.47.0", "sha256": h},
        },
        "text_only_protocol": {
            "text_only": True,
            "modalities": ["text"],
            "temperature": 0,
            "max_tokens": 256,
        },
        "trajectory_metrics": {"first_divergence": 12, "cycle": {"period": 1375, "full_cycles": 1}},
        "response_observation": {
            "response_count": 1,
            "prompt_ids_sha256": h,
            "completion_token_counts": [10],
            "output_ids_sha256": [h],
        },
        "export_index_verification": {
            "keys_validated": True,
            "header_keys_validated": True,
            "validation": "safetensors-header-keys",
            "index_sha256": h,
            "key_count": 3,
            "validated_key_count": 3,
            "shard_count": 1,
            "missing_shards": [],
            "escaped_shards": [],
            "missing_keys": [],
            "unindexed_keys": [],
            "malformed_assignments": [],
            "unsupported_shards": [],
            "invalid_headers": [],
            "validated_shards": ["model.safetensors"],
            "header_sha256": {"model.safetensors": h},
        },
    }
    return evidence


def _ptq_evidence(algorithm="max"):
    evidence = _evidence(algorithm)
    evidence.pop("text_only_protocol")
    evidence.pop("trajectory_metrics")
    evidence.pop("response_observation")
    return evidence


def _native_evidence(algorithm="max"):
    evidence = _evidence(algorithm)
    evidence.update(
        {
            "response_protocol_sha256": "b" * 64,
            "response_thinking_enabled": True,
            "rendered_prompt_ids_sha256": evidence["response_observation"]["prompt_ids_sha256"],
            "response_server_model_path_sha256": "c" * 64,
            "response_loaded_weights_verified": False,
        }
    )
    return evidence


def test_compare_accepts_matched_protocol_and_flags_metadata_only():
    result = runner.compare_arm_evidence(_evidence("max"), _evidence({"method": "mse"}))
    assert result["checks"]["recipe_protocol_equal"]
    assert result["mse_search"] == "metadata-observed-only"
    assert result["checks"]["activation_amax_hashes_equal"]


def test_pair_compare_cli_schema():
    from pair_compare import compare

    result = compare(_evidence("max"), _evidence({"method": "mse"}))
    assert result["schema"] == "matched-ptq-r6-1022"
    assert result["export_reload"] == "not validated"
    assert result["serving_ready"] is False
    assert result["ptq_comparable"] is True
    assert result["replay_ready"] is True


def test_pair_compare_cli_returns_zero_only_when_comparable(monkeypatch):
    import pair_compare

    matching = _ptq_evidence()
    matching_mse = _ptq_evidence({"method": "mse"})
    monkeypatch.setattr(
        pair_compare, "_load", lambda _root, arm: matching if arm == "static-max" else matching_mse
    )
    monkeypatch.setattr(sys, "argv", ["pair_compare", "--evidence-root", "."])
    assert pair_compare.main() == 0


def test_pair_compare_ptq_gate_does_not_require_replay_fields():
    from pair_compare import compare

    result = compare(_ptq_evidence(), _ptq_evidence({"method": "mse"}))
    assert result["ptq_comparable"] is True
    assert result["comparable"] is True
    assert result["replay_ready"] is False
    assert result["checks"]["text_only_protocol_equal"] is False
    assert result["checks"]["trajectory_metrics_present"] is False


def test_pair_compare_cli_rejects_required_check_failures(monkeypatch):
    import pair_compare

    cases = []

    activation_mismatch = _evidence("max")
    activation_mismatch["calibration_gate"]["activation_scales"][0]["amax"]["sha256"] = "different"
    cases.append(activation_mismatch)

    inventory_mismatch = _evidence("max")
    inventory_mismatch["calibration_gate"]["quantizers"].append({"name": "other.weight_quantizer"})
    cases.append(inventory_mismatch)

    export_unvalidated = _evidence("max")
    export_unvalidated["export_index_verification"]["keys_validated"] = False
    cases.append(export_unvalidated)

    for failing_max in cases:
        matching_mse = _evidence({"method": "mse"})
        monkeypatch.setattr(
            pair_compare,
            "_load",
            lambda _root, arm, failing_max=failing_max, matching_mse=matching_mse: (
                failing_max if arm == "static-max" else matching_mse
            ),
        )
        monkeypatch.setattr(sys, "argv", ["pair_compare", "--evidence-root", "."])
        assert pair_compare.main() != 0


def test_pair_compare_requires_identity_but_replay_protocol_is_informational(monkeypatch):
    import pair_compare

    missing = _evidence("max")
    missing.pop("model")
    monkeypatch.setattr(
        pair_compare,
        "_load",
        lambda _root, arm: missing if arm == "static-max" else _evidence({"method": "mse"}),
    )
    monkeypatch.setattr(sys, "argv", ["pair_compare", "--evidence-root", "."])
    assert pair_compare.main() != 0

    missing_protocol = _ptq_evidence()
    monkeypatch.setattr(
        pair_compare,
        "_load",
        lambda _root, arm: (
            missing_protocol if arm == "static-max" else _evidence({"method": "mse"})
        ),
    )
    assert pair_compare.main() == 0


def test_pair_compare_preserves_trajectory_metrics():
    from pair_compare import compare

    result = compare(_evidence("max"), _evidence({"method": "mse"}))
    assert result["checks"]["trajectory_metrics_present"]
    assert result["trajectory_metrics"]["static-max"]["first_divergence"] == 12


def test_pair_compare_marks_replay_ready_only_with_protocol_and_trajectory():
    from pair_compare import compare

    replay_ready = compare(_evidence("max"), _evidence({"method": "mse"}))
    assert replay_ready["replay_ready"] is True

    missing_trajectory = _evidence("max")
    missing_trajectory.pop("trajectory_metrics")
    not_ready = compare(missing_trajectory, _evidence({"method": "mse"}))
    assert not_ready["ptq_comparable"] is True
    assert not_ready["replay_ready"] is False


def test_pair_compare_rejects_wrong_arm_identity_and_incomplete_execution():
    from pair_compare import compare

    wrong = _ptq_evidence()
    wrong["arm"] = "static-max"
    wrong["execution"] = "started"
    result = compare(wrong, _ptq_evidence({"method": "mse"}))
    assert result["checks"]["arms_completed"] is False
    assert result["ptq_comparable"] is False


def test_pair_compare_rejects_both_arms_claiming_max():
    from pair_compare import compare

    max_arm = _ptq_evidence()
    result = compare(max_arm, _ptq_evidence())
    assert result["checks"]["static_mse_algorithm_valid"] is False
    assert result["ptq_comparable"] is False


@pytest.mark.parametrize(
    "field",
    [
        "header_keys_validated",
        "validation",
        "validated_key_count",
        "missing_shards",
        "escaped_shards",
        "missing_keys",
        "unindexed_keys",
        "malformed_assignments",
        "unsupported_shards",
        "invalid_headers",
        "validated_shards",
        "header_sha256",
    ],
)
def test_pair_compare_requires_complete_export_header_report(field):
    from pair_compare import compare

    max_arm = _ptq_evidence()
    max_arm["export_index_verification"].pop(field)
    result = compare(max_arm, _ptq_evidence({"method": "mse"}))
    assert result["checks"]["export_index_keys_validated"] is False
    assert result["ptq_comparable"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("header_keys_validated", False),
        ("validation", "shard-existence"),
        ("key_count", True),
        ("shard_count", True),
        ("validated_key_count", 2),
        ("validated_key_count", True),
        ("index_sha256", "not-a-sha"),
        ("missing_shards", ["missing.safetensors"]),
        ("escaped_shards", ["../escape"]),
        ("missing_keys", ["layer.weight"]),
        ("unindexed_keys", ["extra.weight"]),
        ("malformed_assignments", ["layer.weight"]),
        ("unsupported_shards", ["model.bin"]),
        ("invalid_headers", ["model.safetensors"]),
        ("validated_shards", []),
        ("validated_shards", ["model.safetensors", "model.safetensors"]),
        ("header_sha256", {}),
        ("header_sha256", {"other.safetensors": "a" * 64}),
        ("header_sha256", {"model.safetensors": "not-a-sha"}),
    ],
)
def test_pair_compare_rejects_export_report_contradictions(field, value):
    from pair_compare import compare

    max_arm = _ptq_evidence()
    max_arm["export_index_verification"][field] = value
    result = compare(max_arm, _ptq_evidence({"method": "mse"}))
    assert result["checks"]["export_index_keys_validated"] is False
    assert result["ptq_comparable"] is False


@pytest.mark.parametrize("count", [0, 2, True])
def test_pair_compare_rejects_inconsistent_ordered_calibration_counts(count):
    from pair_compare import compare

    max_arm, mse_arm = _ptq_evidence(), _ptq_evidence({"method": "mse"})
    for arm in (max_arm, mse_arm):
        arm["calibration_batches"]["batch_count"] = count
    result = compare(max_arm, mse_arm)
    assert result["checks"]["calibration_order_equal"] is False
    assert result["ptq_comparable"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "not-a-sha"),
        ("count", True),
        ("finite_count", 0),
        ("finite", False),
        ("positive", False),
        ("min", 0),
        ("min", -1),
        ("min", 2),
        ("max", float("inf")),
        ("max", float("nan")),
    ],
)
@pytest.mark.parametrize("scale", ["amax", "global_amax"])
def test_pair_compare_rejects_incomplete_or_contradictory_scale_summaries(scale, field, value):
    from pair_compare import compare

    max_arm = _ptq_evidence()
    max_arm["calibration_gate"]["quantizers"][0][scale][field] = value
    result = compare(max_arm, _ptq_evidence({"method": "mse"}))
    assert result["checks"]["static_max_scale_evidence_valid"] is False
    assert result["ptq_comparable"] is False


def test_pair_compare_requires_exact_valid_activation_hash_equality():
    from pair_compare import compare

    mse_arm = _ptq_evidence({"method": "mse"})
    mse_arm["calibration_gate"]["activation_scales"][0]["amax"]["sha256"] = "b" * 64
    result = compare(_ptq_evidence(), mse_arm)
    assert result["checks"]["static_mse_scale_evidence_valid"] is True
    assert result["checks"]["activation_amax_hashes_equal"] is False
    assert result["activation_confounder"] is True
    assert result["ptq_comparable"] is False


def test_pair_compare_native_replay_requires_matched_controls_and_prompt():
    from pair_compare import compare

    result = compare(_native_evidence(), _native_evidence({"method": "mse"}))
    assert result["replay_ready"] is True
    assert result["replay_provenance"] == "native-client"
    assert result["checks"]["native_response_protocol_equal"] is True
    assert result["response_loaded_weights_verified"] is False
    assert result["serving_ready"] is False
    assert result["cross_arm_token_comparison"] == "not_performed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_protocol_sha256", "d" * 64),
        ("response_protocol_sha256", "not-a-sha"),
        ("response_thinking_enabled", False),
        ("response_thinking_enabled", 1),
        ("rendered_prompt_ids_sha256", "d" * 64),
        ("response_server_model_path_sha256", "not-a-sha"),
        ("response_loaded_weights_verified", "false"),
    ],
)
def test_pair_compare_rejects_native_replay_control_mismatch(field, value):
    from pair_compare import compare

    mse_arm = _native_evidence({"method": "mse"})
    mse_arm[field] = value
    result = compare(_native_evidence(), mse_arm)
    assert result["checks"]["text_only_protocol_equal"] is True
    assert result["ptq_comparable"] is True
    assert result["replay_ready"] is False
    assert result["replay_provenance"] == "incomplete-or-mixed-native"


@pytest.mark.parametrize(
    "field",
    [
        "response_protocol_sha256",
        "response_thinking_enabled",
        "rendered_prompt_ids_sha256",
        "response_server_model_path_sha256",
        "response_loaded_weights_verified",
    ],
)
def test_pair_compare_rejects_partial_native_provenance(field):
    from pair_compare import compare

    mse_arm = _native_evidence({"method": "mse"})
    mse_arm.pop(field)
    result = compare(_native_evidence(), mse_arm)
    assert result["replay_ready"] is False
    assert result["response_loaded_weights_verified"] is False
    assert result["serving_ready"] is False


def test_pair_compare_does_not_downgrade_mixed_native_and_offline_attachments():
    from pair_compare import compare

    result = compare(_native_evidence(), _evidence({"method": "mse"}))
    assert result["replay_ready"] is False
    assert result["replay_provenance"] == "incomplete-or-mixed-native"


@pytest.mark.parametrize("native", [False, True])
def test_pair_compare_rejects_different_prompts_even_when_protocol_matches(native):
    from pair_compare import compare

    factory = _native_evidence if native else _evidence
    max_arm, mse_arm = factory("max"), factory({"method": "mse"})
    mse_arm["response_observation"]["prompt_ids_sha256"] = "d" * 64
    if native:
        mse_arm["rendered_prompt_ids_sha256"] = "d" * 64
    result = compare(max_arm, mse_arm)
    assert result["checks"]["text_only_protocol_equal"] is True
    assert result["checks"]["response_prompt_hashes_equal"] is False
    assert result["replay_ready"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_ids_sha256", "not-a-sha"),
        ("response_count", True),
        ("response_count", 2),
        ("completion_token_counts", []),
        ("completion_token_counts", [True]),
        ("output_ids_sha256", []),
        ("output_ids_sha256", ["not-a-sha"]),
    ],
)
def test_pair_compare_rejects_incomplete_prompt_observations(field, value):
    from pair_compare import compare

    max_arm, mse_arm = _evidence("max"), _evidence({"method": "mse"})
    for arm in (max_arm, mse_arm):
        arm["response_observation"][field] = value
    result = compare(max_arm, mse_arm)
    assert result["replay_ready"] is False


def test_pair_compare_supports_real_offline_attachment_without_weight_attestation(tmp_path):
    from pair_compare import compare

    arms = []
    for algorithm in ("max", {"method": "mse"}):
        evidence = _ptq_evidence(algorithm)
        evidence["export"] = {"sha256": "e" * 64}
        evidence_path = tmp_path / (evidence["arm"] + "-plan.json")
        response_path = tmp_path / (evidence["arm"] + "-responses.json")
        evidence_path.write_text(json.dumps(evidence))
        response_path.write_text(
            json.dumps(
                {
                    "arm": evidence["arm"],
                    "export_sha256": evidence["export"]["sha256"],
                    "text_only_protocol": {
                        "text_only": True,
                        "modalities": ["text"],
                        "temperature": 0,
                    },
                    "responses": [{"prompt_token_ids": [101, 102], "output_ids": [7, 8, 7, 8]}],
                }
            )
        )
        runner.attach_response_evidence(evidence_path, response_path)
        arms.append(json.loads(evidence_path.read_text()))
    result = compare(*arms)
    assert result["replay_ready"] is True
    assert result["replay_provenance"] == "offline-attachment"
    assert result["checks"]["native_response_protocol_equal"] is False
    assert result["response_loaded_weights_verified"] is False
    assert result["serving_ready"] is False
    assert result["cross_arm_token_comparison"] == "not_performed"
    assert all(arm["trajectory_metrics"]["first_divergence"]["index"] is None for arm in arms)
