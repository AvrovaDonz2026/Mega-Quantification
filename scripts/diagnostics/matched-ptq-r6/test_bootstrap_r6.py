from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bootstrap_r6", HERE / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bootstrap)
maintenance_spec = importlib.util.spec_from_file_location("maintenance_r6", HERE / "maintenance.py")
maintenance = importlib.util.module_from_spec(maintenance_spec)
assert maintenance_spec.loader is not None
maintenance_spec.loader.exec_module(maintenance)
supervisor_spec = importlib.util.spec_from_file_location(
    "pair_supervisor", HERE / "pair_supervisor.py"
)
supervisor = importlib.util.module_from_spec(supervisor_spec)
assert supervisor_spec.loader is not None
sys.modules[supervisor_spec.name] = supervisor
supervisor_spec.loader.exec_module(supervisor)


def inputs(tmp_path: Path) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n')
    (model / "weights-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "calibration.jsonl").write_text('{"text":"fixture"}\n')
    return {
        "model": model,
        "calibration": calibration,
        "output": tmp_path / "output",
        "evidence": tmp_path / "evidence",
        "image_ref": bootstrap.IMAGE_REF,
        "image_id": bootstrap.IMAGE_ID,
    }


def test_package_is_deterministic_and_contains_required_roots():
    first = bootstrap.build_package()
    second = bootstrap.build_package()
    assert first["archive_bytes"] == second["archive_bytes"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert {
        "runner.py",
        "pair_compare.py",
        "pair_supervisor.py",
        "pyproject.toml",
        "recipe.yaml",
    }.issubset(first["manifest"])
    assert "recipes/.r6-package-marker" in first["manifest"]
    with tarfile.open(fileobj=io.BytesIO(first["archive_bytes"]), mode="r:") as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert all(
            member.uid == 0 and member.gid == 0 and member.mtime == 0
            for member in archive.getmembers()
        )


def test_recipe_rewrite_preserves_calibration_policy_flags():
    text = bootstrap.rewrite_recipe().decode()
    assert "  source: /model\n" in text
    assert "  dataset: /calibration/calibration.jsonl\n" in text
    assert "  output_dir: /output/export\n" in text
    for line in (
        "  num_samples: 256",
        "  max_seq_length: 1024",
        "  batch_size: 1",
        "  seed: 42",
        "  with_images: true",
        "  quantize_vision: true",
        "  quantize_mtp: true",
    ):
        assert line in text


def test_plan_binds_expected_hashes_and_fixed_mount_security(tmp_path):
    args = inputs(tmp_path)
    plan = bootstrap.build_plan(**args)
    assert plan["execution_enabled"] is False
    assert set(plan["expected_sha256"]) == {"source", "model", "model_index", "calibration"}
    assert [m["destination"] for m in plan["mounts"]] == [
        "/workspace",
        "/model",
        "/calibration",
        "/output",
        "/evidence",
    ]
    assert [m["read_only"] for m in plan["mounts"]] == [True, True, True, False, False]
    argv = plan["container_argv"]
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert argv[argv.index("--entrypoint") + 1] == "python3"
    assert argv[argv.index("--workdir") + 1] == "/workspace"
    assert argv[argv.index("--tmpfs") + 1] == "/tmp:rw,exec,nosuid,nodev,size=2g"
    assert argv[argv.index("--env", argv.index("--env") + 1) + 1] == "TRANSFORMERS_OFFLINE=1"
    assert "PYTHONPATH=/workspace/src" in argv
    assert "XDG_CACHE_HOME=/tmp/cache" in argv
    assert "TRITON_CACHE_DIR=/tmp/triton" in argv
    assert argv[argv.index("--arm-timeout") + 1] == "14400"
    assert argv[argv.index("--pair-timeout") + 1] == "28800"
    assert argv[argv.index("--compare-script") + 1] == "/workspace/pair_compare.py"
    assert argv[argv.index(args["image_ref"]) + 1] == "/workspace/pair_supervisor.py"
    assert argv[argv.index("--entrypoint") - 1] == "all"
    assert "--execute" not in argv
    assert plan["offline_environment"]["HF_HUB_OFFLINE"] == "1"
    assert plan["image_id"] == args["image_id"]
    json.dumps({k: v for k, v in plan.items() if k != "container_argv"})


def test_hashes_change_when_model_or_calibration_changes(tmp_path):
    args = inputs(tmp_path)
    first = bootstrap.build_plan(**args)
    (args["model"] / "weights-00001.safetensors").write_bytes(b"changed")
    second = bootstrap.build_plan(**args)
    assert first["expected_sha256"]["model"] != second["expected_sha256"]["model"]
    assert first["expected_sha256"]["model_index"] == second["expected_sha256"]["model_index"]
    (args["calibration"] / "calibration.jsonl").write_text('{"text":"changed"}\n')
    third = bootstrap.build_plan(**args)
    assert second["expected_sha256"]["calibration"] != third["expected_sha256"]["calibration"]


@pytest.mark.parametrize(
    "field,value", [("image_ref", "registry.invalid/megaquant:latest"), ("image_id", "latest")]
)
def test_image_must_be_pinned(tmp_path, field, value):
    args = inputs(tmp_path)
    args[field] = value
    with pytest.raises(bootstrap.BootstrapError, match="image"):
        bootstrap.build_plan(**args)


def test_missing_or_ambiguous_index_fails_closed(tmp_path):
    args = inputs(tmp_path)
    (args["model"] / "model.safetensors.index.json").unlink()
    (args["model"] / "one.index.json").write_text("{}")
    (args["model"] / "two.index.json").write_text("{}")
    with pytest.raises(bootstrap.BootstrapError, match="model index"):
        bootstrap.build_plan(**args)


def test_maintenance_plan_uses_strict_schema_budgets_and_no_duplicate_child(tmp_path):
    args = inputs(tmp_path)
    plan = bootstrap.build_plan(**args)
    maintenance_plan = bootstrap.build_maintenance_plan(plan)
    assert set(maintenance_plan) == {
        "job",
        "state_dir",
        "image_ref",
        "image_id",
        "owner",
        "owner_label_key",
        "job_label_key",
        "create_argv",
        "child_argv",
        "arm_timeout_seconds",
        "experiment_ttl",
        "worker_ttl",
        "execution_enabled",
    }
    assert maintenance_plan["arm_timeout_seconds"] == 4 * 60 * 60
    assert maintenance_plan["experiment_ttl"] == 8 * 60 * 60 + 15 * 60
    assert maintenance_plan["worker_ttl"] == 8 * 60 * 60 + 30 * 60
    assert maintenance_plan["execution_enabled"] is False
    image_position = maintenance_plan["create_argv"].index(maintenance_plan["image_ref"])
    assert maintenance_plan["child_argv"] == maintenance_plan["create_argv"][image_position + 1 :]
    assert maintenance.validate_plan(maintenance_plan) == maintenance_plan


def test_supervisor_cli_flags_are_defined_and_execute_is_runner_internal(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    argv = bootstrap.build_plan(**args)["container_argv"]
    assert "--compare-script" in argv
    assert "--arm-timeout" in argv
    assert "--pair-timeout" in argv
    assert "--execute" not in argv
    captured = {}

    def fake_supervise(**kwargs):
        captured.update(kwargs)
        return {"status": "observation_passed"}

    monkeypatch.setattr(supervisor, "supervise_pair", fake_supervise)
    assert (
        supervisor.main(
            [
                "--recipe",
                "/workspace/recipe.yaml",
                "--runner",
                "/workspace/runner.py",
                "--evidence-root",
                "/evidence",
                "--compare-script",
                "/workspace/pair_compare.py",
                "--arm-timeout",
                "14400",
                "--pair-timeout",
                "28800",
            ]
        )
        == 0
    )
    assert captured["timeout_per_arm"] == 14400
    assert captured["pair_timeout"] == 28800
    with pytest.raises(SystemExit):
        supervisor.main(["--execute"])
