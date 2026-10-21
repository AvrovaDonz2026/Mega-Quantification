"""Offline lifecycle tests for the MLP pilot supervisor."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnostics/gdn_parity_20261019/mlp-pilot/supervisor.py"
)


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("mlp_pilot_supervisor_test", SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_controller(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[sys.argv.index('--root') + 1])
root.mkdir(parents=True, exist_ok=False)
if os.environ.get('MLP_SUPERVISOR_CONTROLLER_FAIL') == '1':
    raise SystemExit(7)
phase = sys.argv[sys.argv.index('--phase') + 1]
budget = int(sys.argv[sys.argv.index('--budget') + 1])
variant = sys.argv[sys.argv.index('--source-variant') + 1]
selected = (
    ['control']
    if sys.argv[sys.argv.index('--arms') + 1] == 'control'
    else ['control', 'mlp']
)
revision = {
    'all-beta': '49d3acfd4d54b0cf1afffcfeb369c63af4f56499',
    'original': '4a667614d810f7b5570be23eb75956c603a85c3a',
}[variant]
arms = {name: {
    'model_proof': {'passed': True},
    'source_proof': {'commit': revision, 'exact_archive_tree': True},
    'termination_metadata': {'any_present': True},
    'runtime_source_sha256': 'c' * 64,
    'server_identity': {'passed': True,
        'server_info_sha256': 'd' * 64, 'model_info_sha256': 'e' * 64},
    'trials': [
        {'trial': 1, 'output_ids_sha256': 'a' * 64, 'output_tokens': budget,
         'cached_tokens': 0, 'prompt_echo_exact': True, 'logprobs_finite': True},
        {'trial': 2, 'output_ids_sha256': 'a' * 64, 'output_tokens': budget,
         'cached_tokens': 0, 'prompt_echo_exact': True, 'logprobs_finite': True},
    ], 'repeat_stable': True,
    'cleanup': {'removed': True, 'container_id': 'b' * 64},
} for name in selected}
(root / 'report.json').write_text(json.dumps({
    'status': 'completed',
    'comparison': {'both_repeat_stable': True},
    'source_variant': variant, 'source_revision': revision,
    'selected_arms': selected, 'phase': phase,
    'max_new_tokens': budget,
    'trajectory': 'natural', 'prompt_tokens': 288,
    'native_prefix_tokens': 0, 'input_tokens': 288,
    'controller_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
    'model_gate_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
    'protocol': {'context_length': 65536, 'temperature': 0, 'tp_size': 1,
                 'mtp': False, 'disable_radix_cache': True},
    'arms': arms,
}))
""",
        encoding="utf-8",
    )


def _config(module, tmp_path: Path, controller: Path) -> tuple[Path, dict]:
    remote_root = tmp_path / "remote"
    config_path = tmp_path / "supervisor.py"
    config_path.write_text("", encoding="utf-8")
    row = {
        "Id": "container-full-id",
        "Image": "sha256:glm-image",
        "Name": "/glm53-flash-tf",
        "Created": "2026-10-20T00:00:00Z",
        "Config": {
            "Image": "sha256:glm-image",
            "Labels": {},
            "Cmd": ["python3", "-m", "tensorfold", "serve", "--tp", "2"],
        },
        "HostConfig": {"RestartPolicy": {"Name": "no", "MaximumRetryCount": 0}},
        "State": {"Running": True},
    }
    canonical = module.canonical_sha({"Config": row["Config"], "HostConfig": row["HostConfig"]})
    config = {
        "root_path": str(remote_root),
        "experiment_prefix": "mlp-pilot-r3-test",
        "controller_path": str(controller),
        "model_gate_path": str(controller),
        "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "model_gate_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "expected_glm_id_full": row["Id"],
        "expected_glm_image": row["Image"],
        "expected_glm_name": row["Name"],
        "expected_config_hostconfig_sha256": canonical,
        "base_command": ["python3", str(controller), "--root", str(tmp_path / "placeholder")],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config_path, row


@pytest.mark.parametrize(
    ("controller_fails", "stop_times_out", "original_control"),
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
    ids=("success-two-phases", "controller-failure", "stop-timeout", "original-control-long"),
)
def test_supervisor_always_attempts_glm_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_fails: bool,
    stop_times_out: bool,
    original_control: bool,
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, row = _config(module, tmp_path, controller)
    if original_control:
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        config.update(source_variant="original", arms="control", phases=["long"])
        config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "docker_inspect", lambda _container: row)
    if controller_fails:
        monkeypatch.setenv("MLP_SUPERVISOR_CONTROLLER_FAIL", "1")
    else:
        monkeypatch.delenv("MLP_SUPERVISOR_CONTROLLER_FAIL", raising=False)

    calls: list[list[str]] = []

    def fake_command(argv: list[str], **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["docker", "stop"] and stop_times_out:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        if argv[:2] in (["docker", "stop"], ["docker", "start"]):
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:1] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(module, "command", fake_command)
    monkeypatch.setattr(
        module,
        "wait_glm_stable",
        lambda _config: {"running": True, "health": None, "observed_seconds": 120.0},
    )
    monkeypatch.setattr(module.signal, "signal", lambda *_args: None)
    result = module.main()

    status_path = Path(module_file).parent / "remote" / "supervisor-status.json"
    status = json.loads(status_path.read_text())
    assert result == (1 if controller_fails or stop_times_out else 0)
    assert [call[:2] for call in calls].count(["docker", "start"]) == 1
    assert status["glm_restore_required"] is False
    if not controller_fails and not stop_times_out:
        assert status["state"] == "completed"
        remote_root = Path(module_file).parent / "remote"
        assert (
            remote_root / "mlp-pilot-r3-test-early" / "report.json"
        ).is_file() is not original_control
        assert (remote_root / "mlp-pilot-r3-test-long" / "report.json").is_file()
        report = json.loads((remote_root / "mlp-pilot-r3-test-long/report.json").read_text())
        assert report["selected_arms"] == (["control"] if original_control else ["control", "mlp"])
        assert report["source_variant"] == ("original" if original_control else "all-beta")
    else:
        assert status["state"] == "failed"


def test_config_selectors_and_argv_are_strict(tmp_path: Path) -> None:
    module = _load_supervisor()
    options = module.parse_run_config(
        {"source_variant": "original", "arms": "control", "phases": ["long"]}
    )
    assert options["source_revision"] == "4a667614d810f7b5570be23eb75956c603a85c3a"
    assert options["selected_arms"] == ["control"]
    argv = module.build_phase_argv(
        ["python3", "controller.py", "--root", "old"],
        tmp_path / "evidence",
        "long",
        49152,
        source_variant="original",
        arms="control",
    )
    assert "--source-variant" in argv and "--arms" in argv
    assert argv[-2:] == ["--trajectory", "natural"]
    with pytest.raises(module.SupervisorError):
        module.build_phase_argv(
            ["controller.py", "--root", "x"],
            tmp_path,
            "long",
            1,
            trajectory="historical-prefix",
        )
    with pytest.raises(module.SupervisorError):
        module.parse_run_config({"phases": ["long", "long"]})
    with pytest.raises(module.SupervisorError):
        module.build_phase_argv(
            ["controller.py", "--root", "x", "--arms", "both"], tmp_path, "long", 1
        )


def test_supervisor_reads_revisions_from_source_lock(tmp_path: Path, monkeypatch) -> None:
    module = _load_supervisor()
    lock = {
        "variants": {
            "original": {"commit": "a" * 40},
            "decode-beta": {"commit": "b" * 40},
            "all-beta": {"commit": "c" * 40},
            "shared-decode": {"commit": "d" * 40},
        }
    }
    lock_path = tmp_path / "source-lock.json"
    lock_path.write_text(json.dumps(lock))
    monkeypatch.setattr(module, "SOURCE_LOCK_PATH", lock_path)

    assert module.parse_run_config({"source_variant": "all-beta", "phases": ["long"]})[
        "source_revision"
    ] == "c" * 40


def test_glm_topology_preflight_requires_single_node_proof() -> None:
    module = _load_supervisor()
    single = {"Config": {"Cmd": ["tensorfold", "serve", "--tp", "2"]}}
    summary = module.preflight_glm_topology(single)
    assert summary["topology_class"] == "single_node"
    assert summary["tp"] == 2

    explicit_single = {
        "Config": {
            "Cmd": ["tensorfold", "serve", "--nnodes", "1", "--node-rank", "0", "--tp", "2"]
        }
    }
    assert module.preflight_glm_topology(explicit_single)["reason"] == "explicit_nnodes_one"

    for command in (
        ["tensorfold", "serve", "--tp", "2", "--rank", "1", "--master", "peer"],
        ["tensorfold", "serve", "--tp", "2", "--nnodes", "2", "--node-rank", "1"],
    ):
        with pytest.raises(module.SupervisorError, match="topology"):
            module.preflight_glm_topology({"Config": {"Cmd": command}})

    with pytest.raises(module.SupervisorError, match="flag"):
        module.preflight_glm_topology(
            {"Config": {"Cmd": ["tensorfold", "serve", "--tp", "2", "--rank"]}}
        )
    with pytest.raises(module.SupervisorError, match="single node"):
        module.preflight_glm_topology(
            {"Config": {"Cmd": ["tensorfold", "serve", "--nnodes", "1", "--rank", "1"]}}
        )
    with pytest.raises(module.SupervisorError, match="command"):
        module.preflight_glm_topology({"Config": {"Image": "glm"}})
    with pytest.raises(module.SupervisorError, match="command"):
        module.preflight_glm_topology(
            {"Config": {"Entrypoint": ["tensorfold", "serve", "--tp", "2"]}}
        )


def test_glm_topology_preflight_accepts_shell_command_without_exposing_values() -> None:
    module = _load_supervisor()
    summary = module.preflight_glm_topology(
        {"Config": {"Cmd": "tensorfold serve --tp=2 --context '65536 tokens'"}}
    )
    assert summary["topology_class"] == "single_node"
    assert "context" not in summary


def test_glm_topology_preflight_unwraps_shell_argv_and_entrypoint() -> None:
    module = _load_supervisor()
    shell_argv = {
        "Config": {
            "Cmd": ["sh", "-c", "python -m tensorfold --tp=2 --rank=1 --master=peer"]
        }
    }
    with pytest.raises(module.SupervisorError, match="topology"):
        module.preflight_glm_topology(shell_argv)

    entrypoint_shell = {
        "Config": {
            "Entrypoint": ["bash", "-lc"],
            "Cmd": "python -m tensorfold --tp 2 --rank 1 --master peer",
        }
    }
    with pytest.raises(module.SupervisorError, match="topology"):
        module.preflight_glm_topology(entrypoint_shell)

    unknown_wrapper = {"Config": {"Cmd": ["opaque-launcher", "--model", "glm"]}}
    with pytest.raises(module.SupervisorError, match="topology"):
        module.preflight_glm_topology(unknown_wrapper)


def test_glm_topology_preflight_rejects_empty_master_and_bad_port() -> None:
    module = _load_supervisor()
    for command, message in (
        (["tensorfold", "serve", "--tp", "2", "--master="], "flag"),
        (["tensorfold", "serve", "--tp", "2", "--master-port", "0"], "port"),
        (["tensorfold", "serve", "--tp", "2", "--master-port", "65536"], "port"),
    ):
        with pytest.raises(module.SupervisorError, match=message):
            module.preflight_glm_topology({"Config": {"Cmd": command}})


def test_main_rejects_ambiguous_glm_before_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, row = _config(module, tmp_path, controller)
    row["Config"]["Cmd"] = ["tensorfold", "serve", "--tp", "2", "--rank", "1", "--master", "peer"]
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["expected_config_hostconfig_sha256"] = module.canonical_sha(
        {"Config": row["Config"], "HostConfig": row["HostConfig"]}
    )
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "docker_inspect", lambda _container: row)
    calls: list[list[str]] = []

    def fake_command(argv: list[str], **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(module, "command", fake_command)
    monkeypatch.setattr(module.signal, "signal", lambda *_args: None)
    assert module.main() == 1
    assert not any(argv[:2] in (["docker", "stop"], ["docker", "start"]) for argv in calls)


def test_valid_report_rejects_fake_hashes_and_proofs(tmp_path: Path) -> None:
    module = _load_supervisor()
    report = {
        "status": "completed",
        "comparison": {"both_repeat_stable": True},
        "source_variant": "original",
        "source_revision": "4a667614d810f7b5570be23eb75956c603a85c3a",
        "selected_arms": ["control"],
        "phase": "long",
        "max_new_tokens": 49152,
        "prompt_tokens": 288,
        "native_prefix_tokens": 0,
        "input_tokens": 288,
        "protocol": {
            "context_length": 65536,
            "temperature": 0,
            "tp_size": 1,
            "mtp": False,
            "disable_radix_cache": True,
        },
        "arms": {
            "control": {
                "repeat_stable": True,
                "trials": [
                    {
                        "trial": 1,
                        "output_ids_sha256": "a" * 64,
                        "output_tokens": 1,
                        "cached_tokens": 0,
                        "prompt_echo_exact": True,
                        "logprobs_finite": True,
                    },
                    {
                        "trial": 2,
                        "output_ids_sha256": "a" * 64,
                        "output_tokens": 1,
                        "cached_tokens": 0,
                        "prompt_echo_exact": True,
                        "logprobs_finite": True,
                    },
                ],
                "cleanup": {"removed": True, "container_id": "b" * 64},
                "model_proof": {"passed": True},
                "source_proof": {
                    "commit": "4a667614d810f7b5570be23eb75956c603a85c3a",
                    "exact_archive_tree": True,
                },
                "termination_metadata": {"any_present": True},
                "server_identity": {
                    "passed": True,
                    "server_info_sha256": "d" * 64,
                    "model_info_sha256": "e" * 64,
                },
            }
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    expected = {
        "source_variant": "original",
        "source_revision": report["source_revision"],
        "selected_arms": ["control"],
        "phase": "long",
        "max_new_tokens": 49152,
    }
    assert not module.valid_report(path, expected)
    report["arms"]["control"]["runtime_source_sha256"] = "c" * 64
    report["controller_sha256"] = "d" * 64
    report["model_gate_sha256"] = "e" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    assert module.valid_report(path, expected)
    report.pop("trajectory", None)
    path.write_text(json.dumps(report), encoding="utf-8")
    assert module.valid_report(path, expected)
    report["trajectory"] = "historical-prefix"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not module.valid_report(path, expected)
    report["trajectory"] = "natural"
    for field, bad in [
        ("trial", 1),
        ("output_tokens", 0),
        ("output_tokens", 2),
        ("output_tokens", 49153),
        ("output_tokens", True),
        ("cached_tokens", 99),
        ("prompt_echo_exact", False),
        ("logprobs_finite", False),
        ("output_ids_sha256", "z" * 64),
    ]:
        row = report["arms"]["control"]["trials"][1]
        good = row[field]
        row[field] = bad
        path.write_text(json.dumps(report))
        assert not module.valid_report(path, expected), (field, bad)
        row[field] = good

    for field, bad in [
        ("prompt_tokens", 287),
        ("prompt_tokens", True),
        ("native_prefix_tokens", 1),
        ("native_prefix_tokens", True),
        ("input_tokens", 287),
        ("input_tokens", True),
    ]:
        good = report[field]
        report[field] = bad
        path.write_text(json.dumps(report), encoding="utf-8")
        assert not module.valid_report(path, expected), (field, bad)
        report[field] = good


def test_invalid_config_does_not_touch_containers(tmp_path, monkeypatch):
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, _ = _config(module, tmp_path, controller)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["phases"] = ["long", "long"]
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid config reached Docker or cleanup")

    monkeypatch.setattr(module, "command", unexpected)
    monkeypatch.setattr(module, "cleanup_experiment_containers", unexpected)
    assert module.main() == 1


def test_original_control_cleanup_and_fallback_scope(tmp_path, monkeypatch):
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, glm = _config(module, tmp_path, controller)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(source_variant="original", arms="control", phases=["long"])
    config_path.write_text(json.dumps(config))
    options = module.parse_run_config(config)
    module._ACTIVE_SOURCE_REVISIONS = frozenset({options["source_revision"]})
    module._ACTIVE_JOBS = options["jobs"]
    base = {
        "Image": module.IMAGE,
        "Created": "2026-10-20T00:00:00Z",
        "Config": {
            "Labels": {
                "io.megaquant.parity-v7.owner": module.OWNER,
                "io.megaquant.parity-v7.job": "mlp-pilot-control",
                "io.megaquant.parity-v7.source": options["source_revision"],
            }
        },
    }
    rows = {"eligible": base}
    for name, field, value in [
        ("all-beta", "source", module.SOURCE),
        ("mlp", "job", "mlp-pilot-mlp"),
        ("unrelated", "owner", "someone-else"),
    ]:
        row = copy.deepcopy(base)
        row["Config"]["Labels"][f"io.megaquant.parity-v7.{field}"] = value
        rows[name] = row
    rows["old"] = dict(base, Created="1970-01-01T00:00:00Z")
    rows["other-image"] = dict(base, Image="sha256:other")
    monkeypatch.setattr(module, "container_ids_by_owner", lambda: list(rows))
    monkeypatch.setattr(
        module, "docker_inspect", lambda cid: glm if cid == glm["Id"] else rows[cid]
    )
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module, "command", command)
    monkeypatch.setattr(
        module,
        "wait_glm_stable",
        lambda _config: {"running": True, "health": None, "observed_seconds": 120.0},
    )
    assert module.cleanup_experiment_containers(1) == 1
    assert calls == [["docker", "rm", "-f", "eligible"]]
    calls.clear()
    root = Path(config["root_path"])
    root.mkdir()
    (root / "supervisor-status.json").write_text(
        json.dumps(
            {
                "glm_restore_required": True,
                "supervisor_started_at_epoch": 1,
            }
        )
    )
    monkeypatch.setitem(sys.modules, "supervisor", module)
    spec = importlib.util.spec_from_file_location(
        "fallback_test", SUPERVISOR_PATH.with_name("fallback_restore.py")
    )
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    monkeypatch.setattr(fallback, "__file__", str(tmp_path / "fallback_restore.py"))
    assert fallback.main() == 0
    assert calls == [["docker", "rm", "-f", "eligible"], ["docker", "start", glm["Id"]]]
    status = json.loads((root / "supervisor-status.json").read_text())
    assert status["glm_restore_required"] is False
    assert status["fallback_restore"]["observed_seconds"] == 120.0


def test_fallback_stability_failure_keeps_restore_required(tmp_path, monkeypatch):
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, glm = _config(module, tmp_path, controller)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(source_variant="original", arms="control", phases=["long"])
    config_path.write_text(json.dumps(config))
    options = module.parse_run_config(config)
    module._ACTIVE_SOURCE_REVISIONS = frozenset({options["source_revision"]})
    module._ACTIVE_JOBS = options["jobs"]
    root = Path(config["root_path"])
    root.mkdir()
    (root / "supervisor-status.json").write_text(
        json.dumps({"glm_restore_required": True, "supervisor_started_at_epoch": 1})
    )
    monkeypatch.setitem(sys.modules, "supervisor", module)
    spec = importlib.util.spec_from_file_location(
        "fallback_failure_test", SUPERVISOR_PATH.with_name("fallback_restore.py")
    )
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    monkeypatch.setattr(fallback, "__file__", str(tmp_path / "fallback_restore.py"))
    monkeypatch.setattr(module, "cleanup_experiment_containers", lambda _started: 0)
    monkeypatch.setattr(
        module,
        "command",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    monkeypatch.setattr(module, "docker_inspect", lambda _container: glm)

    def fail_stability(_config):
        raise module.SupervisorError("late exit")

    monkeypatch.setattr(module, "wait_glm_stable", fail_stability)
    assert fallback.main() == 1
    status = json.loads((root / "supervisor-status.json").read_text())
    assert status["glm_restore_required"] is True
    assert status["fallback_restore_error_type"] == "SupervisorError"


def test_wait_glm_stable_rejects_late_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    _module_file, row = _config(module, tmp_path, controller)
    config = json.loads((tmp_path / "config.json").read_text())

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    calls = 0

    def inspect(_container):
        nonlocal calls
        calls += 1
        current = copy.deepcopy(row)
        current["State"] = {"Running": clock.now < 65.0}
        return current

    monkeypatch.setattr(module, "docker_inspect", inspect)
    with pytest.raises(module.SupervisorError, match="exited"):
        module.wait_glm_stable(config, monotonic=clock.monotonic, sleep=clock.sleep)
    assert calls > 1
    assert clock.now == 66.0


def test_wait_glm_stable_accepts_120_seconds_and_health_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    _module_file, row = _config(module, tmp_path, controller)
    config = json.loads((tmp_path / "config.json").read_text())

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(module, "docker_inspect", lambda _container: row)
    snap = module.wait_glm_stable(config, monotonic=clock.monotonic, sleep=clock.sleep)
    assert snap["observed_seconds"] == 120.0

    health_row = copy.deepcopy(row)
    health_row["State"] = {"Running": True, "Health": {"Status": "starting"}}
    monkeypatch.setattr(module, "docker_inspect", lambda _container: health_row)
    with pytest.raises(module.SupervisorError, match="timeout"):
        module.wait_glm_stable(
            config,
            timeout_seconds=8,
            stable_seconds=4,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    missing_status_row = copy.deepcopy(row)
    missing_status_row["State"] = {"Running": True, "Health": {}}
    monkeypatch.setattr(module, "docker_inspect", lambda _container: missing_status_row)
    with pytest.raises(module.SupervisorError, match="timeout"):
        module.wait_glm_stable(
            config,
            timeout_seconds=4,
            stable_seconds=2,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_wait_glm_stable_rejects_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    _module_file, row = _config(module, tmp_path, controller)
    config = json.loads((tmp_path / "config.json").read_text())
    bad_row = copy.deepcopy(row)
    bad_row["Name"] = "/other"
    monkeypatch.setattr(module, "docker_inspect", lambda _container: bad_row)
    with pytest.raises(module.SupervisorError, match="name mismatch"):
        module.wait_glm_stable(config, monotonic=lambda: 0.0, sleep=lambda _seconds: None)
