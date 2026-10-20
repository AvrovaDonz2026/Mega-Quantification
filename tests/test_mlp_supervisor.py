"""Offline lifecycle tests for the MLP pilot supervisor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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
arms = {name: {'trials': [{}, {}], 'cleanup': {'removed': True}}
        for name in ('control', 'mlp')}
(root / 'report.json').write_text(json.dumps({
    'status': 'completed',
    'comparison': {'both_repeat_stable': True},
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
        "Config": {"Image": "sha256:glm-image", "Labels": {}},
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
    ("controller_fails", "stop_times_out"),
    [(False, False), (True, False), (False, True)],
    ids=("success-two-phases", "controller-failure", "stop-timeout"),
)
def test_supervisor_always_attempts_glm_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_fails: bool,
    stop_times_out: bool,
) -> None:
    module = _load_supervisor()
    controller = tmp_path / "controller.py"
    _write_fake_controller(controller)
    module_file, row = _config(module, tmp_path, controller)
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
        assert (remote_root / "mlp-pilot-r3-test-early" / "report.json").is_file()
        assert (remote_root / "mlp-pilot-r3-test-long" / "report.json").is_file()
    else:
        assert status["state"] == "failed"
