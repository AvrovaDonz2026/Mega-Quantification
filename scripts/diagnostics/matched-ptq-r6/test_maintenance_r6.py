import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("maintenance", HERE / "maintenance.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.modules["maintenance"] = m
b_spec = importlib.util.spec_from_file_location("bootstrap_r6_maintenance", HERE / "bootstrap.py")
b = importlib.util.module_from_spec(b_spec)
b_spec.loader.exec_module(b)
w_spec = importlib.util.spec_from_file_location(
    "wait_owned_container", HERE / "wait_owned_container.py"
)
w = importlib.util.module_from_spec(w_spec)
w_spec.loader.exec_module(w)


def plan(tmp_path):
    ref = "registry.invalid/ptq@sha256:" + "a" * 64
    job = "a1b2c3d4e5f61234"
    return {
        "job": job,
        "state_dir": "/var/lib/" + m.OWNER_PREFIX + job,
        "image_ref": ref,
        "image_id": "sha256:" + "b" * 64,
        "owner": m.EXPERIMENT_OWNER,
        "owner_label_key": m.EXPERIMENT_LABEL,
        "job_label_key": m.JOB_LABEL,
        "create_argv": [
            "docker",
            "create",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--name",
            m.OWNER_PREFIX + job,
            "--label",
            m.EXPERIMENT_LABEL + "=" + m.EXPERIMENT_OWNER,
            "--label",
            m.JOB_LABEL + "=" + job,
            ref,
        ],
        "child_argv": ["python3", "/workspace/child.py"],
        "arm_timeout_seconds": 14400,
        "experiment_ttl": 2 * 14400 + 900,
        "worker_ttl": 2 * 14400 + 1800,
        "execution_enabled": False,
    }


class Fake:
    def __init__(self, p):
        self.p, self.row, self.calls, self.states = p, None, [], []

    def service_inspect(self):
        return {
            "exact": True,
            "running": True,
            "restarting": False,
            "oom_killed": False,
            "restart_count": 0,
            "started_at": "t0",
            "health_configured": True,
            "health": "healthy",
        }

    def service_stop(self):
        self.calls.append("service_stop")

    def service_start(self):
        self.calls.append("service_start")

    def attest_image_ref(self, ref):
        return self.p["image_id"]

    def create(self, argv):
        cid = "c" * 64
        self.row = {
            "Id": cid,
            "Image": self.p["image_id"],
            "Name": "/" + m.OWNER_PREFIX + self.p["job"],
            "Config": {
                "Labels": {m.EXPERIMENT_LABEL: m.EXPERIMENT_OWNER, m.JOB_LABEL: self.p["job"]}
            },
        }
        self.calls.append("create")
        return cid

    def start(self, cid):
        self.calls.append("start")

    def inspect(self, cid):
        if self.row is None:
            raise m.DockerAbsent(cid)
        return self.row

    def exists(self, cid):
        return self.row is not None

    def remove(self, cid):
        self.calls.append("remove")
        self.row = None


def test_plan_freezes_ttls_and_strict_argv(tmp_path):
    p = plan(tmp_path)
    m.validate_plan(p)
    p["experiment_ttl"] += 1
    with pytest.raises(m.MaintenanceError):
        m.validate_plan(p)


def test_bootstrap_maintenance_plan_passes_strict_validator(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    (model / "weights.safetensors").write_bytes(b"fixture")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "calibration.jsonl").write_text('{"text":"fixture"}\n')
    outer = b.build_plan(
        model=model,
        calibration=calibration,
        output=tmp_path / "output",
        evidence=tmp_path / "evidence",
        image_ref=b.IMAGE_REF,
        image_id=b.IMAGE_ID,
    )
    maintenance_plan = b.build_maintenance_plan(outer)
    validated = m.validate_plan(maintenance_plan)
    assert validated["execution_enabled"] is False
    assert validated["create_argv"] == outer["container_argv"]
    assert (
        validated["child_argv"]
        == outer["container_argv"][outer["container_argv"].index(outer["image_ref"]) + 1 :]
    )


def test_image_attestation_and_single_owned_container_lifecycle(tmp_path):
    p = plan(tmp_path)
    d = Fake(p)
    owner = m.LocalOwner(p, d, tmp_path / "journal.json")
    owner.arm()
    owner.launch()
    assert d.calls == ["create", "start"]
    owner.cleanup()
    assert d.row is None


def test_image_id_mismatch_fails_before_create(tmp_path):
    p = plan(tmp_path)
    d = Fake(p)
    d.attest_image_ref = lambda ref: "sha256:" + "c" * 64
    owner = m.LocalOwner(p, d, tmp_path / "j.json")
    owner.arm()
    with pytest.raises(m.MaintenanceError, match="attestation"):
        owner.launch()
    assert d.calls == []


def test_foreign_identity_blocks_cleanup(tmp_path):
    p = plan(tmp_path)
    d = Fake(p)
    owner = m.LocalOwner(p, d, tmp_path / "j.json")
    owner.arm()
    owner.launch()
    d.row["Config"]["Labels"][m.JOB_LABEL] = "foreign"
    with pytest.raises(m.MaintenanceError, match="foreign"):
        owner.cleanup()
    assert "remove" not in d.calls


class WaitAdapter:
    def __init__(self, rows):
        self.rows = iter(rows)

    def inspect(self, cid):
        return next(self.rows)


def journal(p):
    return {"plan_sha256": m.digest(p), "experiment_id": "c" * 64}


def row(p, running=True, status=None, code=None):
    return {
        "Id": "c" * 64,
        "Image": p["image_id"],
        "Name": "/" + m.OWNER_PREFIX + p["job"],
        "Config": {"Labels": {m.EXPERIMENT_LABEL: m.EXPERIMENT_OWNER, m.JOB_LABEL: p["job"]}},
        "State": {"Running": running, "Status": status, "ExitCode": code},
    }


def test_wait_running_does_not_pass_then_exited_zero(tmp_path):
    p = plan(tmp_path)
    clock = [0]

    def now():
        return clock[0]

    def sleep(n):
        clock[0] += n

    result = w.wait(
        p,
        journal(p),
        WaitAdapter([row(p), row(p, False, "exited", 0)]),
        timeout=3,
        now=now,
        sleep=sleep,
    )
    assert result["event"] == "SUCCEEDED"


def test_wait_nonzero_exit_and_identity_are_fail_closed(tmp_path):
    p = plan(tmp_path)
    with pytest.raises(m.MaintenanceError, match="failed"):
        w.wait(p, journal(p), WaitAdapter([row(p, False, "exited", 7)]), timeout=1)
    bad = row(p)
    bad["Config"]["Labels"][m.JOB_LABEL] = "other"
    with pytest.raises(m.MaintenanceError, match="identity"):
        w.wait(p, journal(p), WaitAdapter([bad]), timeout=1)

    malformed = journal(p)
    malformed["experiment_id"] = "C" * 64
    with pytest.raises(m.MaintenanceError, match="identity"):
        w.wait(p, malformed, WaitAdapter([]), timeout=1)


def test_wait_timeout_and_unknown_fail_closed(tmp_path):
    p = plan(tmp_path)
    clock = [0]
    with pytest.raises(m.MaintenanceError, match="timeout"):
        w.wait(
            p,
            journal(p),
            WaitAdapter([row(p)] * 5),
            timeout=1,
            now=lambda: clock[0],
            sleep=lambda n: clock.__setitem__(0, clock[0] + n),
        )
    with pytest.raises(m.MaintenanceError, match="unknown"):
        w.wait(p, journal(p), WaitAdapter([row(p, False, "created", None)]), timeout=1)


def test_paired_cleanup_dispatches_both():
    class Remote:
        def __init__(self):
            self.commands = []

        def rpc(self, command):
            self.commands.append(command)
            return {"event": "CLEANED"}

    a, b = Remote(), Remote()
    assert m.paired_rpc([a, b], "CLEANUP", "CLEANED")
    assert a.commands == ["CLEANUP"] and b.commands == ["CLEANUP"]


def test_remote_rpc_retries_only_read_commands_and_bounds_timeout():
    p = plan(None)
    attempts = []

    def flaky(req, timeout):
        attempts.append(req["command"])
        if len(attempts) < 2:
            raise TimeoutError()
        return {"event": "ALIVE", "request_id": req["request_id"]}

    assert m.RemoteRPC(p, flaky).rpc("HEARTBEAT")["event"] == "ALIVE"
    assert attempts == ["HEARTBEAT", "HEARTBEAT"]
    with pytest.raises(m.MaintenanceError, match="invalid_rpc_timeout"):
        m.RemoteRPC(p, flaky).rpc("STATUS", timeout=301)


def test_service_adapter_is_fail_closed_when_snapshot_is_incomplete(tmp_path):
    p = plan(tmp_path)

    class Incomplete(Fake):
        def service_inspect(self):
            return {"exact": True, "running": True}

    with pytest.raises(m.MaintenanceError, match="service_snapshot_incomplete"):
        m.LocalOwner(p, Incomplete(p), tmp_path / "j.json").arm()


def test_serve_rejects_journal_outside_owned_state_dir(tmp_path, monkeypatch):
    p = plan(tmp_path)
    owner = m.LocalOwner(p, Fake(p), tmp_path / "journal.json")
    monkeypatch.setattr(m, "trusted_path", lambda path, regular=False: None)
    with pytest.raises(m.MaintenanceError, match="unsafe_plan"):
        m.serve(p, owner, tmp_path / "worker.sock")


def test_serve_requires_existing_trusted_state_dir(tmp_path, monkeypatch):
    p = plan(tmp_path)
    owner = m.LocalOwner(p, Fake(p), Path(p["state_dir"]) / "journal.json")
    with pytest.raises(m.MaintenanceError, match="unsafe_plan"):
        m.serve(p, owner, tmp_path / "worker.sock")


def test_journal_state_validation_rejects_malformed_lifecycle_fields(tmp_path):
    p = plan(tmp_path)
    expected = m.digest(p)
    bad = {
        "plan_sha256": expected,
        "experiment_id": "not-a-container",
        "create_intent": True,
        "armed": True,
        "cleanup": "not-started",
        "restoration": "required",
        "rpc_nonces": [],
    }
    with pytest.raises(m.MaintenanceError, match="worker_protocol"):
        m._validate_journal_state(bad, expected)


def test_serve_recovers_armed_journal_before_binding_socket(tmp_path, monkeypatch):
    p = plan(tmp_path)
    owner = m.LocalOwner(p, Fake(p), Path(p["state_dir"]) / "journal.json")
    owner.state["armed"] = True
    called = []
    monkeypatch.setattr(m, "_validate_owner_storage", lambda plan, owner: None)
    monkeypatch.setattr(m, "lock_owner", lambda job: open(tmp_path / "lock", "w+"))
    monkeypatch.setattr(
        m, "_load_journal", lambda path, expected: {**owner.state, "plan_sha256": expected}
    )
    monkeypatch.setattr(
        owner, "recover", lambda: called.append(True) or {"event": "VERIFIED_CLOSED"}
    )
    m.serve(p, owner, tmp_path / "worker.sock")
    assert called == [True] and not (tmp_path / "worker.sock").exists()


def test_socket_path_is_never_unlinked_or_replaced(tmp_path, monkeypatch):
    p = plan(tmp_path)
    sock = tmp_path / "worker.sock"
    sock.write_text("sentinel")
    monkeypatch.setattr(m, "lock_owner", lambda job: open(tmp_path / "lock", "w+"))
    monkeypatch.setattr(m, "trusted_path", lambda path, regular=False: None)
    monkeypatch.setattr(m, "_validate_owner_storage", lambda plan, owner: None)
    with pytest.raises(m.MaintenanceError, match="socket_path_exists"):
        m.serve(p, m.LocalOwner(p, Fake(p), tmp_path / "journal.json"), sock, lease=1)
    assert sock.read_text() == "sentinel"
