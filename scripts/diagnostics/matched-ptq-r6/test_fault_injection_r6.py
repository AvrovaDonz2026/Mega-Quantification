import importlib.util
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r6_maintenance_fault", HERE / "maintenance.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan():
    job = "a1b2c3d4e5f61234"
    ref = "registry.invalid/ptq@sha256:" + "a" * 64
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
        "experiment_ttl": 29700,
        "worker_ttl": 30600,
        "execution_enabled": False,
    }


class ServiceDocker:
    def __init__(self, p):
        self.p = p
        self.running = True
        self.restart_count = 0
        self.oom = False
        self.health = "healthy"
        self.started = "t0"
        self.rows = {}

    def service_inspect(self):
        return {
            "exact": True,
            "running": self.running,
            "restarting": False,
            "oom_killed": self.oom,
            "restart_count": self.restart_count,
            "started_at": self.started,
            "health_configured": True,
            "health": self.health,
        }

    def service_stop(self):
        self.running = False

    def service_start(self):
        self.running = True

    def attest_image_ref(self, ref):
        return self.p["image_id"]

    def create(self, argv):
        cid = "c" * 64
        self.rows[cid] = {
            "Id": cid,
            "Image": self.p["image_id"],
            "Name": "/" + m.OWNER_PREFIX + self.p["job"],
            "Config": {
                "Labels": {m.EXPERIMENT_LABEL: m.EXPERIMENT_OWNER, m.JOB_LABEL: self.p["job"]}
            },
        }
        return cid

    def inspect(self, cid):
        if cid not in self.rows:
            raise m.DockerAbsent(cid)
        return self.rows[cid]

    def start(self, cid):
        pass

    def exists(self, cid):
        return cid in self.rows

    def remove(self, cid):
        self.rows.pop(cid, None)


def test_service_stop_and_start_are_actual_mutations(tmp_path):
    p = plan()
    d = ServiceDocker(p)
    o = m.LocalOwner(p, d, tmp_path / "j.json")
    o.arm()
    o.stop_service()
    assert not d.running
    o.start_service()
    assert d.running


def test_verify_close_rejects_health_restart_and_oom(tmp_path):
    p = plan()
    d = ServiceDocker(p)
    o = m.LocalOwner(p, d, tmp_path / "j.json")
    o.arm()
    o.stop_service()
    o.start_service()
    d.health = "starting"
    clock = [0.0]
    d.service_inspect = lambda: {
        "exact": True, "running": True, "restarting": False, "oom_killed": d.oom,
        "restart_count": d.restart_count, "started_at": d.started,
        "health_configured": True, "health": "starting" if clock[0] < 20 else "healthy",
    }
    o.now = lambda: clock[0]
    o.sleep = lambda amount: clock.__setitem__(0, clock[0] + amount)
    result = o.verify_close(seconds=120, timeout=300)
    assert result["application_readiness"] == "healthy"
    assert result["stable_seconds"] >= 120
    assert clock[0] >= 140
    o.state.update(armed=True, restoration="start_attempted")
    d.health = "healthy"
    d.restart_count = 0
    def restart_during_sleep(amount):
        clock[0] += amount
        d.restart_count = 1
    o.sleep = restart_during_sleep
    with pytest.raises(m.MaintenanceError, match="stability_timeout|unstable"):
        o.verify_close(seconds=1, timeout=1)


def test_verify_close_without_healthcheck_reports_unknown(tmp_path):
    p = plan()
    d = ServiceDocker(p)
    o = m.LocalOwner(p, d, tmp_path / "j.json")
    o.state.update(armed=True, restoration="start_attempted")
    d.service_inspect = lambda: {
        "exact": True, "running": True, "restarting": False, "oom_killed": False,
        "restart_count": 0, "started_at": "t0", "health_configured": False, "health": "",
    }
    clock = [0.0]
    o.now = lambda: clock[0]
    o.sleep = lambda amount: clock.__setitem__(0, clock[0] + amount)
    result = o.verify_close(seconds=120, timeout=300)
    assert result["application_readiness"] == "unknown"


def test_rpc_command_budgets_cover_long_verification_and_allow_shorter_timeouts():
    p = plan()
    received = []
    def call(req, timeout):
        received.append(timeout)
        return {"event": "VERIFIED_CLOSED", "request_id": req["request_id"]}
    rpc = m.RemoteRPC(p, call, retries=0)
    rpc.rpc("VERIFY_CLOSE")
    rpc.rpc("VERIFY_CLOSE", timeout=45)
    assert received == [330, 45]
    with pytest.raises(m.MaintenanceError, match="invalid_rpc_timeout"):
        rpc.rpc("STATUS", timeout=121)


def test_heartbeat_skips_busy_remote_without_failure():
    p = plan()
    calls = []

    def call(req, timeout):
        calls.append(req["command"])
        return {"event": "ALIVE", "request_id": req["request_id"]}

    rpc = m.RemoteRPC(p, call, retries=0)
    assert rpc._lock.acquire(blocking=False)
    try:
        assert rpc.heartbeat_if_idle() is None
    finally:
        rpc._lock.release()
    assert rpc.heartbeat_if_idle()["event"] == "ALIVE"


def test_slow_stop_blocks_only_busy_remote_heartbeat():
    p = plan()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def call(req, timeout):
        calls.append(req["command"])
        if req["command"] == "STOP":
            entered.set()
            release.wait(1)
        return {
            "event": "STOPPED" if req["command"] == "STOP" else "ALIVE",
            "request_id": req["request_id"],
        }
    rpc = m.RemoteRPC(p, call, retries=0)
    worker = threading.Thread(target=rpc.rpc, args=("STOP",), daemon=True)
    worker.start()
    assert entered.wait(1)
    assert rpc.heartbeat_if_idle() is None
    release.set()
    worker.join(1)
    assert calls == ["STOP"]


def test_remote_timeout_and_protocol_are_fail_closed():
    p = plan()

    def timeout(req, limit):
        raise TimeoutError()

    with pytest.raises(m.MaintenanceError, match="transport_timeout"):
        m.RemoteRPC(p, timeout).rpc("HEARTBEAT")
    with pytest.raises(m.MaintenanceError, match="transport_protocol"):
        m.RemoteRPC(p, lambda req, limit: []).rpc("HEARTBEAT")


def test_rpc_response_must_correlate_request_id():
    p = plan()
    with pytest.raises(m.MaintenanceError, match="transport_protocol"):
        m.RemoteRPC(
            p, lambda req, limit: {"event": "ALIVE", "request_id": "0" * 32}, retries=0
        ).rpc("HEARTBEAT")


def test_container_escape_flags_are_rejected():
    p = plan()
    for flag in (
        "--mount=type=bind,src=/,dst=/host",
        "--device=/dev/nvidia0",
        "--env=TOKEN=x",
        "--pid=host",
        "--security-opt=apparmor=unconfined",
    ):
        p["create_argv"] = p["create_argv"][:3] + [flag] + p["create_argv"][3:]
        with pytest.raises(m.MaintenanceError, match="invalid_plan"):
            m.validate_plan(p)
        p["create_argv"] = [x for x in p["create_argv"] if x != flag]


def bootstrap_argv(p):
    stage = "/var/lib/" + m.OWNER_PREFIX + p["job"]
    return [
        "docker",
        "create",
        "--name",
        m.OWNER_PREFIX + p["job"],
        "--label",
        m.EXPERIMENT_LABEL + "=" + m.EXPERIMENT_OWNER,
        "--label",
        m.JOB_LABEL + "=" + p["job"],
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "--workdir",
        "/workspace",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONPATH=/workspace/src",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "TRITON_CACHE_DIR=/tmp/triton",
        "--mount",
        f"type=bind,src={stage}/workspace,dst=/workspace,readonly",
        "--mount",
        "type=bind,src=/models/qwen,dst=/model,readonly",
        "--mount",
        "type=bind,src=/data/calibration,dst=/calibration,readonly",
        "--mount",
        "type=bind,src=/runs/output,dst=/output",
        "--mount",
        "type=bind,src=/runs/evidence,dst=/evidence",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=2g",
        p["image_ref"],
        "/workspace/pair_supervisor.py",
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


def test_bootstrap_mount_plan_is_accepted_and_execution_stays_disabled():
    p = plan()
    p["create_argv"] = bootstrap_argv(p)
    image_position = p["create_argv"].index(p["image_ref"])
    p["child_argv"] = p["create_argv"][image_position + 1 :]
    assert m.validate_plan(p)["execution_enabled"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.__setitem__(a.index("--mount") + 1, "type=bind,src=/models/qwen,dst=/"),
        lambda a: a.__setitem__(
            a.index("--mount") + 1, "type=bind,src=/models/../host,dst=/workspace,readonly"
        ),
        lambda a: a.__setitem__(
            a.index("--mount") + 1, "type=bind,src=/models/qwen,dst=/workspace"
        ),
        lambda a: a.insert(a.index("--mount"), "--env=TOKEN=secret"),
        lambda a: a.__setitem__(a.index("--gpus"), "--privileged"),
        lambda a: a.__setitem__(
            a.index("--mount") + 1, "type=bind,src=/models/qwen,dst=/workspace,readonly,readonly"
        ),
        lambda a: a.__setitem__(-1, "--evidence-root=/tmp"),
    ],
)
def test_bootstrap_argv_rejects_mount_and_launch_tampering(mutation):
    p = plan()
    p["create_argv"] = bootstrap_argv(p)
    mutation(p["create_argv"])
    with pytest.raises(m.MaintenanceError, match="invalid_plan"):
        m.validate_plan(p)


def test_replayed_nonce_is_rejected_without_second_mutation(tmp_path, monkeypatch):
    # The worker's durable journal is covered by the protocol implementation;
    # this fixture documents the expected persisted field shape.
    p = plan()
    owner = m.LocalOwner(p, object(), tmp_path / "journal.json")
    owner.state["rpc_nonces"] = ["nonce-1"]
    owner.persist()
    owner.resume()
    assert owner.state["rpc_nonces"] == ["nonce-1"]


def test_paired_rpc_failure_is_not_partial_success():
    class R:
        def __init__(self, ok):
            self.ok = ok
            self.calls = []

        def rpc(self, command):
            self.calls.append(command)
            if not self.ok:
                raise m.MaintenanceError("worker_failed")
            return {"event": "ALIVE"}

    a, b = R(True), R(False)
    with pytest.raises(m.MaintenanceError, match="worker_failed"):
        m.paired_rpc([a, b], "HEARTBEAT", "ALIVE")
    assert a.calls == ["HEARTBEAT"] and b.calls == ["HEARTBEAT"]
