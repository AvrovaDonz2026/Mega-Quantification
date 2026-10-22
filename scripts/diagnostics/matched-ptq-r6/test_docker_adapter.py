import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
m_spec = importlib.util.spec_from_file_location("maintenance", HERE / "maintenance.py")
maintenance = importlib.util.module_from_spec(m_spec)
sys.modules["maintenance"] = maintenance
m_spec.loader.exec_module(maintenance)

a_spec = importlib.util.spec_from_file_location("docker_adapter", HERE / "docker_adapter.py")
adapter = importlib.util.module_from_spec(a_spec)
sys.modules["docker_adapter"] = adapter
a_spec.loader.exec_module(adapter)


IMAGE_REF = "registry.invalid/ptq@sha256:" + "a" * 64
IMAGE_ID = "sha256:" + "b" * 64
CID = "c" * 64
SERVICE_CID = "d" * 64
DOCKER_PREFIX = ["docker", "--host", "unix:///var/run/docker.sock"]
CREATE = (
    "docker",
    "create",
    "--network=none",
    "--read-only",
    "--cap-drop=ALL",
    "--name",
    "megaquant-matched-r6-job0000000000000001",
    "--label",
    "io.megaquant.matched-r6.owner=matched-ptq-r6-1022",
    "--label",
    "io.megaquant.matched-r6.job=job0000000000000001",
    IMAGE_REF,
)


def service_row(*, running=True, config=None, host=None):
    config = config or {"Healthcheck": {"Test": ["CMD-SHELL", "true"]}}
    host = host or {"NetworkMode": "default"}
    return {
        "Id": SERVICE_CID,
        "Name": "/" + adapter.SERVICE_NAME,
        "Image": adapter.SERVICE_IMAGE_ID,
        "Config": config,
        "HostConfig": host,
        "RestartCount": 0,
        "State": {
            "Running": running,
            "Restarting": False,
            "OOMKilled": False,
            "StartedAt": "t0",
            "Health": {"Status": "healthy"},
        },
    }


def config_hash(row):
    import hashlib

    return hashlib.sha256(
        json.dumps(
            {"Config": row["Config"], "HostConfig": row["HostConfig"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


class Runner:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or {}

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        command = argv[3:]
        if argv[:3] != DOCKER_PREFIX:
            raise AssertionError("Docker command did not pin the local socket")
        if command[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, IMAGE_ID + "\n", "")
        if command[:1] == ["create"]:
            return subprocess.CompletedProcess(argv, 0, CID + "\n", "")
        if command[:1] == ["inspect"]:
            cid = command[1]
            if cid not in self.rows:
                return subprocess.CompletedProcess(argv, 1, "", "Error: No such object: missing")
            return subprocess.CompletedProcess(argv, 0, json.dumps([self.rows[cid]]), "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def make(runner, service=None):
    row = service or service_row()
    spec = adapter.ServiceSpec("fixture-node", SERVICE_CID, config_hash(row))
    return adapter.DockerAdapter(spec, IMAGE_REF, IMAGE_ID, CREATE, run=runner)


def test_historical_service_hash_and_exact_snapshot():
    assert adapter.DockerAbsent is maintenance.DockerAbsent
    assert adapter.MaintenanceError is maintenance.MaintenanceError
    row = service_row()
    runner = Runner({SERVICE_CID: row})
    snapshot = make(runner).service_inspect()
    assert snapshot["exact"] is True
    assert snapshot["health"] == "healthy"
    assert runner.calls[0][0][3:] == ["inspect", SERVICE_CID]


def test_missing_health_status_is_normalized_to_string():
    row = service_row()
    row["State"].pop("Health")
    runner = Runner({SERVICE_CID: row})
    snapshot = make(runner).service_inspect()
    assert snapshot["health_configured"] is True
    assert snapshot["health"] == "none"


def test_service_mutation_rechecks_identity_and_uses_bounded_timeout():
    row = service_row()
    runner = Runner({SERVICE_CID: row})
    docker = make(runner)
    docker.service_stop()
    assert runner.calls[-1][0][3:] == ["stop", "--time", "30", SERVICE_CID]
    assert runner.calls[-1][1]["timeout"] == 65.0
    foreign = service_row()
    foreign["Image"] = "sha256:" + "e" * 64
    runner.rows[SERVICE_CID] = foreign
    with pytest.raises(maintenance.MaintenanceError, match="identity_mismatch"):
        docker.service_start()
    assert runner.calls[-1][0][3:] == ["inspect", SERVICE_CID]


def test_image_attestation_create_and_owned_cleanup():
    row = service_row()
    experiment = {
        "Id": CID,
        "Image": IMAGE_ID,
        "Name": "/megaquant-matched-r6-job0000000000000001",
        "Config": {
            "Labels": {
                "io.megaquant.matched-r6.owner": "matched-ptq-r6-1022",
                "io.megaquant.matched-r6.job": "job0000000000000001",
            }
        },
    }
    runner = Runner({SERVICE_CID: row, CID: experiment})
    docker = make(runner)
    assert docker.attest_image_ref(IMAGE_REF) == IMAGE_ID
    assert docker.create(CREATE) == CID
    docker.start(CID)
    docker.remove(CID)
    assert [call[0][3:4] for call in runner.calls][-3:] == [["start"], ["inspect"], ["rm"]]
    assert runner.calls[-1][0][3:] == ["rm", "--force", CID]


def test_foreign_experiment_cannot_be_removed_and_absence_is_safe():
    foreign = {"Id": CID, "Image": IMAGE_ID, "Name": "/other", "Config": {"Labels": {}}}
    runner = Runner({CID: foreign})
    docker = make(runner, service_row())
    with pytest.raises(maintenance.MaintenanceError, match="foreign_experiment"):
        docker.remove(CID)
    del runner.rows[CID]
    assert docker.exists(CID) is False


def test_remove_tolerates_container_disappearing_after_ownership_check():
    experiment = {
        "Id": CID,
        "Image": IMAGE_ID,
        "Name": "/megaquant-matched-r6-job0000000000000001",
        "Config": {
            "Labels": {
                "io.megaquant.matched-r6.owner": "matched-ptq-r6-1022",
                "io.megaquant.matched-r6.job": "job0000000000000001",
            }
        },
    }

    class DisappearsAfterInspect(Runner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            command = argv[3:]
            if command[:1] == ["inspect"] and command[1] == CID and result.returncode == 0:
                self.rows.pop(CID, None)
            if command[:2] == ["rm", "--force"]:
                return subprocess.CompletedProcess(argv, 1, "", "Error: No such container: missing")
            return result

    runner = DisappearsAfterInspect({CID: experiment})
    docker = make(runner)
    docker.remove(CID)
    assert [call[0][3:] for call in runner.calls] == [
        ["inspect", CID],
        ["rm", "--force", CID],
        ["inspect", CID],
    ]


def test_docker_host_environment_cannot_override_pinned_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    runner = Runner({SERVICE_CID: service_row()})
    make(runner).service_inspect()
    argv, kwargs = runner.calls[0]
    assert argv[:3] == DOCKER_PREFIX
    assert "DOCKER_HOST" not in kwargs["env"]
    assert "DOCKER_CONTEXT" not in kwargs["env"]


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired(["docker"], 1), OSError("secret")])
def test_subprocess_failures_are_sanitized(exc):
    def fail(*_args, **_kwargs):
        raise exc

    docker = make(fail)
    with pytest.raises(maintenance.MaintenanceError) as caught:
        docker.service_inspect()
    assert str(caught.value) in {"transport_timeout", "transport_error"}
    assert "secret" not in str(caught.value)
