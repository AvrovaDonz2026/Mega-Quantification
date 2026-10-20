#!/usr/bin/env python3
"""One-shot supervisor for the remote MLP pilot.

This file intentionally has no project imports and performs no network access.
The controller is the only component which talks to the local HTTP server.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OWNER = "aitop-parity-v7"
JOBS = frozenset(("mlp-pilot-control", "mlp-pilot-mlp"))
SOURCE = "49d3acfd4d54b0cf1afffcfeb369c63af4f56499"
IMAGE = "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf"
PHASES = (("early", 4096, 5 * 60 * 60), ("long", 49152, 18 * 60 * 60))
_CHUNK = 8 * 1024 * 1024


class SupervisorError(RuntimeError):
    pass


class Interrupted(SupervisorError):
    pass


def utc_now() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def command(argv: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    # Never expose command output: Docker and nvidia-smi can contain host details.
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def require_command(argv: list[str], *, timeout: float | None = None) -> str:
    result = command(argv, timeout=timeout)
    if result.returncode != 0:
        raise SupervisorError("external command failed")
    return result.stdout


def docker_inspect(container: str) -> dict[str, Any]:
    raw = require_command(["docker", "inspect", container], timeout=20)
    try:
        rows = json.loads(raw)
        row = rows[0]
    except (ValueError, IndexError, TypeError, KeyError) as exc:
        raise SupervisorError("invalid docker inspect response") from exc
    if not isinstance(row, dict):
        raise SupervisorError("invalid docker inspect object")
    return row


def glm_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("Config") if isinstance(row.get("Config"), dict) else {}
    host = row.get("HostConfig") if isinstance(row.get("HostConfig"), dict) else {}
    state = row.get("State") if isinstance(row.get("State"), dict) else {}
    restart = host.get("RestartPolicy") if isinstance(host.get("RestartPolicy"), dict) else {}
    health = state.get("Health") if isinstance(state.get("Health"), dict) else None
    result: dict[str, Any] = {
        "id": row.get("Id"),
        "image": config.get("Image"),
        "image_id": row.get("Image"),
        "name": row.get("Name"),
        "restartpolicy": {
            "name": restart.get("Name"),
            "maximum_retry_count": restart.get("MaximumRetryCount"),
        },
        "running": state.get("Running") is True,
        "config_hostconfig_sha256": canonical_sha({"Config": config, "HostConfig": host}),
    }
    if health is not None:
        result["health"] = health.get("Status")
    return result


def parse_created(row: dict[str, Any]) -> float | None:
    value = row.get("Created")
    if not isinstance(value, str):
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def labels(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("Config")
    return (
        config.get("Labels", {})
        if isinstance(config, dict) and isinstance(config.get("Labels", {}), dict)
        else {}
    )


def container_ids_by_owner() -> list[str]:
    out = require_command(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=io.megaquant.parity-v7.owner={OWNER}",
            "--format",
            "{{.ID}}",
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def matching_owned_rows() -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for cid in container_ids_by_owner():
        try:
            row = docker_inspect(cid)
        except SupervisorError:
            continue
        if labels(row).get("io.megaquant.parity-v7.owner") == OWNER:
            rows.append((cid, row))
    return rows


def preflight_no_residuals() -> None:
    for _cid, row in matching_owned_rows():
        if labels(row).get("io.megaquant.parity-v7.job") in JOBS:
            raise SupervisorError("pre-existing experiment container")


def cleanup_experiment_containers(started_at: float) -> int:
    removed = 0
    for cid, row in matching_owned_rows():
        tag = labels(row)
        created = parse_created(row)
        if (
            tag.get("io.megaquant.parity-v7.job") in JOBS
            and tag.get("io.megaquant.parity-v7.source") == SOURCE
            and row.get("Image") == IMAGE
            and created is not None
            and created >= started_at
        ):
            result = command(["docker", "rm", "-f", cid], timeout=30)
            if result.returncode != 0:
                raise SupervisorError("experiment container cleanup failed")
            removed += 1
    return removed


def gpu_idle_gate() -> None:
    deadline = time.monotonic() + 120
    while True:
        result = command(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], timeout=15
        )
        if result.returncode != 0:
            raise SupervisorError("nvidia-smi query failed")
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not pids:
            return
        if time.monotonic() >= deadline:
            raise SupervisorError("GPU did not become idle")
        time.sleep(min(2.0, max(0.05, deadline - time.monotonic())))


def expected_value(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def verify_glm(config: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    expected_id = expected_value(
        config, "expected_glm_id_full", "expected_glm_full", "expected_glm_id"
    )
    expected_image = expected_value(config, "expected_glm_image")
    expected_image_id = expected_value(
        config, "expected_glm_image_id", "expected_glm_image_full", "expected_glm_image_id_full"
    )
    if (
        not isinstance(expected_id, str)
        or not expected_id
        or not isinstance(expected_image, str)
        or not expected_image
    ):
        raise SupervisorError("GLM identity config missing")
    row = docker_inspect(expected_id)
    actual_id = row.get("Id")
    if actual_id != expected_id:
        raise SupervisorError("GLM container id mismatch")
    config_image = (
        (row.get("Config") or {}).get("Image") if isinstance(row.get("Config"), dict) else None
    )
    image_id = row.get("Image")
    if expected_image.startswith("sha256:"):
        if image_id != expected_image:
            raise SupervisorError("GLM image id mismatch")
    elif config_image != expected_image:
        raise SupervisorError("GLM image reference mismatch")
    if expected_image_id is not None and image_id != expected_image_id:
        raise SupervisorError("GLM image id mismatch")
    expected_name = expected_value(config, "expected_glm_name")
    if not isinstance(expected_name, str) or not expected_name or row.get("Name") != expected_name:
        raise SupervisorError("GLM name mismatch")
    expected_config_host = expected_value(
        config, "expected_config_hostconfig_sha256", "expected_glm_config_hostconfig_sha256"
    )
    if (
        not isinstance(expected_config_host, str)
        or canonical_sha(
            {
                "Config": row.get("Config") if isinstance(row.get("Config"), dict) else {},
                "HostConfig": row.get("HostConfig")
                if isinstance(row.get("HostConfig"), dict)
                else {},
            }
        )
        != expected_config_host
    ):
        raise SupervisorError("GLM config/hostconfig hash mismatch")
    state = row.get("State") if isinstance(row.get("State"), dict) else {}
    if state.get("Running") is not True:
        raise SupervisorError("GLM was not running")
    return expected_id, row, glm_snapshot(row)


def fresh_evidence_roots(root: Path, prefix: str) -> dict[str, Path]:
    paths = {phase: root / f"{prefix}-{phase}" for phase, _, _ in PHASES}
    for path in paths.values():
        if path.exists():
            raise SupervisorError("evidence root is not fresh")
    return paths


def build_phase_argv(base: list[Any], evidence: Path, phase: str, budget: int) -> list[str]:
    argv = [str(x) for x in base]
    if "--phase" in argv or "--budget" in argv:
        raise SupervisorError("base command must not contain phase or budget")
    try:
        root_index = argv.index("--root")
    except ValueError as exc:
        raise SupervisorError("base command has no --root") from exc
    if root_index + 1 >= len(argv) or argv[root_index + 1].startswith("-"):
        raise SupervisorError("invalid --root argument")
    argv[root_index + 1] = str(evidence)
    argv.extend(["--phase", phase, "--budget", str(budget)])
    return argv


def valid_report(path: Path) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict) or report.get("status") != "completed":
        return False
    comparison = report.get("comparison")
    arms = report.get("arms")
    if not isinstance(comparison, dict) or comparison.get("both_repeat_stable") is not True:
        return False
    if not isinstance(arms, dict) or set(arms) != set(JOBS_NAME_TO_ARM):
        return False
    for arm_name in ("control", "mlp"):
        arm = arms.get(arm_name)
        if (
            not isinstance(arm, dict)
            or not isinstance(arm.get("trials"), list)
            or len(arm["trials"]) != 2
        ):
            return False
        cleanup = arm.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("removed") is not True:
            return False
    return True


JOBS_NAME_TO_ARM = {"control": "control", "mlp": "mlp"}


def terminate_child(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            raise SupervisorError("controller did not terminate") from None


def run_phase(
    config: dict[str, Any],
    root: Path,
    evidence: Path,
    phase: str,
    budget: int,
    timeout_seconds: int,
    supervisor_started_at: float,
    status: dict[str, Any],
    status_path: Path,
) -> None:
    log_path = root / f"{evidence.name}.supervisor.log"
    if log_path.exists():
        raise SupervisorError("phase log is not fresh")
    with log_path.open("x", encoding="utf-8") as log:
        argv = build_phase_argv(config["base_command"], evidence, phase, budget)
        status.update({"state": "phase_running", "phase": phase, "pid": None, "at_utc": utc_now()})
        atomic_json(status_path, status)
        proc: subprocess.Popen[Any] | None = None
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status.update({"pid": proc.pid, "at_utc": utc_now()})
            atomic_json(status_path, status)
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                terminate_child(proc)
                raise SupervisorError("phase timeout") from exc
            if proc.returncode != 0:
                raise SupervisorError("controller failed")
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    terminate_child(proc)
                except Exception:
                    pass
            status["pid"] = None
            try:
                atomic_json(status_path, status)
            finally:
                # The controller can be interrupted before its own finally block.
                cleanup_experiment_containers(supervisor_started_at)
    if not valid_report(evidence / "report.json"):
        raise SupervisorError("phase report gate failed")


def signal_handler(signum: int, _frame: Any) -> None:
    raise Interrupted(f"signal {signum}")


def main() -> int:
    supervisor_started_at = time.time()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    config_path = Path(__file__).with_name("config.json")
    status: dict[str, Any] = {"schema_version": 1, "state": "starting", "at_utc": utc_now()}
    lock_handle = None
    glm_id: str | None = None
    glm_was_stopped = False
    status_path: Path | None = None
    restore_error: SupervisorError | None = None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise SupervisorError("config must be an object")
        root = Path(config["root_path"]).expanduser()
        prefix = config.get("experiment_prefix", "mlp-pilot-r3")
        if not isinstance(prefix, str) or not prefix or "/" in prefix or "\\" in prefix:
            raise SupervisorError("invalid experiment prefix")
        root.mkdir(parents=True, exist_ok=True)
        status_path = root / "supervisor-status.json"
        lock_path = root / f".{prefix}.supervisor.lock"
        lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError("another supervisor is running") from exc
        status.update(
            {
                "root": str(root),
                "experiment_prefix": prefix,
                "pid": os.getpid(),
                "supervisor_started_at_epoch": supervisor_started_at,
                "glm_restore_required": False,
                "state": "preflight",
                "at_utc": utc_now(),
            }
        )
        atomic_json(status_path, status)
        evidence = fresh_evidence_roots(root, prefix)

        controller = Path(config["controller_path"]).expanduser()
        model_gate = Path(config["model_gate_path"]).expanduser()
        if not controller.is_file() or not model_gate.is_file():
            raise SupervisorError("gate file missing")
        expected_controller = config.get("controller_sha256")
        expected_gate = config.get("model_gate_sha256")
        if (
            not isinstance(expected_controller, str)
            or sha256_file(controller) != expected_controller
        ):
            raise SupervisorError("controller hash gate failed")
        if not isinstance(expected_gate, str) or sha256_file(model_gate) != expected_gate:
            raise SupervisorError("model gate hash failed")
        base = config.get("base_command")
        if not isinstance(base, list) or not base:
            raise SupervisorError("base command missing")
        base_strings = [str(value) for value in base]
        if str(controller.resolve()) not in [
            str(Path(value).expanduser().resolve())
            for value in base_strings
            if value.endswith(".py")
        ]:
            raise SupervisorError("base command controller differs from controller_path")
        # Validate the command before touching the serving container.
        build_phase_argv(base, evidence["early"], "early", 4096)
        preflight_no_residuals()
        glm_id, before_row, before = verify_glm(config)
        status["glm_before"] = before
        atomic_json(status_path, status)

        status.update({"glm_restore_required": True, "state": "stopping_glm", "at_utc": utc_now()})
        atomic_json(status_path, status)
        glm_was_stopped = True
        result = command(["docker", "stop", "--time", "30", glm_id], timeout=45)
        if result.returncode != 0:
            raise SupervisorError("GLM stop failed")
        status.update({"state": "glm_stopped", "at_utc": utc_now()})
        atomic_json(status_path, status)
        gpu_idle_gate()
        for phase, budget, timeout_seconds in PHASES:
            run_phase(
                config,
                root,
                evidence[phase],
                phase,
                budget,
                timeout_seconds,
                supervisor_started_at,
                status,
                status_path,
            )
            status.update({"state": "phase_completed", "phase": phase, "at_utc": utc_now()})
            atomic_json(status_path, status)
        status.update({"state": "completed", "at_utc": utc_now()})
    except Exception as exc:
        status.update({"state": "failed", "error_type": type(exc).__name__, "at_utc": utc_now()})
        if status_path is not None:
            try:
                atomic_json(status_path, status)
            except Exception:
                pass
    finally:
        try:
            cleanup_experiment_containers(supervisor_started_at)
        except Exception:
            status.update({"cleanup_failed": True, "state": "failed", "at_utc": utc_now()})
        if glm_was_stopped and glm_id is not None:
            restored = False
            for _attempt in range(3):
                try:
                    result = command(["docker", "start", glm_id], timeout=30)
                    if result.returncode == 0:
                        restored = True
                        break
                except Exception:
                    pass
                if _attempt < 2:
                    try:
                        time.sleep(2)
                    except Interrupted:
                        pass
            if restored:
                try:
                    row = docker_inspect(glm_id)
                    snap = glm_snapshot(row)
                    status["glm_restore"] = {
                        "running": snap.get("running"),
                        "health": snap.get("health"),
                    }
                    if snap.get("running") is not True:
                        restored = False
                except Exception:
                    restored = False
            if not restored:
                restore_error = SupervisorError("GLM restore failed")
                status.update(
                    {
                        "state": "failed",
                        "restore_failed": True,
                        "error_type": type(restore_error).__name__,
                        "at_utc": utc_now(),
                    }
                )
            else:
                status["glm_restore_required"] = False
            if status_path is not None:
                try:
                    atomic_json(status_path, status)
                except Exception:
                    pass
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()
    if status_path is not None:
        try:
            atomic_json(status_path, status)
        except Exception:
            restore_error = restore_error or SupervisorError("status persistence failed")
    if restore_error is not None:
        return 1
    return 0 if status.get("state") == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Keep terminal output limited to a safe type name; never print child output.
        sys.stderr.write(f"supervisor failed: {type(exc).__name__}\n")
        raise SystemExit(1) from None
