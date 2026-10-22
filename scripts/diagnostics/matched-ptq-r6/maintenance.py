"""Offline fail-closed lifecycle for the matched PTQ R6 experiment.

The target deployment may provide a small Docker-like adapter.  This module
keeps one owned experiment container for the whole paired run and retains the
R5 service lease/cleanup/recovery ordering.  It never enables execution by
itself.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import socket
import stat
import tempfile
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class MaintenanceError(RuntimeError):
    pass


class DockerAbsent(LookupError):
    pass


OWNER_PREFIX = "megaquant-matched-r6-"
EXPERIMENT_OWNER = "matched-ptq-r6-1022"
EXPERIMENT_LABEL = "io.megaquant.matched-r6.owner"
JOB_LABEL = "io.megaquant.matched-r6.job"
ARMS = ("static-max", "static-MSE")
MIN_ARM_TIMEOUT = 4 * 60 * 60
MAX_ARM_TIMEOUT = 4 * 60 * 60
MIN_MARGIN = 15 * 60
STABLE_SECONDS = 120


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _safe_absolute_path(value: str) -> bool:
    path = Path(value)
    return (
        path.is_absolute() and path != Path("/") and ".." not in path.parts and str(path) == value
    )


def _validate_bootstrap_argv(argv: list[str], plan: Mapping[str, Any]) -> bool:
    """Accept only bootstrap.py's fixed offline launch shape and five mounts."""
    name = OWNER_PREFIX + plan["job"]
    owner_pair = ["--label", plan["owner_label_key"] + "=" + plan["owner"]]
    job_pair = ["--label", plan["job_label_key"] + "=" + plan["job"]]
    fixed = [
        "docker",
        "create",
        "--name",
        name,
        *owner_pair,
        *job_pair,
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
    ]
    if argv[: len(fixed)] != fixed:
        return False
    cursor = len(fixed)
    mounts = []
    while cursor < len(argv) and argv[cursor] == "--mount":
        if cursor + 1 >= len(argv):
            return False
        value = argv[cursor + 1]
        match = re.fullmatch(r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?", value)
        if not match:
            return False
        source, destination, readonly = match.groups()
        if not _safe_absolute_path(source):
            return False
        mounts.append((destination, source, readonly is not None))
        cursor += 2
    destinations = ("/workspace", "/model", "/calibration", "/output", "/evidence")
    if tuple(item[0] for item in mounts) != destinations:
        return False
    if tuple(item[2] for item in mounts) != (True, True, True, False, False):
        return False
    if len({item[1] for item in mounts}) != len(mounts):
        return False
    if mounts[0][1] != "/var/lib/" + name + "/workspace":
        return False
    if (
        cursor + 1 >= len(argv)
        or argv[cursor] != "--tmpfs"
        or argv[cursor + 1] != "/tmp:rw,exec,nosuid,nodev,size=2g"
    ):
        return False
    cursor += 2
    tail = [
        plan["image_ref"],
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
        str(plan["arm_timeout_seconds"]),
        "--pair-timeout",
        "28800",
    ]
    return argv[cursor:] == tail and plan["child_argv"] == tail[1:]


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
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
    if set(plan) != required or not re.fullmatch(r"[a-z0-9]{16,32}", str(plan.get("job", ""))):
        raise MaintenanceError("invalid_plan")
    if plan["execution_enabled"] is not False:
        raise MaintenanceError("execution_disabled")
    if not isinstance(plan["image_ref"], str) or not re.fullmatch(
        r"[^\s@]+(?:/[^\s@]+)*@sha256:[0-9a-f]{64}", plan["image_ref"]
    ):
        raise MaintenanceError("invalid_plan")
    if not isinstance(plan["image_id"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", plan["image_id"]
    ):
        raise MaintenanceError("invalid_plan")
    if not isinstance(plan["owner"], str) or not re.fullmatch(
        r"[a-z][a-z0-9-]{2,48}", plan["owner"]
    ):
        raise MaintenanceError("invalid_plan")
    for key in ("owner_label_key", "job_label_key"):
        if not isinstance(plan[key], str) or not re.fullmatch(r"[A-Za-z0-9._/-]{3,100}", plan[key]):
            raise MaintenanceError("invalid_plan")
    if plan["owner_label_key"] == plan["job_label_key"]:
        raise MaintenanceError("invalid_plan")
    if plan["state_dir"] != "/var/lib/" + OWNER_PREFIX + plan["job"]:
        raise MaintenanceError("invalid_plan")
    arm = plan["arm_timeout_seconds"]
    if type(arm) is not int or arm < MIN_ARM_TIMEOUT or arm > MAX_ARM_TIMEOUT:
        raise MaintenanceError("invalid_plan")
    if (
        plan["experiment_ttl"] != 2 * arm + MIN_MARGIN
        or plan["worker_ttl"] != plan["experiment_ttl"] + MIN_MARGIN
    ):
        raise MaintenanceError("invalid_plan")
    for key in ("create_argv", "child_argv"):
        if (
            not isinstance(plan[key], list)
            or not plan[key]
            or not all(isinstance(x, str) for x in plan[key])
        ):
            raise MaintenanceError("invalid_plan")
    argv = plan["create_argv"]
    if argv[:2] != ["docker", "create"] or argv.count(plan["image_ref"]) != 1:
        raise MaintenanceError("invalid_plan")
    forbidden_exact = {
        "--rm",
        "--privileged",
        "--network=host",
        "host",
        "--pid=host",
        "--ipc=host",
        "--uts=host",
        "--userns=host",
        "--init",
        "--oom-kill-disable",
        "--read-only=false",
    }
    forbidden_prefixes = (
        "--cidfile",
        "-v",
        "--volume",
        "--device",
        "-e",
        "--pid=",
        "--ipc=",
        "--cap-add",
        "--userns",
        "--uts",
        "--hostname",
        "--add-host",
        "--runtime",
        "--privileged=",
        "--sysctl",
        "--publish",
        "-p",
        "--expose",
        "--group-add",
    )
    if any(x in forbidden_exact or x.startswith(forbidden_prefixes) for x in argv):
        raise MaintenanceError("invalid_plan")
    if "--network=none" not in argv or "--read-only" not in argv or "--cap-drop=ALL" not in argv:
        raise MaintenanceError("invalid_plan")
    required = [
        ["--name", OWNER_PREFIX + plan["job"]],
        ["--label", plan["owner_label_key"] + "=" + plan["owner"]],
        ["--label", plan["job_label_key"] + "=" + plan["job"]],
    ]
    if any(sum(argv[i : i + 2] == pair for i in range(len(argv))) != 1 for pair in required):
        raise MaintenanceError("invalid_plan")
    legacy = [
        "docker",
        "create",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--name",
        OWNER_PREFIX + plan["job"],
        "--label",
        plan["owner_label_key"] + "=" + plan["owner"],
        "--label",
        plan["job_label_key"] + "=" + plan["job"],
        plan["image_ref"],
    ]
    if argv != legacy and not _validate_bootstrap_argv(argv, plan):
        raise MaintenanceError("invalid_plan")
    if not plan["child_argv"][0].startswith("/") and plan["child_argv"][0] != "python3":
        raise MaintenanceError("invalid_plan")
    if any(
        ".." in Path(x).parts
        for x in plan["child_argv"]
        if isinstance(x, str) and x.startswith("/")
    ):
        raise MaintenanceError("invalid_plan")
    return dict(plan)


def _identity(row: Mapping[str, Any], plan: Mapping[str, Any], cid: str) -> bool:
    labels = (row.get("Config") or {}).get("Labels") or {}
    return (
        row.get("Id") == cid
        and row.get("Image") == plan["image_id"]
        and row.get("Name") == "/" + OWNER_PREFIX + plan["job"]
        and labels.get(plan["owner_label_key"]) == plan["owner"]
        and labels.get(plan["job_label_key"]) == plan["job"]
    )


class LocalOwner:
    def __init__(
        self,
        plan: Mapping[str, Any],
        docker: Any,
        journal: str | Path,
        now=time.monotonic,
        sleep=time.sleep,
    ):
        self.plan = validate_plan(plan)
        self.docker = docker
        self.journal = Path(journal)
        self.now, self.sleep = now, sleep
        self.state = {
            "plan_sha256": digest(self.plan),
            "experiment_id": None,
            "create_intent": False,
            "armed": False,
            "cleanup": "not-started",
            "restoration": "not-required",
            "rpc_nonces": [],
        }

    def persist(self):
        self.journal.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="journal-", dir=self.journal.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(self.state, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.journal)
        directory_fd = os.open(self.journal.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def resume(self):
        with self.journal.open(encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("plan_sha256") != digest(self.plan):
            raise MaintenanceError("identity_mismatch")
        self.state = state

    def inspect(self):
        cid = self.state.get("experiment_id")
        if cid is None:
            return None
        try:
            row = self.docker.inspect(cid)
        except DockerAbsent:
            return None
        if not isinstance(row, dict) or not _identity(row, self.plan, cid):
            raise MaintenanceError("foreign_experiment")
        return row

    def attest_image(self):
        fn = getattr(self.docker, "attest_image_ref", None)
        if not callable(fn) or fn(self.plan["image_ref"]) != self.plan["image_id"]:
            raise MaintenanceError("image_attestation_failed")

    def arm(self):
        self._service_exact()
        self.state.update(armed=True, restoration="required")
        self.persist()

    def _service_exact(self):
        fn = getattr(self.docker, "service_inspect", None)
        if not callable(fn):
            raise MaintenanceError("service_adapter_missing")
        row = fn()
        required = {
            "exact",
            "running",
            "restarting",
            "oom_killed",
            "restart_count",
            "started_at",
            "health_configured",
            "health",
        }
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise MaintenanceError("service_snapshot_incomplete")
        if row.get("exact") is not True:
            raise MaintenanceError("identity_mismatch")
        if (
            type(row.get("running")) is not bool
            or type(row.get("restarting")) is not bool
            or type(row.get("oom_killed")) is not bool
            or type(row.get("health_configured")) is not bool
            or type(row.get("restart_count")) is not int
            or row["restart_count"] < 0
            or not isinstance(row.get("started_at"), str)
            or not row["started_at"]
            or not isinstance(row.get("health"), str)
        ):
            raise MaintenanceError("service_snapshot_incomplete")
        return row

    def stop_service(self):
        if not self.state.get("armed"):
            raise MaintenanceError("not_armed")
        self._service_exact()
        fn = getattr(self.docker, "service_stop", None)
        if not callable(fn):
            raise MaintenanceError("service_adapter_missing")
        fn()
        after = self._service_exact()
        if isinstance(after, Mapping) and after.get("running") is True:
            raise MaintenanceError("stop_failed")

    def status(self):
        fn = getattr(self.docker, "child_status", None)
        if not callable(fn):
            return {"event": "STATUS", "child": None}
        return {"event": "STATUS", "child": fn(self.state.get("experiment_id"))}

    def launch(self):
        if not self.state.get("armed"):
            raise MaintenanceError("not_armed")
        self.attest_image()
        self.state["create_intent"] = True
        self.persist()
        try:
            cid = self.docker.create(self.plan["create_argv"])
        except BaseException as exc:
            self.persist()
            raise MaintenanceError("create_uncertain") from exc
        if not _id(cid):
            raise MaintenanceError("experiment_identity_malformed")
        self.state["experiment_id"] = cid
        self.persist()
        if self.inspect() is None:
            raise MaintenanceError("experiment_missing")
        self.docker.start(cid)
        self.docker.start_child(self.plan["child_argv"], self.plan["experiment_ttl"]) if hasattr(
            self.docker, "start_child"
        ) else None
        return {"event": "EXPERIMENT_STARTED", "container_identity_sha256": digest(cid)}

    def cleanup(self):
        cid = self.state.get("experiment_id")
        if cid is not None:
            row = self.inspect()
            if row is not None:
                self.docker.remove(cid)
            if self.docker.exists(cid):
                raise MaintenanceError("cleanup_unconfirmed")
        self.state["cleanup"] = "observed-absent"
        self.persist()

    def start_service(self):
        self.cleanup()
        self._service_exact()
        fn = getattr(self.docker, "service_start", None)
        if not callable(fn):
            raise MaintenanceError("service_adapter_missing")
        fn()
        self.state["restoration"] = "start_attempted"
        self.persist()

    def verify_close(self, seconds=STABLE_SECONDS, timeout=5 * 60):
        if self.state.get("restoration") != "start_attempted":
            raise MaintenanceError("not_armed")
        deadline = self.now() + timeout
        since = None
        marker = None
        readiness = None
        while self.now() <= deadline:
            row = self._service_exact()
            if row.get("exact") is not True:
                raise MaintenanceError("identity_mismatch")
            current = (
                row["restart_count"],
                row["started_at"],
                row["oom_killed"],
                row["health_configured"],
            )
            if marker is not None and marker != current:
                raise MaintenanceError("unstable")
            marker = current
            if row["running"] is not True or row["restarting"] or row["oom_killed"]:
                raise MaintenanceError("unstable")
            if row["health_configured"]:
                if row["health"] == "starting":
                    since = None
                    readiness = "healthcheck_starting"
                    self.sleep(min(2, max(0.01, deadline - self.now())))
                    continue
                if row["health"] != "healthy":
                    raise MaintenanceError("unstable")
                readiness = "healthy"
            else:
                readiness = "unknown"
            since = self.now() if since is None else since
            if self.now() - since >= seconds:
                self.state.update(
                    armed=False,
                    restoration="process_stable",
                    application_readiness=readiness,
                )
                self.persist()
                return {
                    "event": "VERIFIED_CLOSED",
                    "stable_seconds": self.now() - since,
                    "application_readiness": readiness,
                }
            self.sleep(min(2, max(0.01, deadline - self.now())))
        raise MaintenanceError("stability_timeout")

    def recover(self):
        if not self.state.get("armed"):
            return {"event": "NOT_ARMED"}
        last = None
        for attempt in range(3):
            try:
                self.start_service()
                return self.verify_close()
            except BaseException as exc:
                last = exc
                if attempt < 2:
                    self.sleep(1)
        self.state.update(
            restoration="unconfirmed", failure=type(last).__name__ if last else "unknown"
        )
        self.persist()
        return {"event": "RESTORATION_UNCONFIRMED"}


def paired_rpc(remotes, command, event):
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [f.result() for f in [pool.submit(r.rpc, command) for r in remotes]]
    if any(r.get("event") != event for r in rows):
        raise MaintenanceError("worker_failed")
    return rows


def paired_rpc_best_effort(remotes, command, event):
    """Issue a paired mutation to both endpoints and report every outcome."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(r.rpc, command) for r in remotes]
        rows = []
        for future in futures:
            try:
                row = future.result()
                if not isinstance(row, Mapping) or row.get("event") != event:
                    raise MaintenanceError("worker_failed")
                rows.append(row)
            except BaseException as exc:
                rows.append({"event": "ERROR", "error": type(exc).__name__})
    if len(rows) != len(remotes) or any(row.get("event") != event for row in rows):
        raise MaintenanceError("paired_mutation_failed")
    return rows


def child_outcome(child):
    if not isinstance(child, Mapping):
        raise MaintenanceError("experiment_failed")
    running = child.get("Running", child.get("running"))
    status = child.get("Status", child.get("SubState"))
    code = child.get("ExitCode", child.get("exit_code"))
    if running is True:
        return "running"
    if status == "exited" and code == 0:
        return "success"
    if running is False:
        raise MaintenanceError("experiment_failed")
    raise MaintenanceError("experiment_unknown")


HEARTBEAT_TTL = 45
HEARTBEAT_ATTEMPTS = 3
SOCKET_TIMEOUT = 10
RPC_RETRIES = 2
RPC_TIMEOUTS = {
    "HEARTBEAT": 60,
    "ARM": 120,
    "STOP": 240,
    "LAUNCH": 240,
    "STATUS": 120,
    "CLEANUP": 120,
    "START": 240,
    "VERIFY_CLOSE": 330,
}


def trusted_path(path: str | Path, regular=False):
    target = Path(path)
    if not target.is_absolute() or ".." in target.parts:
        raise MaintenanceError("unsafe_plan")
    for current in [*reversed(target.parents), target]:
        try:
            info = current.lstat()
        except FileNotFoundError:
            raise MaintenanceError("unsafe_plan") from None
        if (
            info.st_uid != 0
            or info.st_mode & 0o022
            or stat.S_ISLNK(info.st_mode)
            or (current != target and not stat.S_ISDIR(info.st_mode))
        ):
            raise MaintenanceError("unsafe_plan")
    if regular and not stat.S_ISREG(info.st_mode):
        raise MaintenanceError("unsafe_plan")
    if not regular and not stat.S_ISDIR(info.st_mode):
        raise MaintenanceError("unsafe_plan")
    return info


def _validate_owner_storage(plan, owner):
    root = Path(plan["state_dir"])
    if digest(owner.plan) != digest(plan) or owner.journal != root / "journal.json":
        raise MaintenanceError("unsafe_plan")
    trusted_path(root)
    try:
        owner.journal.lstat()
    except FileNotFoundError:
        return
    trusted_path(owner.journal, regular=True)


def _validate_journal_state(state: Any, expected_plan_sha256: str):
    if not isinstance(state, dict) or state.get("plan_sha256") != expected_plan_sha256:
        raise MaintenanceError("identity_mismatch")
    if state.get("experiment_id") is not None and not _id(state["experiment_id"]):
        raise MaintenanceError("worker_protocol")
    if type(state.get("create_intent")) is not bool or type(state.get("armed")) is not bool:
        raise MaintenanceError("worker_protocol")
    if not isinstance(state.get("cleanup"), str) or not isinstance(state.get("restoration"), str):
        raise MaintenanceError("worker_protocol")
    nonces = state.get("rpc_nonces", [])
    if (
        not isinstance(nonces, list)
        or len(nonces) > 128
        or not all(isinstance(nonce, str) and len(nonce) >= 16 for nonce in nonces)
    ):
        raise MaintenanceError("worker_protocol")
    return state


def _load_journal(path: Path, expected_plan_sha256: str):
    info = trusted_path(path, regular=True)
    if info.st_mode & 0o077:
        raise MaintenanceError("unsafe_plan")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        opened = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
            raise MaintenanceError("unsafe_plan")
        state = json.load(stream)
    return _validate_journal_state(state, expected_plan_sha256)


def lock_owner(job: str):
    if not re.fullmatch(r"[a-z0-9]{16,32}", job):
        raise MaintenanceError("invalid_plan")
    path = "/run/lock/" + OWNER_PREFIX + job + ".lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(fd, "a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise MaintenanceError("lock_busy") from exc
    return lock


def _frame(sock, limit=65536):
    sock.settimeout(SOCKET_TIMEOUT)
    data = bytearray()
    while b"\n" not in data:
        try:
            chunk = sock.recv(min(4096, limit + 1 - len(data)))
        except TimeoutError as exc:
            raise MaintenanceError("worker_timeout") from exc
        if not chunk or len(data) + len(chunk) > limit:
            raise MaintenanceError("worker_protocol")
        data.extend(chunk)
    line, _, tail = data.partition(b"\n")
    if tail:
        raise MaintenanceError("worker_protocol")
    return json.loads(line)


def _request_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{32}", value) is not None


def _error_response(request_id: Any, reason: str) -> dict[str, Any]:
    return {
        "event": "ERROR",
        "error": reason,
        "request_id": request_id if _request_id(request_id) else None,
    }


def serve(plan, owner: LocalOwner, socket_path: str | Path, lease=HEARTBEAT_TTL):
    """Target worker implementing the ordered lifecycle over a root socket."""
    validate_plan(plan)
    _validate_owner_storage(plan, owner)
    lock = lock_owner(plan["job"])
    try:
        if owner.journal.exists():
            owner.state = _load_journal(owner.journal, digest(plan))
    except BaseException:
        lock.close()
        raise
    if owner.state.get("armed"):
        try:
            outcome = owner.recover()
            if outcome.get("event") == "RESTORATION_UNCONFIRMED":
                raise MaintenanceError("restoration_unconfirmed")
        finally:
            lock.close()
        return
    path = Path(socket_path)
    server = None
    bound_stat = None
    try:
        trusted_path(path.parent)
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise MaintenanceError("socket_path_exists")
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(path))
        bound_stat = path.lstat()
        os.chmod(path, 0o600)
        server.listen(8)
        server.settimeout(1)
    except BaseException:
        if server is not None:
            server.close()
        if bound_stat is not None:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (bound_stat.st_dev, bound_stat.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
        lock.close()
        raise
    socket_stat = bound_stat
    expires = time.monotonic() + plan["worker_ttl"]
    lease_lock = threading.Lock()
    lease_deadline = [time.monotonic() + lease]
    shutdown = threading.Event()
    operation_lock = threading.Lock()
    in_flight = [None]

    def watchdog():
        while not shutdown.wait(0.25):
            with lease_lock:
                expired = time.monotonic() >= lease_deadline[0]
                operation_deadline = in_flight[0]
                operation_active = operation_deadline is not None
            if expired:
                if operation_active:
                    if time.monotonic() >= operation_deadline:
                        shutdown.set()
                    continue
                try:
                    if operation_lock.acquire(timeout=1):
                        try:
                            if owner.state.get("armed"):
                                owner.recover()
                        finally:
                            operation_lock.release()
                finally:
                    shutdown.set()
                return

    guard = threading.Thread(target=watchdog, name="r6-lease-watchdog", daemon=True)
    guard.start()
    try:
        while time.monotonic() < expires and not shutdown.is_set():
            try:
                client, _ = server.accept()
            except TimeoutError:
                continue
            with client:
                request_id = None
                try:
                    req = _frame(client)
                    request_id = req.get("request_id") if isinstance(req, dict) else None
                    if not isinstance(req, dict):
                        raise MaintenanceError("worker_protocol")
                    if req.get("plan_sha256") != digest(plan):
                        raise MaintenanceError("identity_mismatch")
                    if (
                        not _request_id(request_id)
                        or not isinstance(req.get("nonce"), str)
                        or len(req["nonce"]) < 16
                    ):
                        raise MaintenanceError("worker_protocol")
                    nonce = req["nonce"]
                    seen = owner.state.setdefault("rpc_nonces", [])
                    if nonce in seen:
                        raise MaintenanceError("replayed_request")
                    seen.append(nonce)
                    del seen[:-128]
                    owner.persist()
                    with lease_lock:
                        lease_deadline[0] = time.monotonic() + lease
                    command = req.get("command")
                    if command not in {
                        "HEARTBEAT",
                        "ARM",
                        "STOP",
                        "LAUNCH",
                        "STATUS",
                        "CLEANUP",
                        "START",
                        "VERIFY_CLOSE",
                    }:
                        raise MaintenanceError("worker_protocol")
                    if time.monotonic() >= expires:
                        raise MaintenanceError("worker_ttl_expired")
                    remaining = max(0.01, expires - time.monotonic())
                    if not operation_lock.acquire(timeout=remaining):
                        raise MaintenanceError("operation_timeout")
                    operation_deadline = time.monotonic() + RPC_TIMEOUTS[command]
                    with lease_lock:
                        in_flight[0] = operation_deadline
                    try:
                        try:
                            if command == "HEARTBEAT":
                                response = {"event": "ALIVE", "snapshot": owner._service_exact()}
                            elif command == "ARM":
                                owner.arm()
                                response = {"event": "ARMED"}
                            elif command == "STOP":
                                owner.stop_service()
                                response = {"event": "STOPPED"}
                            elif command == "LAUNCH":
                                response = owner.launch()
                            elif command == "STATUS":
                                response = owner.status()
                            elif command == "CLEANUP":
                                owner.cleanup()
                                response = {"event": "CLEANED"}
                            elif command == "START":
                                owner.start_service()
                                response = {"event": "STARTED"}
                            else:
                                response = owner.verify_close()
                        finally:
                            operation_lock.release()
                    except BaseException as exc:
                        if time.monotonic() >= operation_deadline:
                            try:
                                if owner.state.get("armed"):
                                    owner.recover()
                            finally:
                                with lease_lock:
                                    in_flight[0] = None
                                shutdown.set()
                            raise MaintenanceError("operation_lease_expired") from exc
                        with lease_lock:
                            lease_deadline[0] = time.monotonic() + lease
                            in_flight[0] = None
                        raise
                    with lease_lock:
                        operation_overdue = time.monotonic() >= operation_deadline
                        if not operation_overdue:
                            lease_deadline[0] = time.monotonic() + lease
                            in_flight[0] = None
                    if operation_overdue:
                        try:
                            if owner.state.get("armed"):
                                owner.recover()
                        finally:
                            with lease_lock:
                                in_flight[0] = None
                            shutdown.set()
                        raise MaintenanceError("operation_lease_expired")
                    response = dict(response)
                    response["request_id"] = request_id
                    client.sendall((json.dumps(response) + "\n").encode())
                except (MaintenanceError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    client.sendall(
                        (json.dumps(_error_response(request_id, str(exc))) + "\n").encode()
                    )
                except (TimeoutError, BrokenPipeError, ConnectionError):
                    pass
    finally:
        shutdown.set()
        guard.join(timeout=2)
        try:
            if owner.state.get("armed"):
                owner.recover()
        finally:
            server.close()
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (socket_stat.st_dev, socket_stat.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
            lock.close()


class RemoteRPC:
    """Transport-neutral client; callers inject a reviewed local transport."""

    def __init__(self, plan, call, retries=RPC_RETRIES):
        validate_plan(plan)
        if type(retries) is not int or retries < 0 or retries > RPC_RETRIES:
            raise MaintenanceError("invalid_rpc_policy")
        self.plan, self.call, self.retries = plan, call, retries
        self._lock = threading.Lock()

    def heartbeat_if_idle(self):
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self.rpc("HEARTBEAT", _locked=True)
        finally:
            self._lock.release()

    def rpc(self, command, timeout=None, _locked=False):
        if self.plan.get("execution_enabled") is not False:
            raise MaintenanceError("execution_disabled")
        if command not in {
            "HEARTBEAT",
            "ARM",
            "STOP",
            "LAUNCH",
            "STATUS",
            "CLEANUP",
            "START",
            "VERIFY_CLOSE",
        }:
            raise MaintenanceError("invalid_rpc_command")
        timeout = RPC_TIMEOUTS[command] if timeout is None else timeout
        if (
            type(timeout) not in (int, float)
            or timeout <= 0
            or timeout > RPC_TIMEOUTS[command]
        ):
            raise MaintenanceError("invalid_rpc_timeout")
        if not _locked:
            with self._lock:
                return self.rpc(command, timeout, _locked=True)
        attempts = 1 + (self.retries if command in {"HEARTBEAT", "STATUS"} else 0)
        last = None
        for _ in range(attempts):
            request_id, nonce = secrets.token_hex(16), secrets.token_urlsafe(24)
            try:
                row = self.call(
                    {
                        "plan_sha256": digest(self.plan),
                        "command": command,
                        "request_id": request_id,
                        "nonce": nonce,
                    },
                    timeout,
                )
            except (TimeoutError, ConnectionError) as exc:
                last = exc
                continue
            if not isinstance(row, dict):
                raise MaintenanceError("transport_protocol")
            if row.get("request_id") != request_id:
                raise MaintenanceError("transport_protocol")
            if row.get("event") == "ERROR":
                raise MaintenanceError(str(row.get("error") or "worker_error"))
            return row
        raise MaintenanceError("transport_timeout") from last


def coordinate(remotes, plan, now=time.monotonic, sleep=time.sleep):
    """Paired fail-closed coordinator; execution remains disabled by plan gate."""
    validate_plan(plan)
    if plan.get("execution_enabled") is not False:
        raise MaintenanceError("execution_disabled")
    if len(remotes) != 2:
        raise MaintenanceError("preflight_failed")
    armed = False
    passed = False
    result = {"status": "failed", "restoration": "unconfirmed"}
    heartbeat_stop = threading.Event()
    heartbeat_failure = []

    def heartbeat_loop(targets=remotes):
        while not heartbeat_stop.wait(min(HEARTBEAT_TTL / 3, 15)):
            try:
                rows = []
                for remote in targets:
                    if hasattr(remote, "heartbeat_if_idle"):
                        row = remote.heartbeat_if_idle()
                    else:
                        row = remote.rpc("HEARTBEAT")
                    if row is not None:
                        rows.append(row)
                if any(row.get("event") != "ALIVE" for row in rows):
                    raise MaintenanceError("worker_failed")
            except BaseException as exc:
                heartbeat_failure.append(exc)
                return

    heartbeat_threads = []
    try:
        paired_rpc(remotes, "HEARTBEAT", "ALIVE")
        armed = True
        for index, remote in enumerate(remotes):
            thread = threading.Thread(
                target=lambda r=remote: heartbeat_loop([r]),
                name=f"r6-coordinator-heartbeat-{index}", daemon=True,
            )
            heartbeat_threads.append(thread)
            thread.start()
        paired_rpc(remotes, "ARM", "ARMED")
        paired_rpc(remotes, "STOP", "STOPPED")
        started = remotes[1].rpc("LAUNCH")
        if started.get("event") != "EXPERIMENT_STARTED":
            raise MaintenanceError("experiment_failed")
        deadline = now() + plan["experiment_ttl"]
        while now() < deadline:
            if heartbeat_failure:
                raise MaintenanceError("heartbeat_failed")
            # Every progress poll renews both service leases, including the idle node.
            paired_rpc(remotes, "HEARTBEAT", "ALIVE")
            child = remotes[1].rpc("STATUS").get("child")
            if child_outcome(child) == "success":
                passed = True
                break
            sleep(1)
        if not passed:
            raise MaintenanceError("experiment_timeout")
    except BaseException as exc:
        result["reason"] = type(exc).__name__
    finally:
        if armed:
            failures = []
            try:
                paired_rpc_best_effort(remotes, "CLEANUP", "CLEANED")
            except BaseException as exc:
                failures.append("cleanup:" + type(exc).__name__)
            try:
                paired_rpc_best_effort(remotes, "START", "STARTED")
            except BaseException as exc:
                failures.append("start:" + type(exc).__name__)
            try:
                rows = paired_rpc(remotes, "VERIFY_CLOSE", "VERIFIED_CLOSED")
                if any(r.get("stable_seconds", 0) < STABLE_SECONDS for r in rows):
                    failures.append("verify:unstable")
            except BaseException as exc:
                failures.append("verify:" + type(exc).__name__)
            if failures:
                result["recovery_reason"] = ";".join(failures)
            else:
                result["restoration"] = "process_stable"
        heartbeat_stop.set()
        for thread in heartbeat_threads:
            thread.join(timeout=2)
    if passed and result["restoration"] == "process_stable":
        result["status"] = "observation_passed"
    return result
