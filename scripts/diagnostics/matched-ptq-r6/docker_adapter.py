"""Local, fail-closed Docker adapter for the matched PTQ R6 owner.

The adapter is deliberately small: it translates the methods consumed by
``maintenance.LocalOwner`` into local Docker CLI calls.  It never logs Docker
output and turns subprocess failures into fixed error categories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:  # maintenance must be imported under its canonical module name.
    from maintenance import DockerAbsent, MaintenanceError
except ImportError:  # pragma: no cover - permits direct standalone import.

    class MaintenanceError(RuntimeError):
        pass

    class DockerAbsent(LookupError):
        pass


CID_RE = re.compile(r"[a-f0-9]{64}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
IMAGE_REF_RE = re.compile(r"[^\s@]+(?:/[^\s@]+)*@sha256:[a-f0-9]{64}\Z")
HASH_RE = re.compile(r"[a-f0-9]{64}\Z")
SERVICE_NAME = "fixture-service"
SERVICE_IMAGE_ID = "sha256:" + "f" * 64
DOCKER_SOCKET = "unix:///var/run/docker.sock"
DEFAULT_TIMEOUT = 65.0
INSPECT_TIMEOUT = 20.0


@dataclass(frozen=True)
class ServiceSpec:
    """Exact protected service identity supplied by deployment review."""

    host: str
    cid: str
    config_hash: str


def _config_hash(row: Mapping[str, Any]) -> str:
    """Match the service identity hash byte-for-byte."""
    config = row.get("Config") or {}
    host = row.get("HostConfig") or {}
    return hashlib.sha256(
        json.dumps(
            {"Config": config, "HostConfig": host},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _argv_labels(argv: Sequence[str]) -> tuple[str, dict[str, str]]:
    name = None
    labels: dict[str, str] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--name" and index + 1 < len(argv):
            name = argv[index + 1]
            index += 2
            continue
        if item == "--label" and index + 1 < len(argv):
            value = argv[index + 1]
            if "=" in value:
                key, label = value.split("=", 1)
                labels[key] = label
            index += 2
            continue
        index += 1
    if not name or not labels:
        raise MaintenanceError("invalid_adapter_config")
    return name, labels


class DockerAdapter:
    """Concrete local Docker CLI implementation used by ``LocalOwner``.

    ``create_argv`` is copied and frozen at construction.  A caller must pass
    the same reviewed argv back to :meth:`create`; this prevents a mutable plan
    from changing the image or privileges after review.
    """

    def __init__(
        self,
        service: ServiceSpec | Mapping[str, str],
        image_ref: str,
        image_id: str,
        create_argv: Sequence[str],
        *,
        run: Callable[..., Any] = subprocess.run,
        timeout: float = DEFAULT_TIMEOUT,
        service_name: str = SERVICE_NAME,
        service_image_id: str = SERVICE_IMAGE_ID,
    ) -> None:
        if isinstance(service, Mapping):
            try:
                service = ServiceSpec(service["host"], service["cid"], service["config_hash"])
            except (KeyError, TypeError) as exc:
                raise MaintenanceError("invalid_adapter_config") from exc
        if not isinstance(service, ServiceSpec):
            raise MaintenanceError("invalid_adapter_config")
        if not isinstance(service.host, str) or not service.host or "\x00" in service.host:
            raise MaintenanceError("invalid_adapter_config")
        if not CID_RE.fullmatch(service.cid) or not HASH_RE.fullmatch(service.config_hash):
            raise MaintenanceError("invalid_adapter_config")
        if not isinstance(image_ref, str) or not IMAGE_REF_RE.fullmatch(image_ref):
            raise MaintenanceError("invalid_adapter_config")
        if not isinstance(image_id, str) or not IMAGE_ID_RE.fullmatch(image_id):
            raise MaintenanceError("invalid_adapter_config")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise MaintenanceError("invalid_adapter_config")
        argv = tuple(create_argv)
        if not argv or any(not isinstance(item, str) for item in argv):
            raise MaintenanceError("invalid_adapter_config")
        if argv[:2] != ("docker", "create") or argv.count(image_ref) != 1:
            raise MaintenanceError("invalid_adapter_config")
        name, labels = _argv_labels(argv)
        self.service = service
        self.image_ref = image_ref
        self.image_id = image_id
        self.create_argv = argv
        self._owned_name = "/" + name
        self._owned_labels = labels
        self._run = run
        self.timeout = float(timeout)
        self.service_name = service_name
        self.service_image_id = service_image_id

    def _command(
        self, argv: Sequence[str], *, timeout: float | None = None, check: bool = True
    ) -> Any:
        """Run a bounded command while discarding all command output."""
        limit = self.timeout if timeout is None else float(timeout)
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"DOCKER_HOST", "DOCKER_CONTEXT"}
        }
        try:
            return self._run(
                ["docker", "--host", DOCKER_SOCKET, *argv[1:]],
                capture_output=True,
                text=True,
                timeout=limit,
                check=check,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise MaintenanceError("transport_timeout") from exc
        except subprocess.CalledProcessError as exc:
            raise MaintenanceError("transport_exit") from exc
        except OSError as exc:
            raise MaintenanceError("transport_error") from exc
        except (TypeError, ValueError) as exc:
            raise MaintenanceError("transport_protocol") from exc

    @staticmethod
    def _json_row(result: Any) -> dict[str, Any]:
        try:
            payload = json.loads(result.stdout)
            if (
                not isinstance(payload, list)
                or len(payload) != 1
                or not isinstance(payload[0], dict)
            ):
                raise ValueError
            return payload[0]
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MaintenanceError("transport_protocol") from exc

    def _inspect_row(self, cid: str, *, absent_ok: bool = False) -> dict[str, Any]:
        if not CID_RE.fullmatch(cid):
            raise MaintenanceError("experiment_identity_malformed")
        result = self._command(["docker", "inspect", cid], timeout=INSPECT_TIMEOUT, check=False)
        if getattr(result, "returncode", 0) != 0:
            detail = str(getattr(result, "stderr", "")).lower()
            if absent_ok and ("no such object" in detail or "no such container" in detail):
                raise DockerAbsent(cid)
            raise MaintenanceError("transport_exit")
        return self._json_row(result)

    def inspect(self, cid: str) -> dict[str, Any]:
        """Inspect an experiment container; absence is explicit and safe."""
        return self._inspect_row(cid, absent_ok=True)

    def exists(self, cid: str) -> bool:
        try:
            self._inspect_row(cid, absent_ok=True)
        except DockerAbsent:
            return False
        return True

    def _owned(self, row: Mapping[str, Any], cid: str) -> bool:
        labels = (row.get("Config") or {}).get("Labels") or {}
        return (
            row.get("Id") == cid
            and row.get("Image") == self.image_id
            and row.get("Name") == self._owned_name
            and all(labels.get(key) == value for key, value in self._owned_labels.items())
        )

    def _require_owned(self, cid: str) -> dict[str, Any]:
        row = self._inspect_row(cid)
        if not self._owned(row, cid):
            raise MaintenanceError("foreign_experiment")
        return row

    def attest_image_ref(self, image_ref: str) -> str:
        if image_ref != self.image_ref:
            raise MaintenanceError("image_attestation_failed")
        result = self._command(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.image_ref],
            timeout=INSPECT_TIMEOUT,
        )
        value = getattr(result, "stdout", "").strip()
        if value != self.image_id:
            raise MaintenanceError("image_attestation_failed")
        return value

    def create(self, argv: Sequence[str]) -> str:
        if tuple(argv) != self.create_argv:
            raise MaintenanceError("invalid_plan")
        result = self._command(self.create_argv)
        cid = getattr(result, "stdout", "").strip()
        if not CID_RE.fullmatch(cid):
            raise MaintenanceError("experiment_identity_malformed")
        return cid

    def start(self, cid: str) -> None:
        self._require_owned(cid)
        self._command(["docker", "start", cid])

    def remove(self, cid: str) -> None:
        try:
            self._require_owned(cid)
        except DockerAbsent:
            return
        result = self._command(["docker", "rm", "--force", cid], check=False)
        if getattr(result, "returncode", 0) != 0:
            try:
                self._inspect_row(cid, absent_ok=True)
            except DockerAbsent:
                return
            raise MaintenanceError("transport_exit")

    def _service_row(self) -> dict[str, Any]:
        row = self._inspect_row(self.service.cid)
        state = row.get("State") or {}
        config = row.get("Config") or {}
        health = state.get("Health") or {}
        test = config.get("Healthcheck") or {}
        health_test = test.get("Test") or []
        return {
            "exact": (
                row.get("Id") == self.service.cid
                and row.get("Name") == "/" + self.service_name
                and row.get("Image") == self.service_image_id
                and _config_hash(row) == self.service.config_hash
            ),
            "running": state.get("Running") is True,
            "restarting": state.get("Restarting") is True,
            "oom_killed": state.get("OOMKilled") is True,
            "restart_count": row.get("RestartCount"),
            "started_at": state.get("StartedAt"),
            "health_configured": bool(health_test and health_test[0] != "NONE"),
            "health": health.get("Status") or "none",
        }

    def service_inspect(self) -> dict[str, Any]:
        return self._service_row()

    def _require_service(self) -> None:
        if not self.service_inspect().get("exact"):
            raise MaintenanceError("identity_mismatch")

    def service_stop(self) -> None:
        self._require_service()
        self._command(["docker", "stop", "--time", "30", self.service.cid])

    def service_start(self) -> None:
        self._require_service()
        self._command(["docker", "start", self.service.cid])

    def child_status(self, cid: str | None) -> Mapping[str, Any] | None:
        if cid is None:
            return None
        try:
            row = self._inspect_row(cid, absent_ok=True)
        except DockerAbsent:
            return None
        return row.get("State") if isinstance(row.get("State"), Mapping) else None
