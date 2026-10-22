"""Bounded waiter for the one trusted R6 experiment container."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from maintenance import MaintenanceError, _identity, digest, validate_plan


def wait(
    plan: dict,
    journal: dict,
    adapter,
    timeout: float | None = None,
    now=time.monotonic,
    sleep=time.sleep,
) -> dict:
    validate_plan(plan)
    if journal.get("plan_sha256") != digest(plan):
        raise MaintenanceError("identity_mismatch")
    cid = journal.get("experiment_id")
    if not isinstance(cid, str) or re.fullmatch(r"[0-9a-f]{64}", cid) is None:
        raise MaintenanceError("identity_mismatch")
    deadline = now() + (timeout if timeout is not None else plan["experiment_ttl"])
    while now() <= deadline:
        try:
            row = adapter.inspect(cid)
        except Exception as exc:
            raise MaintenanceError("container_unknown") from exc
        if not isinstance(row, dict) or not _identity(row, plan, cid):
            raise MaintenanceError("identity_mismatch")
        state = row.get("State") or {}
        running = state.get("Running", row.get("Running"))
        if running is True:
            sleep(min(1.0, max(0.01, deadline - now())))
            continue
        status = state.get("Status", row.get("Status"))
        code = state.get("ExitCode", row.get("ExitCode"))
        if status == "exited" and code == 0:
            return {"event": "SUCCEEDED", "container_id": cid}
        if status == "exited" and isinstance(code, int):
            raise MaintenanceError("experiment_failed")
        raise MaintenanceError("container_unknown")
    raise MaintenanceError("experiment_timeout")


def wait_from_files(plan_path: str | Path, journal_path: str | Path, adapter, **kwargs):
    plan = json.loads(Path(plan_path).read_text())
    journal = json.loads(Path(journal_path).read_text())
    return wait(plan, journal, adapter, **kwargs)
