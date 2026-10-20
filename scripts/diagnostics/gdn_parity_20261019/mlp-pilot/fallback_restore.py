#!/usr/bin/env python3
"""systemd ExecStopPost fallback, restricted to this run's exact containers."""

import json
from pathlib import Path

import supervisor as s


def main():
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    run_options = s.parse_run_config(config)
    s._ACTIVE_SOURCE_REVISIONS = frozenset({run_options["source_revision"]})
    s._ACTIVE_JOBS = run_options["jobs"]
    status_path = Path(config["root_path"]) / "supervisor-status.json"
    if not status_path.exists():
        return 0
    status = json.loads(status_path.read_text())
    if status.get("glm_restore_required") is not True:
        return 0
    started = status.get("supervisor_started_at_epoch")
    if type(started) not in (int, float):
        return 1
    try:
        count = s.cleanup_experiment_containers(started)
        status["fallback_cleanup_removed"] = count
    except Exception as exc:
        status["fallback_cleanup_error_type"] = type(exc).__name__
    glm_id = config["expected_glm_id_full"]
    try:
        row = s.docker_inspect(glm_id)
        s.verify_glm_identity(config, row)
        restored = False
        for attempt in range(3):
            try:
                result = s.command(["docker", "start", glm_id], timeout=45)
                if result.returncode == 0:
                    restored = True
                    break
            except Exception:
                pass
            if attempt < 2:
                s.time.sleep(2)
        if restored:
            snap = s.wait_glm_stable(config)
            status["fallback_restore"] = {
                "running": snap.get("running"),
                "health": snap.get("health"),
                "observed_seconds": snap["observed_seconds"],
                "at_utc": s.utc_now(),
            }
        else:
            status["fallback_restore"] = {"running": False, "at_utc": s.utc_now()}
        if restored:
            status["glm_restore_required"] = False
        s.atomic_json(status_path, status)
        return 0 if restored else 1
    except Exception as exc:
        status["fallback_restore_error_type"] = type(exc).__name__
        s.atomic_json(status_path, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
