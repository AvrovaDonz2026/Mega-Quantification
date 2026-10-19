#!/usr/bin/env python3
"""Detached sequential candidate matrix over the existing parity-v7 runner.

Each scheme uses two owned servers for the five fixtures, so source/model/image/runtime
identity and owned-container cleanup remain enforced by that runner. Reports
contain only v7's sanitized summaries; this wrapper never stores native output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RUNNER = HERE.parent / "parity-v7" / "run_v7.py"
SCHEMES = ("w4a4", "w4a8")
CASES = ("python_code", "arithmetic", "red_square", "blue_square", "ocr_3729")
TERMINAL = {"passed", "parity_failed", "failed", "invalid", "crashed", "cleanup_failed"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def jobs() -> list[dict[str, str]]:
    return [{"id": scheme, "scheme": scheme} for scheme in SCHEMES]


def run_job(config: dict[str, Any], root: Path, job: dict[str, str]) -> dict[str, Any]:
    job_dir = root / "jobs" / job["id"]
    runner = resolve(config.get("runner", str(HERE / "matrix_runner.py")), config["config_dir"])
    source = config["sources"][config.get("source_for", "all-beta")]
    model = config["models"][job["scheme"]]
    runner_root = job_dir / "runner"
    client = resolve(config.get("client", str(HERE / "expected_client.py")), config["config_dir"])
    image = config["image"]
    command = [sys.executable, str(runner), "--root", str(runner_root), "--root-job-id", job["scheme"], "--source", str(resolve(source["source"], config["config_dir"])), "--source-revision", str(source["revision"]), "--source-manifest", str(resolve(source["manifest"], config["config_dir"])), "--source-archive", str(resolve(source["archive"], config["config_dir"])), "--model", str(resolve(model["path"], config["config_dir"])), "--model-scheme", job["scheme"], "--model-provenance", str(resolve(model["provenance"], config["config_dir"])), "--fixture-repo", str(resolve(config["fixture_repo"], config["config_dir"])), "--client", str(client), "--image", image, "--image-id", config.get("image_id", image), "--fla-use-fast-ops", config.get("fla_use_fast_ops", "0"), "--force-shared-decode", config.get("force_shared_decode", "0"), "--port", str(config.get("port", 32230) + len(job["id"]))]
    job.update({"state": "running", "started_at_utc": now(), "command": command, "runner_root": str(runner_root)})
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout = (job_dir / "runner.stdout.log").open("w")
    stderr = (job_dir / "runner.stderr.log").open("w")
    child_process = subprocess.Popen(command, cwd=runner.parent, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    status_path = runner_root / "status.json"
    deadline = time.monotonic() + float(config.get("job_timeout_seconds", 86400))
    while time.monotonic() < deadline:
        if status_path.is_file():
            child = read_json(status_path)
            if child.get("state") in TERMINAL:
                report_path = runner_root / "candidate-matrix-report.json"
                job["child_status"] = child
                job["report"] = str(report_path) if report_path.is_file() else None
                if child.get("state") == "passed" and report_path.is_file():
                    job.update({"state": "passed", "exit_code": 0, "ended_at_utc": now()})
                else:
                    job.update({"state": child.get("state") if child.get("state") == "parity_failed" else "failed", "exit_code": 1, "error": f"runner state {child.get('state')}", "ended_at_utc": now()})
                stdout.close(); stderr.close()
                return job
        if child_process.poll() is not None:
            stdout.close(); stderr.close()
            job.update({"state": "failed", "exit_code": child_process.returncode, "error": "runner exited without terminal status", "ended_at_utc": now()})
            return job
        time.sleep(float(config.get("poll_interval_seconds", 2)))
    child_process.terminate()
    stdout.close(); stderr.close()
    job.update({"state": "failed", "exit_code": None, "error": "runner timeout", "ended_at_utc": now()})
    return job


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("launch", "run", "status"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main() -> int:
    args = parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.action == "status":
        print(json.dumps(read_json(root / "status.json") if (root / "status.json").is_file() else {"state": "missing", "root": str(root)}, indent=2, sort_keys=True))
        return 0
    if args.action == "launch":
        if args.config is None or root.exists():
            raise SystemExit("launch requires --config and a fresh root")
        config_path = args.config.expanduser().resolve()
        config = read_json(config_path)
        config["config_dir"] = str(config_path.parent)
        root.mkdir(parents=True)
        selected = jobs()
        write_json(root / "config.json", config)
        write_json(root / "status.json", {"schema_version": 1, "state": "running", "jobs": [{"id": job["id"], "scheme": job["scheme"], "state": "pending"} for job in selected], "started_at_utc": now()})
        command = [sys.executable, str(Path(__file__).resolve()), "--action", "run", "--root", str(root)]
        with (root / "driver.log").open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        status = read_json(root / "status.json")
        status["driver_pid"] = child.pid
        write_json(root / "status.json", status)
        print(json.dumps({"state": "running", "pid": child.pid, "root": str(root)}, indent=2, sort_keys=True))
        return 0
    if not (root / "config.json").is_file():
        raise SystemExit("run requires config.json")
    config = read_json(root / "config.json")
    config["config_dir"] = Path(config["config_dir"])
    status = read_json(root / "status.json")
    job_records = status["jobs"]
    for record in job_records:
        record["state"] = "running"
        write_json(root / "status.json", status)
        job = run_job(config, root, record)
        write_json(root / "jobs" / f"{job['id']}.json", job)
        status["jobs"] = [job if item["id"] == job["id"] else item for item in status["jobs"]]
        write_json(root / "status.json", status)
        if job["state"] == "failed":
            status["state"] = "failed"
            status["ended_at_utc"] = now()
            write_json(root / "status.json", status)
            return 1
    status["state"] = "passed" if all(item["state"] == "passed" for item in status["jobs"]) else "parity_failed"
    status["ended_at_utc"] = now()
    write_json(root / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
