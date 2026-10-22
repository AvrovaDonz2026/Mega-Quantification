"""Run the matched static-max/static-MSE PTQ pair in isolated fresh processes.

The default paths are the paths visible inside the validation container.  Tests and
offline local runs may pass explicit path overrides to :func:`supervise_pair`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARMS = ("static-max", "static-MSE")
DEFAULT_RECIPE = Path("/workspace/recipe.yaml")
DEFAULT_RUNNER = Path("/workspace/runner.py")
DEFAULT_EVIDENCE_ROOT = Path("/evidence")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _terminate_group(process: subprocess.Popen[bytes], grace: float = 1.0) -> bool:
    """Terminate a process group and return whether the leader exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=max(0.05, grace))
        except subprocess.TimeoutExpired:
            pass
    deadline = time.monotonic() + max(0.05, grace)
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except (ProcessLookupError, PermissionError):
            return process.poll() is not None
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return process.poll() is not None
    try:
        process.wait(timeout=max(0.05, grace))
    except subprocess.TimeoutExpired:
        return False
    try:
        os.killpg(process.pid, 0)
    except (ProcessLookupError, PermissionError):
        return True
    return False


def _run_comparator(compare_argv: list[str], log_path: Path, timeout: float) -> dict[str, Any]:
    """Run comparison in its own group so a hung comparator cannot outlive the supervisor."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            compare_argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": "/workspace/src",
            },
            start_new_session=True,
        )
        try:
            process.wait(timeout=max(0.05, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process)
        finally:
            if process.poll() is None:
                _terminate_group(process)
    structured: dict[str, Any] | None = None
    if not timed_out:
        try:
            parsed = json.loads(log_path.read_text())
        except (OSError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            structured = parsed
    return {
        "status": "timeout"
        if timed_out
        else ("completed" if process.returncode == 0 else "failed"),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "process_group_exited": process.returncode is not None,
        "command_sha256": _sha256_json(compare_argv),
        "log_file": "logs/pair_compare.log",
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "result": structured,
    }


def _run_arm(
    arm: str,
    *,
    runner: Path,
    recipe: Path,
    evidence_root: Path,
    timeout: float,
    log_root: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    evidence_dir = evidence_root / arm
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(runner),
        "--recipe",
        str(recipe),
        "--arm",
        arm,
        "--execute",
        "--evidence-dir",
        str(evidence_dir),
    ]
    command_hash = _sha256_json(argv)
    log_path = log_root / f"{arm}.log"
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/workspace/src"})
    if env_overrides:
        env.update(env_overrides)
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    exited = False
    with log_path.open("wb"):
        process = subprocess.Popen(
            argv,
            # Do not persist runner output: stdout can contain prompts/responses
            # and stderr tracebacks can echo source paths or argv values.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
            exited = True
        except subprocess.TimeoutExpired:
            timed_out = True
            exited = _terminate_group(process)
        finally:
            if process.poll() is None:
                exited = _terminate_group(process)
            returncode = process.returncode
    elapsed = time.monotonic() - started
    status = "timeout" if timed_out else ("completed" if returncode == 0 else "failed")
    return {
        "arm": arm,
        "command_sha256": command_hash,
        "input_identity": {
            "recipe": "recipe.yaml",
            "recipe_sha256": _sha256_bytes(recipe.read_bytes()) if recipe.is_file() else None,
        },
        "elapsed_seconds": round(elapsed, 6),
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "process_group_exited": exited and returncode is not None,
        "log_file": f"logs/{arm}.log",
    }


def supervise_pair(
    *,
    recipe: Path = DEFAULT_RECIPE,
    runner: Path = DEFAULT_RUNNER,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    timeout_per_arm: float = 3600.0,
    pair_timeout: float = 7200.0,
    compare_script: Path | None = None,
    compare_timeout: float = 300.0,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run arms sequentially and atomically write ``pair-summary.json``."""
    recipe, runner, evidence_root = Path(recipe), Path(runner), Path(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    for index, arm in enumerate(ARMS):
        remaining = pair_timeout - (time.monotonic() - started)
        if remaining <= 0:
            records.append(
                {"arm": arm, "status": "timeout", "returncode": None, "elapsed_seconds": 0.0}
            )
            break
        result = _run_arm(
            arm,
            runner=runner,
            recipe=recipe,
            evidence_root=evidence_root,
            timeout=min(timeout_per_arm, remaining),
            log_root=evidence_root / "logs",
            env_overrides=env_overrides,
        )
        records.append(result)
        if result["status"] != "completed":
            if index == 0:
                records.append(
                    {
                        "arm": ARMS[1],
                        "status": "skipped",
                        "returncode": None,
                        "elapsed_seconds": 0.0,
                    }
                )
            break
    while len(records) < 2:
        records.append(
            {
                "arm": ARMS[len(records)],
                "status": "skipped",
                "returncode": None,
                "elapsed_seconds": 0.0,
            }
        )

    compare_script = (
        Path(compare_script)
        if compare_script is not None
        else Path(__file__).with_name("pair_compare.py")
    )
    compare: dict[str, Any]
    if all(r.get("status") == "completed" for r in records) and compare_script.is_file():
        compare_log = evidence_root / "logs" / "pair_compare.log"
        compare_argv = [sys.executable, str(compare_script), "--evidence-root", str(evidence_root)]
        compare = _run_comparator(compare_argv, compare_log, compare_timeout)
    elif not all(r.get("status") == "completed" for r in records):
        compare = {"status": "skipped", "returncode": None, "reason": "arm_incomplete"}
    else:
        compare = {"status": "failed", "returncode": None, "reason": "comparator_missing"}
    observation_passed = (
        all(r.get("status") == "completed" for r in records)
        and compare.get("status") == "completed"
        and isinstance(compare.get("result"), dict)
        and compare["result"].get("ptq_comparable") is True
    )
    summary = {
        "schema": "matched-ptq-r6-1022-supervisor-v1",
        "status": "observation_passed" if observation_passed else "failed",
        "input_identity": {
            "recipe": "recipe.yaml",
            "recipe_sha256": _sha256_bytes(recipe.read_bytes()) if recipe.is_file() else None,
        },
        "arms": records,
        "compare": compare,
    }
    _atomic_json(evidence_root / "pair-summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--compare-script", type=Path, default=Path(__file__).with_name("pair_compare.py")
    )
    parser.add_argument("--arm-timeout", type=float, default=3600.0)
    parser.add_argument("--pair-timeout", type=float, default=7200.0)
    args = parser.parse_args(argv)
    summary = supervise_pair(
        recipe=args.recipe,
        runner=args.runner,
        evidence_root=args.evidence_root,
        timeout_per_arm=args.arm_timeout,
        pair_timeout=args.pair_timeout,
        compare_script=args.compare_script,
    )
    return 0 if summary["status"] == "observation_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
