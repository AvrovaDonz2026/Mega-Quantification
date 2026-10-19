#!/usr/bin/env python3
"""Sequential, detached driver for the 2026-10-19 GPU experiment matrix.

The driver owns only its evidence directories and subprocesses.  It never
enumerates or terminates unrelated containers/processes.  The parity runner
does the Docker, source, model, image, and GPU gates for each parity job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
import importlib.util
_LOCK_SPEC = importlib.util.spec_from_file_location("gdn_parity_source_lock", HERE.parent / "source_lock.py")
if _LOCK_SPEC is None or _LOCK_SPEC.loader is None:
    raise RuntimeError("cannot load bundle source lock")
_LOCK_MODULE = importlib.util.module_from_spec(_LOCK_SPEC)
_LOCK_SPEC.loader.exec_module(_LOCK_MODULE)
DEFAULT_ROOT = Path("/home/zhiman/megaquant-aitop-validation-20261019")
DEFAULT_FULL_MODEL_ROOT = Path("/home/zhiman/megaquant-aitop-validation-20261014/full-model")
DEFAULT_IMAGE = "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf"
TERMINAL = {"passed", "parity_failed", "failed", "invalid", "crashed", "cleanup_failed"}
OWNER_LABEL = "io.megaquant.experiment-driver.owner"
OWNER_VALUE = "aitop-experiment-driver"
JOB_LABEL = "io.megaquant.experiment-driver.job"
SOURCE_HASH_PATHS = {
    "packed": Path("python/sglang/kernels/ops/attention/fla/fused_recurrent.py"),
    "verify": Path("python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"),
    "gating": Path("python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py"),
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def config_paths(value: Any, base: Path) -> Any:
    """Resolve path-valued config keys while leaving arbitrary strings alone."""
    if isinstance(value, dict):
        return {key: config_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [config_paths(item, base) for item in value]
    return value


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"links are not allowed in source archive: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)
    children = [item for item in destination.iterdir() if item.name != ".DS_Store"]
    directories = [item for item in children if item.is_dir()]
    return directories[0] if len(children) == 1 and len(directories) == 1 else destination


def source_entry(config: dict[str, Any], name: str) -> dict[str, Any]:
    entries = config.get("source_variants", {})
    if name not in entries or not isinstance(entries[name], dict):
        raise RuntimeError(f"source_variants.{name} is required")
    entry = dict(entries[name])
    package = entry.get("package", config.get("source_package"))
    if package:
        package = resolve(package, config["_config_dir"])
        stem = f"sglang-{name}"
        entry.setdefault("archive", package / f"{stem}.tar.gz")
        entry.setdefault("manifest", package / f"{stem}.manifest.json")
        entry.setdefault("path", package / stem)
    if "archive" not in entry and "path" not in entry:
        raise RuntimeError(f"source_variants.{name} needs archive or path")
    return entry


def materialize_source(config: dict[str, Any], name: str, job_dir: Path) -> dict[str, Any]:
    entry = source_entry(config, name)
    source_archive = resolve(entry.get("archive", entry.get("source_archive")), config["_config_dir"]) if entry.get("archive", entry.get("source_archive")) else None
    supplied_path = entry.get("path", entry.get("source"))
    candidate = resolve(supplied_path, config["_config_dir"]) if supplied_path else None
    if candidate is not None and candidate.is_dir():
        source = candidate
    elif source_archive is not None:
        source = safe_extract(source_archive, job_dir / "source")
    else:
        raise RuntimeError(f"source_variants.{name} has no source path")
    if not source.is_dir():
        raise RuntimeError(f"source path is not a directory: {source}")
    manifest_value = entry.get("manifest", entry.get("source_manifest"))
    if manifest_value:
        manifest = resolve(manifest_value, config["_config_dir"])
    else:
        candidates = [source / "manifest.json", source / "source-manifest.json", source.parent / "manifest.json"]
        manifest = next((candidate for candidate in candidates if candidate.is_file()), None)
    if manifest is None or not manifest.is_file():
        raise RuntimeError(f"source_variants.{name} manifest is missing")
    if source_archive is None:
        archive_value = config.get("source_archive")
        if archive_value:
            source_archive = resolve(archive_value, config["_config_dir"])
    if source_archive is None or not source_archive.is_file():
        raise RuntimeError(f"source_variants.{name} archive is missing (required by parity-v7)")
    revision = entry.get("revision", entry.get("source_revision", config.get("source_revision")))
    if not revision:
        summary_value = entry.get("summary", config.get("source_summary"))
        summary = resolve(summary_value, config["_config_dir"]) if summary_value else source.parent / "summary.json"
        if summary.is_file():
            data = read_json(summary)
            revisions = data.get("revisions", data.get("commits", {})) if isinstance(data, dict) else {}
            if isinstance(revisions, list):
                for item in revisions:
                    if isinstance(item, dict) and item.get("name") in {name, "vision-mtp", "sglang-" + name}:
                        revision = item.get("commit") or item.get("revision")
                        if revision:
                            break
            elif isinstance(revisions, dict):
                revision = revisions.get(name) or revisions.get("sglang-" + name)
            revision = revision or (data.get("revision") if isinstance(data, dict) else None) or (data.get("commit") if isinstance(data, dict) else None)
    if not revision:
        raise RuntimeError(f"source_variants.{name} source revision is missing")
    return {"name": name, "source": source, "archive": source_archive, "manifest": manifest, "revision": str(revision)}


def model_entry(config: dict[str, Any], scheme: str) -> dict[str, Any]:
    models = config.get("models", {})
    value = models.get(scheme, config.get("model"))
    if value is None:
        value = {"path": DEFAULT_FULL_MODEL_ROOT / "models" / scheme, "provenance": DEFAULT_FULL_MODEL_ROOT / "model-identity.json"}
    if not isinstance(value, dict):
        raise RuntimeError(f"models.{scheme} is required")
    result = dict(value)
    for key in ("path", "provenance"):
        if key not in result:
            raise RuntimeError(f"models.{scheme}.{key} is required")
        result[key] = str(resolve(result[key], config["_config_dir"]))
    return result


def make_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for source_name, requested_variant in (("original", "original"), ("decode-beta", "decode-beta"), ("all-beta", "all-beta"), ("shared-decode", "all-beta")):
        jobs.append({"id": f"beta-{source_name}", "kind": "beta", "source": source_name, "variant": requested_variant, "fast_ops": "0", "force_shared_decode": "0", "repetitions": {"baseline": 2, "mtp": 2}})
    for source_name, force in (("decode-beta", "0"), ("all-beta", "0"), ("shared-decode", "0"), ("shared-decode", "1")):
        jobs.append({"id": f"parity-w4a4-python-code-{source_name}-force{force}", "kind": "parity", "source": source_name, "scheme": "w4a4", "case": "python_code", "fast_ops": "0", "force_shared_decode": force, "repetitions": {"baseline": 2, "mtp": 2}})
    return jobs


def select_jobs(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    jobs = make_jobs(config)
    if phase == "local":
        return [job for job in jobs if job["kind"] == "beta"]
    if phase == "parity":
        return [job for job in jobs if job["kind"] == "parity"]
    return jobs


def update_job(root: Path, job: dict[str, Any], **changes: Any) -> None:
    job.update(changes)
    write_json(root / "jobs" / f"{job['id']}.json", job)
    status = read_json(root / "status.json")
    status["jobs"] = [job if item.get("id") == job["id"] else item for item in status.get("jobs", [])]
    status["updated_at_utc"] = now()
    write_json(root / "status.json", status)


def source_gate(source: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    if len(source["revision"]) != 40 or any(char not in "0123456789abcdef" for char in source["revision"]):
        raise RuntimeError("source revision must be a full 40-character hexadecimal commit")
    lock_proof = _LOCK_MODULE.verify_locked_source(source["revision"], source["archive"], source["name"])
    gates = resolve(HERE.parent / "parity-v7" / "input_gates.py", HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment_driver_input_gates", gates)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import parity input gates")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SGLANG_COMMIT = source["revision"]
    module.ARCHIVE_SHA = sha256(source["archive"])
    module._SOURCE_CACHE.clear()
    proof = module.verify_source(source["source"], source["archive"], source["manifest"])
    if proof.get("commit") != source["revision"] or proof.get("exact_archive_tree") is not True:
        raise RuntimeError("source identity gate failed")
    proof["source_file_sha256"] = {
        str(path.relative_to(source["source"])): sha256(path)
        for path in sorted(source["source"].rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    proof["bundle_lock"] = lock_proof
    write_json(job_dir / "source-gate.json", proof)
    return proof


def gpu_idle() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi unavailable; refusing beta GPU launch")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or result.stdout.strip():
        raise RuntimeError("GPU has active compute processes")


def docker_image_id(image: str) -> str:
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("pinned beta image is unavailable")
    return result.stdout.strip()


def run_beta(config: dict[str, Any], root: Path, job: dict[str, Any], source: dict[str, Any], job_dir: Path) -> tuple[str, int | None, str | None]:
    probe = resolve(config.get("beta_probe", str(HERE.parent / "beta-regression" / "probe.py")), config["_config_dir"])
    report = job_dir / "beta-regression-report.json"
    proof = source_gate(source, job_dir)
    gpu_idle()
    image = str(config.get("image", DEFAULT_IMAGE))
    image_id = docker_image_id(image)
    if image_id != str(config.get("image_id", image)):
        raise RuntimeError("beta image identity differs from pinned image")
    report_dir = job_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    container = f"megaquant-experiment-beta-{job['id']}-{uuid.uuid4().hex[:10]}"
    source_mount = f"type=bind,src={source['source']},dst=/sglang-source,readonly"
    beta_mount = f"type=bind,src={probe.parent},dst=/beta,readonly"
    report_mount = f"type=bind,src={report_dir},dst=/report"
    command = ["docker", "run", "--rm", "--gpus", "all", "--name", container, "--label", f"{OWNER_LABEL}={OWNER_VALUE}", "--label", f"{JOB_LABEL}={job['id']}", "--mount", source_mount, "--mount", beta_mount, "--mount", report_mount, "--env", "PYTHONPATH=/sglang-source/python", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--entrypoint", "python", image, "/beta/probe.py", "--variant", job["variant"], "--fla-use-fast-ops", job["fast_ops"], "--source-revision", source["revision"], "--report", f"/report/{report.name}"]
    started = time.time()
    with (job_dir / "stdout.log").open("w") as stdout, (job_dir / "stderr.log").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    cleanup_probe = subprocess.run(["docker", "inspect", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    job["command"] = command
    job["duration_seconds"] = round(time.time() - started, 3)
    job["source_gate"] = proof
    job["report"] = str(report)
    report = report_dir / report.name
    job["report"] = str(report)
    if cleanup_probe.returncode == 0:
        return "failed", result.returncode, "beta container cleanup failed"
    if result.returncode == 0 and report.is_file():
        observed = read_json(report)
        if observed.get("source_revision") != source["revision"]:
            return "failed", result.returncode, "probe source revision differs from explicit revision"
        imported = observed.get("imported_source_hashes")
        expected = {key: proof["source_file_sha256"].get(str(path)) for key, path in SOURCE_HASH_PATHS.items()}
        if not isinstance(imported, dict) or any(imported.get(key) != value for key, value in expected.items()):
            return "failed", result.returncode, "probe imported source hashes differ from exact source tree"
        return "passed", result.returncode, None
    return "failed", result.returncode, "beta probe failed"


def run_parity(config: dict[str, Any], root: Path, job: dict[str, Any], source: dict[str, Any], job_dir: Path) -> tuple[str, int | None, str | None]:
    runner = resolve(config.get("parity_runner", str(HERE.parent / "parity-v7" / "run_v7.py")), config["_config_dir"])
    model = model_entry(config, job["scheme"])
    runner_root = job_dir / "runner"
    fixture_repo = resolve(config.get("fixture_repo", DEFAULT_FULL_MODEL_ROOT / "fixtures"), config["_config_dir"])
    image = str(config.get("image", DEFAULT_IMAGE))
    command = [sys.executable, str(runner), "--action", "launch", "--root", str(runner_root), "--job-id", job["id"], "--source", str(source["source"]), "--source-revision", source["revision"], "--source-variant", source["name"], "--source-manifest", str(source["manifest"]), "--source-archive", str(source["archive"]), "--model", model["path"], "--model-scheme", job["scheme"], "--model-provenance", model["provenance"], "--fixture-repo", str(fixture_repo), "--image", image, "--image-id", str(config.get("image_id", image)), "--case", job["case"], "--fla-use-fast-ops", job["fast_ops"], "--force-shared-decode", job["force_shared_decode"], "--port", str(config.get("port", 31230))]
    launch_out = subprocess.run(command, cwd=runner.parent, capture_output=True, text=True, check=False)
    (job_dir / "launch.stdout.log").write_text(launch_out.stdout)
    (job_dir / "launch.stderr.log").write_text(launch_out.stderr)
    job["command"] = command
    if launch_out.returncode != 0:
        return "failed", launch_out.returncode, "parity runner launch failed"
    timeout = float(config.get("job_timeout_seconds", 86400))
    deadline = time.monotonic() + timeout
    status_path = runner_root / "status.json"
    while time.monotonic() < deadline:
        if status_path.is_file():
            child_status = read_json(status_path)
            state = child_status.get("state")
            if state in TERMINAL:
                job["child_status"] = child_status
                report = runner_root / "parity-v7-report.json"
                job["report"] = str(report) if report.is_file() else None
                if state == "passed":
                    return "passed", 0, None
                if state == "parity_failed":
                    return "parity_failed", 1, "expected parity difference"
                return "failed", 1, f"parity runner ended with {state}"
        time.sleep(float(config.get("poll_interval_seconds", 2)))
    return "failed", None, "parity runner timeout"


def run_all(root: Path, config: dict[str, Any]) -> int:
    status = read_json(root / "status.json")
    jobs = status["jobs"]
    for job in jobs:
        job_dir = root / "jobs" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        update_job(root, job, state="running", started_at_utc=now(), pid=os.getpid())
        try:
            source = materialize_source(config, job["source"], job_dir)
            job["source_identity"] = {key: str(source[key]) for key in ("name", "source", "archive", "manifest", "revision")}
            if job["kind"] == "beta":
                state, exit_code, error = run_beta(config, root, job, source, job_dir)
            else:
                state, exit_code, error = run_parity(config, root, job, source, job_dir)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            state, exit_code, error = "failed", None, f"{type(exc).__name__}: {exc}"
        update_job(root, job, state=state, exit_code=exit_code, error=error, ended_at_utc=now())
        if state == "failed":
            write_json(root / "status.json", {**read_json(root / "status.json"), "state": "failed", "ended_at_utc": now()})
            return 1
    final_state = "passed" if all(job["state"] == "passed" for job in jobs) else "parity_failed"
    write_json(root / "status.json", {**read_json(root / "status.json"), "state": final_state, "ended_at_utc": now()})
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("launch", "run", "status"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("local", "parity", "all"), default="all")
    return parser


def main() -> int:
    args = parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.action == "status":
        print(json.dumps(read_json(root / "status.json") if (root / "status.json").is_file() else {"state": "missing", "root": str(root)}, indent=2, sort_keys=True))
        return 0
    if args.action == "launch":
        if args.config is None:
            raise SystemExit("launch requires --config CONFIG.json")
        if root.exists():
            raise SystemExit(f"launch requires a fresh root: {root}")
        config_path = args.config.expanduser().resolve()
        config = read_json(config_path)
        if not isinstance(config, dict):
            raise SystemExit("config must be a JSON object")
        config["_config_dir"] = str(config_path.parent)
        config = config_paths(config, config_path.parent)
        config["_config_dir"] = str(config_path.parent)
        root.mkdir(parents=True)
        jobs = select_jobs(config, args.phase)
        config["phase"] = args.phase
        for job in jobs:
            job["state"] = "pending"
            write_json(root / "jobs" / f"{job['id']}.json", job)
        write_json(root / "config.json", config)
        write_json(root / "status.json", {"schema_version": 1, "state": "running", "root": str(root), "jobs": jobs, "started_at_utc": now(), "driver_pid": None})
        command = [sys.executable, str(Path(__file__).resolve()), "--action", "run", "--root", str(root)]
        with (root / "driver.log").open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        status = read_json(root / "status.json")
        status["driver_pid"] = child.pid
        write_json(root / "status.json", status)
        print(json.dumps({"state": "running", "pid": child.pid, "root": str(root)}, indent=2, sort_keys=True))
        return 0
    if not root.is_dir() or not (root / "config.json").is_file():
        raise SystemExit(f"run requires a launched root with config.json: {root}")
    config = read_json(root / "config.json")
    config["_config_dir"] = Path(config["_config_dir"])
    return run_all(root, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        raise SystemExit(1)
