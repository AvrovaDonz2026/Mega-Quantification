#!/usr/bin/env python3
"""Prepare and (when explicitly requested) run the Aitop Spark E2E serving check.

The ``plan`` action is read-only apart from the requested JSON plan. The optional
``launch`` action detaches a durable supervisor, and ``run``
runs the same exclusive sequence in the foreground. Cleanup requires the exact
observed container ID and matching ownership labels. This file is an
operator runner for two explicitly classified checkpoint directories;
it is deliberately separate from the public benchmark client.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

MEGA_REVISION = "a0c6163d15601529a424384124471f44ea579310"
SGLANG_REVISION = "4a667614d810f7b5570be23eb75956c603a85c3a"
SGLANG_REPOSITORY = "https://github.com/AvrovaDonz2026/sglang.git"
BASE_IMAGE_ID = "sha256:20f41d6ea8897ac17a3a2c43edbcc1ff051da71607297a0d107d24f9715bacd0"
SERVE_IMAGE = "megaquant-aitop-serve:4a667614-fi071-kernel046"
SERVE_IMAGE_ID = "UNRECORDED: supply --image-id after build"
DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLIENT = DEFAULT_ROOT / "benchmark_generate.py"
DEFAULT_SGLANG_SOURCE = DEFAULT_ROOT / "source" / "sglang-4a667614"
MODEL_PATHS: dict[str, Path] = {}
PROVENANCE: Path | None = None
SOURCE_ARCHIVE: Path | None = None
SOURCE_MANIFEST: Path | None = None
FIXTURE_REPO = DEFAULT_ROOT / "fixtures"
PROBE_ENABLED = False
HOST_NETWORK = False
DEFAULT_HOST_PORT = 31200
INTERNAL_PORT = 30000
SCHEMES = ("w4a4", "w4a8")
OWNER_LABEL = "io.megaquant.e2e.owner"
OWNER_VALUE = "aitop-auxiliary-e2e-20261014"
JOB_LABEL = "io.megaquant.e2e.job"
RUN_LABEL = "io.megaquant.e2e.run"
SOURCE_LABEL = "io.megaquant.e2e.source"
QUANTIZATION = {"w4a4": "modelopt_fp4", "w4a8": "modelopt_mixed"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def git_revision(source: Path) -> str:
    from input_gates import verify_source

    return verify_source(source, SOURCE_ARCHIVE, SOURCE_MANIFEST)["commit"]


def git_origin(source: Path) -> str:
    try:
        return run(["git", "-C", str(source), "remote", "get-url", "origin"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def inspect_image(image: str) -> str:
    try:
        result = run(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect serving image {image!r}") from exc
    actual = result.stdout.strip()
    if actual != SERVE_IMAGE_ID:
        raise RuntimeError(f"serving image changed: {actual!r} != {SERVE_IMAGE_ID!r}")
    return actual


def model_config(export: Path) -> dict[str, Any]:
    config_path = export / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing model config: {config_path}")
    try:
        value = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {config_path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"model config is not an object: {config_path}")
    model_type = value.get("model_type")
    architectures = value.get("architectures")
    text = value.get("text_config") or {}
    text_type = text.get("model_type") if isinstance(text, dict) else None
    allowed_architectures = {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
    if model_type != "qwen3_5" or not isinstance(architectures, list) or not architectures:
        raise RuntimeError(
            "unsupported Qwen3.8 checkpoint alias: expected model_type=qwen3_5 and "
            f"a Qwen3_5 conditional-generation architecture, got "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )
    if architectures[0] not in allowed_architectures:
        raise RuntimeError(
            "unsupported Qwen3.8 checkpoint architecture; refusing to rewrite config: "
            f"{architectures[0]!r}"
        )
    if text_type not in {"qwen3_5_text", "qwen3_5_moe_text", None}:
        raise RuntimeError(
            "unsupported Qwen3.8 text alias; refusing to rewrite config: "
            f"text_config.model_type={text_type!r}"
        )
    return {
        "path": str(config_path),
        "sha256": sha256(config_path),
        "model_type": model_type,
        "architectures": architectures,
        "text_model_type": text_type,
        "vision_config_present": isinstance(value.get("vision_config"), dict),
        "language_model_only": value.get("language_model_only"),
    }


def export_manifest(export: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not export.is_dir():
        raise RuntimeError(f"missing export directory: {export}")
    files = []
    for path in sorted(p for p in export.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(export)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"export directory is empty: {export}")
    return {
        "export": str(export),
        "recorded_at_utc": now(),
        "files": files,
        "file_count": len(files),
        "config": config,
    }


def audit_record(root: Path, scheme: str) -> dict[str, Any]:
    from input_gates import verify_model

    if PROVENANCE is None:
        raise RuntimeError("--model-provenance is required")
    return verify_model(MODEL_PATHS[scheme], scheme, PROVENANCE)


def bind(source: Path, target: str, readonly: bool = True) -> str:
    suffix = ",readonly" if readonly else ""
    return f"type=bind,src={source},dst={target}{suffix}"


def server_argv(export: str, scheme: str, mtp: bool, host_port: int = INTERNAL_PORT) -> list[str]:
    argv = [
        "-m",
        "sglang.launch_server",
        "--model-path",
        export,
        "--host",
        "127.0.0.1" if HOST_NETWORK else "0.0.0.0",
        "--port",
        str(host_port if HOST_NETWORK else INTERNAL_PORT),
        "--trust-remote-code",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.70",
        "--context-length",
        "32768",
        "--chunked-prefill-size",
        "2048",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--tp-size",
        "1",
        "--max-running-requests",
        "1",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-ssm-dtype",
        "float32",
        "--enable-multimodal",
        "--quantization",
        QUANTIZATION[scheme],
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--mamba-full-memory-ratio",
        "4.59",
        "--attention-backend",
        "flashinfer",
        "--sampling-backend",
        "pytorch",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--max-mamba-cache-size",
        "16",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    if mtp:
        argv.extend(
            [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
            ]
        )
    return argv


def docker_server_argv(
    root: Path,
    source: Path,
    client: Path,
    image: str,
    scheme: str,
    mode: str,
    host_port: int,
    job_id: str = "plan-only",
) -> list[str]:
    export = f"/models/Qwen3.8-27B-NVFP4-{scheme.upper()}"
    name = container_name(scheme, mode, job_id)
    command = [
        "docker",
        "run",
        "--detach",
        "--gpus",
        "all",
        "--network",
        "host" if HOST_NETWORK else "bridge",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--name",
        name,
        "--cidfile",
        str(root / "validation" / f"{scheme}-{mode}-container.cid"),
        "--label",
        f"{OWNER_LABEL}={OWNER_VALUE}",
        "--label",
        f"{JOB_LABEL}={job_id}",
        "--label",
        f"{RUN_LABEL}={scheme}-{mode}",
        "--label",
        f"{SOURCE_LABEL}={SGLANG_REVISION}",
        "--publish",
        f"127.0.0.1:{host_port}:{INTERNAL_PORT}",
        "--mount",
        bind(MODEL_PATHS[scheme], export),
        "--mount",
        bind(source, "/sglang-source"),
        "--mount",
        bind(client, "/client/benchmark_generate.py"),
        "--mount",
        bind(root / "validation", "/validation", False),
        "--mount",
        bind(FIXTURE_REPO, "/fixtures"),
        "--env",
        "PYTHONPATH=/sglang-source/python",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "SGLANG_VIT_ENABLE_CUDA_GRAPH=0",
        "--env",
        "SGLANG_KERNEL_API_LOGLEVEL=0",
        "--env",
        "SGLANG_CACHE_DIR=/tmp/megaquant-cache/sglang",
        # The Spark base ships a torch 2.13 ABI-compatible kernel (0.4.6.post1).
        # Newer 0.4.9 wheels require torch 2.14.1 and fail to import on this base.
        # Keep the exact pinned source while explicitly recording this compatibility
        # gate until the base image moves to torch 2.14.1.
        "--env",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--entrypoint",
        "python",
        image_ref(image),
    ]
    if HOST_NETWORK:
        publish_index = command.index("--publish")
        del command[publish_index : publish_index + 2]
    if PROBE_ENABLED:
        probe_dir = Path(__file__).resolve().parent / "execution-probe"
        state = root / "validation" / f"{scheme}-{mode}-probe-state"
        state.mkdir(parents=True, exist_ok=True)
        if (state / "READY").exists() or list(state.glob("*.json")):
            raise RuntimeError("Probe state must be fresh")
        command[command.index("PYTHONPATH=/sglang-source/python")] = (
            "PYTHONPATH=/probe:/sglang-source/python"
        )
        pos = command.index("--entrypoint")
        command[pos:pos] = [
            "--mount",
            bind(probe_dir, "/probe"),
            "--mount",
            bind(state, "/probe-state", False),
            "--env",
            "MEGAQUANT_PROBE_STATE=/probe-state/worker.json",
            "--env",
            "MEGAQUANT_PROBE_READY=/probe-state/READY",
        ]
    command.extend(server_argv(export, scheme, mode == "mtp", host_port))
    return command


def container_name(scheme: str, mode: str, job: str) -> str:
    suffix = "probe" if PROBE_ENABLED else "ordinary"
    return f"megaquant-aitop-{scheme}-{mode}-{suffix}-4a66761-{job[:8]}"


def probe_argv(container: str, scheme: str, mode: str, host_port: int = INTERNAL_PORT) -> list[str]:
    command = [
        "docker",
        "exec",
        container,
        "python",
        "/probe/probe_client.py",
        "--base-url",
        f"http://127.0.0.1:{host_port if HOST_NETWORK else INTERNAL_PORT}",
        "--state-dir",
        "/probe-state",
        "--output",
        "/probe-state/request.json",
        "--model",
        f"/models/Qwen3.8-27B-NVFP4-{scheme.upper()}",
    ]
    if mode == "mtp":
        command.append("--mtp")
    return command


def client_argv(
    container: str,
    scheme: str,
    mode: str,
    host_port: int,
    root: Path,
    client: Path,
) -> list[str]:
    export = f"/models/Qwen3.8-27B-NVFP4-{scheme.upper()}"
    report = f"/validation/{scheme}-{mode}.json"
    command = [
        "docker",
        "exec",
        container,
        "python",
        "/client/benchmark_generate.py",
        "--base-url",
        f"http://127.0.0.1:{host_port if HOST_NETWORK else INTERNAL_PORT}",
        "--repo",
        "/fixtures",
        "--model",
        export,
        "--mode",
        mode,
        "--run-metadata",
        f"/validation/{scheme}-{mode}-run-metadata.json",
        "--output",
        report,
        "--warmup",
        "2",
        "--repetitions",
        "5",
        "--timeout",
        "300",
    ]
    if mode == "mtp":
        command.extend(["--baseline", f"/validation/{scheme}-baseline.json"])
    return command


def validate_ports(ports: list[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port already has a listener; refusing to touch it: {port}")


def names_in_docker() -> set[str]:
    try:
        return set(run(["docker", "ps", "-a", "--format", "{{.Names}}"]).stdout.splitlines())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot inspect Docker containers") from exc


def plan(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    source = args.sglang_source.resolve()
    client = args.client.resolve()
    source_revision = None
    source_origin = None
    source_error = None
    try:
        source_revision = git_revision(source)
        source_origin = git_origin(source)
    except RuntimeError as exc:
        source_error = str(exc)
    client_hash = sha256(client) if client.is_file() else None
    exports = {}
    for scheme in SCHEMES:
        export = MODEL_PATHS[scheme]
        config = model_config(export)
        audit = audit_record(root, scheme)
        exports[scheme] = {
            "path": str(export),
            "config": config,
            "audit": audit,
        }
    plan_data: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": now(),
        "model_generation_claims": {k: exports[k]["audit"]["generation"] for k in SCHEMES},
        "probe_enabled": PROBE_ENABLED,
        "source_proof": __import__("input_gates").verify_source(
            source, SOURCE_ARCHIVE, SOURCE_MANIFEST
        ),
        "scope": (
            "Sequential native /generate text+vision sanity for explicitly recorded W4A4 and W4A8 "
            "exports; not GPQA, quality, or throughput."
        ),
        "megaquant_revision": MEGA_REVISION,
        "sglang_repository": SGLANG_REPOSITORY,
        "sglang_revision": SGLANG_REVISION,
        "sglang_source": str(source),
        "sglang_source_revision": source_revision,
        "sglang_source_origin": source_origin,
        "sglang_source_error": source_error,
        "base_image_id": BASE_IMAGE_ID,
        "container_image_tag": args.image,
        "container_image": image_ref(args.image),
        "container_image_id_expected": SERVE_IMAGE_ID,
        "benchmark_client": str(client),
        "benchmark_client_sha256": client_hash,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "job_id": getattr(args, "job_id", None) or "plan-only",
        "ports": {"baseline": args.host_port_base, "mtp": args.host_port_base + 1},
        "server": {
            "target_endpoint": "/generate",
            "internal_port": None if HOST_NETWORK else INTERNAL_PORT,
            "network_mode": "host" if HOST_NETWORK else "bridge",
            "listen_host": "127.0.0.1" if HOST_NETWORK else "0.0.0.0",
            "context_length": 32768,
            "temperature": 0,
            "enable_thinking": False,
            "attention_backend": "flashinfer",
            "fp4_gemm_backend": "flashinfer_cutlass",
            "mamba_ssm_dtype": "float32",
            "graph_backends": {"decode": "disabled", "prefill": "disabled"},
            "quantization": QUANTIZATION,
        },
        "exports": exports,
        "runs": [],
    }
    for scheme in SCHEMES:
        for mode, port in (("baseline", args.host_port_base), ("mtp", args.host_port_base + 1)):
            name = container_name(scheme, mode, getattr(args, "job_id", None) or "plan-only")
            run_metadata = {
                "schema_version": 1,
                "scheme": scheme,
                "mode": mode,
                "sglang_revision": SGLANG_REVISION,
                "sglang_repository": SGLANG_REPOSITORY,
                "container_image_tag": args.image,
                "container_image": image_ref(args.image),
                "container_image_id_expected": SERVE_IMAGE_ID,
                "base_image_id": BASE_IMAGE_ID,
                "benchmark_client_sha256": client_hash,
                "runner_sha256": sha256(Path(__file__).resolve()),
                "job_id": getattr(args, "job_id", None) or "plan-only",
                "server_container": name,
                "server_command": docker_server_argv(
                    root,
                    source,
                    client,
                    args.image,
                    scheme,
                    mode,
                    port,
                    getattr(args, "job_id", None) or "plan-only",
                ),
                "benchmark_command": probe_argv(name, scheme, mode, port)
                if PROBE_ENABLED
                else client_argv(name, scheme, mode, port, root, client),
                "native_endpoint": "/generate",
                "port": port,
                "model_config_sha256": exports[scheme]["config"]["sha256"],
                "created_at_utc": now(),
            }
            plan_data["runs"].append(run_metadata)
    return plan_data


def image_ref(image: str) -> str:
    # This is a local Docker image ID, not a repository digest. The tag is
    # inspected immediately before launch, then only this ID is executed.
    if image.startswith("sha256:") and image != SERVE_IMAGE_ID:
        raise RuntimeError(f"unexpected local image ID: {image}")
    return SERVE_IMAGE_ID


def write_manifests(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    for scheme in SCHEMES:
        export = MODEL_PATHS[scheme]
        config = model_config(export)
        audit_record(root, scheme)
        value = export_manifest(export, config)
        json_dump(root / "validation" / f"{scheme}-export-manifest.json", value)


def write_plan(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    data = plan(args)
    path = root / "validation" / "e2e-plan.json"
    json_dump(path, data)
    for entry in data["runs"]:
        metadata = root / "validation" / f"{entry['scheme']}-{entry['mode']}-run-metadata.json"
        json_dump(metadata, entry)
    return path


def owned_container(container_id: str, job_id: str, run_key: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise RuntimeError("refusing a non-full container ID")
    value = json.loads(run(["docker", "inspect", container_id]).stdout)
    if not isinstance(value, list) or len(value) != 1:
        raise RuntimeError(f"unexpected container inspection for {container_id}")
    item = value[0]
    labels = item.get("Config", {}).get("Labels") or {}
    expected = {
        OWNER_LABEL: OWNER_VALUE,
        JOB_LABEL: job_id,
        RUN_LABEL: run_key,
        SOURCE_LABEL: SGLANG_REVISION,
    }
    if item.get("Id") != container_id or item.get("Image") != SERVE_IMAGE_ID:
        raise RuntimeError(f"container/image identity mismatch: {container_id}")
    if any(labels.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"container ownership labels mismatch: {container_id}")
    return item


def get_json(port: int, endpoint: str) -> dict[str, Any]:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{endpoint} did not return a JSON object")
    return value


def verify_server_info(
    info: dict[str, Any],
    model: dict[str, Any],
    scheme: str,
    mode: str,
    host_port: int = INTERNAL_PORT,
) -> dict[str, Any]:
    expected = {
        "host": "127.0.0.1" if HOST_NETWORK else "0.0.0.0",
        "port": host_port if HOST_NETWORK else INTERNAL_PORT,
        "model_path": f"/models/Qwen3.8-27B-NVFP4-{scheme.upper()}",
        "quantization": QUANTIZATION[scheme],
        "fp4_gemm_runner_backend": "flashinfer_cutlass",
        "attention_backend": "flashinfer",
        "sampling_backend": "pytorch",
        "linear_attn_backend": "triton",
        "mamba_backend": "triton",
        "mamba_ssm_dtype": "float32",
        "context_length": 32768,
        "tp_size": 1,
        "max_running_requests": 1,
        "enable_multimodal": True,
        "enable_torch_compile": False,
        "speculative_algorithm": "EAGLE" if mode == "mtp" else None,
    }
    if mode == "mtp":
        expected.update(
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
            speculative_draft_model_quantization=QUANTIZATION[scheme],
        )
    errors = {
        key: {"expected": value, "actual": info.get(key)}
        for key, value in expected.items()
        if info.get(key) != value
    }
    graph = info.get("cuda_graph_config") or {}
    for phase in ("decode", "prefill"):
        actual = graph.get(phase, {}).get("backend")
        if actual != "disabled":
            errors[f"cuda_graph_config.{phase}.backend"] = {
                "expected": "disabled",
                "actual": actual,
            }
    if model.get("model_path") != expected["model_path"]:
        errors["model_info.model_path"] = {
            "expected": expected["model_path"],
            "actual": model.get("model_path"),
        }
    if model.get("has_image_understanding") is not True:
        errors["model_info.has_image_understanding"] = {
            "expected": True,
            "actual": model.get("has_image_understanding"),
        }
    return {"passed": not errors, "expected": expected, "mismatches": errors}


def wait_for_health(
    port: int,
    container_id: str,
    job_id: str,
    run_key: str,
    timeout: float = 900.0,
) -> dict[str, Any]:
    import urllib.error

    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        item = owned_container(container_id, job_id, run_key)
        if not item.get("State", {}).get("Running"):
            raise RuntimeError(
                f"owned server container exited during startup: "
                f"{container_id}, state={item.get('State')}"
            )
        try:
            return get_json(port, "/server_info")
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy on port {port}: {last_error}")


def runtime_probe(container: str, output: Path) -> None:
    code = (
        "import json, pathlib, sys; "
        "import sglang, torch, flashinfer, sgl_kernel, importlib.metadata as md; "
        "value={'python':sys.executable,'sglang_file':sglang.__file__,"
        "'sglang_version':getattr(sglang,'__version__',None),"
        "'torch_version':torch.__version__,'sglang_kernel_version':md.version('sglang-kernel'),"
        "'flashinfer_version':getattr(flashinfer,'__version__',None)}; "
        "print(json.dumps(value, sort_keys=True)); "
        "assert '/sglang-source/python/sglang/' in "
        "str(pathlib.Path(sglang.__file__).resolve()), value; "
        "assert torch.__version__ == '2.13.0+cu130'; "
        "assert md.version('sglang-kernel') == '0.4.6.post1'; "
        "assert flashinfer.__version__ == '0.7.1'; "
        "assert torch.cuda.is_available()"
    )
    result = subprocess.run(
        ["docker", "exec", container, "python", "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    report: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if lines:
        try:
            report["import_report"] = json.loads(lines[-1])
        except json.JSONDecodeError:
            report["import_report_error"] = "probe did not emit a final JSON object"
    json_dump(output, report)
    if result.returncode:
        raise RuntimeError(f"runtime source/backend probe failed for {container}; inspect {output}")


def capture_logs(container: str, output: Path) -> None:
    result = subprocess.run(
        ["docker", "logs", container],
        check=False,
        text=True,
        capture_output=True,
    )
    output.write_text(result.stdout + result.stderr)


def cleanup_owned(container_id: str, job_id: str, run_key: str, log: Path, evidence: Path) -> None:
    report: dict[str, Any] = {"container_id": container_id, "started_at_utc": now()}
    try:
        item = owned_container(container_id, job_id, run_key)
        report["observed_labels"] = item["Config"]["Labels"]
        report["observed_image_id"] = item["Image"]
        capture_logs(container_id, log)
        result = subprocess.run(
            ["docker", "rm", "-f", container_id], text=True, capture_output=True
        )
        report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f"failed to remove owned container {container_id}")
        report["removed"] = True
    except Exception as exc:
        report.update(removed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["ended_at_utc"] = now()
        json_dump(evidence, report)


def run_one(args: argparse.Namespace, scheme: str, mode: str, port: int) -> None:
    root = args.root.resolve()
    source = args.sglang_source.resolve()
    client = args.client.resolve()
    image = args.image
    image_id = inspect_image(image)
    if git_revision(source) != SGLANG_REVISION:
        raise RuntimeError("source revision changed between plan and launch")
    model_config(MODEL_PATHS[scheme])
    audit_record(root, scheme)
    validate_ports([port])
    name = container_name(scheme, mode, args.job_id)
    if name in names_in_docker():
        raise RuntimeError(f"owned container name already exists; inspect before retry: {name}")
    run_key = f"{scheme}-{mode}"
    job_id = args.job_id
    cidfile = root / "validation" / f"{run_key}-container.cid"
    if cidfile.exists():
        raise RuntimeError(f"previous container evidence exists; inspect before retry: {cidfile}")
    command = docker_server_argv(root, source, client, image_id, scheme, mode, port, job_id)
    log = root / "validation" / f"{scheme}-{mode}-server.log"
    root.joinpath("validation").mkdir(parents=True, exist_ok=True)
    metadata_path = root / "validation" / f"{scheme}-{mode}-run-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if sha256(client) != metadata["benchmark_client_sha256"]:
        raise RuntimeError("benchmark client changed since the recorded plan")
    container_id = None
    try:
        started = subprocess.run(command, check=False, text=True, capture_output=True)
        (root / "validation" / f"{run_key}-docker-launch.log").write_text(
            started.stdout + started.stderr
        )
        if cidfile.is_file():
            observed_id = cidfile.read_text().strip()
            owned_container(observed_id, job_id, run_key)
            container_id = observed_id
        if started.returncode:
            raise RuntimeError(
                f"docker launch failed for {run_key}: returncode={started.returncode}"
            )
        if container_id is None or started.stdout.strip() != container_id:
            raise RuntimeError(f"Docker stdout/cidfile identity mismatch for {run_key}")
        metadata.update(
            {
                "container_image_id_observed": image_id,
                "server_container_id_observed": container_id,
                "server_command": command,
                "server_log": str(log),
                "launched_at_utc": now(),
            }
        )
        json_dump(metadata_path, metadata)
        info = wait_for_health(port, container_id, job_id, run_key)
        model_info = get_json(port, "/model_info")
        identity = verify_server_info(info, model_info, scheme, mode, port)
        json_dump(
            root / "validation" / f"{run_key}-server-info.json",
            {"server_info": info, "model_info": model_info, "identity_gate": identity},
        )
        if not identity["passed"]:
            raise RuntimeError(f"resolved server identity/configuration gate failed for {run_key}")
        runtime_probe(container_id, root / "validation" / f"{run_key}-runtime-import.json")
        benchmark = (
            probe_argv(container_id, scheme, mode, port)
            if PROBE_ENABLED
            else client_argv(container_id, scheme, mode, port, root, client)
        )
        metadata["benchmark_command"] = benchmark
        json_dump(metadata_path, metadata)
        result = subprocess.run(benchmark, text=True, capture_output=True)
        (root / "validation" / f"{scheme}-{mode}-client.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode:
            raise RuntimeError(
                f"benchmark failed for {scheme}/{mode}; inspect "
                f"{root / 'validation' / f'{scheme}-{mode}.json'}"
            )
        if PROBE_ENABLED:
            checked = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_id,
                    "python",
                    "/probe/validate_probe.py",
                    "--report",
                    "/probe-state/request.json",
                    "--scheme",
                    scheme,
                    "--mode",
                    mode,
                ],
                text=True,
                capture_output=True,
            )
            (root / "validation" / f"{run_key}-probe-validation.log").write_text(
                checked.stdout + checked.stderr
            )
            if checked.returncode:
                raise RuntimeError("Quantized auxiliary apply probe did not pass")
    finally:
        # Docker can create a container before its client command fails. Recover
        # only its full CID, and recheck all labels before logs/removal. Names
        # are never used as a fallback ownership or cleanup boundary.
        if container_id is None and cidfile.is_file():
            container_id = cidfile.read_text().strip()
        if container_id:
            cleanup_owned(
                container_id,
                job_id,
                run_key,
                log,
                root / "validation" / f"{run_key}-cleanup.json",
            )


def run_all(args: argparse.Namespace) -> None:
    if os.environ.get("MEGAQUANT_ALLOW_E2E_RUN") != "1":
        raise RuntimeError("set MEGAQUANT_ALLOW_E2E_RUN=1 for the explicit GPU/container run")
    root = args.root.resolve()
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    status_path = validation / "e2e-job-status.json"
    with (validation / "e2e-job.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise RuntimeError(f"previous job evidence exists; inspect before retry: {status_path}")
        args.job_id = args.job_id or uuid.uuid4().hex
        status: dict[str, Any] = {
            "schema_version": 1,
            "job_id": args.job_id,
            "supervisor_pid": os.getpid(),
            "started_at_utc": now(),
            "state": "running",
            "runs": [],
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        json_dump(status_path, status)

        def on_terminate(signum: int, _frame: Any) -> None:
            raise RuntimeError(
                f"supervisor received signal {signum}; cleaning only owned containers"
            )

        signal.signal(signal.SIGTERM, on_terminate)
        signal.signal(signal.SIGINT, on_terminate)
        try:
            inspect_image(args.image)
            active = run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
            ).stdout.strip()
            if active:
                raise RuntimeError("GPU already has compute processes; leaving services untouched")
            git_revision(args.sglang_source.resolve())
            validate_ports([args.host_port_base, args.host_port_base + 1])
            existing = names_in_docker()
            expected = {
                container_name(scheme, mode, args.job_id)
                for scheme in SCHEMES
                for mode in ("baseline", "mtp")
            }
            if existing & expected:
                raise RuntimeError(f"run names already exist: {sorted(existing & expected)}")
            path = write_plan(args)
            status["plan"] = str(path)
            json_dump(status_path, status)
            print(json.dumps({"plan": str(path)}, indent=2), flush=True)
            # The same-export passing baseline is required before its MTP run.
            # Keep one full model in the shared GB10 memory at a time.
            for scheme in SCHEMES:
                for mode, port in (
                    ("baseline", args.host_port_base),
                    ("mtp", args.host_port_base + 1),
                ):
                    entry = {"scheme": scheme, "mode": mode, "started_at_utc": now()}
                    status["runs"].append(entry)
                    json_dump(status_path, status)
                    print(json.dumps({"run_started": entry}), flush=True)
                    run_one(args, scheme, mode, port)
                    entry.update(state="passed", ended_at_utc=now())
                    json_dump(status_path, status)
            status["state"] = "passed"
        except BaseException as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            if status["runs"] and "state" not in status["runs"][-1]:
                status["runs"][-1].update(state="failed", ended_at_utc=now())
            raise
        finally:
            status["ended_at_utc"] = now()
            json_dump(status_path, status)


def launch_detached(args: argparse.Namespace) -> None:
    if os.environ.get("MEGAQUANT_ALLOW_E2E_RUN") != "1":
        raise RuntimeError("set MEGAQUANT_ALLOW_E2E_RUN=1 for the explicit GPU/container run")
    root = args.root.resolve()
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    status = validation / "e2e-job-status.json"
    launch_record = validation / "e2e-launch.json"
    log = validation / "e2e-supervisor.log"
    if any(path.exists() for path in (status, launch_record, log)):
        raise RuntimeError("previous launch evidence exists; inspect/archive it before retry")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--root",
        str(root),
        "--sglang-source",
        str(args.sglang_source.resolve()),
        "--client",
        str(args.client.resolve()),
        "--image",
        args.image,
        "--host-port-base",
        str(args.host_port_base),
        "--job-id",
        uuid.uuid4().hex,
    ]
    command.extend(
        [
            "--image-id",
            SERVE_IMAGE_ID,
            "--source-archive",
            str(SOURCE_ARCHIVE),
            "--source-manifest",
            str(SOURCE_MANIFEST),
            "--w4a4-model",
            str(MODEL_PATHS["w4a4"]),
            "--w4a8-model",
            str(MODEL_PATHS["w4a8"]),
            "--model-provenance",
            str(PROVENANCE),
            "--fixture-repo",
            str(FIXTURE_REPO),
        ]
    )
    if PROBE_ENABLED:
        command.append("--execution-probe")
    if HOST_NETWORK:
        command.append("--host-network")
    with log.open("x") as handle:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    report = {
        "job_id": command[command.index("--job-id") + 1],
        "supervisor_pid": child.pid,
        "command": command,
        "launched_at_utc": now(),
        "durable_log": str(log),
        "status": str(status),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "benchmark_client_sha256": sha256(args.client.resolve()),
    }
    json_dump(launch_record, report)
    print(json.dumps(report, indent=2))


def show_status(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    path = root / "validation" / "e2e-job-status.json"
    if path.is_file():
        value = json.loads(path.read_text())
        value["lock_held"] = False
        with (root / "validation" / "e2e-job.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                value["lock_held"] = True
        if value.get("state") == "running" and not value["lock_held"]:
            value["supervisor_warning"] = "status says running but supervisor lock is not held"
        print(json.dumps(value, indent=2))
    else:
        print(json.dumps({"state": "not-started", "expected_status": str(path)}, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=("manifest", "plan", "launch", "run", "status"))
    value.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    value.add_argument("--sglang-source", type=Path, default=DEFAULT_SGLANG_SOURCE)
    value.add_argument("--client", type=Path, default=DEFAULT_CLIENT)
    value.add_argument("--image", default=SERVE_IMAGE)
    value.add_argument("--host-port-base", type=int, default=DEFAULT_HOST_PORT)
    value.add_argument(
        "--host-network",
        action="store_true",
        help="Use isolated loopback host ports for first-JIT network access",
    )
    value.add_argument("--image-id", help="Exact local image ID recorded after compatible build")
    value.add_argument("--source-archive", type=Path)
    value.add_argument("--source-manifest", type=Path)
    value.add_argument("--w4a4-model", type=Path)
    value.add_argument("--w4a8-model", type=Path)
    value.add_argument("--model-provenance", type=Path)
    value.add_argument("--fixture-repo", type=Path, default=FIXTURE_REPO)
    value.add_argument(
        "--execution-probe",
        action="store_true",
        help="Separate instrumented four-mode job in a new root",
    )
    value.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    global \
        SERVE_IMAGE_ID, \
        MODEL_PATHS, \
        PROVENANCE, \
        SOURCE_ARCHIVE, \
        SOURCE_MANIFEST, \
        FIXTURE_REPO, \
        PROBE_ENABLED, \
        HOST_NETWORK
    if args.action != "status":
        if not args.image_id or not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
            raise SystemExit("Supply the recorded --image-id; a mutable tag alone is insufficient")
        required = (
            args.w4a4_model,
            args.w4a8_model,
            args.model_provenance,
            args.source_archive,
            args.source_manifest,
        )
        if any(value is None for value in required):
            raise SystemExit(
                "Explicit model paths, provenance, source archive and source manifest are required"
            )
        SERVE_IMAGE_ID = args.image_id
        MODEL_PATHS = {"w4a4": args.w4a4_model.resolve(), "w4a8": args.w4a8_model.resolve()}
        PROVENANCE = args.model_provenance.resolve()
        SOURCE_ARCHIVE = args.source_archive.resolve()
        SOURCE_MANIFEST = args.source_manifest.resolve()
        FIXTURE_REPO = args.fixture_repo.resolve()
        PROBE_ENABLED = args.execution_probe
        HOST_NETWORK = args.host_network
        if not (FIXTURE_REPO / "scripts/check_sglang_multimodal.py").is_file():
            raise SystemExit("Missing fixed public fixture helper")
        if any(root == p or root in p.parents or p in root.parents for p in MODEL_PATHS.values()):
            raise SystemExit("Evidence root and models must be distinct independent directories")
    if args.action == "manifest":
        write_manifests(args)
        return 0
    if args.action == "status":
        show_status(args)
        return 0
    if not args.client.is_file():
        raise SystemExit(f"benchmark client is missing: {args.client}")
    if args.action == "plan":
        if (root / "validation" / "e2e-job-status.json").exists() or (
            root / "validation" / "e2e-launch.json"
        ).exists():
            raise SystemExit("job evidence exists; refusing to overwrite the recorded run plan")
        print(json.dumps({"plan": str(write_plan(args))}, indent=2))
    elif args.action == "launch":
        launch_detached(args)
    else:
        run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
