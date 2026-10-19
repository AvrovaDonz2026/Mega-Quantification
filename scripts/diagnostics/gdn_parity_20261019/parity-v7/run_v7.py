#!/usr/bin/env python3
"""Detached first-divergence parity runner for one W4A4/W4A8 fixture.

Only sanitized summaries are persisted. Native responses are held in memory
while the two trials are compared and are never written to a file or log.
"""
from __future__ import annotations

import argparse, ast, datetime as dt, fcntl, hashlib, json, math, os, re, shutil, signal, socket, subprocess, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
import importlib.util
_LOCK_SPEC = importlib.util.spec_from_file_location("gdn_parity_source_lock", HERE.parent / "source_lock.py")
if _LOCK_SPEC is None or _LOCK_SPEC.loader is None:
    raise RuntimeError("cannot load bundle source lock")
_LOCK_MODULE = importlib.util.module_from_spec(_LOCK_SPEC)
_LOCK_SPEC.loader.exec_module(_LOCK_MODULE)
OWNER_KEY = "io.megaquant.parity-v7.owner"
OWNER_VALUE = "aitop-parity-v7"
JOB_KEY = "io.megaquant.parity-v7.job"
RUN_KEY = "io.megaquant.parity-v7.run"
SOURCE_KEY = "io.megaquant.parity-v7.source"
DEFAULT_IMAGE = "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf"
SCHEMES = ("w4a4", "w4a8")
CASES = ("python_code", "arithmetic", "red_square", "blue_square", "ocr_3729")
RUNTIME_MODULES = {
    "sglang.kernels.ops.attention.fla.op": "python/sglang/kernels/ops/attention/fla/op.py",
    "sglang.kernels.ops.attention.fla.fused_gdn_gating": "python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py",
    "sglang.kernels.ops.attention.fla.fused_recurrent": "python/sglang/kernels/ops/attention/fla/fused_recurrent.py",
    "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent": "python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py",
    "sglang.srt.layers.attention.linear.gdn_backend": "python/sglang/srt/layers/attention/linear/gdn_backend.py",
    "sglang.srt.environ": "python/sglang/srt/environ.py",
}

class GateError(RuntimeError):
    pass

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())

def run(command: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True, timeout=timeout)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    tmp.replace(path)

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value

def progress(args: argparse.Namespace, role: str, milestone: str, trial: int | None = None) -> None:
    path = args.root / "progress.json"
    current = read_json(path) if path.is_file() else {"schema_version": 1, "job_id": args.job_id, "milestones": []}
    entry: dict[str, Any] = {"at_utc": now(), "role": role, "milestone": milestone}
    if trial is not None:
        entry["trial"] = trial
    current.setdefault("milestones", []).append(entry)
    write_json(path, current)

def gpu_idle() -> None:
    if shutil.which("nvidia-smi") is None:
        raise GateError("nvidia-smi unavailable; refusing launch")
    result = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], check=False)
    if result.returncode or result.stdout.strip():
        raise GateError("GPU has active compute processes; no containers were touched")

def inspect(container: str) -> dict[str, Any]:
    value = json.loads(run(["docker", "inspect", container]).stdout)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError("unexpected Docker inspect")
    return value[0]

def owned(container: str, job: str, role: str, image_id: str, revision: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise RuntimeError("container ID is not a full ID")
    item = inspect(container)
    labels = item.get("Config", {}).get("Labels") or {}
    expected = {OWNER_KEY: OWNER_VALUE, JOB_KEY: job, RUN_KEY: role, SOURCE_KEY: revision}
    if item.get("Id") != container or item.get("Image") != image_id or any(labels.get(k) != v for k, v in expected.items()):
        raise GateError("container ownership/image/source identity mismatch")
    return item

def cleanup(container: str | None, job: str, role: str, image_id: str, revision: str) -> dict[str, Any] | None:
    if not container:
        return None
    owned(container, job, role, image_id, revision)
    result = run(["docker", "rm", "-f", container], check=False)
    if result.returncode:
        raise RuntimeError("owned container cleanup failed")
    return {"removed": True, "container_id": container, "role": role}

def bind(source: Path, target: str) -> str:
    return f"type=bind,src={source},dst={target},readonly"

def server_args(model: str, scheme: str, mtp: bool, port: int) -> list[str]:
    args = ["-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1", "--port", str(port), "--trust-remote-code", "--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.70", "--context-length", "32768", "--chunked-prefill-size", "2048", "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", "--tp-size", "1", "--max-running-requests", "1", "--mamba-radix-cache-strategy", "extra_buffer_lazy", "--mamba-ssm-dtype", "float32", "--enable-multimodal", "--quantization", "modelopt_fp4" if scheme == "w4a4" else "modelopt_mixed", "--fp4-gemm-backend", "flashinfer_cutlass", "--mamba-full-memory-ratio", "4.59", "--attention-backend", "flashinfer", "--sampling-backend", "pytorch", "--linear-attn-backend", "triton", "--mamba-backend", "triton", "--max-mamba-cache-size", "16", "--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"]
    if mtp:
        args += ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", "2", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "3"]
    return args

def server_command(args: argparse.Namespace, job: str, role: str, port: int) -> list[str]:
    export = f"/models/Qwen3.8-27B-NVFP4-{args.model_scheme.upper()}"
    name = f"megaquant-parity-v7-{args.model_scheme}-{role}-{job[:10]}"
    command = ["docker", "run", "--detach", "--gpus", "all", "--network", "host", "--ipc", "host", "--log-driver", "none", "--shm-size", "16g", "--name", name, "--label", f"{OWNER_KEY}={OWNER_VALUE}", "--label", f"{JOB_KEY}={job}", "--label", f"{RUN_KEY}={role}", "--label", f"{SOURCE_KEY}={args.source_revision}", "--mount", bind(args.model, export), "--mount", bind(args.source, "/sglang-source"), "--env", "PYTHONPATH=/sglang-source/python", "--env", "PYTHONUNBUFFERED=1", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "SGLANG_VIT_ENABLE_CUDA_GRAPH=0", "--env", "SGLANG_KERNEL_API_LOGLEVEL=0", "--env", "SGLANG_CACHE_DIR=/tmp/megaquant-cache/sglang", "--env", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1", "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", f"FLA_USE_FAST_OPS={args.fla_use_fast_ops}", "--env", f"SGLANG_GDN_FORCE_SHARED_DECODE={args.force_shared_decode}", "--env", f"MEGAQUANT_EXPECTED_SGLANG_REVISION={args.source_revision}", "--entrypoint", "python", args.image]
    command += server_args(export, args.model_scheme, role == "mtp", port)
    return command

def client_command(args: argparse.Namespace, job: str, role: str, cidfile: Path) -> list[str]:
    name = f"megaquant-parity-v7-{args.model_scheme}-{role}-client-{job[:10]}"
    return ["docker", "run", "--detach", "--network", "host", "--log-driver", "none", "--name", name, "--cidfile", str(cidfile), "--label", f"{OWNER_KEY}={OWNER_VALUE}", "--label", f"{JOB_KEY}={job}", "--label", f"{RUN_KEY}={role}-client", "--label", f"{SOURCE_KEY}={args.source_revision}", "--mount", bind(args.model, "/model"), "--mount", bind(args.fixture_repo, "/fixtures"), "--mount", bind(args.client, "/client.py"), "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--entrypoint", "python", args.image, "-c", "import time; time.sleep(86400)"]

def wait_health(args: argparse.Namespace, cid: str, role: str, port: int) -> dict[str, Any]:
    deadline = time.monotonic() + args.health_timeout
    while time.monotonic() < deadline:
        item = owned(cid, args.job_id, role, args.image_id, args.source_revision)
        if not item.get("State", {}).get("Running"):
            raise RuntimeError("owned server exited before health")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=5) as response:
                info = json.load(response)
            if isinstance(info, dict):
                return info
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(2)
    raise RuntimeError("server did not become healthy")

def verify_identity(args: argparse.Namespace, info: dict[str, Any], role: str, port: int) -> None:
    import serve_e2e
    serve_e2e.HOST_NETWORK = True
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=5) as response:
        model_info = json.load(response)
    checked = serve_e2e.verify_server_info(info, model_info, args.model_scheme, "mtp" if role == "mtp" else "baseline", port)
    if not checked.get("passed"):
        raise GateError(json.dumps(checked.get("mismatches", {}), sort_keys=True))
    if info.get("dump_requests_folder") or info.get("crash_dump_folder") or info.get("log_requests"):
        raise GateError("server request/output logging is enabled")

def runtime_source_hashes(args: argparse.Namespace, container: str) -> dict[str, Any]:
    expected = {relative: sha256_file(args.source / relative) for relative in RUNTIME_MODULES.values() if (args.source / relative).is_file()}
    if len(expected) != len(RUNTIME_MODULES):
        return {"expected": expected, "observed": None, "passed": False, "error": "required runtime source module missing", "source_file_count": len(expected)}
    code = (
        "import hashlib,importlib,json,os,pathlib\n"
        f"mods={RUNTIME_MODULES!r}\n"
        "observed={}; errors={}\n"
        "for name,relative in mods.items():\n"
        " try:\n"
        "  module=importlib.import_module(name); path=pathlib.Path(module.__file__).resolve(); observed[name]={'file':str(path),'relative':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        " except Exception as exc: errors[name]=type(exc).__name__\n"
        "op=importlib.import_module('sglang.kernels.ops.attention.fla.op'); wanted=os.environ.get('FLA_USE_FAST_OPS',''); "
        "binding=((wanted=='1' and op.exp is op.tldevice.fast_expf) or (wanted=='0' and op.exp is op.tl.exp)); "
        "print(json.dumps({'env':{'FLA_USE_FAST_OPS':wanted,'SGLANG_GDN_FORCE_SHARED_DECODE':os.environ.get('SGLANG_GDN_FORCE_SHARED_DECODE')},'modules':observed,'errors':errors,'fla_op_binding_matches':binding},sort_keys=True))"
    )
    result = run(["docker", "exec", container, "python", "-c", code], check=False)
    observed = None
    if result.returncode == 0:
        try: observed = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError): pass
    observed_modules = observed.get("modules") if isinstance(observed, dict) else None
    module_ok = isinstance(observed_modules, dict) and all(
        value.get("relative") == relative and value.get("file") == f"/sglang-source/{relative}" and value.get("sha256") == expected[relative]
        for module, relative in RUNTIME_MODULES.items()
        for value in [observed_modules.get(module, {})]
    )
    env_ok = isinstance(observed, dict) and observed.get("env", {}).get("FLA_USE_FAST_OPS") == args.fla_use_fast_ops and observed.get("env", {}).get("SGLANG_GDN_FORCE_SHARED_DECODE") == args.force_shared_decode
    binding_ok = isinstance(observed, dict) and observed.get("fla_op_binding_matches") is True
    return {"expected": expected, "observed": observed, "passed": module_ok and env_ok and binding_ok, "source_file_count": len(expected)}

def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def summarize(wrapper: dict[str, Any]) -> dict[str, Any]:
    response = wrapper.get("response")
    if not isinstance(response, dict): raise RuntimeError("native response is not an object")
    ids, meta = response.get("output_ids"), response.get("meta_info")
    if not isinstance(ids, list) or not ids or not isinstance(meta, dict): raise RuntimeError("response metadata missing")
    n = meta.get("completion_tokens")
    token_lp, top = meta.get("output_token_logprobs"), meta.get("output_top_logprobs")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or len(ids) != n or not isinstance(token_lp, list) or not isinstance(top, list) or len(token_lp) != n or len(top) != n: raise RuntimeError("response lengths inconsistent")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in ids): raise RuntimeError("invalid output IDs")
    if not isinstance(response.get("prompt_token_ids"), list): raise RuntimeError("prompt IDs missing")
    reason = meta.get("finish_reason")
    if not isinstance(reason, dict) or reason.get("type") not in {"stop", "length"}: raise RuntimeError("invalid stop reason")
    scalar_keys = {"completion_tokens", "cached_tokens", "spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_accept_rate", "spec_accept_length", "latency", "ttft", "prefill_time", "decode_time", "e2e_latency", "time_to_first_token"}
    scalars = {key: meta[key] for key in scalar_keys if key in meta and (finite(meta[key]) or (isinstance(meta[key], int) and not isinstance(meta[key], bool)))}
    return {"output_ids": ids, "prompt_ids": response["prompt_token_ids"], "output_len": n, "prompt_len": len(response["prompt_token_ids"]), "token_logprobs": token_lp, "top_logprobs": top, "finish_reason_type": reason["type"], "scalars": scalars, "fixture_file_sha256": wrapper.get("fixture_file_sha256"), "rendered_prompt_sha256": wrapper.get("rendered_prompt_sha256"), "image_sha256": wrapper.get("image_sha256"), "code_ast": ast_summary(response.get("text"))}

def ast_summary(text: Any) -> dict[str, Any]:
    if not isinstance(text, str): return {"available": False}
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    source = match.group(1) if match else text
    try: tree = ast.parse(source)
    except SyntaxError: return {"available": True, "valid": False}
    required = {"fibonacci", "factorial", "is_prime"}
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    }
    function_details: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        function = definitions.get(name)
        if function is None:
            continue
        function_details[name] = {
            "args": [arg.arg for arg in function.args.args],
            "docstring_nonempty": bool((ast.get_docstring(function) or "").strip()),
            "has_loop": any(isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in ast.walk(function)),
            "distinct_assert_calls": 0,
        }
    # Assertions outside implementations are tests; count distinct assertion
    # expressions that call each required function.
    implementation_nodes = {id(node) for function in definitions.values() for node in ast.walk(function)}
    assertions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assert) and id(node) not in implementation_nodes
    ]
    assert_tests = {name: set() for name in required}
    for assertion in assertions:
        called = {
            node.func.id
            for node in ast.walk(assertion.test)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for name in assert_tests.keys() & called:
            assert_tests[name].add(ast.dump(assertion.test, include_attributes=False))
    for name, tests in assert_tests.items():
        if name in function_details:
            function_details[name]["distinct_assert_calls"] = len(tests)
    return {
        "available": True,
        "valid": True,
        "functions": {name: details["args"] for name, details in function_details.items()},
        "function_details": function_details,
        "assert_count": len(assertions),
        "loop_count": sum(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in ast.walk(tree)),
    }

def margin(row: Any) -> float | None:
    try:
        value = float(row[0][0]) - float(row[1][0])
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, IndexError): return None

def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    left, right = a["output_ids"], b["output_ids"]
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]: prefix += 1
    mismatch = prefix if prefix < len(left) and prefix < len(right) else None
    result = {"baseline_output_len": len(left), "mtp_output_len": len(right), "shared_prefix_tokens": prefix, "first_divergence_index": mismatch, "baseline_token_fingerprint": digest(left), "mtp_token_fingerprint": digest(right), "baseline_stop": a["finish_reason_type"], "mtp_stop": b["finish_reason_type"], "prompt_len": a["prompt_len"], "prompt_ids_equal": a["prompt_ids"] == b["prompt_ids"], "rendered_prompt_sha256_equal": a["rendered_prompt_sha256"] == b["rendered_prompt_sha256"], "fixture_file_sha256_equal": a["fixture_file_sha256"] == b["fixture_file_sha256"], "image_sha256_equal": a["image_sha256"] == b["image_sha256"], "equal_outputs": left == right, "baseline_ast": a["code_ast"], "mtp_ast": b["code_ast"], "scalar_whitelist": {"baseline": a["scalars"], "mtp": b["scalars"]}}
    if mismatch is not None:
        selected_a = a["token_logprobs"][mismatch][0] if a["token_logprobs"][mismatch] else None
        selected_b = b["token_logprobs"][mismatch][0] if b["token_logprobs"][mismatch] else None
        result["first_divergence"] = {"baseline": {"token_fingerprint": digest(left[mismatch]), "top5_margin": margin(a["top_logprobs"][mismatch]), "selected_logprob": selected_a if finite(selected_a) else None, "selected_probability": math.exp(selected_a) if finite(selected_a) else None}, "mtp": {"token_fingerprint": digest(right[mismatch]), "top5_margin": margin(b["top_logprobs"][mismatch]), "selected_logprob": selected_b if finite(selected_b) else None, "selected_probability": math.exp(selected_b) if finite(selected_b) else None}}
    return result

def parity_gate(args: argparse.Namespace, comparison: dict[str, Any], sides: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
    prompt_identity = bool(comparison["prompt_ids_equal"] and comparison["rendered_prompt_sha256_equal"] and comparison["fixture_file_sha256_equal"] and comparison["image_sha256_equal"])
    equal_outputs = bool(comparison["equal_outputs"])
    stability = all(len(values) == 2 and values[0]["output_ids"] == values[1]["output_ids"] and values[0]["prompt_ids"] == values[1]["prompt_ids"] and values[0]["fixture_file_sha256"] == values[1]["fixture_file_sha256"] and values[0]["rendered_prompt_sha256"] == values[1]["rendered_prompt_sha256"] and values[0]["image_sha256"] == values[1]["image_sha256"] for values in sides.values())
    normal_stop = all(
        summary["finish_reason_type"] == "stop"
        for values in sides.values()
        for summary in values
    )
    ast_gate = True
    if args.case == "python_code":
        ast_gate = all(
            summary["code_ast"].get("valid") is True
            and {"fibonacci", "factorial", "is_prime"}.issubset(summary["code_ast"].get("functions", {}))
            and all(
                details.get("docstring_nonempty") is True
                and details.get("has_loop") is True
                and details.get("distinct_assert_calls", 0) >= 2
                for details in summary["code_ast"].get("function_details", {}).values()
            )
            for values in sides.values()
            for summary in values
        )
    image_gate = all((values[0]["image_sha256"] is None and values[1]["image_sha256"] is None) if args.case in {"python_code", "arithmetic"} else all(isinstance(value["image_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["image_sha256"]) for value in values) for values in sides.values())
    return {"prompt_identity": prompt_identity, "image_identity": image_gate, "equal_outputs": equal_outputs, "stability": stability, "normal_stop": normal_stop, "ast_gate": ast_gate}

def validate_inputs(args: argparse.Namespace) -> None:
    if args.model_scheme not in SCHEMES or args.case not in CASES: raise SystemExit("invalid model scheme or case")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision): raise SystemExit("--source-revision must be a full 40-character commit")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id) or args.image_id != args.image: raise SystemExit("--image and --image-id must be the exact sha256 image ID")
    for path in (args.source, args.source_archive, args.source_manifest, args.model, args.model_provenance, args.fixture_repo, args.client):
        if not path.exists(): raise SystemExit(f"missing path: {path}")
    if not (args.fixture_repo / "scripts/check_sglang_multimodal.py").is_file(): raise SystemExit("fixture helper missing")
    if args.root == HERE or HERE in args.root.parents or args.root in HERE.parents: raise SystemExit("fresh evidence root must be outside parity-v7 bundle")

def source_gate(args: argparse.Namespace) -> dict[str, Any]:
    lock_proof = _LOCK_MODULE.verify_locked_source(args.source_revision, args.source_archive, args.source_variant)
    import input_gates
    input_gates.SGLANG_COMMIT = args.source_revision
    input_gates.ARCHIVE_SHA = sha256_file(args.source_archive)
    input_gates._SOURCE_CACHE.clear()
    proof = input_gates.verify_source(args.source.resolve(), args.source_archive.resolve(), args.source_manifest.resolve())
    if proof.get("commit") != args.source_revision: raise GateError("source revision gate failed")
    proof["source_file_hashes"] = {str(p.relative_to(args.source)): sha256_file(p) for p in sorted(args.source.rglob("*")) if p.is_file()}
    proof["bundle_lock"] = lock_proof
    return proof

def model_gate(args: argparse.Namespace) -> dict[str, Any]:
    import input_gates
    return input_gates.verify_model(args.model.resolve(), args.model_scheme, args.model_provenance.resolve())

def run_side(args: argparse.Namespace, role: str) -> list[dict[str, Any]]:
    port = args.port if role == "baseline" else args.port + 1
    server = client = None
    cidfile = args.root / f".{role}-client.cid"
    summaries: list[dict[str, Any]] = []
    try:
        progress(args, role, "starting")
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0: raise RuntimeError("diagnostic port occupied")
        expected = run(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], check=False).stdout.strip()
        if expected != args.image_id: raise GateError("serving image identity differs from pinned image")
        gpu_idle()
        launched = run(server_command(args, args.job_id, role, port), check=False)
        if launched.returncode or not re.fullmatch(r"[0-9a-f]{64}", launched.stdout.strip()): raise RuntimeError("server launch failed")
        server = launched.stdout.strip(); owned(server, args.job_id, role, args.image_id, args.source_revision)
        info = wait_health(args, server, role, port); progress(args, role, "healthy"); verify_identity(args, info, role, port)
        source_hashes = runtime_source_hashes(args, server)
        write_json(args.root / f"{role}-runtime-source.json", {"schema_version": 1, "role": role, "job_id": args.job_id, "source_revision": args.source_revision, "image_id": args.image_id, **source_hashes})
        if not source_hashes["passed"]: raise GateError("runtime imported source hash mismatch")
        started = run(client_command(args, args.job_id, role, cidfile), check=False)
        if started.returncode or not cidfile.is_file(): raise RuntimeError("client launch failed")
        client = cidfile.read_text().strip(); owned(client, args.job_id, role + "-client", args.image_id, args.source_revision)
        for _trial in (1, 2):
            progress(args, role, "request", _trial)
            result = run(["docker", "exec", client, "python", "/client.py", "--base-url", f"http://127.0.0.1:{port}", "--model", "/model", "--fixture-repo", "/fixtures", "--case", args.case, "--timeout", str(args.request_timeout)], check=False, timeout=args.request_timeout + 30)
            if result.returncode: raise RuntimeError("diagnostic client failed")
            wrapper = json.loads(result.stdout)
            summary = summarize(wrapper); summary["source_import_hashes"] = source_hashes; summary["env"] = {"FLA_USE_FAST_OPS": args.fla_use_fast_ops, "SGLANG_GDN_FORCE_SHARED_DECODE": args.force_shared_decode}
            summaries.append(summary)
        return summaries
    finally:
        if client is None and cidfile.is_file(): client = cidfile.read_text().strip()
        try:
            cleanup(client, args.job_id, role + "-client", args.image_id, args.source_revision)
        finally:
            try:
                cleanup(server, args.job_id, role, args.image_id, args.source_revision)
            finally:
                progress(args, role, "cleanup")
        cidfile.unlink(missing_ok=True)

def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    args.root.mkdir(parents=True, exist_ok=True)
    source = source_gate(args); model = model_gate(args)
    sides: dict[str, list[dict[str, Any]]] = {"baseline": [], "mtp": []}
    for role in ("baseline", "mtp"):
        values = run_side(args, role)
        if len(values) != 2 or values[0]["output_ids"] != values[1]["output_ids"] or values[0]["prompt_ids"] != values[1]["prompt_ids"]:
            raise RuntimeError(f"{role} repeated fixture is unstable")
        sides[role].extend(values)
    comparison = compare(sides["baseline"][0], sides["mtp"][0])
    gates = parity_gate(args, comparison, sides)
    status = "passed" if all(gates.values()) else "parity_failed"
    runtime_refs = {role: {"path": str(args.root / f"{role}-runtime-source.json"), "sha256": sha256_file(args.root / f"{role}-runtime-source.json")} for role in sides}
    report = {"schema_version": 7, "status": status, "created_at_utc": now(), "job_id": args.job_id, "source_revision": args.source_revision, "source_proof": source, "model_scheme": args.model_scheme, "model_metadata_gate": model, "image_id": args.image_id, "case": args.case, "max_new_tokens": 768 if args.case == "python_code" else 32, "repetitions": 2, "fla_use_fast_ops": args.fla_use_fast_ops, "force_shared_decode": args.force_shared_decode, "comparison": comparison, "parity_gates": gates, "runtime_source_hashes": runtime_refs, "stability": {role: {"stable": True, "output_fingerprint": digest(values[0]["output_ids"])} for role, values in sides.items()}, "environment": {"python": sys.version.split()[0], "image_id": args.image_id, "source_revision": args.source_revision, "FLA_USE_FAST_OPS": args.fla_use_fast_ops, "SGLANG_GDN_FORCE_SHARED_DECODE": args.force_shared_decode}}
    write_json(args.root / "parity-v7-report.json", report)
    return report

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("launch", "run", "status"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path); p.add_argument("--source-revision"); p.add_argument("--source-variant"); p.add_argument("--source-manifest", type=Path); p.add_argument("--source-archive", type=Path)
    p.add_argument("--model", type=Path); p.add_argument("--model-scheme", choices=SCHEMES); p.add_argument("--model-provenance", type=Path)
    p.add_argument("--fixture-repo", type=Path); p.add_argument("--client", type=Path, default=HERE / "client.py")
    p.add_argument("--image", default=DEFAULT_IMAGE); p.add_argument("--image-id", default=DEFAULT_IMAGE); p.add_argument("--fla-use-fast-ops", choices=("0", "1"), default="0"); p.add_argument("--force-shared-decode", choices=("0", "1"), default="0")
    p.add_argument("--case", choices=CASES, default="python_code"); p.add_argument("--port", type=int, default=31230); p.add_argument("--job-id", default=None); p.add_argument("--health-timeout", type=float, default=1200); p.add_argument("--request-timeout", type=float, default=1200)
    return p

def main() -> int:
    args = parser().parse_args(); args.root = args.root.resolve(); args.job_id = args.job_id or uuid.uuid4().hex
    if args.action == "status":
        path = args.root / "status.json"
        print(json.dumps(read_json(path) if path.is_file() else {"state": "missing", "root": str(args.root)}, indent=2, sort_keys=True)); return 0
    required = ("source", "source_revision", "source_manifest", "source_archive", "model", "model_scheme", "model_provenance", "fixture_repo")
    if any(getattr(args, key) is None for key in required):
        raise SystemExit("run/launch require source, source-revision, source-manifest, source-archive, model, model-scheme, model-provenance, and fixture-repo")
    args.source = args.source.resolve(); args.source_archive = args.source_archive.resolve(); args.source_manifest = args.source_manifest.resolve(); args.model = args.model.resolve(); args.model_provenance = args.model_provenance.resolve(); args.fixture_repo = args.fixture_repo.resolve(); args.client = args.client.resolve(); args.fla_use_fast_ops = str(args.fla_use_fast_ops); args.force_shared_decode = str(args.force_shared_decode); args.image_id = args.image_id.strip()
    validate_inputs(args)
    if args.action == "launch":
        if args.root.exists(): raise RuntimeError("launch requires a fresh nonexistent root")
        args.root.mkdir(parents=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--action", "run", "--job-id", args.job_id]
        for key, value in vars(args).items():
            if key in {"action", "job_id"}: continue
            flag = "--" + key.replace("_", "-")
            command += [flag, str(value)]
        write_json(args.root / "launch.json", {"schema_version": 1, "job_id": args.job_id, "command": command, "state": "prepared", "created_at_utc": now()})
        with (args.root / "supervisor.log").open("x") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write_json(args.root / "status.json", {"state": "running", "job_id": args.job_id, "supervisor_pid": child.pid, "started_at_utc": now()}); print(json.dumps({"state": "running", "pid": child.pid, "root": str(args.root)}, indent=2)); return 0
    write_json(args.root / "status.json", {"state": "running", "job_id": args.job_id, "started_at_utc": now()})
    try:
        report = run_probe(args); state = report["status"]; write_json(args.root / "status.json", {"state": state, "job_id": args.job_id, "ended_at_utc": now()}); return 0 if state == "passed" else 1
    except BaseException as exc:
        status = {"state": "failed", "job_id": args.job_id, "error_type": type(exc).__name__, "ended_at_utc": now()}
        if isinstance(exc, GateError):
            status["reason"] = str(exc)
        write_json(args.root / "status.json", status); raise

if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except BaseException as exc: sys.stderr.write(type(exc).__name__ + "\n"); raise SystemExit(1)
