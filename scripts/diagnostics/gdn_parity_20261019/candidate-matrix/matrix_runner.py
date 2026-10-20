#!/usr/bin/env python3
"""One scheme, two owned servers, five cases, and sanitized checkpoints."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
R2 = Path(os.environ.get("GDN_PARITY_RUNNER_DIR", str(HERE.parent / "parity-v7"))).resolve()
sys.path.insert(0, str(R2))
import run_v7 as v  # noqa: E402

CASES = ("python_code", "arithmetic", "red_square", "blue_square", "ocr_3729")
DEFAULT_EXPECTED = {
    "arithmetic": "391",
    "red_square": "red",
    "blue_square": "blue",
    "ocr_3729": "3729",
}
SOURCE_LOCK = {
    "commit": "49d3acfd4d54b0cf1afffcfeb369c63af4f56499",
    "archive_sha256": "1f8612963f5322395ade27bcc503baff28d436cb85555341b7f500740d4a3033",
}


FIXTURE_SHA256 = "f0e8ff85f40b4aaa9847ab95dd9098591845f137faab574ca6d146f0be3f04da"


def fixture_expected(repo: Path) -> dict[str, str]:
    path = repo / "scripts/check_sglang_multimodal.py"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if FIXTURE_SHA256 and actual != FIXTURE_SHA256:
        raise RuntimeError("fixture sha256 differs from pinned candidate fixture")
    tree = ast.parse(path.read_text())
    result = dict(DEFAULT_EXPECTED)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TEXT_CASES" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            for item in value:
                if item.get("name") == "arithmetic":
                    result["arithmetic"] = item["expected"]
    result["fixture_sha256"] = actual
    return result


def answer_check(wrapper: dict[str, Any], case: str, expected: dict[str, str]) -> dict[str, Any]:
    response = wrapper.get("response", {})
    text = response.get("text") if isinstance(response, dict) else None
    wanted = expected.get(case)
    passed = wanted is None or (isinstance(text, str) and text.strip() == wanted)
    return {
        "passed": passed,
        "expected": wanted,
        "scope": "ast_structure_only" if case == "python_code" else "exact_text",
        "text_sha256": v.sha256_bytes(text.encode()) if isinstance(text, str) else None,
    }


def sanitized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_check": row["answer_check"],
        "output_len": row["output_len"],
        "output_fingerprint": v.digest(row["output_ids"]),
        "prompt_len": row["prompt_len"],
        "prompt_fingerprint": v.digest(row["prompt_ids"]),
        "finish_reason_type": row["finish_reason_type"],
        "scalars": row["scalars"],
        "code_ast": row["code_ast"],
        "fixture_file_sha256": row["fixture_file_sha256"],
        "rendered_prompt_sha256": row["rendered_prompt_sha256"],
        "image_sha256": row["image_sha256"],
        "requested_max_new_tokens": row["requested_max_new_tokens"],
    }


def run_side(
    args: argparse.Namespace, role: str, expected: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    port = args.port if role == "baseline" else args.port + 1
    server = client = None
    cidfile = args.root / f".{role}-client.cid"
    summaries: dict[str, list[dict[str, Any]]] = {case: [] for case in CASES}
    try:
        v.progress(args, role, "starting")
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("diagnostic port occupied")
        expected_image = v.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], check=False
        ).stdout.strip()
        if expected_image != args.image_id:
            raise v.GateError("serving image identity differs from pinned image")
        v.gpu_idle()
        launched = v.run(v.server_command(args, args.job_id, role, port), check=False)
        if launched.returncode or not re.fullmatch(r"[0-9a-f]{64}", launched.stdout.strip()):
            raise RuntimeError("server launch failed")
        server = launched.stdout.strip()
        v.owned(server, args.job_id, role, args.image_id, args.source_revision)
        info = v.wait_health(args, server, role, port)
        v.verify_identity(args, info, role, port)
        v.progress(args, role, "healthy")
        source_hashes = v.runtime_source_hashes(args, server)
        v.write_json(args.root / f"{role}-runtime-source.json", source_hashes)
        if not source_hashes["passed"]:
            raise v.GateError("runtime imported source hash mismatch")
        started = v.run(v.client_command(args, args.job_id, role, cidfile), check=False)
        if started.returncode or not cidfile.is_file():
            raise RuntimeError("client launch failed")
        client = cidfile.read_text().strip()
        v.owned(client, args.job_id, role + "-client", args.image_id, args.source_revision)
        for case in CASES:
            args.case = case
            for _trial in (1, 2):
                v.progress(args, role, f"request:{case}", _trial)
                result = v.run(
                    [
                        "docker",
                        "exec",
                        client,
                        "python",
                        "/client.py",
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--model",
                        "/model",
                        "--fixture-repo",
                        "/fixtures",
                        "--case",
                        case,
                        "--timeout",
                        str(args.request_timeout),
                    ],
                    check=False,
                    timeout=args.request_timeout + 30,
                )
                if result.returncode:
                    raise RuntimeError(f"diagnostic client failed: {role}/{case}")
                wrapper = json.loads(result.stdout)
                budget = 768 if case == "python_code" else 32
                if (
                    wrapper.get("case") != case
                    or wrapper.get("requested_max_new_tokens") != budget
                    or wrapper.get("fixture_file_sha256") != FIXTURE_SHA256
                ):
                    raise v.GateError("request case, budget, or fixture identity differs")
                answer = answer_check(wrapper, case, expected)
                summary = v.summarize(wrapper)
                summary["answer_check"] = answer
                summary["requested_max_new_tokens"] = budget
                summary["source_import_hashes"] = source_hashes
                summaries[case].append(summary)
                v.write_json(
                    args.root / f"{role}-checkpoint.json",
                    {
                        name: [sanitized_row(row) for row in rows]
                        for name, rows in summaries.items()
                    },
                )
        return summaries
    finally:
        if client is None and cidfile.is_file():
            client = cidfile.read_text().strip()
        cleanup = {}
        try:
            cleanup["client"] = v.cleanup(
                client, args.job_id, role + "-client", args.image_id, args.source_revision
            )
        finally:
            try:
                cleanup["server"] = v.cleanup(
                    server, args.job_id, role, args.image_id, args.source_revision
                )
            finally:
                v.write_json(args.root / f"{role}-cleanup.json", cleanup)
                v.progress(args, role, "cleanup")
                cidfile.unlink(missing_ok=True)


def comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result = v.compare(a, b)
    return result


def execution_gate(sides: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    mtp_rows = [row for rows in sides["mtp"].values() for row in rows]
    baseline_rows = [row for rows in sides["baseline"].values() for row in rows]
    mtp_values = [row.get("scalars", {}).get("spec_verify_ct") for row in mtp_rows]
    baseline_values = [row.get("scalars", {}).get("spec_verify_ct", 0) for row in baseline_rows]
    finite_mtp = all(
        isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
        for value in mtp_values
    )
    return {
        "mtp_spec_verify_ct_present": all(isinstance(value, (int, float)) for value in mtp_values),
        "mtp_spec_verify_ct_positive": all(
            value > 0 for value in mtp_values if isinstance(value, (int, float))
        ),
        "mtp_spec_verify_ct_finite": finite_mtp,
        "baseline_spec_verify_ct_zero_or_absent": all(
            value in (None, 0) for value in baseline_values
        ),
        "passed": bool(mtp_values)
        and finite_mtp
        and all(isinstance(value, (int, float)) and value > 0 for value in mtp_values)
        and all(value in (None, 0) for value in baseline_values),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-scheme", choices=("w4a4", "w4a8"), required=True)
    p.add_argument("--model-provenance", type=Path, required=True)
    p.add_argument("--fixture-repo", type=Path, required=True)
    p.add_argument("--client", type=Path, default=HERE / "expected_client.py")
    p.add_argument("--image", required=True)
    p.add_argument("--image-id", required=True)
    p.add_argument("--root-job-id", default=None)
    p.add_argument("--port", type=int, default=32240)
    p.add_argument("--health-timeout", type=float, default=1200)
    p.add_argument("--request-timeout", type=float, default=1200)
    p.add_argument("--fla-use-fast-ops", choices=("0", "1"), default="0")
    p.add_argument("--force-shared-decode", choices=("0", "1"), default="0")
    args = p.parse_args()
    args.source_variant = "all-beta"
    args.root = args.root.resolve()
    args.job_id = args.root_job_id or f"candidate-{args.model_scheme}"
    args.source = args.source.resolve()
    args.source_archive = args.source_archive.resolve()
    args.source_manifest = args.source_manifest.resolve()
    args.model = args.model.resolve()
    args.model_provenance = args.model_provenance.resolve()
    args.fixture_repo = args.fixture_repo.resolve()
    args.client = args.client.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    v.write_json(
        args.root / "status.json",
        {"state": "running", "job_id": args.job_id, "started_at_utc": v.now()},
    )
    args.case = "python_code"
    v.validate_inputs(args)
    if args.force_shared_decode != "0" or args.fla_use_fast_ops != "0":
        raise v.GateError("candidate protocol requires shared decode off and FLA_USE_FAST_OPS=0")
    if (
        args.source_revision != SOURCE_LOCK["commit"]
        or v.sha256_file(args.source_archive) != SOURCE_LOCK["archive_sha256"]
    ):
        raise v.GateError("candidate source does not match pinned all-beta lock")
    source = v.source_gate(args)
    model = v.model_gate(args)
    expected = fixture_expected(args.fixture_repo)
    sides: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in ("baseline", "mtp"):
        sides[role] = run_side(args, role, expected)
    reports: dict[str, Any] = {}
    for case in CASES:
        left, right = sides["baseline"][case], sides["mtp"][case]
        comp = comparison(left[0], right[0])
        gates = v.parity_gate(SimpleNamespace(case=case), comp, {"baseline": left, "mtp": right})
        answer_gate = all(row.get("answer_check", {}).get("passed") for row in left + right)
        gates["expected_answer"] = answer_gate
        reports[case] = {
            "case": case,
            "status": "passed" if all(gates.values()) else "parity_failed",
            "comparison": comp,
            "parity_gates": gates,
            "repetitions": 2,
            "max_new_tokens": 768 if case == "python_code" else 32,
            "trials": {
                "baseline": [sanitized_row(row) for row in left],
                "mtp": [sanitized_row(row) for row in right],
            },
        }
    execution = execution_gate(sides)
    report = {
        "schema_version": 1,
        "status": "passed"
        if all(item["status"] == "passed" for item in reports.values()) and execution["passed"]
        else "parity_failed",
        "job_id": args.job_id,
        "model_scheme": args.model_scheme,
        "source_revision": args.source_revision,
        "source_proof": source,
        "model_metadata_gate": model,
        "image_id": args.image_id,
        "expected_answers": expected,
        "mtp_execution_gate": execution,
        "scope": (
            "functional parity and MTP execution; code is checked for AST structure, "
            "not executed; visual/MTP quantized apply probe is not performed"
        ),
        "cases": reports,
        "created_at_utc": v.now(),
        "environment": {
            "FLA_USE_FAST_OPS": args.fla_use_fast_ops,
            "SGLANG_GDN_FORCE_SHARED_DECODE": args.force_shared_decode,
        },
    }
    v.write_json(args.root / "candidate-matrix-report.json", report)
    v.write_json(
        args.root / "status.json",
        {"state": report["status"], "job_id": args.job_id, "ended_at_utc": v.now()},
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        try:
            root_arg = next(
                (sys.argv[index + 1] for index, value in enumerate(sys.argv) if value == "--root"),
                None,
            )
            if root_arg:
                v.write_json(
                    Path(root_arg) / "status.json",
                    {
                        "state": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "ended_at_utc": v.now(),
                    },
                )
        except Exception:
            pass
        sys.stderr.write(type(exc).__name__ + "\n")
        raise SystemExit(1) from None
    raise SystemExit(exit_code)
