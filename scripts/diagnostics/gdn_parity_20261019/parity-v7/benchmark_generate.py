#!/usr/bin/env python3
"""Small deterministic /generate sanity with warmup and uncached request timings.

Uses the repository's public generated fixtures, never calibration examples.
Requires a quiet server: every request flushes the prefix cache outside its timer.
This is not a representative model-quality or throughput benchmark.
Schema 3 persists only explicit summaries, hashes, counts, timings, and checks.
Prompts, requests, responses, output text, and token IDs remain in memory only.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import math
import platform
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def value_digest(value) -> str:
    """Hash a value using canonical JSON; never store its serialized contents."""
    return digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    )


def safe_error(exc: Exception) -> dict:
    """Exception messages can contain request/response contents or model output."""
    result = {"type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        result["http_status"] = exc.code
    return result


def metadata_summary(metadata: dict) -> dict:
    """Whitelist provenance instead of persisting arbitrary caller metadata."""
    result = {}
    for key in ("sglang_revision",):
        value = metadata.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
            result[key] = value
    for key in (
        "container_image",
        "container_image_id_expected",
        "base_image_id",
        "container_image_id_observed",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            result[key] = value
    for key in (
        "benchmark_client_sha256",
        "runner_sha256",
        "model_config_sha256",
        "server_container_id_observed",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            result[key] = value
    for key, choices in (
        ("scheme", {"w4a4", "w4a8"}),
        ("mode", {"baseline", "mtp"}),
        ("native_endpoint", {"/generate"}),
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value in choices:
            result[key] = value
    port = metadata.get("port")
    if nonnegative_int(port) and 0 < port < 65536:
        result["port"] = port
    command = metadata.get("server_command")
    result["server_command_sha256"] = value_digest(command)
    if isinstance(command, (list, str)):
        result["server_command_count"] = len(command)
    return result


def decode_json(data):
    # Retain invalid numeric constants as strings so failed finite-value checks
    # can still be saved in a valid JSON report instead of losing the evidence.
    return json.loads(data, parse_constant=lambda value: value)


def request(base_url: str, path: str, timeout: float, payload: dict | None = None):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        # Do not read or relay server error bodies; they may contain inputs.
        raise exc


def flush_cache(base_url: str, timeout: float) -> dict:
    status, body = request(base_url, "/flush_cache?timeout=30", timeout, {})
    message = body.decode(errors="replace")
    # The pinned SGLang HTTP server returns plain text, not a JSON object.
    if status != 200 or not message.startswith("Cache flushed."):
        raise RuntimeError("Cannot confirm prefix-cache flush")
    return {"status": status, "confirmed": True, "included_in_timing": False}


def load_fixtures(repo: Path):
    path = repo / "scripts/check_sglang_multimodal.py"
    spec = importlib.util.spec_from_file_location("megaquant_multimodal_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import fixed fixtures from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def create_cases(fixtures, tokenizer) -> list[dict]:
    cases = [dict(case) for case in fixtures.TEXT_CASES]
    # The code fixture is a structural sanity check.  Give it enough room to
    # finish so a length cutoff cannot be mistaken for a valid response.
    for case in cases:
        if case["name"] == "python_code":
            case["max_new_tokens"] = 512
    for name, prompt, expected in (
        ("red_square", fixtures.COLOR_PROMPT, "red"),
        ("blue_square", fixtures.COLOR_PROMPT, "blue"),
        ("ocr_3729", fixtures.OCR_PROMPT, "3729"),
    ):
        raw = fixtures.fixture_png(name)
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "expected": expected,
                "max_new_tokens": 32,
                "image_data": "data:image/png;base64," + base64.b64encode(raw).decode(),
                "image_sha256": digest(raw),
                "image_bytes": len(raw),
            }
        )
    for case in cases:
        content = [{"type": "text", "text": case["prompt"]}]
        if "image_data" in case:
            content.insert(0, {"type": "image"})
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        payload = {
            "text": rendered,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": case["max_new_tokens"],
            },
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        if "image_data" in case:
            payload["image_data"] = case["image_data"]
        case["request"] = payload
        case["prompt_sha256"] = digest(case["prompt"].encode())
        case["rendered_prompt_sha256"] = digest(rendered.encode())
        case["request_sha256"] = value_digest(payload)
    return cases


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def function_body_nodes(function: ast.FunctionDef):
    """Inspect the function's own body, excluding nested definition bodies."""
    pending = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))


def code_structure(output: str) -> dict:
    """Check requested code structure with AST only; never execute model output."""
    required = ("fibonacci", "factorial", "is_prime")
    checks = {
        "code_one_python_block": False,
        "code_valid_ast": False,
        "code_required_functions_and_signatures": False,
        "code_function_docstrings": False,
        "code_function_loops": False,
        "code_function_returns": False,
        "code_six_assert_tests": False,
        "code_two_distinct_assert_tests_per_function": False,
    }
    result = {
        "checks": checks,
        "validation": "AST structure only; assertions and generated functions are never executed.",
        "docstring_validation": (
            "Nonempty docstrings are required; natural-language sentence quality is not scored."
        ),
    }
    block = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\n(.*?)\n?```\s*", output, re.DOTALL)
    checks["code_one_python_block"] = block is not None and output.count("```") == 2
    if not checks["code_one_python_block"]:
        return result
    try:
        tree = ast.parse(block.group(1), filename="<generated-code>", mode="exec")
    except (SyntaxError, ValueError) as exc:
        result["parse_error_type"] = type(exc).__name__
        return result
    checks["code_valid_ast"] = True
    definitions = {
        name: [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
        for name in required
    }
    functions = {
        name: nodes[0]
        for name, nodes in definitions.items()
        if len(nodes) == 1 and isinstance(nodes[0], ast.FunctionDef)
    }
    checks["code_required_functions_and_signatures"] = len(functions) == len(required) and all(
        [arg.arg for arg in function.args.posonlyargs + function.args.args] == ["n"]
        and not function.args.kwonlyargs
        and function.args.vararg is None
        and function.args.kwarg is None
        for function in functions.values()
    )
    result["functions"] = {}
    for name, function in functions.items():
        nodes = list(function_body_nodes(function))
        docstring = ast.get_docstring(function)
        result["functions"][name] = {
            "docstring_nonempty": isinstance(docstring, str) and bool(docstring.strip()),
            # Comprehensions are loops too, for example all(n % i for i in ...).
            "has_loop": any(
                isinstance(node, (ast.For, ast.While, ast.comprehension)) for node in nodes
            ),
            "has_return": any(isinstance(node, ast.Return) for node in nodes),
        }
    complete = len(functions) == len(required)
    checks["code_function_docstrings"] = complete and all(
        info["docstring_nonempty"] for info in result["functions"].values()
    )
    checks["code_function_loops"] = complete and all(
        info["has_loop"] for info in result["functions"].values()
    )
    checks["code_function_returns"] = complete and all(
        info["has_return"] for info in result["functions"].values()
    )
    # Assertions inside the implementations are preconditions, not tests of
    # those functions. Test helpers/main guards outside the implementations count.
    implementation_nodes = {
        id(node) for function in functions.values() for node in ast.walk(function)
    }
    assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert) and id(node) not in implementation_nodes
    ]
    tests = {name: set() for name in required}
    for assertion in assertions:
        called = {
            node.func.id
            for node in ast.walk(assertion.test)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for name in tests.keys() & called:
            tests[name].add(ast.dump(assertion.test, include_attributes=False))
    result["assert_count"] = len(assertions)
    result["distinct_assert_tests_per_function"] = {
        name: len(items) for name, items in tests.items()
    }
    checks["code_six_assert_tests"] = len(assertions) >= 6
    checks["code_two_distinct_assert_tests_per_function"] = all(
        len(items) >= 2 for items in tests.values()
    )
    return result


def response_checks(case: dict, response: dict, fixtures, mode: str) -> dict:
    output = response.get("text")
    checks = {"nonempty": isinstance(output, str) and bool(output.strip())}
    if "expected" in case:
        # These prompts ask for only an integer, lowercase color word, or digits.
        # Case conversion/punctuation trimming would conceal instruction failures.
        checks["expected_answer"] = isinstance(output, str) and output.strip() == case["expected"]
    if case["name"] == "python_code":
        checks.update(code_structure(output if isinstance(output, str) else "")["checks"])
    meta = response.get("meta_info")
    checks["native_meta_info"] = isinstance(meta, dict)
    meta = meta if isinstance(meta, dict) else {}
    finish_reason = meta.get("finish_reason")
    checks["terminal_finish_reason_present"] = isinstance(finish_reason, dict) and isinstance(
        finish_reason.get("type"), str
    )
    checks["not_length_terminated"] = (
        isinstance(finish_reason, dict) and finish_reason.get("type") != "length"
    )
    checks["successful_stop"] = (
        isinstance(finish_reason, dict) and finish_reason.get("type") == "stop"
    )
    ids = response.get("output_ids")
    checks["native_output_ids"] = (
        isinstance(ids, list) and bool(ids) and all(nonnegative_int(token) for token in ids)
    )
    logprobs = meta.get("output_token_logprobs")
    checks["finite_output_logprobs"] = (
        isinstance(logprobs, list)
        and bool(logprobs)
        and all(
            isinstance(item, list)
            and len(item) >= 2
            and finite_number(item[0])
            and item[0] <= 1e-5
            and nonnegative_int(item[1])
            for item in logprobs
        )
    )
    checks["logprobs_match_native_output_ids"] = (
        checks["native_output_ids"]
        and checks["finite_output_logprobs"]
        and [item[1] for item in logprobs] == ids
    )
    completion = meta.get("completion_tokens")
    checks["positive_completion_tokens"] = nonnegative_int(completion) and completion > 0
    checks["completion_tokens_match_native_output_ids"] = (
        checks["positive_completion_tokens"]
        and checks["native_output_ids"]
        and completion == len(ids)
    )
    cached = meta.get("cached_tokens")
    checks["no_cached_prompt_tokens"] = nonnegative_int(cached) and cached == 0
    counters = {
        key: meta.get(key)
        for key in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
    }
    if mode == "mtp":
        verify, correct, proposed = counters.values()
        typed_counts = all(nonnegative_int(value) for value in counters.values())
        checks["mtp_executed"] = nonnegative_int(verify) and verify > 0
        checks["spec_draft_counts_valid"] = (
            typed_counts and proposed > 0 and 0 <= correct <= proposed
        )
        rate, length = meta.get("spec_accept_rate"), meta.get("spec_accept_length")
        checks["spec_accept_rate_matches_counts"] = (
            checks["spec_draft_counts_valid"]
            and finite_number(rate)
            and 0 <= rate <= 1
            and math.isclose(rate, correct / proposed, rel_tol=1e-6, abs_tol=1e-8)
        )
        checks["spec_accept_length_matches_counts"] = (
            checks["mtp_executed"]
            and checks["positive_completion_tokens"]
            and finite_number(length)
            and length > 0
            and math.isclose(length, completion / verify, rel_tol=1e-6, abs_tol=1e-8)
        )
    else:
        checks["baseline_has_no_spec_execution"] = all(
            key not in meta or (nonnegative_int(value) and value == 0)
            for key, value in counters.items()
        )
    return checks


def summarize(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def output_summary(response: dict, checks: dict) -> dict:
    """Keep token fingerprints and text/logprob statistics without their values."""
    output = response.get("text")
    ids = response.get("output_ids")
    result = {
        "output_ids_sha256": value_digest(ids) if checks["native_output_ids"] else None,
        "output_ids_count": len(ids) if isinstance(ids, list) else None,
    }
    if isinstance(output, str):
        data = output.encode()
        result.update(
            output_text_sha256=digest(data),
            output_text_chars=len(output),
            output_text_utf8_bytes=len(data),
        )
    meta = response.get("meta_info")
    meta = meta if isinstance(meta, dict) else {}
    logprobs = meta.get("output_token_logprobs")
    result["output_logprob_count"] = len(logprobs) if isinstance(logprobs, list) else None
    if checks["finite_output_logprobs"]:
        values = [item[0] for item in logprobs]
        result["output_logprob_min"] = min(values)
        result["output_logprob_max"] = max(values)
    return result


def spec_summary(meta: dict) -> dict:
    result = {}
    for key in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts"):
        if nonnegative_int(value := meta.get(key)):
            result[key] = value
    for key in ("spec_accept_rate", "spec_accept_length"):
        if finite_number(value := meta.get(key)):
            result[key] = value
    histogram = meta.get("spec_correct_drafts_histogram")
    if isinstance(histogram, list) and all(nonnegative_int(value) for value in histogram):
        result["spec_correct_drafts_histogram"] = histogram
    return result


def finish_report(report: dict) -> None:
    all_trials = [trial for case in report["cases"] for trial in case["trials"]]
    measured = [trial for trial in all_trials if trial["phase"] == "measured"]
    passed = (
        bool(all_trials)
        and not report.get("error")
        and all(
            not trial.get("error") and all(trial.get("checks", {}).values()) for trial in all_trials
        )
    )
    for case in report["cases"]:
        valid = [
            trial
            for trial in case["trials"]
            if trial["phase"] == "measured"
            and not trial.get("error")
            and all(trial.get("checks", {}).values())
        ]
        summary = {
            "measured_trials": len(valid),
            "required_measured_trials": report["measured_repetitions"],
        }
        if valid:
            summary["client_wall_seconds"] = summarize([trial["wall_seconds"] for trial in valid])
            summary["end_to_end_completion_tokens_per_second"] = summarize(
                [trial["completion_tokens_per_second"] for trial in valid]
            )
            summary["completion_tokens"] = summarize(
                [trial["completion_tokens"] for trial in valid]
            )
            first = valid[0]["output_summary"]
            summary["greedy_output_ids_identical_across_measured_trials"] = all(
                trial["output_summary"]["output_ids_sha256"] == first["output_ids_sha256"]
                and trial["output_summary"]["output_ids_count"] == first["output_ids_count"]
                for trial in valid
            )
            summary["output_comparison_method"] = "canonical_json_sha256_and_count"
            passed = passed and summary["greedy_output_ids_identical_across_measured_trials"]
            common_timing_keys = set.intersection(
                *(set(trial["server_timing_metrics"]) for trial in valid)
            )
            summary["server_timing_metrics"] = {
                key: summarize([trial["server_timing_metrics"][key] for trial in valid])
                for key in sorted(common_timing_keys)
            }
        passed = passed and len(valid) == report["measured_repetitions"]
        case["summary"] = summary
    by_name = {case["name"]: case for case in report["cases"]}
    report["color_counterfactual_passed"] = all(
        name in by_name
        and len(
            trials := [trial for trial in by_name[name]["trials"] if trial["phase"] == "measured"]
        )
        == report["measured_repetitions"]
        and all(trial.get("checks", {}).get("expected_answer") is True for trial in trials)
        for name in ("red_square", "blue_square")
    )
    passed = passed and report["color_counterfactual_passed"]
    if report["mode"] == "mtp":
        totals = {
            key: sum(
                value
                for trial in measured
                if nonnegative_int(value := trial.get("spec_metrics", {}).get(key))
            )
            for key in ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
        }
        proposed = totals["spec_num_proposed_drafts"]
        totals["aggregate_spec_accept_rate"] = (
            totals["spec_num_correct_drafts"] / proposed if proposed else None
        )
        report["measured_spec_metrics"] = totals
        report["greedy_outputs_identical"] = bool(measured) and all(
            trial.get("baseline_comparison", {}).get("exact_output_ids") is True
            for trial in measured
        )
        passed = (
            passed
            and totals["spec_num_correct_drafts"] > 0
            and bool(report.get("baseline_report"))
            and report["greedy_outputs_identical"]
        )
    required_names = {"arithmetic", "python_code", "red_square", "blue_square", "ocr_3729"}
    report["validation_passed"] = (
        passed and len(report["cases"]) == len(required_names) and set(by_name) == required_names
    )


def save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--repo",
        "--fixture-repo",
        dest="repo",
        type=Path,
        required=True,
        help="MegaQuant source containing the fixed fixture helper",
    )
    parser.add_argument("--model", required=True, help="Offline local tokenizer/model directory")
    parser.add_argument("--mode", choices=("baseline", "mtp"), required=True)
    parser.add_argument(
        "--run-metadata",
        type=Path,
        required=True,
        help="JSON with sglang_revision, container_image, server_command",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Required in MTP mode: passing baseline report from this same client",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1 or args.timeout <= 0:
        parser.error("warmup, repetitions, and timeout must be positive")
    if (args.mode == "mtp") != bool(args.baseline):
        parser.error("MTP mode requires --baseline; baseline mode must omit it")
    raw_metadata = decode_json(args.run_metadata.read_text())
    metadata = metadata_summary(raw_metadata) if isinstance(raw_metadata, dict) else {}
    if (
        not isinstance(metadata, dict)
        or any(
            not metadata.get(key)
            for key in ("sglang_revision", "container_image", "server_command_sha256")
        )
        or not raw_metadata.get("server_command")
    ):
        parser.error(
            "run metadata must include full sglang_revision, sha256 container_image, "
            "and server_command"
        )
    fixtures, fixture_path = load_fixtures(args.repo)
    # Import only when running, so --help has no Torch or tokenizer dependency.
    from PIL import __version__ as pillow_version
    from transformers import AutoTokenizer
    from transformers import __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = create_cases(fixtures, tokenizer)
    model_config = Path(args.model) / "config.json"
    model_config_sha256 = digest(model_config.read_bytes()) if model_config.is_file() else None
    references = {}
    if args.baseline:
        baseline = decode_json(args.baseline.read_text())
        if (
            not baseline.get("validation_passed")
            or baseline.get("mode") != "baseline"
            or baseline.get("schema_version") != 3
            or baseline.get("fixture_source_sha256") != digest(fixture_path.read_bytes())
            or baseline.get("model") != args.model
            or baseline.get("model_config_sha256") != model_config_sha256
        ):
            parser.error("--baseline must be a passing baseline report from this client")
        for key in ("sglang_revision", "container_image"):
            if baseline.get("run_metadata", {}).get(key) != metadata[key]:
                parser.error(f"Baseline run metadata mismatch: {key}")
        references = {case["name"]: case for case in baseline["cases"]}
        for case in cases:
            reference = references.get(case["name"], {})
            if any(
                reference.get(key) != case.get(key)
                for key in (
                    "prompt_sha256",
                    "rendered_prompt_sha256",
                    "image_sha256",
                    "request_sha256",
                )
            ):
                parser.error(f"Baseline fixture mismatch for {case['name']}")
    report = {
        "schema_version": 3,
        "persistence_policy": (
            "Explicit summaries only: no prompt, rendered prompt, payload text, output text, "
            "output token IDs, raw responses, raw requests, or arbitrary exception messages."
        ),
        "output_id_digest_encoding": "SHA256 of canonical JSON with compact separators",
        "scope": (
            "Five fixed deterministic text/vision sanity cases; code checks inspect AST structure "
            "without executing generated code or proving its algorithmic correctness. "
            "No general model-quality or performance claim."
        ),
        "timing_definition": (
            "Client wall time for one native /generate call, including preprocessing, prefill, "
            "decode, and transport; prefix-cache flush excluded. Server timing fields are "
            "reported only when provided, without inferring separate prefill/decode times."
        ),
        "endpoint": "/generate",
        "mode": args.mode,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "model_config_sha256": model_config_sha256,
        "run_metadata": metadata,
        "run_metadata_sha256": digest(args.run_metadata.read_bytes()),
        "fixture_source": str(fixture_path),
        "fixture_source_sha256": digest(fixture_path.read_bytes()),
        "python_version": platform.python_version(),
        "pillow_version": pillow_version,
        "transformers_version": transformers_version,
        "concurrency": 1,
        "warmup_repetitions": args.warmup,
        "measured_repetitions": args.repetitions,
        "temperature": 0,
        "enable_thinking": False,
        "baseline_report": str(args.baseline) if args.baseline else None,
        "baseline_report_sha256": digest(args.baseline.read_bytes()) if args.baseline else None,
        "baseline_parity_required": args.mode == "mtp",
        "cases": [],
    }
    try:
        status, body = request(args.base_url, "/get_server_info", args.timeout)
        report["server_info"] = {
            "http_status": status,
            "response_sha256": digest(body),
            "response_bytes": len(body),
            "native_object": isinstance(decode_json(body), dict),
        }
    except Exception as exc:
        report["server_info_error"] = safe_error(exc)
    save(args.output, report)
    try:
        for case in cases:
            case_result = {
                key: case[key]
                for key in (
                    "name",
                    "prompt_sha256",
                    "rendered_prompt_sha256",
                    "request_sha256",
                    "max_new_tokens",
                    "image_sha256",
                    "image_bytes",
                )
                if key in case
            }
            case_result["trials"] = []
            report["cases"].append(case_result)
            for index in range(args.warmup + args.repetitions):
                phase = "warmup" if index < args.warmup else "measured"
                trial = {"phase": phase, "index": index + 1}
                case_result["trials"].append(trial)
                trial["prefix_cache_flush"] = flush_cache(args.base_url, args.timeout)
                start = time.perf_counter()
                try:
                    status, body = request(
                        args.base_url, "/generate", args.timeout, case["request"]
                    )
                finally:
                    trial["wall_seconds"] = time.perf_counter() - start
                response = decode_json(body)
                if status != 200 or not isinstance(response, dict):
                    raise RuntimeError("Expected one successful native /generate response object")
                trial["checks"] = response_checks(case, response, fixtures, args.mode)
                trial["output_summary"] = output_summary(response, trial["checks"])
                if case["name"] == "python_code":
                    output = response.get("text")
                    trial["code_structure"] = code_structure(
                        output if isinstance(output, str) else ""
                    )
                meta = response.get("meta_info")
                meta = meta if isinstance(meta, dict) else {}
                trial["completion_tokens"] = (
                    value if nonnegative_int(value := meta.get("completion_tokens")) else None
                )
                trial["cached_tokens"] = (
                    value if nonnegative_int(value := meta.get("cached_tokens")) else None
                )
                trial["spec_metrics"] = spec_summary(meta)
                trial["server_timing_metrics"] = {
                    key: value
                    for key, value in meta.items()
                    if key
                    in {
                        "latency",
                        "ttft",
                        "prefill_time",
                        "decode_time",
                        "e2e_latency",
                        "e2e_latency_seconds",
                        "time_to_first_token",
                        "prefill_time_seconds",
                        "decode_time_seconds",
                    }
                    and finite_number(value)
                }
                if finite_number(trial["completion_tokens"]) and trial["completion_tokens"] > 0:
                    trial["completion_tokens_per_second"] = (
                        trial["completion_tokens"] / trial["wall_seconds"]
                    )
                if references:
                    reference_trial = next(
                        item
                        for item in references[case["name"]]["trials"]
                        if item["phase"] == "measured"
                    )
                    current = trial["output_summary"]
                    reference = reference_trial["output_summary"]
                    comparison = {
                        "method": "canonical_json_sha256_and_count",
                        "exact_output_ids": (
                            trial["checks"]["native_output_ids"]
                            and current["output_ids_sha256"] == reference["output_ids_sha256"]
                            and current["output_ids_count"] == reference["output_ids_count"]
                        ),
                        "baseline_tokens": reference["output_ids_count"],
                        "current_tokens": current["output_ids_count"],
                        "baseline_output_ids_sha256": reference["output_ids_sha256"],
                        "current_output_ids_sha256": current["output_ids_sha256"],
                    }
                    trial["baseline_comparison"] = comparison
                    trial["checks"]["baseline_output_ids_identical"] = (
                        trial["checks"]["native_output_ids"] and comparison["exact_output_ids"]
                    )
                save(args.output, report)
                print(
                    json.dumps(
                        {
                            "case": case["name"],
                            "phase": phase,
                            "index": index + 1,
                            "wall_seconds": trial["wall_seconds"],
                            "completion_tokens": trial["completion_tokens"],
                            "cached_tokens": trial["cached_tokens"],
                            "checks": trial["checks"],
                            "spec_metrics": trial["spec_metrics"],
                        }
                    ),
                    flush=True,
                )
                if not trial["checks"]["no_cached_prompt_tokens"]:
                    raise RuntimeError(
                        "Cached prompt tokens were nonzero or not reported; stopping timings"
                    )
    except Exception as exc:
        report["error"] = safe_error(exc)
        if report["cases"] and report["cases"][-1]["trials"]:
            report["cases"][-1]["trials"][-1]["error"] = report["error"]
        print(json.dumps({"error": report["error"]}), flush=True)
    report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    finish_report(report)
    save(args.output, report)
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
