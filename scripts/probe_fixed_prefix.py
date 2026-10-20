#!/usr/bin/env python3
"""Compare reconstructed prefixes using SGLang's native next-token logprobs.

The private bundle contains token IDs, never text requests. It has schema_version=1,
context_length, tokenization (retokenized_journal or original_token_ids), tokenizer_sha256,
source_journal_sha256 (a list), and cases. Each case has positive integer case/prefix
labels, stage (before_loop, loop_onset, inside_loop), input_ids and candidate_token_ids.
Candidate groups are loop, thinking_end, eos, reference_next or alternative, each a
nonempty list of IDs. Group scores mean the best *single-token* logprob in the group.

Example (synthetic IDs, not a model fixture)::

    {"case": 1, "prefix": 1, "stage": "loop_onset", "input_ids": [10, 11],
     "candidate_token_ids": {"loop": [12], "thinking_end": [13], "eos": [14]}}

Use --max-new-tokens 256/2048 for a bounded diagnostic rollout. This does not modify
formal journals or constitute a benchmark rerun. Private output contains requests,
server metadata and responses; the report has only hashes, counts and score statistics.
Both outputs must be new files. Journal retokenization does not recover original IDs
or original incremental decode states; even original IDs are replayed via prefill.

API checked against the frozen SGLang 0.5.20 io_struct.py and tokenizer_manager.py:
logprob options are top-level fields; output top/target entries are per-position lists
of [logprob, token_id, text]. return_text_in_logprobs=False removes detokenized text.
Greedy sampling returns log_softmax scores, not a temperature-zero distribution.
These are logprobs of the server's processed logits, not an unmodified tensor dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STAGES = {"before_loop", "loop_onset", "inside_loop"}
ROLES = {"loop", "thinking_end", "eos", "reference_next", "alternative"}
MODEL_LABELS = (
    "mse", "max", "mse_mixed_control", "lm_head_bf16", "attention_bf16", "combined_bf16", "bf16",
)
SERVER_KEYS = (
    "version", "context_length", "dtype", "kv_cache_dtype", "mamba_ssm_dtype", "tp_size",
    "attention_backend", "disable_radix_cache", "chunked_prefill_size", "disable_cuda_graph",
    "max_running_requests", "enable_deterministic_inference", "allow_auto_truncate",
    "speculative_algorithm",
)
LIMITATIONS = [
    "Prefill replay does not restore original incremental decode states.",
    "Candidate scores measure individual next tokens, not complete ending sequences.",
    "A next-token ranking change alone does not establish rollout stopping recovery.",
]


class DiagnosticError(ValueError):
    """An invalid input, response or replay environment."""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def valid_int(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def require_hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DiagnosticError(f"{name} must be a lowercase SHA256")


def validate_bundle(bundle: dict, max_new_tokens: int) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise DiagnosticError("Expected bundle schema_version=1")
    if not valid_int(max_new_tokens, 1) or max_new_tokens > 4096:
        raise DiagnosticError("max_new_tokens must be between 1 and 4096")
    if not valid_int(bundle.get("context_length"), 2):
        raise DiagnosticError("context_length must be an integer >=2")
    if bundle.get("tokenization") not in {"retokenized_journal", "original_token_ids"}:
        raise DiagnosticError("tokenization must identify the provenance of input IDs")
    require_hash(bundle.get("tokenizer_sha256"), "tokenizer_sha256")
    journals = bundle.get("source_journal_sha256")
    if not isinstance(journals, list) or not journals:
        raise DiagnosticError("source_journal_sha256 must be a nonempty list")
    for journal in journals:
        require_hash(journal, "source_journal_sha256 entry")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DiagnosticError("cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise DiagnosticError("Each prefix must be an object")
        if not valid_int(case.get("case"), 1) or not valid_int(case.get("prefix"), 1):
            raise DiagnosticError("case and prefix must be positive integer anonymous labels")
        identity = (case["case"], case["prefix"])
        if identity in seen:
            raise DiagnosticError("Duplicate case/prefix labels")
        seen.add(identity)
        if case.get("stage") not in STAGES:
            raise DiagnosticError("Unknown prefix stage")
        ids = case.get("input_ids")
        if not isinstance(ids, list) or not ids or not all(valid_int(i) for i in ids):
            raise DiagnosticError("input_ids must be a nonempty list of nonnegative integers")
        if len(ids) + max_new_tokens > bundle["context_length"]:
            raise DiagnosticError("Prefix plus diagnostic output would exceed context_length")
        groups = case.get("candidate_token_ids")
        if not isinstance(groups, dict) or not {"loop", "thinking_end", "eos"} <= groups.keys():
            raise DiagnosticError("Candidate groups must include loop, thinking_end and eos")
        if groups.keys() - ROLES:
            raise DiagnosticError("Unknown candidate role")
        for ids in groups.values():
            if (
                not isinstance(ids, list) or not 1 <= len(ids) <= 32
                or not all(valid_int(i) for i in ids) or len(set(ids)) != len(ids)
            ):
                raise DiagnosticError("Each candidate role needs 1..32 unique token IDs")


def request_payload(case: dict, max_new_tokens: int, top_logprobs_num: int) -> dict:
    return {
        "input_ids": case["input_ids"],
        "sampling_params": {
            "temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "max_new_tokens": max_new_tokens, "min_new_tokens": 0,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "ignore_eos": False, "skip_special_tokens": False, "no_stop_trim": True,
        },
        "return_logprob": True, "logprob_start_len": -1,
        "top_logprobs_num": top_logprobs_num,
        "token_ids_logprob": sorted({
            i for ids in case["candidate_token_ids"].values() for i in ids
        }),
        "return_text_in_logprobs": False, "stream": False, "no_logs": True,
    }


def http_json(base_url: str, path: str, timeout: float, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise DiagnosticError("Expected a single native response object")
    return result


def server_snapshot(info: dict, context_length: int) -> dict:
    # Some SGLang builds wrap the command-line fields, others expose them directly.
    args = info.get("server_args", info)
    if not isinstance(args, dict):
        raise DiagnosticError("Missing server arguments")
    if args.get("context_length") != context_length:
        raise DiagnosticError("Actual server context differs from the fixed prefix bundle")
    if "speculative_algorithm" not in args or args["speculative_algorithm"] is not None:
        raise DiagnosticError("Fixed-prefix diagnostics require speculative decoding disabled")
    if args.get("allow_auto_truncate") is not False:
        raise DiagnosticError("Fixed-prefix diagnostics require auto-truncate disabled")
    if args.get("mamba_ssm_dtype") not in {"float32", "fp32"}:
        raise DiagnosticError("Fixed-prefix diagnostics require FP32 SSM states")
    selected = {}
    for key in SERVER_KEYS:
        value = args.get(key, info.get(key))
        if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", value):
            raise DiagnosticError("Unexpected server setting format")
        if value is not None and type(value) not in (str, int, float, bool):
            raise DiagnosticError("Unexpected server setting type")
        if type(value) is float and not math.isfinite(value):
            raise DiagnosticError("Nonfinite server setting")
        selected[key] = value
    return selected


def logprob_map(entries: Any, name: str) -> dict[int, float]:
    if not isinstance(entries, list) or not entries:
        raise DiagnosticError(f"Missing {name}")
    result = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
            raise DiagnosticError(f"Malformed {name} entry")
        value, token_id = entry[:2]
        if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-5:
            raise DiagnosticError(f"Invalid {name} logprob")
        if not valid_int(token_id) or token_id in result:
            raise DiagnosticError(f"Invalid or duplicate {name} token ID")
        result[token_id] = float(value)
    return result


def repetition_stats(ids: list[int]) -> dict:
    """Count exact output-ID repeats without retaining the output-ID sequence."""
    longest = current = 0
    previous = None
    for token_id in ids:
        current = current + 1 if token_id == previous else 1
        longest = max(longest, current)
        previous = token_id
    windows = [tuple(ids[i:i + 32]) for i in range(0, len(ids) - 31, 16)]
    seen = set()
    repeated = 0
    for window in windows:
        repeated += window in seen
        seen.add(window)
    tail = ids[-4096:]
    period = suffix = cycles = 0
    for candidate in range(1, min(1024, len(tail) // 3) + 1):
        pos = len(tail) - candidate - 1
        while pos >= 0 and tail[pos] == tail[pos + candidate]:
            pos -= 1
        length = len(tail) - pos - 1
        if length >= max(32, 3 * candidate):
            period, suffix, cycles = candidate, length, length // candidate
            break
    return {
        "longest_same_token_run": longest,
        "repeated_32_token_windows_stride16": repeated,
        "sampled_32_token_windows_stride16": len(windows),
        "repeated_32_token_window_fraction": repeated / len(windows) if windows else 0.0,
        "checked_suffix_tokens": len(tail), "suffix_shortest_period_tokens": period or None,
        "suffix_periodic_tokens": suffix, "suffix_full_cycles": cycles,
        "suffix_rule": "shortest exact period with >=3 cycles and >=32 tokens in last4096 IDs",
    }


def summarize_response(case: dict, response: dict, max_new_tokens: int) -> dict:
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise DiagnosticError("Missing response metadata")
    if meta.get("prompt_tokens") != len(case["input_ids"]):
        raise DiagnosticError("Server consumed a different prefix token count")
    completed = meta.get("completion_tokens")
    if not valid_int(completed, 1) or completed > max_new_tokens:
        raise DiagnosticError("Unexpected completion token count")
    if meta.get("spec_verify_ct", 0) or meta.get("spec_num_proposed_drafts", 0):
        raise DiagnosticError("Speculative decoding ran during the probe")
    selected_entries = meta.get("output_token_logprobs")
    if not isinstance(selected_entries, list) or len(selected_entries) != completed:
        raise DiagnosticError("Output logprobs do not align with completion token count")
    selected = logprob_map([selected_entries[0]], "selected token")
    selected_id, selected_score = next(iter(selected.items()))
    top_rows = meta.get("output_top_logprobs")
    target_rows = meta.get("output_token_ids_logprobs")
    if not isinstance(top_rows, list) or len(top_rows) != completed:
        raise DiagnosticError("Top logprobs do not align with output positions")
    if not isinstance(target_rows, list) or len(target_rows) != completed:
        raise DiagnosticError("Target logprobs do not align with output positions")
    top = logprob_map(top_rows[0], "top tokens")
    targets = logprob_map(target_rows[0], "target tokens")
    expected = {i for ids in case["candidate_token_ids"].values() for i in ids}
    if set(targets) != expected:
        raise DiagnosticError("Server omitted or added requested candidate IDs")
    ordered = sorted(top, key=lambda token_id: (-top[token_id], token_id))
    if len(ordered) < 2 or selected_id not in top:
        raise DiagnosticError("Need at least two top candidates including the sampled token")
    if abs(top[selected_id] - selected_score) > 1e-5:
        raise DiagnosticError("Selected and top logprobs disagree")
    if abs(top[ordered[0]] - selected_score) > 1e-5:
        raise DiagnosticError("Selected next token is not a greedy top candidate")
    for token_id in top.keys() & targets.keys():
        if abs(top[token_id] - targets[token_id]) > 1e-5:
            raise DiagnosticError("Top and targeted logprobs disagree")
    output_ids = response.get("output_ids")
    if not isinstance(output_ids, list) or len(output_ids) != completed:
        raise DiagnosticError("Native output_ids must include all generated tokens")
    if not all(valid_int(i) for i in output_ids):
        raise DiagnosticError("Invalid native output_ids")
    logged_ids = []
    for entry in selected_entries:
        logged_ids.append(next(iter(logprob_map([entry], "selected token"))))
    if output_ids != logged_ids:
        raise DiagnosticError("Native output_ids differ from output logprob IDs")
    candidates = {}
    for role, ids in sorted(case["candidate_token_ids"].items()):
        score = max(targets[token_id] for token_id in ids)
        ranks = [ordered.index(i) + 1 for i in ids if i in top]
        candidates[role] = {
            "token_ids_sha256": digest(ids), "token_count": len(ids),
            "best_single_token_logprob": score,
            "selected_minus_best_logprob": selected_score - score,
            "best_top_rank": min(ranks) if ranks else None,
            "selected_in_group": selected_id in ids,
        }
    finish = meta.get("finish_reason", {})
    finish_type = finish.get("type") if isinstance(finish, dict) else None
    if finish_type == "abort":
        raise DiagnosticError("Aborted generation cannot provide a successful probe result")
    if finish_type not in {"length", "stop"}:
        raise DiagnosticError("Unknown native finish reason")
    eos = set(case["candidate_token_ids"]["eos"])
    if finish_type == "stop":
        # SGLang also labels string/regex stops and its "NaN happened" path
        # as type=stop. Only an actual, preserved EOS match validates stopping
        # under this request's ignore_eos=False and no_stop_trim=True protocol.
        matched = finish.get("matched")
        if not valid_int(matched) or matched not in eos:
            raise DiagnosticError("Stop finish must match an integer declared EOS token")
        if output_ids[-1] != matched:
            raise DiagnosticError("Matched EOS must be preserved as the last native output token")
        if any(token in eos for token in output_ids[:-1]):
            raise DiagnosticError("Native output continued after an EOS token")
    else:
        if completed != max_new_tokens:
            raise DiagnosticError("Length finish occurred before the diagnostic token budget")
        if any(token in eos for token in output_ids) or finish.get("matched") is not None:
            raise DiagnosticError("Length finish cannot contain an EOS token or match")
        if "length" in finish and (
            not valid_int(finish["length"], 1) or finish["length"] != max_new_tokens
        ):
            raise DiagnosticError("Native length finish differs from the diagnostic token budget")
    result = {
        "case": case["case"], "prefix": case["prefix"], "stage": case["stage"],
        "input_ids_sha256": digest(case["input_ids"]), "prompt_tokens": meta["prompt_tokens"],
        "selected_token_sha256": digest(selected_id), "selected_logprob": selected_score,
        "top1_logprob": top[ordered[0]], "top2_logprob": top[ordered[1]],
        "top1_minus_top2_logprob": top[ordered[0]] - top[ordered[1]],
        "candidates": candidates,
        "rollout": {
            "completion_tokens": completed, "finish_type": finish_type,
            "output_ids_sha256": digest(output_ids),
            "thinking_end_token_occurrences": sum(
                i in case["candidate_token_ids"]["thinking_end"] for i in output_ids
            ),
            "eos_token_occurrences": sum(
                i in case["candidate_token_ids"]["eos"] for i in output_ids
            ),
            "repetition": repetition_stats(output_ids),
        },
    }
    if {"loop", "thinking_end"} <= candidates.keys():
        result["loop_minus_thinking_end_logprob"] = (
            candidates["loop"]["best_single_token_logprob"]
            - candidates["thinking_end"]["best_single_token_logprob"]
        )
    return result


def compare_reports(current: dict, reference: dict) -> dict:
    for key in ("schema_version", "source", "protocol", "server_settings"):
        if current.get(key) != reference.get(key):
            raise DiagnosticError(f"Reference differs in {key}; controlled comparison rejected")
    old = {(r["case"], r["prefix"]): r for r in reference["cases"]}
    new = {(r["case"], r["prefix"]): r for r in current["cases"]}
    if old.keys() != new.keys():
        raise DiagnosticError("Reference and current prefix labels differ")
    comparisons = []
    for identity, row in new.items():
        other = old[identity]
        if "error_type" in row or "error_type" in other:
            comparisons.append({"case": identity[0], "prefix": identity[1], "valid": False})
            continue
        if row["input_ids_sha256"] != other["input_ids_sha256"]:
            raise DiagnosticError("Reference uses different input IDs")
        if row["candidates"].keys() != other["candidates"].keys():
            raise DiagnosticError("Reference uses different candidate roles")
        deltas = {}
        for role, candidate in row["candidates"].items():
            baseline = other["candidates"][role]
            if candidate["token_ids_sha256"] != baseline["token_ids_sha256"]:
                raise DiagnosticError("Reference uses different candidate IDs")
            deltas[role] = {
                "best_single_token_logprob_change": candidate["best_single_token_logprob"]
                - baseline["best_single_token_logprob"],
                "selected_minus_best_logprob_change": candidate["selected_minus_best_logprob"]
                - baseline["selected_minus_best_logprob"],
            }
        comparisons.append({
            "case": identity[0], "prefix": identity[1], "valid": True,
            "selected_next_token_changed": row["selected_token_sha256"]
            != other["selected_token_sha256"],
            "top1_minus_top2_logprob_change": row["top1_minus_top2_logprob"]
            - other["top1_minus_top2_logprob"],
            "loop_minus_thinking_end_logprob_change": row["loop_minus_thinking_end_logprob"]
            - other["loop_minus_thinking_end_logprob"],
            "candidate_deltas": deltas,
            "exact_rollout_ids": row["rollout"]["output_ids_sha256"]
            == other["rollout"]["output_ids_sha256"],
        })
    return {"reference_model_label": reference["model_label"], "cases": comparisons}


def reserve_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)


def write_json(path: Path, value: dict) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        temporary.write(raw)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Private fixed input-ID bundle")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--server-info-path", choices=("/get_server_info", "/server_info"),
                        default="/get_server_info")
    parser.add_argument("--model-label", choices=MODEL_LABELS, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-logprobs-num", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if not 2 <= args.top_logprobs_num <= 128:
        parser.error("--top-logprobs-num must be between 2 and 128")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    paths = [args.bundle, args.private_output, args.report_output]
    if args.reference_report:
        paths.append(args.reference_report)
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Input and output paths must all differ")
    bundle = json.loads(args.bundle.read_text())
    validate_bundle(bundle, args.max_new_tokens)
    reserve_output(args.private_output)
    try:
        reserve_output(args.report_output)
    except Exception:
        args.private_output.unlink()
        raise
    private = {"model_label": args.model_label, "bundle": bundle, "cases": []}
    report = {
        "schema_version": 1, "model_label": args.model_label,
        "source": {
            "bundle_sha256": digest(bundle), "tokenizer_sha256": bundle["tokenizer_sha256"],
            "source_journal_sha256": bundle["source_journal_sha256"],
            "tokenization": bundle["tokenization"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "protocol": {
            "endpoint": "/generate", "max_new_tokens": args.max_new_tokens,
            "top_logprobs_num": args.top_logprobs_num, "concurrency": 1,
            "sampling_params": request_payload(bundle["cases"][0], args.max_new_tokens,
                                               args.top_logprobs_num)["sampling_params"],
            "candidate_group_semantics": "best single-token logprob, not sequence probability",
        },
        "limitations": LIMITATIONS + (
            ["Journal text was retokenized; original IDs are unavailable."]
            if bundle["tokenization"] == "retokenized_journal" else []
        ),
        "cases": [], "complete": False,
    }
    failed = False
    try:
        info = http_json(args.base_url, args.server_info_path, args.timeout)
        private["server_info"] = info
        report["server_settings"] = server_snapshot(info, bundle["context_length"])
        for case in bundle["cases"]:
            payload = request_payload(case, args.max_new_tokens, args.top_logprobs_num)
            entry = {"case": case["case"], "prefix": case["prefix"], "request": payload}
            start = time.monotonic()
            try:
                response = http_json(args.base_url, "/generate", args.timeout, payload)
                entry["response"] = response
                row = summarize_response(case, response, args.max_new_tokens)
                row["request_seconds"] = time.monotonic() - start
                report["cases"].append(row)
            except Exception as exc:
                failed = True
                entry["error"] = str(exc)
                report["cases"].append({
                    "case": case["case"], "prefix": case["prefix"], "stage": case["stage"],
                    "error_type": type(exc).__name__,
                })
                print(json.dumps({"case": case["case"], "prefix": case["prefix"],
                                  "error_type": type(exc).__name__}), flush=True)
            private["cases"].append(entry)
            write_json(args.private_output, private)
            write_json(args.report_output, report)
            print(json.dumps({"completed_prefixes": len(report["cases"]), "failed": failed}),
                  flush=True)
        report["complete"] = not failed
        if args.reference_report:
            reference = json.loads(args.reference_report.read_text())
            report["comparison"] = compare_reports(report, reference)
    except Exception as exc:
        failed = True
        report["complete"] = False
        report["error_type"] = type(exc).__name__
        private["error"] = str(exc)
        print(json.dumps({"error_type": type(exc).__name__}), flush=True)
    finally:
        write_json(args.private_output, private)
        write_json(args.report_output, report)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
