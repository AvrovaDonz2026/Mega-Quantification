#!/usr/bin/env python3
"""Score two candidate tokens at one fixed prefix using a fresh prefill."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from typing import Any

SCHEMA_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def token_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def post(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("native response is not an object")
    return value


def flush_cache(base_url: str, timeout: float) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/flush_cache?timeout=30",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if response.status != 200 or not body.startswith(b"Cache flushed."):
            raise RuntimeError("cache flush was not confirmed")


def read_control() -> dict[str, Any]:
    value = json.load(sys.stdin)
    if not isinstance(value, dict):
        raise ValueError("control must be an object")
    prefix = value.get("prefix_ids")
    candidates = value.get("candidate_ids")
    if not isinstance(prefix, list) or not prefix or not all(token_id(item) for item in prefix):
        raise ValueError("control prefix IDs are invalid")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 2
        or not all(token_id(item) for item in candidates)
        or candidates[0] == candidates[1]
    ):
        raise ValueError("control candidate IDs are invalid")
    prefix_role = value.get("prefix_role", "unspecified")
    if not isinstance(prefix_role, str) or prefix_role not in {"baseline", "mtp", "unspecified"}:
        raise ValueError("prefix_role is invalid")
    index = value.get("target_index")
    if index is not None and not token_id(index):
        raise ValueError("target_index is invalid")
    return {
        "prefix_ids": prefix,
        "candidate_ids": candidates,
        "prefix_role": prefix_role,
        "target_index": index,
    }


def parse_score_row(
    response: dict[str, Any], candidates: list[int], expected_prompt: list[int]
) -> dict[str, Any]:
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("response meta_info is missing")
    rows = meta.get("input_token_ids_logprobs")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("input candidate logprob row count is not exactly one")
    row = rows[0]
    if not isinstance(row, list):
        raise RuntimeError("input candidate logprob row is not a list")
    scores: dict[int, float] = {}
    occurrences: dict[int, int] = {}
    for item in row:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not finite(item[0])
            or item[0] > 1e-5
            or not token_id(item[1])
            or item[2] is not None
        ):
            raise RuntimeError("input candidate logprob row contains an invalid item")
        occurrences[item[1]] = occurrences.get(item[1], 0) + 1
        scores[item[1]] = float(item[0])
    if any(occurrences.get(candidate) != 1 for candidate in candidates):
        raise RuntimeError("input candidate row does not contain each requested ID exactly once")
    prompt_ids = response.get("prompt_token_ids")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or not all(token_id(item) for item in prompt_ids)
    ):
        raise RuntimeError("prompt token IDs are missing or invalid")
    if prompt_ids != expected_prompt:
        raise RuntimeError("returned prompt token IDs do not match the submitted input IDs")
    return {
        "scores": [scores[candidate] for candidate in candidates],
        "prompt_len": len(prompt_ids),
        "finish_reason_type": meta.get("finish_reason", {}).get("type")
        if isinstance(meta.get("finish_reason"), dict)
        else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    control = read_control()
    prefix = control["prefix_ids"]
    candidates = control["candidate_ids"]
    input_ids = prefix + [candidates[0]]
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "stream": False,
        "return_logprob": True,
        "logprob_start_len": len(prefix),
        "token_ids_logprob": candidates,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
        "return_prompt_token_ids": True,
        "log_metrics": False,
    }
    flush_cache(args.base_url, args.timeout)
    parsed = parse_score_row(
        post(args.base_url, "/generate", payload, args.timeout), candidates, input_ids
    )
    values = parsed["scores"]
    difference = values[0] - values[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "probe_kind": "same_prefix_candidate_logits_fresh_prefill",
        "decode_state_aligned": False,
        "prefix_role": control["prefix_role"],
        "target_index": control["target_index"],
        "prefix_len": len(prefix),
        "prefix_sha256": digest(prefix),
        "candidate_ids_sha256": digest(candidates),
        "candidate_count": 2,
        "input_ids_len": len(input_ids),
        "logprob_start_len": len(prefix),
        "input_logprob_row_count": 1,
        "row_position": "appended_candidate_zero_at_input_index_logprob_start_len",
        "prompt_len_returned": parsed["prompt_len"],
        "finish_reason_type": parsed["finish_reason_type"],
        "candidate_scores": [
            {
                "token_fingerprint": digest(candidate),
                "logprob": value,
                "probability": math.exp(value),
                "finite": finite(value),
            }
            for candidate, value in zip(candidates, values, strict=True)
        ],
        "candidate_logprob_difference": difference,
        "candidate_logprob_abs_difference": abs(difference),
        "candidate_argmax_index": 0 if values[0] > values[1] else 1,
        "candidate_argmax_tie": values[0] == values[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    try:
        report = run(args)
    except BaseException as exc:
        sys.stderr.write(type(exc).__name__ + "\n")
        return 1
    sys.stdout.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
