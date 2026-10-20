#!/usr/bin/env python3
"""Run one deterministic fixture; raw response stays in the controller pipe."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

FIXTURE_SHA256 = "f0e8ff85f40b4aaa9847ab95dd9098591845f137faab574ca6d146f0be3f04da"


def post(base: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("native response is not an object")
    return value


def load_cases(repo: Path):
    path = repo / "scripts/check_sglang_multimodal.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise RuntimeError("fixture file differs from the pinned validation fixture")
    spec = importlib.util.spec_from_file_location("v7_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing fixture loader")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    cases = [dict(x) for x in module.TEXT_CASES]
    for name, prompt in (
        ("red_square", module.COLOR_PROMPT),
        ("blue_square", module.COLOR_PROMPT),
        ("ocr_3729", module.OCR_PROMPT),
    ):
        raw = module.fixture_png(name)
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "max_new_tokens": 32,
                "image_data": "data:image/png;base64," + base64.b64encode(raw).decode(),
                "image_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return path, cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture-repo", type=Path, required=True)
    parser.add_argument(
        "--case",
        default="python_code",
        choices=("python_code", "arithmetic", "red_square", "blue_square", "ocr_3729"),
    )
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    fixture_path, cases = load_cases(args.fixture_repo)
    selected = next(case for case in cases if case["name"] == args.case)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    content = [{"type": "text", "text": selected["prompt"]}]
    if "image_data" in selected:
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
            "max_new_tokens": (
                768 if args.case == "python_code" else selected.get("max_new_tokens", 32)
            ),
        },
        "stream": False,
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 5,
        "return_text_in_logprobs": False,
        "return_prompt_token_ids": True,
        "log_metrics": False,
    }
    if "image_data" in selected:
        payload["image_data"] = selected["image_data"]
    flush = urllib.request.Request(
        args.base_url.rstrip("/") + "/flush_cache?timeout=30",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(flush, timeout=args.timeout) as response:
        if response.status != 200 or not response.read().startswith(b"Cache flushed."):
            raise RuntimeError("cache flush was not confirmed")
    value = post(args.base_url, "/generate", payload, args.timeout)
    sys.stdout.write(
        json.dumps(
            {
                "response": value,
                "case": selected["name"],
                "fixture_file_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "image_sha256": selected.get("image_sha256"),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        sys.stderr.write(type(exc).__name__ + "\n")
        raise SystemExit(1) from None
