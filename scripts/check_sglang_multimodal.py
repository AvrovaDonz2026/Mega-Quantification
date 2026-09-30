#!/usr/bin/env python3
"""Run deterministic visual and embedded-MTP checks against an existing server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

COLOR_PROMPT = (
    "What is the color of the square in this image? Answer exactly one lowercase color word."
)
OCR_PROMPT = "Read the four digits printed in this image. Answer only the digits."
TEXT_CASES = [
    {
        "name": "arithmetic",
        "prompt": "Calculate 17 * 23. Answer only the integer, without explanation.",
        "expected": "391",
        "max_new_tokens": 32,
    },
    {
        "name": "python_code",
        "prompt": (
            "Write Python code defining fibonacci(n), factorial(n), and is_prime(n). "
            "Use loops, include a one-sentence docstring and two assert tests for each function. "
            "Return only one Python code block."
        ),
        "max_new_tokens": 192,
    },
]
SPEC_KEYS = [
    "spec_verify_ct",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_accept_rate",
    "spec_accept_length",
    "spec_correct_drafts_histogram",
]


def fixture_png(name: str) -> bytes:
    if name in {"red_square", "blue_square"}:
        image = Image.new("RGB", (320, 320), "white")
        ImageDraw.Draw(image).rectangle((48, 48, 271, 271), fill=name.split("_")[0])
    else:
        image = Image.new("RGB", (640, 320), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default(size=160)
        except TypeError:
            font = ImageFont.load_default()
            glyphs = Image.new("RGB", (40, 16), "white")
            ImageDraw.Draw(glyphs).text((0, 0), "3729", font=font, fill="black")
            image.paste(glyphs.resize((560, 224), Image.Resampling.NEAREST), (40, 48))
        else:
            box = draw.textbbox((0, 0), "3729", font=font)
            x = (image.width - box[2] + box[0]) // 2 - box[0]
            y = (image.height - box[3] + box[1]) // 2 - box[1]
            draw.text((x, y), "3729", font=font, fill="black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected one response object, got {type(result).__name__}")
    return result


def output_checks(case: dict, response: dict) -> dict:
    output = response.get("text", "")
    expected = case.get("expected")
    checks = {"nonempty": bool(output.strip())}
    if expected is not None:
        normalized = output.strip().lower().strip(".\n\r\t `\"'")
        checks["expected_answer"] = normalized == expected
    logprobs = response.get("meta_info", {}).get("output_token_logprobs", [])
    if logprobs:
        checks["finite_output_logprobs"] = all(
            isinstance(item[0], (int, float)) and math.isfinite(item[0]) for item in logprobs
        )
    return checks


def compare_ids(current: list[int], reference: list[int]) -> dict:
    common = next(
        (i for i, (a, b) in enumerate(zip(current, reference, strict=False)) if a != b),
        min(len(current), len(reference)),
    )
    return {
        "exact_output_ids": current == reference,
        "common_prefix_tokens": common,
        "baseline_tokens": len(reference),
        "current_tokens": len(current),
        "first_mismatch": None if current == reference else common,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="Offline local tokenizer/model directory")
    parser.add_argument("--mode", choices=["baseline", "mtp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--chat-fallback", action="store_true")
    args = parser.parse_args()
    if args.mode == "mtp" and args.baseline is None:
        parser.error("--mode mtp requires --baseline")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    references = {case["name"]: case for case in baseline["cases"]} if baseline else {}
    cases = [dict(case) for case in TEXT_CASES]
    for name, prompt, expected in [
        ("red_square", COLOR_PROMPT, "red"),
        ("blue_square", COLOR_PROMPT, "blue"),
        ("ocr_3729", OCR_PROMPT, "3729"),
    ]:
        raw = fixture_png(name)
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "expected": expected,
                "max_new_tokens": 32,
                "image_data": "data:image/png;base64," + base64.b64encode(raw).decode(),
                "image_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    report = {"mode": args.mode, "cases": [], "timeout_per_request_seconds": args.timeout}
    for case in cases:
        content = [{"type": "text", "text": case["prompt"]}]
        if "image_data" in case:
            content.insert(0, {"type": "image"})
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        payload = {
            "text": text,
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
        result = {
            "name": case["name"],
            "expected": case.get("expected"),
            "request": payload,
            "image_sha256": case.get("image_sha256"),
            "endpoint": "/generate",
        }
        start = time.perf_counter()
        try:
            response = post(args.base_url, "/generate", payload, args.timeout)
            result["response"] = response
            result["checks"] = output_checks(case, response)
            meta = response.get("meta_info", {})
            result["spec_metrics"] = {key: meta[key] for key in SPEC_KEYS if key in meta}
            if args.mode == "mtp":
                result["checks"]["mtp_executed"] = meta.get("spec_verify_ct", 0) > 0
                result["accepted_drafts"] = meta.get("spec_num_correct_drafts", 0)
                reference = references.get(case["name"], {}).get("response", {})
                if "output_ids" in response and "output_ids" in reference:
                    result["baseline_comparison"] = compare_ids(
                        response["output_ids"], reference["output_ids"]
                    )
                else:
                    result["baseline_comparison"] = {
                        "error": "Native output_ids missing from one response"
                    }
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if args.chat_fallback and "image_data" in case:
                chat_payload = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": case["image_data"]}},
                                {"type": "text", "text": case["prompt"]},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": case["max_new_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream": False,
                }
                result["fallback_request"] = chat_payload
                try:
                    chat = post(args.base_url, "/v1/chat/completions", chat_payload, args.timeout)
                    result["fallback_response"] = chat
                    message = chat["choices"][0]["message"]
                    result["fallback_checks"] = output_checks(
                        case, {"text": message.get("content") or ""}
                    )
                except Exception as fallback_exc:
                    result["fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
        result["wall_seconds"] = time.perf_counter() - start
        response = result.get("response", {})
        completion_tokens = response.get("meta_info", {}).get("completion_tokens")
        if completion_tokens:
            result["end_to_end_tokens_per_second"] = completion_tokens / result["wall_seconds"]
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
        print(
            json.dumps(
                {
                    "name": result["name"],
                    "text": response.get("text"),
                    "checks": result.get("checks"),
                    "spec_metrics": result.get("spec_metrics"),
                    "baseline_comparison": result.get("baseline_comparison"),
                    "error": result.get("error"),
                    "wall_seconds": result["wall_seconds"],
                }
            ),
            flush=True,
        )
    by_name = {case["name"]: case for case in report["cases"]}
    report["color_counterfactual_passed"] = all(
        by_name[name].get("checks", {}).get("expected_answer", False)
        for name in ("red_square", "blue_square")
    )
    if args.mode == "mtp":
        report["total_verify_ct"] = sum(
            case.get("spec_metrics", {}).get("spec_verify_ct", 0) for case in report["cases"]
        )
        report["total_correct_drafts"] = sum(
            case.get("spec_metrics", {}).get("spec_num_correct_drafts", 0)
            for case in report["cases"]
        )
        report["total_proposed_drafts"] = sum(
            case.get("spec_metrics", {}).get("spec_num_proposed_drafts", 0)
            for case in report["cases"]
        )
        proposed = report["total_proposed_drafts"]
        report["aggregate_spec_accept_rate"] = (
            report["total_correct_drafts"] / proposed if proposed else None
        )
        report["greedy_outputs_identical"] = all(
            case.get("baseline_comparison", {}).get("exact_output_ids", False)
            for case in report["cases"]
        )
    report["functional_passed"] = all(
        not case.get("error") and all(case.get("checks", {}).values()) for case in report["cases"]
    )
    report["validation_passed"] = report["functional_passed"]
    if args.mode == "mtp":
        report["validation_passed"] = (
            report["validation_passed"]
            and report["greedy_outputs_identical"]
            and report["total_correct_drafts"] > 0
        )
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
