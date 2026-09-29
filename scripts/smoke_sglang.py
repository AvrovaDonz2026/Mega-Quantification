#!/usr/bin/env python3
"""Send text and image requests to a running SGLang server; save full evidence."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path


def request_json(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")
    models = request_json(f"{base}/v1/models")
    model = models["data"][0]["id"]

    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (224, 224), (255, 0, 0)).save(png, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    cases = [
        ("text", "What is 2 + 2? Answer with just the number.", "4"),
        (
            "vision",
            [
                {"type": "text", "text": "What color is this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            "red",
        ),
    ]
    report = {"base_url": base, "model": model, "cases": []}
    failed = False
    for name, content, expected in cases:
        started = time.monotonic()
        try:
            response = request_json(
                f"{base}/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            answer = response["choices"][0]["message"].get("content") or ""
            passed = expected in answer.lower()
            result = {"case": name, "passed": passed, "expected": expected, "response": response}
        except Exception as exc:
            passed = False
            result = {"case": name, "passed": False, "error": str(exc)}
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["cases"].append(result)
        failed |= not passed
    # Preserve speculative decoding counters when supported. HTTP success alone
    # does not prove MTP was used or that its acceptance rate is useful.
    try:
        report["server_info"] = request_json(f"{base}/get_server_info")
    except Exception as exc:
        report["server_info_error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": not failed}))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
