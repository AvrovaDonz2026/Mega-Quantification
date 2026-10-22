"""Collect one local SGLang text response and attach sanitized replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "runner.py"
MAX_TIMEOUT = 1800.0
MAX_NEW_TOKENS = 262144


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, new_url):
        return None


def _runner():
    spec = importlib.util.spec_from_file_location("r6_native_response_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the paired evidence runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loopback_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("SGLang base URL must use HTTP on a loopback host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("SGLang base URL must not contain credentials, query, or fragment")
    return base_url.rstrip("/")


def _request_json(
    url: str, *, timeout: float, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        # Ignore ambient proxy settings and never follow a redirect away from loopback.
        opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("local SGLang request failed") from exc
    if not isinstance(value, dict):
        raise RuntimeError("local SGLang response was not a JSON object")
    return value


def _evidence_identity(evidence_path: Path) -> tuple[dict[str, Any], Path, str, str]:
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("could not read arm evidence") from exc
    if not isinstance(evidence, dict) or evidence.get("execution") != "complete":
        raise ValueError("native response requires completed arm evidence")
    if evidence.get("arm") not in {"static-max", "static-MSE"}:
        raise ValueError("arm evidence has an unknown arm")
    export = evidence.get("export")
    if (
        not isinstance(export, dict)
        or not isinstance(export.get("path"), str)
        or not isinstance(export.get("sha256"), str)
    ):
        raise ValueError("arm evidence does not contain an exported path and fingerprint")
    if (
        not isinstance(evidence.get("export_index_verification"), dict)
        or evidence["export_index_verification"].get("keys_validated") is not True
    ):
        raise ValueError("arm evidence does not contain validated export keys")
    path = Path(export["path"]).expanduser().resolve()
    if not path.exists():
        raise ValueError("the exported arm directory is missing")
    tree = _runner()._tree_hash(path)
    if tree.get("sha256") != export["sha256"]:
        raise ValueError("the exported arm files no longer match the evidence fingerprint")
    return evidence, path, str(export["sha256"]), str(evidence["arm"])


def _load_tokenizer(export_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to tokenize the local prompt") from exc
    return AutoTokenizer.from_pretrained(
        str(export_path), trust_remote_code=True, local_files_only=True
    )


def _attach_in_memory(
    evidence: dict[str, Any], record: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    runner = _runner()
    bundle = {
        "arm": evidence["arm"],
        "export_sha256": evidence["export"]["sha256"],
        "text_only_protocol": protocol,
    }
    records = [record]
    evidence["text_only_protocol"] = runner._text_only_protocol(bundle, records)
    evidence["response_observation"] = runner._response_observation(records)
    evidence["trajectory_metrics"] = runner._trajectory_metrics(records)
    return evidence


def collect(
    *,
    evidence_path: Path,
    base_url: str,
    prompt: str,
    timeout: float = 300.0,
    max_new_tokens: int = 256,
    enable_thinking: bool = True,
    output_path: Path | None = None,
    tokenizer: Any | None = None,
    request_json=_request_json,
) -> Path:
    if not 0 < timeout <= MAX_TIMEOUT:
        raise ValueError(f"timeout must be in (0, {MAX_TIMEOUT:g}]")
    if not 0 < max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be in [1, {MAX_NEW_TOKENS}]")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")

    base = _loopback_url(base_url)
    evidence, export_path, export_sha256, arm = _evidence_identity(evidence_path.resolve())
    tokenizer = tokenizer or _load_tokenizer(export_path)
    expected_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if hasattr(expected_ids, "tolist"):
        expected_ids = expected_ids.tolist()
    if expected_ids and isinstance(expected_ids[0], list):
        expected_ids = expected_ids[0]
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError("tokenizer produced no prompt token IDs")

    # SGLang reports model_path as server configuration; this does not prove
    # which weight bytes were actually loaded into the serving process.
    info = request_json(base + "/get_model_info", timeout=timeout)
    model_path = info.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).expanduser().resolve() != export_path:
        raise ValueError("live SGLang model_path does not match the evidence export path")

    protocol = {
        "text_only": True,
        "modalities": ["text"],
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
    }
    response = request_json(
        base + "/generate",
        timeout=timeout,
        payload={
            "input_ids": expected_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
            "stream": False,
            "return_prompt_token_ids": True,
        },
    )
    record = response.get("response") if isinstance(response.get("response"), dict) else response
    # Require native prompt IDs to match the locally tokenized request exactly.
    prompt_ids = record.get("prompt_token_ids", record.get("prompt_ids"))
    if prompt_ids != expected_ids:
        raise ValueError("SGLang prompt token IDs do not match local tokenizer output")
    if not any(key in record for key in ("output_ids", "completion_ids", "token_ids")):
        raise ValueError("SGLang response is missing native output token IDs")
    attached = _attach_in_memory(evidence, response, protocol)
    # Bind protocol hash to a stable canonical encoding using the runner's hash.
    attached["response_protocol_sha256"] = _runner()._digest_json(protocol)
    attached["response_thinking_enabled"] = enable_thinking
    attached["response_server_model_path_sha256"] = hashlib.sha256(
        str(export_path).encode("utf-8")
    ).hexdigest()
    attached["rendered_prompt_ids_sha256"] = _runner()._ids_sha256(expected_ids)
    attached["response_loaded_weights_verified"] = False
    target = (output_path or evidence_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(attached, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-file", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help=(
            "default 256 for a short smoke request; up to 262144 for extended repetition "
            "diagnostics"
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="disable model thinking for a short smoke request",
    )
    args = parser.parse_args(argv)
    prompt = (
        args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    )
    result = collect(
        evidence_path=args.evidence_file,
        base_url=args.base_url,
        prompt=prompt,
        timeout=args.timeout,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=not args.disable_thinking,
        output_path=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__, file=__import__("sys").stderr)
        raise SystemExit(1) from None
