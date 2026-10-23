"""Offline regression coverage for the multimodal SGLang checker payload."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_sglang_multimodal.py"
SPEC = importlib.util.spec_from_file_location("check_sglang_multimodal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class _FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "rendered prompt"


def _response(case: dict, *, mtp: bool) -> dict:
    expected = case.get("expected")
    return {
        "text": expected or "generated code",
        "output_ids": [17, 23],
        "meta_info": {
            "completion_tokens": 2,
            "output_token_logprobs": [],
            "spec_verify_ct": 1 if mtp else 0,
            "spec_num_correct_drafts": 1 if mtp else 0,
            "spec_num_proposed_drafts": 1 if mtp else 0,
        },
    }


def test_main_sends_native_output_id_request_and_validates_mtp_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        checker.AutoTokenizer,
        "from_pretrained",
        lambda model, local_files_only: _FakeTokenizer(),
    )

    requests: list[dict] = []
    expected_by_order = ["391", None, "red", "blue", "3729"]
    mtp = False

    def fake_post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
        assert base_url == "http://offline"
        assert path == "/generate"
        assert timeout == 7
        requests.append(payload)
        case = {"expected": expected_by_order[(len(requests) - 1) % len(expected_by_order)]}
        return _response(case, mtp=mtp)

    monkeypatch.setattr(checker, "post", fake_post)

    baseline_path = tmp_path / "baseline.json"
    baseline_args = [
        "check_sglang_multimodal.py",
        "--base-url",
        "http://offline",
        "--model",
        str(tmp_path / "model"),
        "--mode",
        "baseline",
        "--output",
        str(baseline_path),
        "--timeout",
        "7",
    ]
    monkeypatch.setattr("sys.argv", baseline_args)
    assert checker.main() == 0
    baseline = json.loads(baseline_path.read_text())
    assert len(baseline["cases"]) == len(checker.TEXT_CASES) + 3
    assert all(payload["return_output_ids"] is True for payload in requests)
    assert all("output_ids" in case["response"] for case in baseline["cases"])

    requests.clear()
    mtp = True
    mtp_path = tmp_path / "mtp.json"
    mtp_args = [
        "check_sglang_multimodal.py",
        "--base-url",
        "http://offline",
        "--model",
        str(tmp_path / "model"),
        "--mode",
        "mtp",
        "--output",
        str(mtp_path),
        "--baseline",
        str(baseline_path),
        "--timeout",
        "7",
    ]
    monkeypatch.setattr("sys.argv", mtp_args)
    assert checker.main() == 0
    report = json.loads(mtp_path.read_text())
    assert report["validation_passed"] is True
    assert report["greedy_outputs_identical"] is True
    assert report["total_verify_ct"] == len(report["cases"])
    assert report["total_correct_drafts"] == len(report["cases"])
    assert all(
        case["baseline_comparison"]["exact_output_ids"]
        and case["checks"]["mtp_executed"]
        for case in report["cases"]
    )
    assert all(payload["return_output_ids"] is True for payload in requests)
