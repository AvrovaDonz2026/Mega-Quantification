"""Offline tests for the privacy-preserving R6 response evidence producer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r6_runner_responses", HERE / "runner.py")
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _evidence(tmp_path: Path) -> Path:
    path = tmp_path / "static_max-plan.json"
    path.write_text(
        json.dumps(
            {
                "arm": "static-max",
                "execution": "complete",
                "export": {"sha256": "export-sha"},
                "export_index_verification": {"keys_validated": True},
            }
        )
    )
    return path


def _bundle(tmp_path: Path, *, arm: str = "static-max", export_sha256: str = "export-sha") -> Path:
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps(
            {
                "arm": arm,
                "export_sha256": export_sha256,
                "text_only_protocol": {
                    "text_only": True,
                    "modalities": ["text"],
                    "temperature": 0,
                    "top_p": 0.95,
                    "max_new_tokens": 32,
                },
                "responses": [
                    {
                        "prompt_token_ids": [101, 102],
                        "output_ids": [7, 8, 7, 8, 7, 8, 7, 8, 7, 8],
                        "text": "must not be persisted",
                    }
                ],
            }
        )
    )
    return path


def test_attach_response_evidence_is_hash_only_and_detects_cycle(tmp_path):
    evidence_path = _evidence(tmp_path)
    runner.attach_response_evidence(evidence_path, _bundle(tmp_path))

    evidence = json.loads(evidence_path.read_text())
    assert evidence["text_only_protocol"]["schema"] == "r6-text-only-v1"
    assert evidence["response_observation"]["prompt_ids_sha256"]
    assert evidence["trajectory_metrics"]["cycle"]["found"] is True
    assert "must not be persisted" not in evidence_path.read_text()
    assert "output_ids" not in evidence


def test_attach_response_evidence_requires_export_binding(tmp_path):
    with pytest.raises(runner.RunnerError, match="not bound"):
        runner.attach_response_evidence(
            _evidence(tmp_path), _bundle(tmp_path, export_sha256="wrong-export")
        )


def test_attach_response_evidence_rejects_decoded_text_without_native_ids(tmp_path):
    bundle = json.loads(_bundle(tmp_path).read_text())
    bundle["responses"][0].pop("output_ids")
    bundle["responses"][0]["text"] = "decoded output only"
    path = tmp_path / "text-only.json"
    path.write_text(json.dumps(bundle))

    with pytest.raises(runner.RunnerError, match="token IDs"):
        runner.attach_response_evidence(_evidence(tmp_path), path)


def test_attach_response_evidence_rejects_nonzero_temperature(tmp_path):
    bundle = json.loads(_bundle(tmp_path).read_text())
    bundle["text_only_protocol"]["temperature"] = 1
    path = tmp_path / "sampled.json"
    path.write_text(json.dumps(bundle))

    with pytest.raises(runner.RunnerError, match="temperature=0"):
        runner.attach_response_evidence(_evidence(tmp_path), path)


def test_trajectory_metrics_records_divergence_without_token_payload(tmp_path):
    bundle = json.loads(_bundle(tmp_path).read_text())
    bundle["responses"].append({"prompt_token_ids": [101, 102], "output_ids": [7, 8, 9]})
    path = tmp_path / "two-responses.json"
    path.write_text(json.dumps(bundle))
    runner.attach_response_evidence(_evidence(tmp_path), path)

    # The helper itself returns only aggregate metrics and no ID sequence.
    loaded = json.loads(path.read_text())
    result = runner._trajectory_metrics(loaded["responses"])
    assert result["first_divergence"]["index"] == 2
    assert "output_ids" not in result
