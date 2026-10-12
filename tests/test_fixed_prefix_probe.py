"""Safety and score semantics for the controlled native SGLang prefix probe."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fixed_prefix.py"
SPEC = importlib.util.spec_from_file_location("probe_fixed_prefix", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def bundle() -> dict:
    return {
        "schema_version": 1, "context_length": 64, "tokenization": "retokenized_journal",
        "tokenizer_sha256": "a" * 64, "source_journal_sha256": ["b" * 64],
        "cases": [{
            "case": 1, "prefix": 1, "stage": "loop_onset", "input_ids": [10, 11],
            "candidate_token_ids": {"loop": [12], "thinking_end": [13], "eos": [14, 15]},
            "raw_private_text": "private prompt must never enter the report",
        }],
    }


def response() -> dict:
    return {
        "text": "private model output", "output_ids": [12],
        "meta_info": {
            "prompt_tokens": 2, "completion_tokens": 1, "finish_reason": {"type": "length"},
            "output_token_logprobs": [[-0.2, 12, None]],
            "output_top_logprobs": [[[-0.2, 12, None], [-0.3, 13, None]]],
            "output_token_ids_logprobs": [[
                [-0.2, 12, None], [-0.3, 13, None], [-4.0, 14, None], [-5.0, 15, None],
            ]],
        },
    }


def test_native_payload_avoids_text_and_preserves_greedy_candidates():
    item = bundle()["cases"][0]
    probe.validate_bundle(bundle(), 1)
    payload = probe.request_payload(item, 1, 32)
    assert "text" not in payload
    assert payload["input_ids"] == [10, 11]
    assert payload["token_ids_logprob"] == [12, 13, 14, 15]
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == -1
    assert payload["return_text_in_logprobs"] is False
    assert payload["sampling_params"]["temperature"] == 0
    assert payload["sampling_params"]["repetition_penalty"] == 1
    assert payload["sampling_params"]["no_stop_trim"] is True


@pytest.mark.parametrize("mutation", [
    lambda x: x["cases"][0].update(input_ids=[True]),
    lambda x: x["cases"].append(copy.deepcopy(x["cases"][0])),
    lambda x: x["cases"][0].update(stage="raw dataset question"),
    lambda x: x["cases"][0]["candidate_token_ids"].update(eos=[14, 14]),
    lambda x: x["cases"][0]["candidate_token_ids"].update(private_answer=[16]),
    lambda x: x.update(tokenizer_sha256="missing provenance"),
    lambda x: x.update(context_length=2),
])
def test_bundle_rejects_invalid_or_uncontrolled_inputs(mutation):
    data = bundle()
    mutation(data)
    with pytest.raises(probe.DiagnosticError):
        probe.validate_bundle(data, 1)


def test_report_score_margin_and_no_private_content():
    row = probe.summarize_response(bundle()["cases"][0], response(), 1)
    assert row["loop_minus_thinking_end_logprob"] == pytest.approx(0.1)
    assert row["top1_minus_top2_logprob"] == pytest.approx(0.1)
    assert row["candidates"]["eos"]["best_single_token_logprob"] == -4
    assert row["candidates"]["eos"]["selected_minus_best_logprob"] == pytest.approx(3.8)
    assert row["candidates"]["thinking_end"]["best_top_rank"] == 2
    rendered = json.dumps(row)
    assert "private" not in rendered
    assert "output_ids\"" not in rendered
    assert "input_ids\"" not in rendered
    assert "token_id\"" not in rendered


@pytest.mark.parametrize("mutation", [
    lambda x: x["meta_info"].update(prompt_tokens=1),
    lambda x: x["meta_info"].update(spec_verify_ct=1),
    lambda x: x["meta_info"].update(output_top_logprobs=[[[-0.1, 13, None], [-0.2, 12, None]]]),
    lambda x: x["meta_info"]["output_token_ids_logprobs"][0].pop(),
    lambda x: x["meta_info"]["output_token_ids_logprobs"][0][1].__setitem__(0, float("nan")),
    lambda x: x.update(output_ids=[]),
    lambda x: x.update(output_ids=[13]),
])
def test_response_rejects_truncation_speculation_and_inconsistent_scores(mutation):
    data = response()
    mutation(data)
    with pytest.raises(probe.DiagnosticError):
        probe.summarize_response(bundle()["cases"][0], data, 1)


def test_eos_probe_counts_generated_stopping_id_even_if_text_empty():
    data = response()
    data["text"] = ""
    data["output_ids"] = [14]
    data["meta_info"].update(
        output_token_logprobs=[[-0.1, 14, None]],
        output_top_logprobs=[[[-0.1, 14, None], [-0.2, 12, None]]],
        output_token_ids_logprobs=[[
            [-0.2, 12, None], [-0.3, 13, None], [-0.1, 14, None], [-5, 15, None],
        ]],
        finish_reason={"type": "stop", "matched": 14},
    )
    row = probe.summarize_response(bundle()["cases"][0], data, 1)
    assert row["rollout"]["eos_token_occurrences"] == 1
    assert row["rollout"]["finish_type"] == "stop"


def terminal_response(output_ids: list[int], finish: dict) -> dict:
    """Aligned native IDs/logprobs so failures isolate terminal semantics."""
    data = response()
    data["output_ids"] = output_ids
    meta = data["meta_info"]
    meta["completion_tokens"] = len(output_ids)
    meta["finish_reason"] = finish
    meta["output_token_logprobs"] = [[-0.2, token, None] for token in output_ids]
    meta["output_top_logprobs"] = [
        [[-0.2, token, None], [-0.3, 13, None]] for token in output_ids
    ]
    meta["output_token_ids_logprobs"] = [
        [[-0.2 if candidate == token else -0.3 if candidate == 13 else -5.0,
          candidate, None] for candidate in [12, 13, 14, 15]]
        for token in output_ids
    ]
    return data


@pytest.mark.parametrize("matched", ["NaN happened", "stop", "14", True, [14], None, 99])
def test_stop_rejects_non_integer_or_undeclared_eos_even_when_final_id_is_eos(matched):
    data = terminal_response([14], {"type": "stop", "matched": matched})
    with pytest.raises(probe.DiagnosticError, match="integer declared EOS"):
        probe.summarize_response(bundle()["cases"][0], data, 8)


@pytest.mark.parametrize("output_ids,matched", [([12], 14), ([14], 15)])
def test_stop_requires_matched_eos_preserved_as_final_native_id(output_ids, matched):
    data = terminal_response(output_ids, {"type": "stop", "matched": matched})
    with pytest.raises(probe.DiagnosticError, match="preserved as the last"):
        probe.summarize_response(bundle()["cases"][0], data, 8)


def test_stop_rejects_output_continuing_after_an_earlier_eos():
    data = terminal_response([14, 12, 15], {"type": "stop", "matched": 15})
    with pytest.raises(probe.DiagnosticError, match="continued after an EOS"):
        probe.summarize_response(bundle()["cases"][0], data, 8)


@pytest.mark.parametrize("eos", [14, 15])
def test_genuine_eos_can_stop_before_budget_without_changing_report_schema(eos):
    data = terminal_response([12, eos], {"type": "stop", "matched": eos})
    row = probe.summarize_response(bundle()["cases"][0], data, 8)
    assert row["rollout"]["completion_tokens"] == 2
    assert row["rollout"]["finish_type"] == "stop"
    assert row["rollout"]["eos_token_occurrences"] == 1
    assert set(row["rollout"]) == {
        "completion_tokens", "finish_type", "output_ids_sha256",
        "thinking_end_token_occurrences", "eos_token_occurrences", "repetition",
    }


def test_length_finish_must_reach_requested_budget():
    with pytest.raises(probe.DiagnosticError, match="before the diagnostic token budget"):
        probe.summarize_response(bundle()["cases"][0], response(), 8)


@pytest.mark.parametrize("finish", [{"type": "length"}, {"type": "length", "length": 2}])
def test_length_finish_accepts_legacy_and_native_metadata_at_budget(finish):
    data = terminal_response([12, 12], finish)
    row = probe.summarize_response(bundle()["cases"][0], data, 2)
    assert row["rollout"]["completion_tokens"] == 2
    assert row["rollout"]["finish_type"] == "length"


def test_length_finish_rejects_generated_eos():
    data = terminal_response([12, 14], {"type": "length", "length": 2})
    with pytest.raises(probe.DiagnosticError, match="cannot contain an EOS"):
        probe.summarize_response(bundle()["cases"][0], data, 2)


@pytest.mark.parametrize("length", [1, True, None])
def test_length_metadata_must_match_budget_when_present(length):
    data = terminal_response([12, 12], {"type": "length", "length": length})
    with pytest.raises(probe.DiagnosticError, match="length finish differs"):
        probe.summarize_response(bundle()["cases"][0], data, 2)


def test_aborted_request_cannot_be_reported_complete(tmp_path, monkeypatch):
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle()))
    private, public = tmp_path / "private.json", tmp_path / "report.json"

    def fake_http(_url, path, _timeout, payload=None):
        if path == "/get_server_info":
            return server()
        data = response()
        # An abort may still include valid token IDs and logprobs from partial decode.
        data["meta_info"]["finish_reason"] = {"type": "abort", "message": "private detail"}
        return data

    monkeypatch.setattr(probe, "http_json", fake_http)
    result = probe.main(["--bundle", str(source), "--model-label", "mse",
                         "--private-output", str(private), "--report-output", str(public)])
    report_data = json.loads(public.read_text())
    assert result == 1
    assert report_data["complete"] is False
    assert report_data["cases"][0]["error_type"] == "DiagnosticError"
    assert "selected_token_sha256" not in report_data["cases"][0]
    assert "private detail" not in public.read_text()
    assert "private detail" in private.read_text()


def test_repetition_stats_count_exact_period_in_tokens():
    stats = probe.repetition_stats([99, 98] + [1, 2, 3] * 40)
    assert stats["suffix_shortest_period_tokens"] == 3
    assert stats["suffix_full_cycles"] == 40
    assert stats["suffix_periodic_tokens"] == 120
    assert stats["longest_same_token_run"] == 1
    assert probe.repetition_stats(list(range(120)))["suffix_shortest_period_tokens"] is None
    assert probe.repetition_stats([1] * 40)["longest_same_token_run"] == 40


def server() -> dict:
    return {
        "context_length": 64, "speculative_algorithm": None, "allow_auto_truncate": False,
        "mamba_ssm_dtype": "float32", "model_path": "/private/checkpoint",
        "version": "0.5.20", "attention_backend": "flashinfer", "api_key": "private-key",
    }


def test_server_snapshot_excludes_paths_and_secrets():
    settings = probe.server_snapshot(server(), 64)
    assert "model_path" not in settings and "api_key" not in settings
    assert "private" not in json.dumps(settings)
    with pytest.raises(probe.DiagnosticError):
        probe.server_snapshot(server(), 65)
    for key, value in [("speculative_algorithm", "EAGLE"), ("allow_auto_truncate", True),
                       ("mamba_ssm_dtype", "bfloat16")]:
        info = server()
        info[key] = value
        with pytest.raises(probe.DiagnosticError):
            probe.server_snapshot(info, 64)


def report() -> dict:
    return {
        "schema_version": 1, "model_label": "mse", "source": {"bundle_sha256": "a" * 64},
        "protocol": {"temperature": 0}, "server_settings": {"mamba_ssm_dtype": "float32"},
        "cases": [probe.summarize_response(bundle()["cases"][0], response(), 1)],
    }


def test_comparison_rejects_changed_prefix_candidates_and_protocol():
    old = report()
    new = copy.deepcopy(old)
    new["model_label"] = "lm_head_bf16"
    new["cases"][0]["selected_token_sha256"] = "f" * 64
    new["cases"][0]["loop_minus_thinking_end_logprob"] = -0.1
    comparison = probe.compare_reports(new, old)
    row = comparison["cases"][0]
    assert row["selected_next_token_changed"] is True
    assert row["loop_minus_thinking_end_logprob_change"] == pytest.approx(-0.2)
    for mutation in [
        lambda x: x["source"].update(bundle_sha256="b" * 64),
        lambda x: x["protocol"].update(temperature=1),
        lambda x: x["cases"][0].update(input_ids_sha256="c" * 64),
        lambda x: x["cases"][0]["candidates"]["loop"].update(token_ids_sha256="d" * 64),
    ]:
        changed = copy.deepcopy(new)
        mutation(changed)
        with pytest.raises(probe.DiagnosticError):
            probe.compare_reports(changed, old)


def test_main_preserves_existing_output(tmp_path):
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle()))
    private = tmp_path / "private.json"
    private.write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        probe.main(["--bundle", str(source), "--model-label", "mse",
                    "--private-output", str(private), "--report-output", str(tmp_path / "report")])
    assert private.read_text() == "do not overwrite"


def test_main_report_cannot_leak_http_error_body(tmp_path, monkeypatch):
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle()))
    private, public = tmp_path / "private.json", tmp_path / "report.json"

    def fake_http(_url, path, _timeout, payload=None):
        if path == "/get_server_info":
            return server()
        raise RuntimeError("private trace and endpoint credentials")

    monkeypatch.setattr(probe, "http_json", fake_http)
    result = probe.main(["--bundle", str(source), "--model-label", "mse",
                         "--private-output", str(private), "--report-output", str(public)])
    assert result == 1
    assert "private trace" in private.read_text()
    assert "private trace" not in public.read_text()
    assert json.loads(public.read_text())["complete"] is False
    assert private.stat().st_mode & 0o777 == 0o600
