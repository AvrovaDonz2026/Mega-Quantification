"""Offline regression tests for the stdlib-only MLP pilot helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

CONTROLLER = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnostics/gdn_parity_20261019/mlp-pilot/controller.py"
)
SPEC = importlib.util.spec_from_file_location("mlp_pilot_controller", CONTROLLER)
assert SPEC is not None and SPEC.loader is not None
CONTROLLER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLLER_MODULE)


def test_known_period_stats_matches_historical_long_suffix_scalars() -> None:
    lag = 2032
    onset = 33131
    budget = 49152
    prefix = [1_000_000 + index for index in range(onset)]
    period = [2_000_000 + index for index in range(lag)]
    tokens = prefix + [period[index % lag] for index in range(budget - onset)]

    stats = CONTROLLER_MODULE.known_period_stats(tokens, lag=lag)

    assert stats == {
        "periodic_lag": 2032,
        "periodic_onset": 33131,
        "periodic_count": 13989,
        "periodic_extent": 16021,
        "periodic_complete_periods": 7,
        "periodic_confirmed": True,
    }


@pytest.mark.parametrize(
    ("length", "expected_periods", "confirmed"),
    [(2 * 5, 2, True), (2 * 5 - 1, 1, False)],
)
def test_known_period_stats_two_cycle_confirmation_boundary(
    length: int, expected_periods: int, confirmed: bool
) -> None:
    lag = 5
    period = [10, 20, 30, 40, 50]
    tokens = [period[index % lag] for index in range(length)]

    stats = CONTROLLER_MODULE.known_period_stats(tokens, lag=lag)

    assert stats["periodic_onset"] == 0
    assert stats["periodic_count"] == length - lag
    assert stats["periodic_extent"] == length
    assert stats["periodic_complete_periods"] == expected_periods
    assert stats["periodic_confirmed"] is confirmed


def test_known_period_stats_keeps_longest_run_after_a_break() -> None:
    lag = 3
    first_period = [100, 101, 102]
    second_period = [200, 201, 202]
    tokens = (
        [10, 11, 12]
        + [first_period[index % lag] for index in range(3 * lag)]
        + [999, 998, 997]
        + [second_period[index % lag] for index in range(2 * lag + 1)]
    )

    stats = CONTROLLER_MODULE.known_period_stats(tokens, lag=lag)

    assert stats["periodic_onset"] == 3
    assert stats["periodic_count"] == 2 * lag
    assert stats["periodic_extent"] == 3 * lag
    assert stats["periodic_complete_periods"] == 3
    assert stats["periodic_confirmed"] is True


def test_known_period_stats_only_checks_requested_lag() -> None:
    tokens = [index % 2 for index in range(100)]

    stats = CONTROLLER_MODULE.known_period_stats(tokens, lag=3)

    assert stats["periodic_lag"] == 3
    assert stats["periodic_count"] == 0
    assert stats["periodic_onset"] is None
    assert stats["periodic_extent"] == 0
    assert stats["periodic_confirmed"] is False


def test_termination_stats_natural_stop_requires_thinking_end_before_terminal_eos() -> None:
    out = [11, 22, 77, 33, 99]
    stats = CONTROLLER_MODULE.termination_stats(
        out, {"finish_reason": {"type": "stop"}}, {99}, {77}, budget=8
    )

    assert stats == {
        "finish_reason_type": "stop",
        "contains_eos": True,
        "contains_thinking_end": True,
        "thinking_end_count": 1,
        "terminal_eos": True,
        "first_thinking_end_index": 2,
        "natural_termination": True,
        "finish_reason_matched": {
            "match_kind": "none",
            "matched_is_eos": False,
            "matched_equals_terminal": False,
        },
    }


def test_termination_stats_length_requires_exact_budget() -> None:
    out = [1, 2, 3, 4]
    assert (
        CONTROLLER_MODULE.termination_stats(
            out, {"finish_reason": {"type": "length"}}, {99}, {77}, budget=4
        )["finish_reason_type"]
        == "length"
    )
    assert (
        CONTROLLER_MODULE.termination_stats(
            out,
            {"finish_reason": {"type": "length", "length": 4}},
            {99},
            {77},
            budget=4,
        )["finish_reason_type"]
        == "length"
    )

    with pytest.raises(RuntimeError, match="length finish"):
        CONTROLLER_MODULE.termination_stats(
            out[:-1], {"finish_reason": {"type": "length"}}, {99}, {77}, budget=4
        )
    with pytest.raises(RuntimeError, match="length finish"):
        CONTROLLER_MODULE.termination_stats(
            out,
            {"finish_reason": {"type": "length", "length": 3}},
            {99},
            {77},
            budget=4,
        )


@pytest.mark.parametrize(
    "meta",
    [
        {"finish_reason": {"type": "abort"}},
        {"finish_reason": "stop"},
        {},
    ],
)
def test_termination_stats_rejects_invalid_finish_reason(meta: dict) -> None:
    with pytest.raises(RuntimeError, match="invalid finish reason"):
        CONTROLLER_MODULE.termination_stats([1], meta, {99}, {77}, budget=1)


def test_termination_stats_distinguishes_contained_eos_from_terminal_eos() -> None:
    out = [99, 77, 12]
    stats = CONTROLLER_MODULE.termination_stats(
        out, {"finish_reason": {"type": "stop"}}, {99}, {77}, budget=8
    )

    assert stats["contains_eos"] is True
    assert stats["terminal_eos"] is False
    assert stats["contains_thinking_end"] is True
    assert stats["first_thinking_end_index"] == 1
    assert stats["natural_termination"] is False


def _write_model_metadata(
    tmp_path: Path,
    *,
    config: dict,
    generation: dict | None,
    tokenizer_config: dict,
    added_tokens: list[dict],
) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (model / "generation_config.json").write_text(json.dumps(generation))
    (model / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (model / "tokenizer.json").write_text(json.dumps({"added_tokens": added_tokens}))
    return model


def test_load_eos_metadata_unions_generation_and_tokenizer_sources(tmp_path: Path) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"text_config": {"eos_token_id": 7}},
        generation={"eos_token_id": [8, 9]},
        tokenizer_config={"eos_token": {"content": "<|end|>"}, "additional_stop_token_ids": [99]},
        added_tokens=[{"id": 10, "content": "<|end|>"}, {"id": 11, "content": "<|eom_id|>"}],
    )

    eos, proof = CONTROLLER_MODULE.load_eos_metadata(model)

    assert eos == {7, 8, 9, 10, 11}
    assert [row["source"] for row in proof["sources"]] == [
        "generation_config",
        "config",
        "tokenizer",
        "tokenizer_additional_stop",
    ]
    assert all(set(row) == {"source", "count", "sha256", "present"} for row in proof["sources"])
    rendered_proof = json.dumps(proof)
    assert "<|end|>" not in rendered_proof
    assert all(value not in proof.values() for value in ({7, 8, 9, 10, 11}, "<|end|>"))


def test_load_eos_metadata_falls_back_to_config_when_generation_missing(tmp_path: Path) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"text_config": {"eos_token_id": [7, 8]}},
        generation=None,
        tokenizer_config={"additional_stop_token_ids": [True]},
        added_tokens=[],
    )

    eos, proof = CONTROLLER_MODULE.load_eos_metadata(model)

    assert eos == {7, 8}
    assert [row["source"] for row in proof["sources"]] == [
        "generation_config",
        "config",
        "tokenizer",
        "tokenizer_additional_stop",
    ]
    assert proof["sources"][0]["present"] is False


def test_load_eos_metadata_reads_additional_stops_without_tokenizer_eos(tmp_path: Path) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"text_config": {"eos_token_id": 7}},
        generation=None,
        tokenizer_config={},
        added_tokens=[{"id": 11, "content": "<|eom_id|>"}],
    )

    eos, proof = CONTROLLER_MODULE.load_eos_metadata(model)

    assert eos == {7, 11}
    assert proof["sources"][-1]["count"] == 1


def test_load_eos_metadata_prefers_root_config_eos_over_text_config(tmp_path: Path) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"eos_token_id": 7, "text_config": {"eos_token_id": 8}},
        generation=None,
        tokenizer_config={},
        added_tokens=[],
    )

    eos, _ = CONTROLLER_MODULE.load_eos_metadata(model)

    assert eos == {7}


def test_load_eos_metadata_uses_root_config_eos_when_text_config_has_none(tmp_path: Path) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"eos_token_id": 7, "text_config": {}},
        generation=None,
        tokenizer_config={},
        added_tokens=[],
    )

    eos, _ = CONTROLLER_MODULE.load_eos_metadata(model)

    assert eos == {7}


@pytest.mark.parametrize("bad", [[-1], [True], ["9"], {"bad": 1}])
def test_load_eos_metadata_rejects_malformed_ids(tmp_path: Path, bad) -> None:
    model = _write_model_metadata(
        tmp_path,
        config={"text_config": {"eos_token_id": 7}},
        generation={"eos_token_id": bad},
        tokenizer_config={},
        added_tokens=[],
    )

    with pytest.raises(RuntimeError, match="malformed generation_config EOS metadata"):
        CONTROLLER_MODULE.load_eos_metadata(model)


def test_termination_stats_finish_match_summary_does_not_expose_raw_token() -> None:
    stats = CONTROLLER_MODULE.termination_stats(
        [11, 22, 99], {"finish_reason": {"type": "stop", "matched": 99}}, {99}, {22}, budget=8
    )

    assert stats["finish_reason_matched"] == {
        "match_kind": "id",
        "matched_is_eos": True,
        "matched_equals_terminal": True,
    }
    assert 99 not in stats["finish_reason_matched"].values()


def test_termination_stats_requires_explicit_match_to_equal_terminal() -> None:
    matched = CONTROLLER_MODULE.termination_stats(
        [77, 11, 10], {"finish_reason": {"type": "stop", "matched": 10}}, {10}, {77}, budget=8
    )
    assert matched["natural_termination"] is True
    stats = CONTROLLER_MODULE.termination_stats(
        [77, 11, 10], {"finish_reason": {"type": "stop", "matched": 11}}, {10}, {77}, budget=8
    )
    assert stats["natural_termination"] is False
