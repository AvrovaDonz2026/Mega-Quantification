"""Offline regression tests for the stdlib-only MLP pilot helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
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


def test_period_stats_across_boundary_requires_generated_matches() -> None:
    lag = 4
    prefix = [index % lag for index in range(3 * lag)]
    unmatched = [100 + index for index in range(10)]
    stats = CONTROLLER_MODULE.period_stats_across_boundary(prefix, unmatched, lag)
    assert stats["continuation_match_count"] == 0
    assert stats["confirmed_across_boundary"] is False

    one_match = unmatched.copy()
    one_match[0] = prefix[-lag]
    stats = CONTROLLER_MODULE.period_stats_across_boundary(prefix, one_match, lag)
    assert stats["continuation_match_count"] == 1
    assert stats["confirmed_across_boundary"] is False


def test_period_stats_across_boundary_requires_two_generated_periods() -> None:
    lag = 4
    prefix = [index % lag for index in range(20 * lag)]
    short = [index % lag for index in range(2 * lag - 1)]
    stats = CONTROLLER_MODULE.period_stats_across_boundary(prefix, short, lag)
    assert stats["continuation_matched_prefix_tokens"] == 2 * lag - 1
    assert stats["confirmed_across_boundary"] is False

    long = [index % lag for index in range(2 * lag)]
    stats = CONTROLLER_MODULE.period_stats_across_boundary(prefix, long, lag)
    assert stats["continuation_matched_prefix_tokens"] == 2 * lag
    assert stats["confirmed_across_boundary"] is True
    assert stats["crosses_input_continuation_boundary"] is True


def test_period_stats_across_boundary_rejects_invalid_lag_or_prefix() -> None:
    with pytest.raises(ValueError, match="lag must be positive"):
        CONTROLLER_MODULE.period_stats_across_boundary([1, 2], [1], 0)
    with pytest.raises(ValueError, match="at least one lag"):
        CONTROLLER_MODULE.period_stats_across_boundary([1, 2], [1], 3)


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


def test_source_revision_and_paths_are_variant_specific(tmp_path: Path) -> None:
    lock = tmp_path / "source-lock.json"
    variants = {
        name: {"commit": f"{index:040x}", "archive_sha256": "a" * 64}
        for index, name in enumerate(("original", "decode-beta", "all-beta", "shared-decode"), 1)
    }
    lock.write_text(json.dumps({"variants": variants}))
    assert CONTROLLER_MODULE.source_revision_for_variant("original", lock) == "1".zfill(40)
    assert CONTROLLER_MODULE.source_revision_for_variant("all-beta", lock) == "3".zfill(40)
    paths = CONTROLLER_MODULE.source_paths(tmp_path, "original")
    assert paths == (
        tmp_path / "source-package/sglang-original",
        tmp_path / "source-package/sglang-original.tar.gz",
        tmp_path / "source-package/sglang-original.manifest.json",
    )


def test_source_variant_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unknown source variant"):
        CONTROLLER_MODULE.source_revision_for_variant("decode-beta", Path("missing"))


def test_single_arm_comparison_omits_cross_arm_fields() -> None:
    outputs = {"control": [[1, 2], [1, 2]]}
    comparison = CONTROLLER_MODULE.build_comparison(outputs, "control")
    assert comparison == {"both_repeat_stable": True, "selected_arm_count": 1}


def test_cli_rejects_unknown_variant_and_keeps_default_flags() -> None:
    parser = CONTROLLER_MODULE.build_parser()
    required = [
        "--root",
        "r",
        "--remote-root",
        "rr",
        "--export",
        "e",
        "--bf16",
        "b",
        "--control",
        "c",
        "--mlp",
        "m",
        "--ablation-tool",
        "a",
        "--native-private",
        "n",
    ]
    defaults = parser.parse_args(required)
    assert defaults.trajectory == "natural"
    assert defaults.source_variant == "all-beta"
    assert defaults.arms == "both"
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--source-variant", "decode-beta"])


def test_historical_prefix_trajectory_rejects_long_phase_and_out_of_range_budget() -> None:
    with pytest.raises(ValueError, match="requires --phase early"):
        CONTROLLER_MODULE.validate_trajectory_options("historical-prefix", "long", 49152)
    with pytest.raises(ValueError, match="between 512 and 4096"):
        CONTROLLER_MODULE.validate_trajectory_options("historical-prefix", "early", 8192)
    CONTROLLER_MODULE.validate_trajectory_options("historical-prefix", "early", 512)
    CONTROLLER_MODULE.validate_trajectory_options("historical-prefix", "early", 4096)


@pytest.mark.parametrize("arms", ["both", "control"])
@pytest.mark.parametrize("source_variant", ["original", "all-beta"])
@pytest.mark.parametrize("trajectory", ["natural", "historical-prefix"])
def test_main_fake_lifecycle_preserves_source_and_selected_arm_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arms: str,
    source_variant: str,
    trajectory: str,
) -> None:
    module = CONTROLLER_MODULE
    image = module.IMAGE
    prompt = list(range(288))
    reference = (
        [index % CONTROLLER_MODULE.KNOWN_PERIOD for index in range(49152)]
        if trajectory == "historical-prefix"
        else list(range(49152))
    )
    module.NATIVE_SHA = "native-sha"
    module.PROMPT_SHA = "prompt-sha"
    remote = tmp_path / "remote"
    (remote / "candidate-bundle-r1" / "parity-v7").mkdir(parents=True)
    native_path = tmp_path / "native.json"
    native_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": 3,
                        "request": {"input_ids": prompt},
                        "response": {"output_ids": reference},
                    }
                ]
            }
        )
    )
    for name in ("export", "bf16", "control", "mlp", "ablation-tool"):
        (tmp_path / name).mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 7, "content": "</think>"}]})
    )
    lock = tmp_path / "source-lock.json"
    commits = {
        "original": "1" * 40,
        "decode-beta": "2" * 40,
        "all-beta": "3" * 40,
        "shared-decode": "4" * 40,
    }
    lock.write_text(json.dumps({"variants": {k: {"commit": v} for k, v in commits.items()}}))
    source_dir = remote / "source-package" / f"sglang-{source_variant}"
    source_dir.mkdir(parents=True)
    source_archive = source_dir.with_suffix(".tar.gz")
    source_archive.write_bytes(b"archive")
    source_dir.with_suffix(".manifest.json").write_text("{}")

    calls = {
        "gates": [],
        "owned": [],
        "runtime": [],
        "cleanup": [],
        "models": [],
        "requests": [],
    }
    fake = types.ModuleType("run_v7")
    fake._LOCK_MODULE = types.SimpleNamespace(LOCK_PATH=lock)
    fake.now = lambda: "now"
    fake.sha256_file = lambda path: "native-sha" if Path(path) == native_path else "file-sha"
    fake.digest = lambda value: "prompt-sha" if value == prompt else "digest"
    fake.write_json = lambda path, value: Path(path).write_text(json.dumps(value))
    fake.source_gate = lambda a: (
        calls["gates"].append((a.source, a.source_revision)) or {"passed": True}
    )
    fake.server_command = lambda a, *args: ["server", "--context-length", "1", a.source_revision]
    fake.run = lambda command: types.SimpleNamespace(
        stdout=image if command[:2] == ["docker", "image"] else "container"
    )
    fake.gpu_idle = lambda: None
    fake.wait_health = lambda *args: {
        "host": "127.0.0.1",
        "port": 32262,
        "model_path": "/models/Qwen3.8-27B-NVFP4-MSE-" + args[0].job_id.rsplit("-", 1)[-1].upper(),
        "quantization": "modelopt_mixed",
        "fp4_gemm_runner_backend": "flashinfer_cutlass",
        "attention_backend": "flashinfer",
        "sampling_backend": "pytorch",
        "linear_attn_backend": "triton",
        "mamba_backend": "triton",
        "mamba_ssm_dtype": "float32",
        "kv_cache_dtype": "fp8_e4m3",
        "context_length": 65536,
        "tp_size": 1,
        "max_running_requests": 1,
        "enable_multimodal": True,
        "speculative_algorithm": None,
        "disable_radix_cache": True,
        "allow_auto_truncate": False,
        "random_seed": module.RANDOM_SEED,
        "cuda_graph_config": {
            "prefill": {"backend": "disabled"},
            "decode": {"backend": "disabled"},
        },
    }
    fake.owned = lambda *args: calls["owned"].append(args)
    fake.runtime_source_hashes = lambda a, server: (
        calls["runtime"].append((a.source, a.source_revision)) or {"passed": True}
    )
    fake.cleanup = lambda *args: calls["cleanup"].append(args) or {"removed": True}
    monkeypatch.setitem(sys.modules, "run_v7", fake)
    monkeypatch.setitem(
        sys.modules,
        "model_gate",
        types.SimpleNamespace(
            verify_model=lambda *args: calls["models"].append(args) or {"passed": True}
        ),
    )
    monkeypatch.setattr(module, "load_eos_metadata", lambda model: ({99}, {"sources": []}))
    monkeypatch.setattr(module, "termination_stats", lambda *args: {"natural_termination": True})
    monkeypatch.setattr(module, "lcp", lambda left, right: 0)

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def connect_ex(self, *args):
            return 1

    monkeypatch.setattr(module.socket, "socket", FakeSocket)

    model_info_calls = {"count": 0}

    class Response:
        status = 200

        def __init__(self, payload=None):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            if self.payload is None:
                return b"Cache flushed."
            return json.dumps(self.payload).encode()

        def __iter__(self):
            return iter(())

        def json(self):
            return self.payload

    def urlopen(request, timeout=0):
        url = request if isinstance(request, str) else request.full_url
        if "/model_info" in url:
            arm = "CONTROL" if model_info_calls["count"] == 0 else "MLP"
            model_info_calls["count"] += 1
            return Response({"model_path": f"/models/Qwen3.8-27B-NVFP4-MSE-{arm}"})
        if "/flush_cache" in url:
            return Response()
        calls["requests"].append(json.loads(request.data))
        request_prompt = prompt + reference[:41472] if trajectory == "historical-prefix" else prompt
        return Response(
            {
                "output_ids": [1, 2],
                "meta_info": {
                    "completion_tokens": 2,
                    "prompt_tokens": len(request_prompt),
                    "cached_tokens": 0,
                    "output_token_logprobs": [[-1.0], [-1.0]],
                    "finish_reason": {"type": "stop"},
                },
                "prompt_token_ids": request_prompt,
            }
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    argv = [
        "controller.py",
        "--root",
        str(tmp_path / "out"),
        "--remote-root",
        str(remote),
        "--export",
        str(tmp_path / "export"),
        "--bf16",
        str(tmp_path / "bf16"),
        "--control",
        str(model),
        "--mlp",
        str(model),
        "--ablation-tool",
        str(tmp_path / "ablation-tool"),
        "--native-private",
        str(native_path),
        "--source-variant",
        source_variant,
        "--arms",
        arms,
        "--trajectory",
        trajectory,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["trajectory"] == trajectory
    assert report["prompt_tokens"] == 288
    assert report["input_tokens"] == (41760 if trajectory == "historical-prefix" else 288)
    assert report["native_prefix_tokens"] == (41472 if trajectory == "historical-prefix" else 0)
    expected_input_hash = "digest" if trajectory == "historical-prefix" else "prompt-sha"
    assert report["input_ids_sha256"] == expected_input_hash
    if trajectory == "historical-prefix":
        assert report["fresh_prefill"] is True
        assert report["incremental_state_reconstructed"] is False
        assert all(
            request["input_ids"] == prompt + reference[:41472] for request in calls["requests"]
        )
        assert all(
            "continuation_reference_lcp" in trial and "period_2032_across_input_prefix" in trial
            for arm in report["arms"].values()
            for trial in arm["trials"]
        )
    else:
        assert "fresh_prefill" not in report
        assert all(request["input_ids"] == prompt for request in calls["requests"])
    assert report["selected_arms"] == (["control"] if arms == "control" else ["control", "mlp"])
    assert len(calls["gates"]) == len(report["selected_arms"])
    assert all(
        isinstance(path, Path) and revision == commits[source_variant]
        for path, revision in calls["gates"]
    )
    assert all(revision == commits[source_variant] for _, revision in calls["runtime"])
    assert all(args[-1] == commits[source_variant] for args in calls["owned"] + calls["cleanup"])
    assert len(calls["models"]) == len(report["selected_arms"])
    if arms == "control":
        assert all(args[3] == "control" for args in calls["models"])
        assert set(report["arms"]) == {"control"}
    else:
        assert {args[3] for args in calls["models"]} == {"control", "mlp"}
