import importlib.util
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts/diagnostics/gdn_parity_20261019/parity-v7/run_v7.py"
SPEC = importlib.util.spec_from_file_location("parity_v7_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER_MODULE)


def _summary(ids, *, finish="stop"):
    count = len(ids)
    rows = [[-0.1, token] for token in ids]
    top = [[[-0.1, token, None], [-0.2, token + 100, None]] for token in ids]
    return {
        "output_ids": ids,
        "prompt_ids": [7, 8],
        "output_len": count,
        "prompt_len": 2,
        "token_logprobs": rows,
        "top_logprobs": top,
        "finish_reason_type": finish,
        "scalars": {},
        "fixture_file_sha256": "a" * 64,
        "rendered_prompt_sha256": "b" * 64,
        "image_sha256": None,
        "code_ast": {"available": False},
    }


def test_compare_records_bounded_divergence_window_and_topk_without_text():
    left = _summary(list(range(10)))
    right = _summary(list(range(5)) + [99] + list(range(6, 10)))
    result = RUNNER_MODULE.compare(left, right)

    assert result["first_divergence_index"] == 5
    divergence = result["first_divergence"]
    assert divergence["baseline"]["token_window"] == {
        "start_index": 1,
        "end_index": 10,
        "before_ids": [1, 2, 3, 4],
        "at_id": 5,
        "after_ids": [6, 7, 8, 9],
    }
    assert divergence["mtp"]["token_window"]["at_id"] == 99
    assert divergence["baseline"]["top_k_logprobs"]["k"] == 2
    assert divergence["baseline"]["top_k_logprobs"]["entries"][0] == {
        "token_id": 5,
        "logprob": -0.1,
    }
    assert all(
        "text" not in item for item in divergence["baseline"]["top_k_logprobs"]["entries"]
    )


def test_compare_records_end_of_sequence_divergence():
    result = RUNNER_MODULE.compare(
        _summary([1, 2, 3]), _summary([1, 2, 3, 4], finish="length")
    )

    assert result["first_divergence_index"] == 3
    assert result["first_divergence"]["baseline"]["token_window"]["at_id"] is None
    assert result["first_divergence"]["mtp"]["token_window"]["at_id"] == 4
    assert result["first_divergence"]["baseline"]["top_k_logprobs"]["entries"] == []
