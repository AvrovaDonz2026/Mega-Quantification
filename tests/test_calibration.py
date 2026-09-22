"""Calibration helpers: Nemotron splits, text field, sized/re-iterable batches."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from megaquant.calibration import (
    CalibrationBatches,
    DummyCalibIter,
    _extract_text,
    _load_hf_dataset,
    _nemotron_splits,
    build_calibration_iter,
)
from megaquant.config import recipe_from_mapping

NEMOTRON_ROW = {
    "uuid": "x",
    "license": "cc",
    "generator": "test",
    "version": "1",
    "category": "chat",
    "reasoning": "off",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ],
}


class _FakeTokenizer:
    pad_token = None
    pad_token_id = None
    eos_token = "<|endoftext|>"
    eos_token_id = 1

    def __call__(self, texts, **kwargs):  # noqa: ANN001
        n = len(texts)
        return {
            "input_ids": [[1, 2, 3]] * n,
            "attention_mask": [[1, 1, 1]] * n,
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _w4a8_recipe(dataset: str, **calib: Any):
    payload = {
        "name": "calib-test",
        "backend": "modelopt",
        "scheme": "nvfp4_w4a8",
        "algorithm": "max",
        "kv_cache": "fp8",
        "family": "qwen3_5",
        "extra_ignore": [],
        "model": {"source": "Qwen/Qwen3.8-27B"},
        "calibration": {
            "dataset": dataset,
            "num_samples": 2,
            "max_seq_length": 32,
            "batch_size": 1,
            "seed": 0,
            **calib,
        },
        "export": {"output_dir": "outputs/test"},
    }
    return recipe_from_mapping(payload)


def test_nemotron_splits_put_chat_before_train() -> None:
    splits = _nemotron_splits("nvidia/Nemotron-Post-Training-Dataset-v2")
    assert splits[0] == "chat"
    assert splits.index("chat") < splits.index("train")


def test_nemotron_v2_load_uses_chat_split_not_config_name(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeDS:
        def __iter__(self):
            return iter([NEMOTRON_ROW])

    def fake_load_dataset(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("name") == "chat":
            raise AssertionError("Nemotron 'chat' is a split, not a BuilderConfig name")
        if kwargs.get("split") == "train":
            raise ValueError("Unknown split 'train'. Should be one of ['stem', 'chat', ...]")
        if kwargs.get("split") == "chat" and kwargs.get("name") is None:
            return FakeDS()
        raise ValueError(f"unexpected kwargs {kwargs}")

    fake_mod = ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    fake_mod.get_dataset_config_names = lambda *_a, **_k: ["default"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    ds = _load_hf_dataset("nvidia/Nemotron-Post-Training-Dataset-v2")
    assert list(ds)[0]["messages"][0]["content"] == "What is 2+2?"
    assert calls, "load_dataset was never called"
    assert calls[0].get("split") == "chat"
    assert "name" not in calls[0]


def test_extract_text_from_nemotron_messages() -> None:
    text = _extract_text(NEMOTRON_ROW, None, _FakeTokenizer())
    assert text is not None
    assert "2+2" in text


def test_jsonl_calib_iter_has_len_and_is_reusable(tmp_path) -> None:
    path = tmp_path / "calib.jsonl"
    path.write_text('{"text": "alpha"}\n{"text": "beta"}\n{"text": "gamma"}\n')
    recipe = _w4a8_recipe(str(path), num_samples=2, batch_size=1)
    tok = _FakeTokenizer()
    it = build_calibration_iter(recipe, tok)
    assert isinstance(it, CalibrationBatches)
    assert len(it) == 2
    first = list(it)
    second = list(it)
    assert len(first) == len(second) == 2
    assert "input_ids" in first[0]
    assert tok.pad_token == tok.eos_token


def test_dummy_calib_iter_has_len() -> None:
    dummy = DummyCalibIter(3)
    assert len(dummy) == 3
    assert len(list(dummy)) == 3
