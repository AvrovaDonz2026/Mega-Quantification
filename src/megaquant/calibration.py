"""Calibration iterators for PTQ (CPU-safe; no GPU required)."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from megaquant.config import Recipe
from megaquant.exceptions import CalibrationError


class DummyCalibIter:
    """n batches of zeros shaped ``[1, 16]`` (for tests; no real data)."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import torch
        except ImportError:
            for _ in range(self.n):
                yield {
                    "input_ids": [[0] * 16],
                    "attention_mask": [[0] * 16],
                }
            return
        for _ in range(self.n):
            yield {
                "input_ids": torch.zeros(1, 16, dtype=torch.long),
                "attention_mask": torch.zeros(1, 16, dtype=torch.long),
            }


def _messages_to_text(messages: Any, tokenizer: Any | None) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except (TypeError, ValueError, KeyError):
            pass
    parts: list[str] = []
    for item in messages:
        if isinstance(item, Mapping):
            content = item.get("content", item.get("value", ""))
            if isinstance(content, list):
                content = " ".join(
                    str(p.get("text", p) if isinstance(p, Mapping) else p) for p in content
                )
            role = item.get("role", item.get("from", "user"))
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _extract_text(example: Mapping[str, Any], text_field: str | None, tokenizer: Any) -> str | None:
    if text_field:
        if text_field not in example:
            return None
        value = example[text_field]
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _messages_to_text(value, tokenizer)
        return str(value)

    for key in ("text", "prompt", "article", "content", "input", "question"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            extra = example.get("completion") or example.get("output") or example.get("response")
            if isinstance(extra, str) and extra.strip():
                return f"{value}\n{extra}"
            return value

    for key in ("messages", "conversations", "conversation"):
        if key in example:
            text = _messages_to_text(example[key], tokenizer)
            if text.strip():
                return text
    return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload, dict):
            yield payload
            return
        raise CalibrationError(f"JSON calibration file must be an object or array: {path}")
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise CalibrationError(f"JSONL rows must be objects ({path}:{line_no})")
            yield row


def _load_hf_dataset(dataset_id: str) -> Any:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise CalibrationError(
            "The datasets library is required for Hugging Face calibration. "
            "Install with: pip install megaquant[hf]"
        ) from exc

    attempts: list[dict[str, Any]] = []
    if dataset_id == "cnn_dailymail":
        attempts.append({"path": dataset_id, "name": "3.0.0", "split": "train", "streaming": True})
    elif dataset_id == "HuggingFaceH4/ultrachat_200k" or dataset_id.endswith("ultrachat_200k"):
        attempts.append({"path": dataset_id, "split": "train_sft", "streaming": True})
        attempts.append({"path": dataset_id, "split": "train", "streaming": True})
    elif dataset_id == "nvidia/Nemotron-Post-Training-Dataset-v2":
        attempts.append({"path": dataset_id, "split": "train", "streaming": True})
        attempts.append({"path": dataset_id, "name": "chat", "split": "train", "streaming": True})
    else:
        attempts.append({"path": dataset_id, "split": "train", "streaming": True})

    errors: list[str] = []
    for kwargs in attempts:
        try:
            return load_dataset(**kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")

    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as exc:
        configs = []
        errors.append(str(exc))
    for name in configs[:8]:
        try:
            return load_dataset(dataset_id, name=name, split="train", streaming=True)
        except Exception as exc:
            errors.append(f"name={name}: {exc}")

    detail = errors[-1] if errors else "unknown error"
    raise CalibrationError(f"Could not load calibration dataset '{dataset_id}': {detail}")


def _tokenize_batch(tokenizer: Any, texts: list[str], max_seq_length: int) -> dict[str, Any]:
    try:
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise CalibrationError(f"Tokenizer failed: {exc}") from exc
    if not isinstance(encoded, dict) and hasattr(encoded, "data"):
        encoded = dict(encoded)
    if "input_ids" not in encoded:
        raise CalibrationError("Tokenizer did not return input_ids")
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = [[1] * len(ids) for ids in encoded["input_ids"]]
    return encoded


def build_calibration_iter(recipe: Recipe, tokenizer: Any) -> Iterator[dict[str, Any]]:
    """Yield tokenized batches with ``input_ids`` / ``attention_mask``."""
    if tokenizer is None:
        raise CalibrationError("A tokenizer is required to build the calibration iterator")

    calib = recipe.calibration
    dataset = calib.dataset
    path = Path(dataset).expanduser()
    examples: list[dict[str, Any]]

    if path.exists() and path.is_file():
        rows = list(_iter_jsonl(path))
        rng = random.Random(calib.seed)
        rng.shuffle(rows)
        examples = rows[: calib.num_samples]
    else:
        try:
            ds = _load_hf_dataset(dataset)
        except CalibrationError:
            raise
        except ImportError as exc:
            raise CalibrationError(
                "The datasets library is required for Hugging Face calibration. "
                "Install with: pip install megaquant[hf]"
            ) from exc
        if hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(seed=calib.seed, buffer_size=10_000)
            except TypeError:
                try:
                    ds = ds.shuffle(seed=calib.seed)
                except Exception:
                    pass
        examples = []
        for row in ds:
            if not isinstance(row, Mapping):
                continue
            examples.append(dict(row))
            if len(examples) >= calib.num_samples:
                break

    texts: list[str] = []
    for example in examples:
        text = _extract_text(example, calib.text_field, tokenizer)
        if text and text.strip():
            texts.append(text)
    if not texts:
        raise CalibrationError(
            f"No text examples extracted from '{dataset}' "
            f"(num_samples={calib.num_samples}, text_field={calib.text_field!r})"
        )

    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= calib.batch_size:
            yield _tokenize_batch(tokenizer, batch, calib.max_seq_length)
            batch = []
    if batch:
        yield _tokenize_batch(tokenizer, batch, calib.max_seq_length)
