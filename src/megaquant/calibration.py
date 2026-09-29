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


class CalibrationBatches:
    """Reusable, sized iterable of already-tokenized PTQ batches.

    ModelOpt ``forward_loop`` (especially Local-Hessian) may iterate calibration
    data more than once. A generator would be exhausted after the first pass, and
    some ModelOpt / tqdm paths call ``len()``.
    """

    def __init__(self, batches: list[dict[str, Any]], *, num_image_samples: int = 0):
        self.batches = list(batches)
        self.num_image_samples = num_image_samples

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.batches)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.batches[index]


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


# Nemotron-Post-Training-Dataset-v2/v3 have splits chat/stem/math/code/…, not ``train``.
# ``chat`` is a split name, not a BuilderConfig (``name=``).
_NEMOTRON_PREFIX = "nvidia/Nemotron-Post-Training-Dataset"
_SPLIT_CANDIDATES = (
    "train",
    "train_sft",
    "chat",
    "stem",
    "math",
    "code",
    "validation",
    "test",
)


def _is_dataset_dict(ds: Any) -> bool:
    if ds is None or isinstance(ds, dict):
        # Row dicts are mappings; HuggingFace DatasetDict is a dedicated type.
        return False
    name = type(ds).__name__
    if name in {"DatasetDict", "IterableDatasetDict"}:
        return True
    keys_fn = getattr(ds, "keys", None)
    getitem = getattr(ds, "__getitem__", None)
    if not callable(keys_fn) or not callable(getitem):
        return False
    # A streaming IterableDataset is iterable but is not a split mapping.
    if hasattr(ds, "column_names") or hasattr(ds, "features"):
        return False
    try:
        keys = list(keys_fn())
    except Exception:
        return False
    return bool(keys) and all(isinstance(k, str) for k in keys)


def _unwrap_dataset_dict(ds: Any, preferred: tuple[str, ...] = _SPLIT_CANDIDATES) -> Any:
    if not _is_dataset_dict(ds):
        return ds
    keys = list(ds.keys())
    for split in preferred:
        if split in keys:
            return ds[split]
    return ds[keys[0]]


def _nemotron_splits(dataset_id: str) -> tuple[str, ...]:
    if _NEMOTRON_PREFIX in dataset_id:
        return ("chat", "stem", "math", "code", "train", "train_sft")
    if dataset_id == "cnn_dailymail":
        return ("train",)
    if dataset_id.endswith("ultrachat_200k"):
        return ("train_sft", "train")
    return _SPLIT_CANDIDATES


def _load_hf_dataset(dataset_id: str) -> Any:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise CalibrationError(
            "The datasets library is required for Hugging Face calibration. "
            "Install with: pip install megaquant[hf]"
        ) from exc

    splits = _nemotron_splits(dataset_id)
    attempts: list[dict[str, Any]] = []
    if dataset_id == "cnn_dailymail":
        attempts.append({"path": dataset_id, "name": "3.0.0", "split": "train", "streaming": True})
    elif dataset_id.endswith("ultrachat_200k"):
        for split in splits:
            attempts.append({"path": dataset_id, "split": split, "streaming": True})
    else:
        for split in splits:
            attempts.append({"path": dataset_id, "split": split, "streaming": True})
        # README for Nemotron v2 shows a config named "SFT"; try it after default.
        if _NEMOTRON_PREFIX in dataset_id:
            for split in splits:
                attempts.append(
                    {"path": dataset_id, "name": "SFT", "split": split, "streaming": True}
                )

    errors: list[str] = []
    for kwargs in attempts:
        try:
            return _unwrap_dataset_dict(load_dataset(**kwargs), splits)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")

    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as exc:
        configs = []
        errors.append(str(exc))
    for name in configs[:8]:
        for split in splits:
            try:
                return _unwrap_dataset_dict(
                    load_dataset(dataset_id, name=name, split=split, streaming=True),
                    splits,
                )
            except Exception as exc:
                errors.append(f"name={name} split={split}: {exc}")
        try:
            return _unwrap_dataset_dict(load_dataset(dataset_id, name=name, streaming=True), splits)
        except Exception as exc:
            errors.append(f"name={name} (no split): {exc}")

    detail = errors[-1] if errors else "unknown error"
    gated = any("gated" in item.lower() for item in errors)
    if gated:
        raise CalibrationError(
            f"Could not load calibration dataset '{dataset_id}' (gated on the Hub). "
            "Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN, or switch calibration.dataset to "
            "HuggingFaceH4/ultrachat_200k "
            "(recipes/qwen3.8-27b-nvfp4-w4a8.public-calib.yaml) or a local JSONL. "
            f"Last error: {detail}"
        )
    raise CalibrationError(f"Could not load calibration dataset '{dataset_id}': {detail}")


def _ensure_pad_token(tokenizer: Any) -> None:
    """Qwen3.8 ``text_config.pad_token_id`` is null; padding=True would raise."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_tok = getattr(tokenizer, "pad_token", None)
    if pad_id is not None and pad_tok is not None:
        return
    eos = getattr(tokenizer, "eos_token", None)
    if eos is not None:
        try:
            tokenizer.pad_token = eos
            return
        except (TypeError, ValueError, AttributeError):
            pass
    unk = getattr(tokenizer, "unk_token", None)
    if unk is not None:
        try:
            tokenizer.pad_token = unk
        except (TypeError, ValueError, AttributeError):
            pass


def _tokenize_batch(tokenizer: Any, texts: list[str], max_seq_length: int) -> dict[str, Any]:
    _ensure_pad_token(tokenizer)
    pad = True if len(texts) > 1 else False
    try:
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=pad,
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


def _image_reference(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("path", "url", "image"):
            if key in value:
                return value[key]
        raise CalibrationError("Image objects must contain a path, url, or image")
    return value


def _multimodal_example(
    example: Mapping[str, Any], text_field: str | None, tokenizer: Any
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Normalize local image rows and OpenAI-style image messages for a processor."""
    raw_messages = example.get(text_field) if text_field else None
    if not isinstance(raw_messages, list):
        raw_messages = next(
            (
                example[key]
                for key in ("messages", "conversations", "conversation")
                if isinstance(example.get(key), list)
            ),
            None,
        )
    images: list[Any] = []
    messages: list[dict[str, Any]] = []
    if raw_messages is not None:
        for message in raw_messages:
            if not isinstance(message, Mapping):
                raise CalibrationError("Multimodal messages must be objects")
            role = message.get("role", message.get("from", "user"))
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            content = message.get("content", message.get("value", ""))
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                raise CalibrationError("Multimodal message content must be text or a list")
            normalized = []
            for item in content:
                if not isinstance(item, Mapping):
                    raise CalibrationError("Multimodal content items must be objects")
                kind = item.get("type", "text")
                if kind in {"image", "image_url"}:
                    ref = item.get("image", item.get("image_url", item.get("path")))
                    if ref is None:
                        ref = item.get("url")
                    if ref is None:
                        raise CalibrationError("Each image content item needs a local image path")
                    images.append(_image_reference(ref))
                    normalized.append({"type": "image"})
                elif kind == "text":
                    normalized.append({"type": "text", "text": str(item.get("text", ""))})
                else:
                    raise CalibrationError(f"Unsupported calibration content type: {kind!r}")
            messages.append({"role": role, "content": normalized})
    else:
        text = _extract_text(example, text_field, tokenizer)
        if text:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    top_images = example.get("images", example.get("image"))
    if top_images is not None:
        if images:
            raise CalibrationError("Use either top-level images or message image content, not both")
        if not isinstance(top_images, list):
            top_images = [top_images]
        images = [_image_reference(value) for value in top_images]
        if not messages:
            messages = [{"role": "user", "content": []}]
        target = next((message for message in messages if message["role"] == "user"), None)
        if target is None:
            raise CalibrationError("Top-level images need a user message")
        target["content"] = [{"type": "image"} for _ in images] + target["content"]
    return messages, images


def _load_calibration_image(reference: Any, base_dir: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CalibrationError("Image calibration requires Pillow: pip install pillow") from exc
    if isinstance(reference, Image.Image):
        return reference.convert("RGB")
    if not isinstance(reference, (str, Path)):
        raise CalibrationError("Calibration images must be local paths or PIL images")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"Cannot open calibration image '{path}': {exc}") from exc


def _build_multimodal_batches(
    recipe: Recipe,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    base_dir: Path,
    processor: Any | None,
) -> CalibrationBatches:
    calib = recipe.calibration
    rows = [_multimodal_example(row, calib.text_field, tokenizer) for row in examples]
    rows = [(messages, images) for messages, images in rows if messages]
    image_samples = sum(bool(images) for _, images in rows)
    if not image_samples:
        raise CalibrationError(
            "calibration.with_images=true but no images were found in the selected samples; "
            "provide image/images paths or image content in messages"
        )
    if processor is None:
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                recipe.model.source, trust_remote_code=recipe.model.trust_remote_code
            )
        except Exception as exc:
            raise CalibrationError(f"Could not load the multimodal processor: {exc}") from exc
    _ensure_pad_token(getattr(processor, "tokenizer", tokenizer))
    batches = []
    for offset in range(0, len(rows), calib.batch_size):
        selected = rows[offset : offset + calib.batch_size]
        images: list[Any] = []
        try:
            texts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages, _ in selected
            ]
            for _, references in selected:
                for reference in references:
                    images.append(_load_calibration_image(reference, base_dir))
            # Truncating expanded image tokens can leave a different token count
            # from image_grid_thw, or remove the image entirely. Reject oversize
            # examples below instead of silently skipping visual calibration.
            kwargs: dict[str, Any] = {
                "text": texts,
                "padding": len(texts) > 1,
                "truncation": False,
                "return_tensors": "pt",
            }
            if images:
                kwargs["images"] = images
            encoded = dict(processor(**kwargs))
        except CalibrationError:
            raise
        except Exception as exc:
            raise CalibrationError(f"Multimodal processor failed: {exc}") from exc
        finally:
            for image in images:
                image.close()
        if "input_ids" not in encoded:
            raise CalibrationError("Multimodal processor did not return input_ids")
        if images and "pixel_values" not in encoded:
            raise CalibrationError("Multimodal processor did not return pixel_values for images")
        if any(len(ids) > calib.max_seq_length for ids in encoded["input_ids"]):
            raise CalibrationError(
                f"Multimodal sample exceeds max_seq_length={calib.max_seq_length}; "
                "increase max_seq_length or use smaller images/shorter text. "
                "Image tokens cannot be safely truncated."
            )
        batches.append(encoded)
    return CalibrationBatches(batches, num_image_samples=image_samples)


def build_calibration_iter(
    recipe: Recipe, tokenizer: Any, *, processor: Any | None = None
) -> CalibrationBatches:
    """Return tokenized batches with ``input_ids`` / ``attention_mask``.

    The result is sized (``len``) and re-iterable so ModelOpt can run the
    forward loop more than once (Local-Hessian) without a second Hub pass.
    """
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
        ds = _unwrap_dataset_dict(ds)
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

    if calib.with_images:
        base_dir = path.resolve().parent if path.is_file() else Path.cwd()
        return _build_multimodal_batches(recipe, tokenizer, examples, base_dir, processor)

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

    batches: list[dict[str, Any]] = []
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= calib.batch_size:
            batches.append(_tokenize_batch(tokenizer, batch, calib.max_seq_length))
            batch = []
    if batch:
        batches.append(_tokenize_batch(tokenizer, batch, calib.max_seq_length))
    return CalibrationBatches(batches)
