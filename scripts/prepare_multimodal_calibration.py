#!/usr/bin/env python3
"""Build reproducible image/text calibration from a local COCO caption parquet.

Requires ``pyarrow`` and ``Pillow``. Images are materialized beside the JSONL so
the complete output directory can be mounted as ``/data`` in the quantizer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def caption_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for field in ("raw", "caption", "text", "sent", "sentences"):
            if field in value:
                text = caption_text(value[field])
                if text:
                    return text
    if isinstance(value, list):
        for item in value:
            text = caption_text(item)
            if text:
                return text
    return ""


def text_example(row: Mapping[str, Any]) -> str:
    for field in ("text", "prompt", "article", "content", "input", "question"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            extra = row.get("completion") or row.get("output") or row.get("response")
            return f"{value}\n{extra}" if isinstance(extra, str) and extra.strip() else value
    for field in ("messages", "conversations", "conversation"):
        if not isinstance(row.get(field), list):
            continue
        parts = []
        for message in row[field]:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role", message.get("from", "user"))
            content = message.get("content", message.get("value", ""))
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, Mapping) and item.get("type", "text") == "text"
                )
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
    return ""


def _open_image(value: Any, source_dir: Path) -> Any:
    from PIL import Image

    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        value = value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(value))
    if isinstance(value, str):
        path = Path(value).expanduser()
        return Image.open(path if path.is_absolute() else source_dir / path)
    raise ValueError("Image column must contain encoded bytes or a local path")


def prepare(
    parquet: Path,
    output_dir: Path,
    *,
    source_revision: str,
    source_dataset: str = "COCO-Caption2017",
    image_samples: int = 128,
    max_image_side: int = 448,
    seed: int = 42,
    text_jsonl: Path | None = None,
    text_samples: int = 128,
    max_text_chars: int = 600,
    image_column: str = "image",
    caption_column: str | None = None,
    expected_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps

    if image_samples < 1 or text_samples < 0 or max_image_side < 1 or max_text_chars < 1:
        raise ValueError(
            "Sample counts and size limits must be positive (text samples may be zero)"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty to avoid stale samples: {output_dir}")
    parquet_sha256 = sha256_file(parquet)
    if expected_parquet_sha256 and parquet_sha256 != expected_parquet_sha256.lower():
        raise ValueError(
            f"Parquet SHA256 mismatch: expected {expected_parquet_sha256}, got {parquet_sha256}"
        )
    source = pq.ParquetFile(parquet)
    columns = source.schema_arrow.names
    if image_column not in columns:
        raise ValueError(f"Image column {image_column!r} not found; available columns: {columns}")
    if caption_column is None:
        caption_column = next(
            (
                column
                for column in ("caption", "captions", "answer", "sentences", "text", "description")
                if column in columns
            ),
            None,
        )
    if caption_column not in columns:
        raise ValueError(f"Caption column not found; available columns: {columns}")
    total_rows = source.metadata.num_rows
    if image_samples > total_rows:
        raise ValueError(f"Requested {image_samples} images but parquet only has {total_rows} rows")
    selected = set(random.Random(seed).sample(range(total_rows), image_samples))

    texts: list[dict[str, Any]] = []
    if text_jsonl is not None and text_samples:
        for line_no, line in enumerate(text_jsonl.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"Text JSONL row {line_no} is not an object")
            text = text_example(row).strip()
            if text:
                texts.append({"text": text[:max_text_chars], "source_line": line_no})
        if text_samples > len(texts):
            raise ValueError(f"Requested {text_samples} texts but found only {len(texts)}")
        texts = random.Random(seed).sample(texts, text_samples)
    elif text_jsonl is None:
        text_samples = 0

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    calibration: list[dict[str, Any]] = []
    image_manifest: list[dict[str, Any]] = []
    row_index = 0
    read_columns = [image_column, caption_column]
    if "question" in columns:
        read_columns.append("question")
    for batch in source.iter_batches(batch_size=16, columns=read_columns):
        for row in batch.to_pylist():
            index = row_index
            row_index += 1
            if index not in selected:
                continue
            caption = caption_text(row[caption_column])[:max_text_chars]
            if not caption:
                raise ValueError(f"Selected parquet row {index} has no caption")
            target = image_dir / f"{index:08d}.png"
            with _open_image(row[image_column], parquet.parent) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                original_size = list(image.size)
                image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
                image.save(target, format="PNG", optimize=False)
                size = list(image.size)
                image.close()
            relative = target.relative_to(output_dir).as_posix()
            calibration.append(
                {
                    "image": relative,
                    "messages": [
                        {
                            "role": "user",
                            "content": str(row.get("question") or "Describe this image.")[
                                :max_text_chars
                            ],
                        },
                        {"role": "assistant", "content": caption},
                    ],
                }
            )
            image_manifest.append(
                {
                    "source_row": index,
                    "path": relative,
                    "sha256": sha256_file(target),
                    "original_size": original_size,
                    "size": size,
                }
            )
    if len(image_manifest) != image_samples:
        raise ValueError(f"Expected {image_samples} images; extracted {len(image_manifest)}")
    calibration.extend({"text": row["text"]} for row in texts)
    random.Random(seed).shuffle(calibration)
    calibration_path = output_dir / "calibration.jsonl"
    calibration_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in calibration),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "source": {
            "dataset": source_dataset,
            "revision": source_revision,
            "parquet": str(parquet.resolve()),
            "sha256": parquet_sha256,
            "num_rows": total_rows,
            "image_column": image_column,
            "caption_column": caption_column,
        },
        "parameters": {
            "seed": seed,
            "image_samples": image_samples,
            "text_samples": len(texts),
            "max_image_side": max_image_side,
            "max_text_chars": max_text_chars,
        },
        "calibration": {
            "path": calibration_path.name,
            "sha256": sha256_file(calibration_path),
            "num_samples": len(calibration),
        },
        "images": image_manifest,
    }
    if text_jsonl is not None:
        manifest["text_source"] = {
            "path": str(text_jsonl.resolve()),
            "sha256": sha256_file(text_jsonl),
            "selected_lines": [row["source_line"] for row in texts],
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Pinned source dataset commit")
    parser.add_argument("--source-dataset", default="COCO-Caption2017")
    parser.add_argument("--image-samples", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-jsonl", type=Path)
    parser.add_argument("--text-samples", type=int, default=128)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column")
    parser.add_argument("--expected-parquet-sha256")
    args = parser.parse_args()
    try:
        manifest = prepare(**vars(args))
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"Calibration preparation failed: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
