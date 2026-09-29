"""Reproducible real-image calibration preparation from caption parquet."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_multimodal_calibration.py"
_SPEC = importlib.util.spec_from_file_location("prepare_multimodal_calibration", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
prepare_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_module)


def _parquet(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    image_module = pytest.importorskip("PIL.Image")
    rows = []
    for index in range(4):
        buffer = io.BytesIO()
        image_module.new("RGB", (96, 48), color=(index * 40, 20, 10)).save(buffer, format="PNG")
        rows.append(
            {
                "image": {"bytes": buffer.getvalue(), "path": f"image-{index}.png"},
                "question": "Describe this image in detail.",
                "answer": [f"Caption {index}.", "An alternative caption."],
            }
        )
    source = tmp_path / "source.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), source, row_group_size=2)
    return source


def test_prepare_real_image_bytes_and_text_reproducibly(tmp_path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    source = _parquet(tmp_path)
    texts = tmp_path / "text.jsonl"
    texts.write_text("\n".join(json.dumps({"text": "text-" + str(i) * 20}) for i in range(4)))
    manifests = []
    for folder in ("first", "second"):
        output = tmp_path / folder
        manifests.append(
            prepare_module.prepare(
                source,
                output,
                source_revision="pinned-commit",
                image_samples=2,
                max_image_side=32,
                text_jsonl=texts,
                text_samples=2,
                max_text_chars=12,
                expected_parquet_sha256=prepare_module.sha256_file(source),
            )
        )
        rows = [
            json.loads(line) for line in (output / "calibration.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 4
        assert sum("image" in row for row in rows) == 2
        assert sum("text" in row for row in rows) == 2
        assert all(len(row["text"]) <= 12 for row in rows if "text" in row)
        for row in rows:
            if "image" not in row:
                continue
            with image_module.open(output / row["image"]) as image:
                assert image.size == (32, 16)
            assert row["messages"][1]["content"].startswith("Caption ")
        manifest = manifests[-1]
        assert manifest["source"]["caption_column"] == "answer"
        assert manifest["source"]["revision"] == "pinned-commit"
        assert manifest["calibration"]["num_samples"] == 4
        assert len(manifest["images"]) == 2
        for image in manifest["images"]:
            assert image["sha256"] == prepare_module.sha256_file(output / image["path"])
    assert manifests[0] == manifests[1]


def test_prepare_rejects_checksum_mismatch_and_insufficient_text(tmp_path) -> None:
    source = _parquet(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_module.prepare(
            source,
            tmp_path / "bad-hash",
            source_revision="revision",
            image_samples=1,
            expected_parquet_sha256="0" * 64,
        )
    texts = tmp_path / "text.jsonl"
    texts.write_text('{"text":"only one"}\n')
    with pytest.raises(ValueError, match="Requested 2 texts"):
        prepare_module.prepare(
            source,
            tmp_path / "not-enough",
            source_revision="revision",
            image_samples=1,
            text_jsonl=texts,
            text_samples=2,
        )


def test_caption_and_conversation_text_extraction() -> None:
    assert prepare_module.caption_text({"raw": ["Caption one", "Caption two"]}) == "Caption one"
    assert (
        prepare_module.text_example(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        )
        == "user: Hello\nassistant: Hi"
    )
