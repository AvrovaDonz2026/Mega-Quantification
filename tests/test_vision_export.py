"""Vision weights must survive language-only PTQ export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaquant.cli import main
from megaquant.exceptions import BackendError
from megaquant.mtp_export import RawTensor, write_safetensors
from megaquant.vision_export import VISION_SHARD_NAME, restore_vision, restore_vision_from_recipe


def test_restores_source_vision_and_preserves_index(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    weights = {
        "model.visual.blocks.0.attn.qkv.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "model.visual.patch_embed.proj.weight": torch.arange(4, dtype=torch.bfloat16),
    }
    save_file(weights, str(source / "source.safetensors"))
    (source / "preprocessor_config.json").write_text(
        '{"image_processor_type":"Qwen3_5ImageProcessor"}'
    )
    (source / "video_preprocessor_config.json").write_text(
        '{"video_processor_type":"Qwen3_5VideoProcessor"}'
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "source.safetensors" for key in weights}})
    )
    (export / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {}, "language_model_only": True})
    )
    (export / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": 16}, "weight_map": {"mtp.weight": "mtp.safetensors"}}
        )
    )
    recipe = {"model": {"source": str(source), "quantize_vision": False}}

    note = restore_vision_from_recipe(export, recipe)
    assert note == {"vision_tensors": 2, "restored_tensors": 2, "method": "hf-source"}
    assert json.loads((export / "config.json").read_text())["language_model_only"] is False
    for filename in ("preprocessor_config.json", "video_preprocessor_config.json"):
        assert (export / filename).read_bytes() == (source / filename).read_bytes()
    index = json.loads((export / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.weight"] == "mtp.safetensors"
    assert all(index["weight_map"][key] == VISION_SHARD_NAME for key in weights)
    assert index["metadata"]["total_size"] == 16 + sum(
        t.numel() * t.element_size() for t in weights.values()
    )
    with safetensors.safe_open(str(export / VISION_SHARD_NAME), framework="pt") as shard:
        assert set(shard.keys()) == set(weights)
        for key, original in weights.items():
            assert torch.equal(shard.get_tensor(key), original)

    again = restore_vision_from_recipe(export, recipe)
    assert again == {"vision_tensors": 2, "restored_tensors": 0, "method": "already-present"}
    assert json.loads((export / "config.json").read_text())["language_model_only"] is False
    assert json.loads((export / "model.safetensors.index.json").read_text()) == index
    assert restore_vision(export, source) == again
    assert main(["restore-vision", str(export), "--source", str(source)]) == 0


def test_missing_vision_in_source_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (export / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "vision_config": {}}))
    (export / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

    with pytest.raises(BackendError, match="no model.visual"):
        restore_vision_from_recipe(export, {"model": {"source": str(source)}})


def test_missing_image_processor_fails(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    key = "model.visual.patch_embed.proj.weight"
    save_file({key: torch.ones(2, dtype=torch.bfloat16)}, str(source / "source.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "source.safetensors"}})
    )
    (export / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {}, "language_model_only": True})
    )

    with pytest.raises(BackendError, match="preprocessor_config.json"):
        restore_vision(export, source)
    assert json.loads((export / "config.json").read_text())["language_model_only"] is True


def test_other_models_are_unchanged(tmp_path: Path) -> None:
    export = tmp_path / "export"
    export.mkdir()
    config = export / "config.json"
    config.write_text(json.dumps({"model_type": "other_vlm", "vision_config": {}}))
    assert restore_vision(export, "unused") is None
    recipe = {"model": {"source": "unused", "quantize_vision": True}}
    assert restore_vision_from_recipe(export, recipe) is None


def _quantized_vision_dirs(tmp_path: Path) -> tuple[Path, Path, dict]:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    module = "model.visual.blocks.0.attn.qkv"
    # No source shard is needed: this path must keep the packed ModelOpt weights.
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {module + ".weight": "source.safetensors"}})
    )
    tensors = {
        module + ".weight": RawTensor(dtype="U8", shape=(2, 8), data=bytes(16)),
        module + ".weight_scale": RawTensor(dtype="F8_E4M3", shape=(2, 1), data=bytes(2)),
        module + ".weight_scale_2": RawTensor(dtype="F32", shape=(), data=bytes(4)),
    }
    write_safetensors(export / "packed.safetensors", tensors)
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "packed.safetensors" for key in tensors}})
    )
    (export / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {}, "language_model_only": True})
    )
    for filename in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    ):
        (source / filename).write_text(json.dumps({"source": filename}))
    (source / "chat_template.jinja").write_text("source template")
    recipe = {"model": {"source": str(source), "quantize_vision": True}}
    return source, export, recipe


def test_quantized_vision_copies_processor_and_keeps_packed_weights(tmp_path: Path) -> None:
    source, export, recipe = _quantized_vision_dirs(tmp_path)
    (export / "chat_template.jinja").write_text("exported tokenizer template")
    unchanged = {
        name: (export / name).read_bytes()
        for name in ("packed.safetensors", "model.safetensors.index.json")
    }

    note = restore_vision_from_recipe(export, recipe)
    assert note == {"vision_tensors": 3, "restored_tensors": 0, "method": "modelopt-export"}
    assert json.loads((export / "vision_export.json").read_text()) == note
    assert json.loads((export / "config.json").read_text())["language_model_only"] is False
    for filename in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
    ):
        assert (export / filename).read_bytes() == (source / filename).read_bytes()
    assert (export / "chat_template.jinja").read_text() == "exported tokenizer template"
    assert not (export / VISION_SHARD_NAME).exists()
    assert restore_vision_from_recipe(export, recipe) == note
    assert {name: (export / name).read_bytes() for name in unchanged} == unchanged


@pytest.mark.parametrize("missing", ["tensor", "shard"])
def test_quantized_vision_rejects_incomplete_export(tmp_path: Path, missing: str) -> None:
    _, export, recipe = _quantized_vision_dirs(tmp_path)
    if missing == "shard":
        (export / "packed.safetensors").unlink()
    else:
        index_path = export / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].pop("model.visual.blocks.0.attn.qkv.weight")
        index_path.write_text(json.dumps(index))
    with pytest.raises(BackendError, match="Quantized vision export is missing"):
        restore_vision_from_recipe(export, recipe)
    assert not (export / "vision_export.json").exists()
    assert not (export / VISION_SHARD_NAME).exists()


def test_quantized_vision_requires_image_preprocessor(tmp_path: Path) -> None:
    source, export, recipe = _quantized_vision_dirs(tmp_path)
    (source / "preprocessor_config.json").unlink()
    with pytest.raises(BackendError, match="preprocessor_config.json"):
        restore_vision_from_recipe(export, recipe)
    assert not (export / "vision_export.json").exists()
