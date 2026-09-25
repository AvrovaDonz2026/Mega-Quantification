"""BF16 MTP restore + 1-layer SGLang draft (no GPU, no 27B download)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from megaquant.cli import main
from megaquant.exceptions import BackendError, MegaQuantError
from megaquant.mtp_export import (
    MTP_SHARD_NAME,
    RawTensor,
    collect_mtp_from_model,
    collect_mtp_from_source,
    config_mtp_layers,
    restore_bf16_mtp,
    restore_bf16_mtp_from_recipe,
    to_hf_mtp_name,
    write_mtp_draft,
    write_safetensors,
)
from megaquant.sglang_export import quantized_layers_from_weight_map, rewrite_sglang_mixed_export

_F32 = struct.pack("<ff", 1.0, -2.5)
_MTP_A = "mtp.fc.weight"
_MTP_B = "mtp.layers.0.mlp.gate_proj.weight"


def _tensor() -> RawTensor:
    return RawTensor(dtype="F32", shape=(2,), data=_F32)


def _write_index(path: Path, weight_map: dict[str, str], *, total_size: int | None = None) -> None:
    payload: dict = {"weight_map": weight_map}
    if total_size is not None:
        payload["metadata"] = {"total_size": total_size}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _lm_weight_map() -> dict[str, str]:
    return {
        "model.language_model.layers.0.mlp.gate_proj.weight": "a.safetensors",
        "model.language_model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "lm_head.weight": "a.safetensors",
    }


def _source_dir(root: Path) -> Path:
    src = root / "bf16"
    src.mkdir()
    tensors = {
        _MTP_A: _tensor(),
        _MTP_B: _tensor(),
        "model.layers.0.mlp.gate_proj.weight": _tensor(),
    }
    write_safetensors(src / "model-00018-of-00018.safetensors", tensors)
    (src / "model-00001-of-00018.safetensors").write_bytes(b"not-mtp")
    _write_index(
        src / "model.safetensors.index.json",
        {
            "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00018.safetensors",
            _MTP_A: "model-00018-of-00018.safetensors",
            _MTP_B: "model-00018-of-00018.safetensors",
        },
    )
    (src / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1}}),
        encoding="utf-8",
    )
    return src


def _export_dir(root: Path, *, mtp_layers: int = 1) -> Path:
    dest = root / "export"
    dest.mkdir()
    _write_index(dest / "model.safetensors.index.json", _lm_weight_map(), total_size=12)
    (dest / "a.safetensors").write_bytes(b"weights")
    (dest / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 64,
                    "full_attention_interval": 4,
                    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                    "mtp_num_hidden_layers": mtp_layers,
                },
                "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            }
        ),
        encoding="utf-8",
    )
    (dest / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}}),
        encoding="utf-8",
    )
    (dest / "tokenizer.json").write_text("{}", encoding="utf-8")
    (dest / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return dest


def test_to_hf_mtp_name_strips_module_prefixes() -> None:
    assert to_hf_mtp_name("model.language_model.mtp.fc.weight") == "mtp.fc.weight"
    assert to_hf_mtp_name("mtp.layers.0.mlp.gate_proj.weight") == _MTP_B
    assert to_hf_mtp_name("mtp.layers.0.mlp.gate_proj.weight_scale") is None
    assert to_hf_mtp_name("model.layers.0.mlp.gate_proj.weight") is None


def test_config_mtp_layers_reads_text_config() -> None:
    assert config_mtp_layers({"text_config": {"mtp_num_hidden_layers": 1}}) == 1
    assert config_mtp_layers({"mtp_num_hidden_layers": 0}) == 0
    assert config_mtp_layers({}) == 0


def test_restore_from_local_source_writes_shard_and_draft(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    note = restore_bf16_mtp(export, source=src)
    assert note["method"] == "hf-source"
    assert note["mtp_tensors"] == 2
    assert note["shard"] == MTP_SHARD_NAME
    assert (export / MTP_SHARD_NAME).is_file()
    index = json.loads((export / "model.safetensors.index.json").read_text())
    assert index["weight_map"][_MTP_A] == MTP_SHARD_NAME
    assert index["weight_map"][_MTP_B] == MTP_SHARD_NAME
    assert index["metadata"]["total_size"] == 12 + 8 + 8
    assert note["draft_dir"] == "export-draft"
    draft = tmp_path / "export-draft"
    assert (draft / MTP_SHARD_NAME).is_file()
    assert not (draft / "hf_quant_config.json").exists()
    cfg = json.loads((draft / "config.json").read_text())
    assert "quantization_config" not in cfg
    assert cfg["text_config"]["num_hidden_layers"] == 1
    assert cfg["text_config"]["full_attention_interval"] == 1
    assert cfg["text_config"]["layer_types"] == ["full_attention"]
    draft_index = json.loads((draft / "model.safetensors.index.json").read_text())
    assert set(draft_index["weight_map"]) == {_MTP_A, _MTP_B}


def test_restore_keeps_mtp_out_of_quantized_layers(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    restore_bf16_mtp(export, source=src, write_draft=False)
    layers = rewrite_sglang_mixed_export(export)
    joined = " ".join(layers)
    assert "mtp" not in joined
    hf = json.loads((export / "hf_quant_config.json").read_text())
    assert "mtp*" in hf["quantization"]["exclude_modules"]
    weight_map = json.loads((export / "model.safetensors.index.json").read_text())["weight_map"]
    assert _MTP_A in weight_map
    assert "mtp" not in " ".join(quantized_layers_from_weight_map(weight_map))


def test_restore_is_idempotent(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    restore_bf16_mtp(export, source=src, write_draft=False)
    first = (export / MTP_SHARD_NAME).stat().st_mtime_ns
    note = restore_bf16_mtp(export, source=src, write_draft=False)
    assert note["method"] == "already-present"
    assert (export / MTP_SHARD_NAME).stat().st_mtime_ns == first


def test_restore_repairs_missing_indexed_mtp_shard(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    index_path = export / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({_MTP_A: MTP_SHARD_NAME, _MTP_B: MTP_SHARD_NAME})
    index["metadata"]["total_size"] += 16
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(BackendError, match="indexed MTP shard.*missing"):
        restore_bf16_mtp(export, write_draft=False)

    note = restore_bf16_mtp(export, source=src, write_draft=False)
    assert note["method"] == "hf-source"
    assert (export / MTP_SHARD_NAME).is_file()
    assert json.loads(index_path.read_text()) == index
    assert restore_bf16_mtp(export, source=src, write_draft=False)["method"] == "already-present"


def test_quantize_mtp_skips_bf16_graft(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    note = restore_bf16_mtp_from_recipe(
        export,
        {"model": {"source": str(src), "quantize_mtp": True}},
    )
    assert note is not None
    assert note["method"] == "skipped-quantize-mtp"
    assert not (export / MTP_SHARD_NAME).exists()


def test_no_mtp_architecture_is_noop(tmp_path: Path) -> None:
    export = _export_dir(tmp_path, mtp_layers=0)
    cfg = json.loads((export / "config.json").read_text())
    cfg["text_config"].pop("mtp_num_hidden_layers")
    (export / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    note = restore_bf16_mtp(export)
    assert note["method"] == "skipped-no-mtp"
    assert not (export / MTP_SHARD_NAME).exists()


def test_missing_mtp_raises_when_config_declares_layers(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    with pytest.raises(BackendError, match="mtp_num_hidden_layers"):
        restore_bf16_mtp(export)


def test_collect_from_in_memory_model(tmp_path: Path) -> None:
    class Fake:
        def named_parameters(self):
            yield "language_model.mtp.fc.weight", _tensor()
            yield "mtp.layers.0.mlp.gate_proj.weight_scale", _tensor()
            yield "lm_head.weight", _tensor()

    tensors = collect_mtp_from_model(Fake())
    assert set(tensors) == {_MTP_A}
    export = _export_dir(tmp_path)
    note = restore_bf16_mtp(export, model=Fake(), write_draft=False)
    assert note["method"] == "in-memory"
    assert json.loads((export / "model.safetensors.index.json").read_text())["weight_map"][
        _MTP_A
    ] == MTP_SHARD_NAME


def test_hub_source_reads_only_mtp_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _source_dir(tmp_path)
    calls: list[str] = []

    def fake_download(repo_id: str, filename: str) -> Path:
        calls.append(filename)
        if filename == "model.safetensors.index.json":
            return src / filename
        if filename == "model-00018-of-00018.safetensors":
            return src / filename
        raise AssertionError(f"unexpected hub download {filename}")

    monkeypatch.setattr("megaquant.mtp_export._hub_download", fake_download)
    tensors = collect_mtp_from_source("Qwen/Qwen3.8-27B")
    assert set(tensors) == {_MTP_A, _MTP_B}
    assert calls == [
        "model.safetensors.index.json",
        "model-00018-of-00018.safetensors",
    ]


def test_write_mtp_draft_requires_tensors(tmp_path: Path) -> None:
    export = _export_dir(tmp_path)
    with pytest.raises(MegaQuantError, match="no mtp"):
        write_mtp_draft(export)


def test_cli_restore_mtp(tmp_path: Path) -> None:
    src = _source_dir(tmp_path)
    export = _export_dir(tmp_path)
    code = main(["restore-mtp", str(export), "--source", str(src)])
    assert code == 0
    assert (export / MTP_SHARD_NAME).is_file()
    assert (tmp_path / "export-draft" / "config.json").is_file()
    code = main(["write-mtp-draft", str(export), "--output", str(tmp_path / "custom-draft")])
    assert code == 0
    assert (tmp_path / "custom-draft" / MTP_SHARD_NAME).is_file()
