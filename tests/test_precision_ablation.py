"""Check tensor preservation and safe HF indexing using tiny real payloads."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prepare_precision_ablation",
    Path(__file__).parents[1] / "scripts" / "prepare_precision_ablation.py",
)
ablation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ablation)

ATTN = "model.language_model.layers.0.self_attn.q_proj"
LINEAR = "model.language_model.layers.1.linear_attn.in_proj_qkv"
MLP = "model.language_model.layers.0.mlp.gate_proj"
MLP_PROJECTIONS = tuple(
    f"model.language_model.layers.0.mlp.{projection}"
    for projection in ("gate_proj", "up_proj", "down_proj")
)
MLP_LAYER1_PROJECTIONS = tuple(
    f"model.language_model.layers.1.mlp.{projection}"
    for projection in ("gate_proj", "up_proj", "down_proj")
)
VISION = "model.visual.blocks.0.attn.qkv"
MTP = "mtp.layers.0.self_attn.q_proj"


def _json(path, value):
    path.write_text(json.dumps(value))


def _shard(root, filename, tensors):
    header, payload = {}, bytearray()
    for i, (name, (dtype, shape)) in enumerate(tensors.items(), 1):
        start = len(payload)
        payload.extend(bytes([i]) * (math.prod(shape) * ablation.DTYPE_BYTES[dtype]))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / filename).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _packed(module):
    return {
        module + ".weight": ("U8", [2, 8]),
        module + ".weight_scale": ("F8_E4M3", [2, 1]),
        module + ".weight_scale_2": ("F32", []),
        module + ".input_scale": ("F32", []),
    }


@pytest.fixture
def checkpoints(tmp_path):
    export, source = tmp_path / "mse", tmp_path / "bf16"
    export.mkdir()
    source.mkdir()
    edited = {**_packed("lm_head"), **_packed(ATTN), **_packed(LINEAR),
              **{name + suffix: value for name in MLP_PROJECTIONS
                 for suffix, value in ((".weight", ("U8", [2, 8])),
                                       (".weight_scale", ("F8_E4M3", [2, 1])),
                                       (".weight_scale_2", ("F32", [])),
                                       (".input_scale", ("F32", [])))},
              "model.language_model.norm.weight": ("BF16", [2]),
              "model.language_model.layers.1.linear_attn.in_proj_a.weight": ("BF16", [2, 16])}
    preserved = {**_packed(VISION), **_packed(MTP)}
    _shard(export, "edited.safetensors", edited)
    _shard(export, "preserved.safetensors", preserved)
    _json(export / ablation.INDEX_NAME, {"weight_map": {
        **dict.fromkeys(edited, "edited.safetensors"),
        **dict.fromkeys(preserved, "preserved.safetensors"),
    }})
    bf16 = {
        "model.language_model.lm_head.weight": ("BF16", [2, 16]),
        ATTN + ".weight": ("BF16", [2, 16]),
        LINEAR + ".weight": ("BF16", [2, 16]),
        **{name + ".weight": ("BF16", [2, 16]) for name in MLP_PROJECTIONS},
    }
    _shard(source, "model.safetensors", bf16)
    # Existing bogus aliases must not label BF16 weights or survive restoration.
    quant = {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": None,
             "quantized_layers": {"model.language_model.norm": {"quant_algo": "NVFP4"},
                                  "model.language_model.lm_head": {"quant_algo": "NVFP4"}},
             "exclude_modules": ["model.visual.*", "mtp.*", "*.in_proj_a"]}
    _json(export / "config.json", {"model_type": "qwen3_5", "vision_config": {},
                                   "quantization_config": {**quant, "quant_method": "modelopt"}})
    _json(export / "hf_quant_config.json", {"producer": {"name": "modelopt"},
                                            "quantization": quant})
    _json(export / "provenance.json", {"git_sha": "1" * 40})
    _json(export / "tokenizer_config.json", {})
    (export / ".env").write_text("PRIVATE_PLACEHOLDER=1\n")
    return export, source


def _payloads(root):
    tensors, _ = ablation.read_checkpoint(root)
    return {name: (tensor.dtype, tensor.shape, ablation._stream((tensor,)))
            for name, tensor in tensors.items()}


@pytest.mark.parametrize("variant,restored", [
    ("control", set()), ("lm-head", {"lm_head"}),
    ("attention", {ATTN, LINEAR}), ("both", {"lm_head", ATTN, LINEAR}),
    ("mlp", set(MLP_PROJECTIONS)),
])
def test_variants_preserve_all_unselected_payloads_and_branches(checkpoints, tmp_path,
                                                               variant, restored):
    export, source = checkpoints
    before, source_before = _payloads(export), _payloads(source)
    output = tmp_path / variant
    summary = ablation.write_ablation(export, source, output, variant)
    after = _payloads(output)
    for name, info in before.items():
        module, suffix = name.rsplit(".", 1)
        if module in restored and suffix in ablation.SCALE_SUFFIXES:
            assert name not in after
        elif module in restored:
            key = "model.language_model.lm_head.weight" if module == "lm_head" else name
            assert after[name] == source_before[key]
        else:
            assert after[name] == info
    assert _payloads(export) == before
    assert _payloads(source) == source_before
    assert (output / "preserved.safetensors").read_bytes() == (
        export / "preserved.safetensors").read_bytes()
    assert not (output / ".env").exists()
    config = json.loads((output / "config.json").read_text())
    hf = json.loads((output / "hf_quant_config.json").read_text())["quantization"]
    layers = hf["quantized_layers"]
    assert config["quantization_config"]["quantized_layers"] == layers
    assert config["quantization_config"]["quant_algo"] == hf["quant_algo"] == "MIXED_PRECISION"
    assert VISION in layers and MTP in layers
    assert "model.language_model.norm" not in layers
    assert "model.language_model.lm_head" not in layers
    assert not restored.intersection(layers)
    assert not {"model.visual.*", "mtp.*"}.intersection(hf["exclude_modules"])
    assert summary["payloads_verified"] is True
    assert summary["restored_modules"] == len(restored)
    note = (output / "precision_ablation.json").read_text()
    assert str(export) not in note and str(source) not in note


def test_selected_never_matches_vision_mtp_or_small_bf16_gates():
    assert ablation.selected("lm_head", "lm-head")
    assert ablation.selected(ATTN, "attention")
    assert not ablation.selected("mtp.lm_head", "lm-head")
    assert not ablation.selected("model.mtp.layers.0.self_attn.q_proj", "attention")
    assert not ablation.selected(VISION, "attention")
    for module in MLP_PROJECTIONS:
        assert ablation.selected(module, "mlp")
        assert not ablation.selected(module, "attention")
    assert not ablation.selected("model.language_model.layers.0.mlp.other_proj", "mlp")
    assert not ablation.selected("model.visual.layers.0.mlp.gate_proj", "mlp")
    assert not ablation.selected("mtp.layers.0.mlp.gate_proj", "mlp")


def test_mlp_export_restores_only_language_mlp_and_updates_metadata(checkpoints, tmp_path):
    export, source = checkpoints
    output = tmp_path / "mlp"
    summary = ablation.write_ablation(export, source, output, "mlp")
    tensors, _ = ablation.read_checkpoint(output)
    for module in MLP_PROJECTIONS:
        weight = tensors[module + ".weight"]
        assert weight.dtype == "BF16"
        assert weight.shape == (2, 16)
        assert module + ".weight_scale" not in tensors
        assert module + ".input_scale" not in tensors
    assert MLP_PROJECTIONS[0] + ".weight" in tensors
    assert VISION + ".weight" in tensors and tensors[VISION + ".weight"].dtype == "U8"
    assert MTP + ".weight" in tensors and tensors[MTP + ".weight"].dtype == "U8"
    config = json.loads((output / "config.json").read_text())
    quant = config["quantization_config"]
    assert quant["quant_algo"] == "MIXED_PRECISION"
    assert all(module not in quant["quantized_layers"] for module in MLP_PROJECTIONS)
    assert VISION in quant["quantized_layers"] and MTP in quant["quantized_layers"]
    assert summary["restored_modules"] == 3


@pytest.mark.parametrize("projection", ("gate_proj", "up_proj", "down_proj"))
def test_mlp_projection_selector_restores_only_one_family(checkpoints, tmp_path, projection):
    export, source = checkpoints
    output = tmp_path / projection
    summary = ablation.write_ablation(
        export, source, output, "mlp", mlp_projections=[projection]
    )
    tensors, _ = ablation.read_checkpoint(output)
    for candidate in ("gate_proj", "up_proj", "down_proj"):
        module = f"model.language_model.layers.0.mlp.{candidate}"
        if candidate == projection:
            assert tensors[module + ".weight"].dtype == "BF16"
            assert module + ".weight_scale" not in tensors
        else:
            assert tensors[module + ".weight"].dtype == "U8"
            assert module + ".weight_scale" in tensors
    assert summary["restored_modules"] == 1
    assert summary["selection"] == {
        "mlp_projections": [projection], "layer_start": None, "layer_end": None,
    }


def test_mlp_layer_interval_is_half_open_and_rejects_empty_selection(checkpoints, tmp_path):
    export, source = checkpoints
    output = tmp_path / "band"
    summary = ablation.write_ablation(
        export, source, output, "mlp", layer_start=0, layer_end=1
    )
    assert summary["restored_modules"] == 3
    with pytest.raises(ValueError, match="matched no packed"):
        ablation.prepare_plan(export, source, "mlp", layer_start=1, layer_end=2)


@pytest.fixture
def multi_layer_checkpoints(tmp_path):
    export, source = tmp_path / "multi-mse", tmp_path / "multi-bf16"
    export.mkdir()
    source.mkdir()
    mlp = MLP_PROJECTIONS + MLP_LAYER1_PROJECTIONS + tuple(
        f"model.language_model.layers.2.mlp.{projection}"
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    edited = {
        name + suffix: value
        for name in mlp
        for suffix, value in (
            (".weight", ("U8", [2, 8])),
            (".weight_scale", ("F8_E4M3", [2, 1])),
            (".weight_scale_2", ("F32", [])),
            (".input_scale", ("F32", [])),
        )
    }
    preserved = _packed(VISION)
    _shard(export, "edited.safetensors", edited)
    _shard(export, "preserved.safetensors", preserved)
    _json(export / ablation.INDEX_NAME, {"weight_map": {
        **dict.fromkeys(edited, "edited.safetensors"),
        **dict.fromkeys(preserved, "preserved.safetensors"),
    }})
    _shard(source, "model.safetensors", {
        **{name + ".weight": ("BF16", [2, 16]) for name in mlp},
    })
    quant = {"quant_algo": "NVFP4", "group_size": 16,
             "quantized_layers": {VISION: {"quant_algo": "NVFP4"}}}
    _json(export / "config.json", {"model_type": "qwen3_5",
                                   "quantization_config": {**quant, "quant_method": "modelopt"}})
    _json(export / "hf_quant_config.json", {"producer": {"name": "modelopt"},
                                            "quantization": quant})
    _json(export / "provenance.json", {"git_sha": "1" * 40})
    return export, source


def test_mlp_layer_band_preserves_outside_and_unselected_projection_payloads(
    multi_layer_checkpoints, tmp_path
):
    export, source = multi_layer_checkpoints
    before = _payloads(export)
    source_before = _payloads(source)
    output = tmp_path / "band-write"
    plan = ablation.prepare_plan(
        export, source, "mlp", mlp_projections=["down_proj", "gate_proj"],
        layer_start=1, layer_end=2,
    )
    summary = ablation.write_ablation(
        export, source, output, "mlp", mlp_projections=["down_proj", "gate_proj"],
        layer_start=1, layer_end=2,
    )
    after = _payloads(output)
    selected_modules = {
        "model.language_model.layers.1.mlp.down_proj",
        "model.language_model.layers.1.mlp.gate_proj",
    }
    for name, info in before.items():
        module, suffix = name.rsplit(".", 1)
        if module in selected_modules and suffix == "weight":
            assert after[name] == source_before[name]
        elif module in selected_modules and suffix in ablation.SCALE_SUFFIXES:
            assert name not in after
        else:
            assert after[name] == info
    assert summary["restored_modules"] == 2
    assert summary["selection"] == {
        "mlp_projections": ["gate_proj", "down_proj"], "layer_start": 1, "layer_end": 2,
    }
    record = json.loads((output / "precision_ablation.json").read_text())
    assert record["selection"] == summary["selection"]
    assert plan["summary"]["selection"] == summary["selection"]
    assert _payloads(export) == before
    assert _payloads(source) == source_before


def test_mlp_layer_band_boundaries_and_canonical_projection_order(multi_layer_checkpoints):
    export, source = multi_layer_checkpoints
    summary = ablation.prepare_plan(
        export, source, "mlp", mlp_projections=["down_proj", "gate_proj"],
        layer_start=0, layer_end=1,
    )["summary"]
    assert summary["restored_modules"] == 2
    assert summary["selection"]["mlp_projections"] == ["gate_proj", "down_proj"]
    with pytest.raises(ValueError, match="matched no packed"):
        ablation.prepare_plan(export, source, "mlp", layer_start=3, layer_end=4)


@pytest.mark.parametrize("kwargs, message", [
    ({"layer_start": 0}, "supplied together"),
    ({"layer_start": 1, "layer_end": 1}, "0 <= start < end"),
    ({"layer_start": -1, "layer_end": 1}, "0 <= start < end"),
    ({"mlp_projections": ["gate_proj", "gate_proj"]}, "duplicates"),
    ({"mlp_projections": ["gate_proj", "other_proj"]}, "Unknown MLP projection"),
    ({"mlp_projections": [], "layer_start": 0, "layer_end": 1}, "cannot be empty"),
])
def test_mlp_selector_rejects_invalid_conditions(checkpoints, kwargs, message):
    export, source = checkpoints
    with pytest.raises(ValueError, match=message):
        ablation.prepare_plan(export, source, "mlp", **kwargs)


def test_mlp_selector_is_rejected_for_non_mlp_variant(checkpoints):
    export, source = checkpoints
    with pytest.raises(ValueError, match="require --variant mlp"):
        ablation.prepare_plan(export, source, "control", layer_start=0, layer_end=1)


@pytest.mark.parametrize("variant", ("control", "attention", "both", "lm-head"))
def test_selected_rejects_mlp_selectors_for_non_mlp_variant(variant):
    with pytest.raises(ValueError, match="require --variant mlp"):
        ablation.selected(ATTN, variant, mlp_projections=["gate_proj"])


def test_cli_passes_mlp_selector_to_dry_run(checkpoints, tmp_path, capsys):
    export, source = checkpoints
    assert ablation.main([
        "--export", str(export), "--bf16", str(source), "--variant", "mlp",
        "--mlp-projection", "up_proj", "--layer-start", "0", "--layer-end", "1",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["restored_modules"] == 1
    assert summary["selection"] == {
        "mlp_projections": ["up_proj"], "layer_start": 0, "layer_end": 1,
    }


def test_cli_write_preserves_selected_plan(multi_layer_checkpoints, tmp_path, capsys):
    export, source = multi_layer_checkpoints
    output = tmp_path / "cli-band"
    args = [
        "--export", str(export), "--bf16", str(source), "--variant", "mlp",
        "--mlp-projection", "down_proj", "--layer-start", "1", "--layer-end", "2",
    ]
    assert ablation.main(args) == 0
    planned = json.loads(capsys.readouterr().out)
    assert ablation.main([*args, "--write", "--output", str(output)]) == 0
    written = json.loads(capsys.readouterr().out)
    record = json.loads((output / "precision_ablation.json").read_text())
    assert planned["restored_modules"] == written["restored_modules"] == 1
    assert planned["selection"] == written["selection"] == record["selection"]


def test_dry_run_reads_headers_without_creating_output(checkpoints, tmp_path, capsys):
    export, source = checkpoints
    output = tmp_path / "new"
    assert ablation.main(["--export", str(export), "--bf16", str(source),
                         "--variant", "both", "--output", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["restored_modules"] == 3


def test_output_overlap_and_existing_output_are_rejected(checkpoints, tmp_path):
    export, source = checkpoints
    for output in (export, export / "child", source / "child", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            ablation.write_ablation(export, source, output, "both")
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        ablation.write_ablation(export, source, existing, "both")
    assert marker.read_bytes() == b"original"


def test_mismatched_source_shape_fails_before_any_output(checkpoints, tmp_path):
    export, source = checkpoints
    _shard(source, "model.safetensors", {"lm_head.weight": ("BF16", [2, 32])})
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="logical shape"):
        ablation.write_ablation(export, source, output, "lm-head")
    assert not output.exists()
    assert not list(tmp_path.glob(".bad.staging-*"))


def test_hf_ghost_key_and_escaping_shard_fail_closed(checkpoints):
    export, _ = checkpoints
    index = json.loads((export / ablation.INDEX_NAME).read_text())
    index["weight_map"]["ghost.weight"] = "edited.safetensors"
    _json(export / ablation.INDEX_NAME, index)
    with pytest.raises(ValueError, match="index/header disagreement"):
        ablation.read_checkpoint(export)
    index["weight_map"] = {"lm_head.weight": "../bf16/model.safetensors"}
    _json(export / ablation.INDEX_NAME, index)
    with pytest.raises(ValueError, match="Unsafe shard"):
        ablation.read_checkpoint(export)


def test_exception_cleans_staging_and_preserves_inputs(checkpoints, tmp_path, monkeypatch):
    export, source = checkpoints
    before = _payloads(export)

    def fail(*args):
        raise ValueError("injected metadata failure")

    monkeypatch.setattr(ablation, "_metadata", fail)
    with pytest.raises(ValueError, match="injected"):
        ablation.write_ablation(export, source, tmp_path / "failed", "both")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
    assert _payloads(export) == before


def test_explicit_fused_projection_uses_original_row_order(checkpoints, tmp_path):
    export, source = checkpoints
    base = "model.language_model.layers.0.self_attn"
    split = {f"{base}.{part}_proj.weight": ("BF16", [2, 16]) for part in ("q", "k", "v")}
    _shard(source, "model.safetensors", split)
    source_tensors, _ = ablation.read_checkpoint(source)
    parts = ablation._bf16_parts(base + ".qkv_proj.weight", (6, 16),
                                 {ablation.canonical(k): v for k, v in source_tensors.items()})
    target = tmp_path / "fused.safetensors"
    name = base + ".qkv_proj.weight"
    tensor = ablation.Tensor(parts[0].path, "BF16", (6, 16), parts[0].offset,
                             sum(p.nbytes for p in parts))
    ablation._write_shard(target, {name: (tensor, parts)})
    actual = ablation.read_header(target)[name]
    with target.open("rb") as handle:
        handle.seek(actual.offset)
        assert handle.read(actual.nbytes) == bytes([1]) * 64 + bytes([2]) * 64 + bytes([3]) * 64
