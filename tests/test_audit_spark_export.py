"""Exercise the public audit with small real safetensors files, without Torch."""

import importlib.util
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audit_spark_export", Path(__file__).parents[1] / "scripts" / "audit_spark_export.py"
)
audit_spark_export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_spark_export)
REVISION = "1" * 40


def _json(path, value):
    path.write_text(json.dumps(value))


def _checkpoint(root, tensors):
    root.mkdir(exist_ok=True)
    header, payload = {}, bytearray()
    for name, (dtype, shape, values) in tensors.items():
        start = len(payload)
        if values is None:
            payload.extend(bytes(math.prod(shape) * audit_spark_export.DTYPE_BYTES[dtype]))
        else:
            payload.extend(struct.pack("<" + "f" * len(values), *values))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    _json(
        root / "model.safetensors.index.json",
        {"weight_map": dict.fromkeys(tensors, "model.safetensors")},
    )


@pytest.fixture
def export_fixture(tmp_path):
    def create(scheme="nvfp4_w4a8"):
        source, export = tmp_path / "source", tmp_path / "export"
        source_weights = {
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [2, 16], None),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", [2, 16], None),
            "model.mtp.fc.weight": ("BF16", [2, 16], None),
        }
        exported = {}
        layers = {}
        for module in (
            "model.layers.0.self_attn.q_proj",
            "model.visual.blocks.0.attn.qkv_proj",
            "mtp.fc",
        ):
            fp8 = scheme == "nvfp4_w4a8" and "self_attn" in module
            exported[module + ".weight"] = ("F8_E4M3" if fp8 else "U8", [2, 16 if fp8 else 8], None)
            exported[module + ".weight_scale"] = ("F32", [2, 1], [1.0, 1.0])
            exported[module + ".input_scale"] = ("F32", [1], [1.0])
            if not fp8:
                exported[module + ".weight_scale_2"] = ("F32", [1], [1.0])
            layers[module] = {"quant_algo": "FP8" if fp8 else "NVFP4"}
        _checkpoint(source, source_weights)
        _checkpoint(export, exported)
        quant = {
            "quant_algo": "NVFP4" if scheme == "nvfp4_w4a4" else "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": layers,
        }
        _json(export / "hf_quant_config.json", {"quantization": quant})
        _json(
            export / "config.json",
            {"model_type": "qwen3_5", "vision_config": {}, "quantization_config": quant},
        )
        _json(
            export / "calibration_coverage.json",
            {
                "vision": {"model.visual.blocks.0.attn.qkv.input_quantizer": 128},
                "mtp": {"model.mtp.fc.input_quantizer": 256},
            },
        )
        _json(export / "vision_export.json", {"method": "modelopt-export", "restored_tensors": 0})
        _json(export / "mtp_export.json", {"method": "modelopt-export", "mtp_quantized": True})
        _json(export / "tokenizer_config.json", {})
        _json(export / "preprocessor_config.json", {})
        _json(
            export / "provenance.json",
            {"git_sha": REVISION, "scheme": scheme, "calibration": {"with_images": True}},
        )
        return (
            SimpleNamespace(
                export_dir=export, source_dir=source, scheme=scheme, expected_revision=REVISION
            ),
            source_weights,
            exported,
        )

    return create


@pytest.mark.parametrize("scheme", ["nvfp4_w4a8", "nvfp4_w4a4"])
def test_complete_quantized_language_vision_mtp_export_passes(export_fixture, scheme):
    args, _, _ = export_fixture(scheme)
    report = audit_spark_export.run(args)
    assert report["passed"], report["errors"]
    assert report["git_sha"] == report["expected_revision"] == REVISION
    assert report["source_tensor_counts"] == {"language": 1, "vision": 1, "mtp": 1}
    assert report["calibration_coverage"]["mtp"]["min_calls"] == 256
    assert report["scale_payload_scan"]["tensors_checked"] == (9 if scheme.endswith("4a4") else 8)


def test_revision_mismatch_fails_without_rewriting_record(export_fixture, capsys):
    args, _, _ = export_fixture()
    path = args.export_dir / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["git_sha"] = "2" * 40
    _json(path, provenance)
    before = path.read_bytes()
    result_path = args.export_dir / "audit.json"
    result = audit_spark_export.main(
        [
            str(args.export_dir),
            str(args.source_dir),
            args.scheme,
            "--expected-revision",
            REVISION,
            "--output-json",
            str(result_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(result_path.read_text())
    assert result == 1 and not report["passed"]
    assert report["git_sha"] == "2" * 40
    assert any("provenance revision mismatch" in error for error in report["errors"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("revision_args", [[], ["--expected-revision", "15561d8"]])
def test_cli_requires_explicit_full_revision(tmp_path, revision_args):
    with pytest.raises(SystemExit) as exc:
        audit_spark_export.main(
            [
                str(tmp_path),
                str(tmp_path),
                "nvfp4_w4a4",
                "--output-json",
                str(tmp_path / "x.json"),
                *revision_args,
            ]
        )
    assert exc.value.code == 2


def test_missing_source_language_tensor_is_detected(export_fixture):
    args, source, _ = export_fixture()
    source["model.norm.weight"] = ("BF16", [2], None)
    _checkpoint(args.source_dir, source)
    report = audit_spark_export.run(args)
    assert not report["passed"]
    assert any("source language tensor absent" in error for error in report["errors"])


def test_nvfp4_wrong_block_shape_is_detected(export_fixture):
    args, _, exported = export_fixture()
    exported["mtp.fc.weight_scale"] = ("F32", [2, 2], [1.0] * 4)
    _checkpoint(args.export_dir, exported)
    report = audit_spark_export.run(args)
    assert not report["passed"]
    assert any("not group 16: mtp.fc" in error for error in report["errors"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scale_payload_is_detected(export_fixture, value):
    args, _, exported = export_fixture()
    exported["mtp.fc.input_scale"] = ("F32", [1], [value])
    _checkpoint(args.export_dir, exported)
    report = audit_spark_export.run(args)
    assert not report["passed"]
    assert any("nonfinite/invalid scale tensor: mtp.fc.input_scale" in e for e in report["errors"])


@pytest.mark.parametrize("calls", [0, False, "256"])
def test_unvisited_or_invalid_mtp_calibration_is_detected(export_fixture, calls):
    args, _, _ = export_fixture()
    _json(
        args.export_dir / "calibration_coverage.json",
        {
            "vision": {"model.visual.blocks.0.attn.qkv.input_quantizer": 128},
            "mtp": {"model.mtp.fc.input_quantizer": calls},
        },
    )
    report = audit_spark_export.run(args)
    assert not report["passed"]
    assert "unvisited/invalid mtp input quantizers" in report["errors"]


def test_unsafe_index_path_fails_closed(export_fixture):
    args, _, _ = export_fixture()
    _json(
        args.export_dir / "model.safetensors.index.json",
        {"weight_map": {"mtp.fc.weight": "../outside.safetensors"}},
    )
    report = audit_spark_export.run(args)
    assert not report["passed"]
    assert any("unsafe shard filename" in error for error in report["errors"])
