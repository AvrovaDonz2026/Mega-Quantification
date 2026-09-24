"""SGLang MIXED_PRECISION export rewrite (no GPU / ModelOpt)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaquant.sglang_export import (
    inspect_sglang_quant_config,
    is_sglang_mixed_scheme,
    quantized_layers_from_weight_map,
    rewrite_sglang_mixed_export,
    sglang_quant_snapshot,
)

QWEN38_WEIGHT_MAP = {
    "model.language_model.layers.0.mlp.gate_proj.weight": "a.safetensors",
    "model.language_model.layers.0.mlp.up_proj.weight": "a.safetensors",
    "model.language_model.layers.0.mlp.down_proj.weight": "a.safetensors",
    "model.language_model.layers.0.mlp.gate_proj.weight_scale": "a.safetensors",
    "model.language_model.layers.0.self_attn.q_proj.weight": "a.safetensors",
    "model.language_model.layers.0.self_attn.k_proj.weight": "a.safetensors",
    "model.language_model.layers.0.self_attn.v_proj.weight": "a.safetensors",
    "model.language_model.layers.0.self_attn.o_proj.weight": "a.safetensors",
    "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": "b.safetensors",
    "model.language_model.layers.1.linear_attn.in_proj_z.weight": "b.safetensors",
    "model.language_model.layers.1.linear_attn.out_proj.weight": "b.safetensors",
    "model.language_model.layers.1.linear_attn.conv1d.weight": "b.safetensors",
    "model.language_model.layers.1.linear_attn.in_proj_a.weight": "b.safetensors",
    "model.language_model.embed_tokens.weight": "a.safetensors",
    "visual.patch_embed.proj.weight": "c.safetensors",
    "mtp.layers.0.mlp.gate_proj.weight": "d.safetensors",
    "lm_head.weight": "e.safetensors",
}


def test_is_sglang_mixed_scheme() -> None:
    assert is_sglang_mixed_scheme("nvfp4_w4a8")
    assert is_sglang_mixed_scheme("nvfp4-mixed")
    assert not is_sglang_mixed_scheme("w4a8_nvfp4_fp8")
    assert not is_sglang_mixed_scheme("nvfp4_w4a4")


def test_quantized_layers_map_matches_nvidia_card() -> None:
    layers = quantized_layers_from_weight_map(QWEN38_WEIGHT_MAP)
    assert layers["model.language_model.layers.0.mlp.gate_proj"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert layers["model.language_model.layers.0.self_attn.q_proj"] == {
        "quant_algo": "FP8",
    }
    assert layers["model.language_model.layers.1.linear_attn.in_proj_qkv"] == {
        "quant_algo": "FP8",
    }
    assert layers["lm_head"] == {"quant_algo": "NVFP4", "group_size": 16}
    joined = " ".join(layers)
    assert "visual" not in joined
    assert "embed_tokens" not in joined
    assert "conv1d" not in joined
    assert "in_proj_a" not in joined
    assert "mtp" not in joined


def test_rewrite_writes_mixed_precision(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": QWEN38_WEIGHT_MAP}),
        encoding="utf-8",
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": "modelopt", "version": "0.46.1"},
                "quantization": {"quant_algo": "NVFP4", "group_size": 16},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    layers = rewrite_sglang_mixed_export(tmp_path)
    hf = json.loads((tmp_path / "hf_quant_config.json").read_text())
    cfg = json.loads((tmp_path / "config.json").read_text())
    quant = hf["quantization"]
    assert quant["quant_algo"] == "MIXED_PRECISION"
    assert quant["kv_cache_quant_algo"] == "FP8"
    assert quant["quantized_layers"] == layers
    assert "mtp*" in quant["exclude_modules"]
    assert cfg["quantization_config"]["quant_algo"] == "MIXED_PRECISION"
    assert cfg["quantization_config"]["quant_method"] == "modelopt"
    assert cfg["quantization_config"]["quantized_layers"]["lm_head"]["quant_algo"] == "NVFP4"


def test_rewrite_requires_weight_map(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rewrite_sglang_mixed_export(tmp_path)


def test_rewrite_sglang_cli(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": QWEN38_WEIGHT_MAP}),
        encoding="utf-8",
    )
    (tmp_path / "hf_quant_config.json").write_text("{}\n", encoding="utf-8")
    from megaquant.cli import main

    code = main(["rewrite-sglang", str(tmp_path)])
    assert code == 0
    hf = json.loads((tmp_path / "hf_quant_config.json").read_text())
    assert hf["quantization"]["quant_algo"] == "MIXED_PRECISION"


def _write_compressed_tensors_export(root: Path) -> dict[str, bytes]:
    prefix = "model.language_model.layers.0"
    weight_map = {
        f"{prefix}.mlp.down_proj.weight_packed": "a.safetensors",
        f"{prefix}.mlp.down_proj.weight_global_scale": "a.safetensors",
        f"{prefix}.self_attn.o_proj.weight": "a.safetensors",
        f"{prefix}.self_attn.o_proj.weight_scale": "a.safetensors",
        "lm_head.weight_packed": "a.safetensors",
    }
    files = {
        "model.safetensors.index.json": json.dumps({"weight_map": weight_map}),
        "config.json": json.dumps(
            {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {}}}
        ),
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return {name: (root / name).read_bytes() for name in files}


def test_rewrite_refuses_compressed_tensors_export(tmp_path: Path) -> None:
    before = _write_compressed_tensors_export(tmp_path)
    with pytest.raises(ValueError, match="compressed-tensors"):
        rewrite_sglang_mixed_export(tmp_path)
    assert {name: (tmp_path / name).read_bytes() for name in before} == before
    assert not (tmp_path / "hf_quant_config.json").exists()


def test_rewrite_sglang_cli_reports_compressed_tensors_cleanly(tmp_path: Path, capsys) -> None:
    _write_compressed_tensors_export(tmp_path)
    from megaquant.cli import main

    assert main(["rewrite-sglang", str(tmp_path)]) == 1
    assert "compressed-tensors" in capsys.readouterr().err


def test_auto_llmcompressor_fallback_notes_mixed_export(monkeypatch) -> None:
    from megaquant import pipeline
    from megaquant.config import Recipe

    recipe = Recipe.model_validate(
        {
            "name": "auto-mixed",
            "scheme": "nvfp4_w4a8",
            "model": {"source": "dummy/model"},
            "calibration": {},
            "export": {"output_dir": "outputs/x"},
        }
    )
    monkeypatch.setattr(pipeline, "list_backends", lambda: ["llmcompressor"])
    monkeypatch.setattr(pipeline, "get_backend", lambda name: object())
    monkeypatch.setattr(pipeline, "_backend_supports", lambda backend, scheme: True)
    notes: list[str] = []
    assert pipeline._pick_backend(recipe, notes) == "llmcompressor"
    assert any("compressed-tensors" in note for note in notes)


def _write_hf_quant(tmp_path: Path, quantization: dict) -> None:
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": quantization}),
        encoding="utf-8",
    )


def test_inspect_mixed_precision_with_layers_is_sglang_ok(tmp_path: Path) -> None:
    _write_hf_quant(
        tmp_path,
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "lm_head": {"quant_algo": "NVFP4", "group_size": 16},
            },
        },
    )
    info = inspect_sglang_quant_config(tmp_path)
    assert info["quant_algo"] == "MIXED_PRECISION"
    assert info["has_quantized_layers"] is True
    assert info["sglang_ok"] is True
    assert info["warning"] is None
    assert sglang_quant_snapshot(str(tmp_path)) == info


def test_inspect_w4a8_nvfp4_fp8_not_sglang_ok(tmp_path: Path) -> None:
    _write_hf_quant(tmp_path, {"quant_algo": "W4A8_NVFP4_FP8", "group_size": 32})
    info = inspect_sglang_quant_config(tmp_path)
    assert info["quant_algo"] == "W4A8_NVFP4_FP8"
    assert info["sglang_ok"] is False
    assert info["has_quantized_layers"] is False
    w4afp8 = tmp_path / "w4afp8"
    w4afp8.mkdir()
    _write_hf_quant(w4afp8, {"quant_algo": "W4AFP8"})
    assert inspect_sglang_quant_config(w4afp8)["sglang_ok"] is False


def test_inspect_nvfp4_without_quantized_layers_warns(tmp_path: Path) -> None:
    _write_hf_quant(tmp_path, {"quant_algo": "NVFP4", "group_size": 16})
    info = inspect_sglang_quant_config(tmp_path)
    assert info["quant_algo"] == "NVFP4"
    assert info["has_quantized_layers"] is False
    assert info["sglang_ok"] is True
    assert info["warning"] is not None
    assert "rewrite-sglang" in info["warning"]
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_hf_quant(mixed, {"quant_algo": "MIXED_PRECISION"})
    mixed_info = inspect_sglang_quant_config(mixed)
    assert mixed_info["sglang_ok"] is True
    assert mixed_info["warning"] is not None
    assert "rewrite-sglang" in mixed_info["warning"]


def test_inspect_missing_config_does_not_crash(tmp_path: Path) -> None:
    info = inspect_sglang_quant_config(tmp_path)
    assert info["quant_algo"] is None
    assert info["has_quantized_layers"] is False
    assert info["sglang_ok"] is True
    assert info["warning"] is None
    missing = inspect_sglang_quant_config(tmp_path / "does-not-exist")
    assert missing["sglang_ok"] is True
    (tmp_path / "hf_quant_config.json").write_text("not-json", encoding="utf-8")
    broken = inspect_sglang_quant_config(tmp_path)
    assert broken["sglang_ok"] is True
    assert sglang_quant_snapshot("/ckpt") is None
    assert sglang_quant_snapshot("Qwen/Qwen3.8-27B") is None
