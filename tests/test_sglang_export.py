"""SGLang MIXED_PRECISION export rewrite (no GPU / ModelOpt)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaquant.sglang_export import (
    is_sglang_mixed_scheme,
    quantized_layers_from_weight_map,
    rewrite_sglang_mixed_export,
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
