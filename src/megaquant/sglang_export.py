"""Rewrite ModelOpt HF exports so SGLang loads mixed NVFP4/FP8.

SGLang's ModelOpt loader accepts:

* ``quant_algo: NVFP4`` (uniform W4A4, group_size 16)
* ``quant_algo: MIXED_PRECISION`` plus a non-empty ``quantized_layers`` map
  (NVFP4 gs16 on MLP + ``lm_head``, FP8 on attention). Current SGLang main
  routes that map to ``modelopt_mixed``.

It **rejects** ``quant_algo: W4A8_NVFP4_FP8`` (uniform NVFP4 block 32 +
FP8 activations). That encoding stays TensorRT-LLM-only
(``w4a8_nvfp4_fp8``).

ModelOpt 0.46 ``export_hf_checkpoint`` of a mixed quant_cfg often writes a
single ``NVFP4`` / ``W4A8_NVFP4_FP8`` tag. This module rebuilds the NVIDIA
card layout from the exported weight names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SGLANG_MIXED_ALGOS = frozenset({"nvfp4_w4a8", "nvfp4_mixed"})

_NVFP4_LAYER = {"quant_algo": "NVFP4", "group_size": 16}
_FP8_LAYER = {"quant_algo": "FP8"}

_SKIP_SUBSTR = (
    "visual",
    "vision",
    "mtp",
    "embed_tokens",
    "embed_positions",
    "conv1d",
    "in_proj_a",
    "in_proj_b",
    "patch_embed",
    "merger",
)

_SCALE_MARKERS = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".input_global_scale",
    ".output_scale",
    ".amax",
    "inv_freq",
)

_NVFP4_TAILS = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "lm_head",
)

_FP8_TAILS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
)


def is_sglang_mixed_scheme(scheme_name: str | None) -> bool:
    key = (scheme_name or "").strip().lower().replace("-", "_")
    return key in SGLANG_MIXED_ALGOS


def _module_from_tensor(name: str) -> str | None:
    if any(marker in name for marker in _SCALE_MARKERS):
        return None
    if name.endswith(".weight"):
        return name[: -len(".weight")]
    if name.endswith(".bias"):
        return name[: -len(".bias")]
    return None


def _skipped(module: str) -> bool:
    lowered = module.lower()
    return any(token in lowered for token in _SKIP_SUBSTR)


def _ends_with_tail(module: str, tail: str) -> bool:
    return module == tail or module.endswith("." + tail)


def layer_entry_for_module(module: str) -> dict[str, Any] | None:
    if _skipped(module):
        return None
    if any(_ends_with_tail(module, tail) for tail in _NVFP4_TAILS):
        return dict(_NVFP4_LAYER)
    if any(_ends_with_tail(module, tail) for tail in _FP8_TAILS):
        return dict(_FP8_LAYER)
    return None


def quantized_layers_from_weight_map(weight_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers: dict[str, dict[str, Any]] = {}
    for tensor_name in weight_map:
        module = _module_from_tensor(str(tensor_name))
        if module is None:
            continue
        entry = layer_entry_for_module(module)
        if entry is None:
            continue
        layers[module] = entry
        if _ends_with_tail(module, "lm_head") and module != "lm_head":
            layers.setdefault("lm_head", dict(_NVFP4_LAYER))
    return dict(sorted(layers.items()))


def load_weight_map(export_dir: str | Path) -> dict[str, Any]:
    root = Path(export_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if isinstance(weight_map, dict):
            return weight_map
    single = root / "model.safetensors"
    if single.is_file():
        try:
            from safetensors import safe_open
        except ImportError:
            return {}
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    return {}


def build_mixed_quant_section(
    *,
    weight_map: dict[str, Any],
    producer: dict[str, Any] | None = None,
    kv_cache_quant_algo: str | None = "FP8",
    exclude_modules: list[str] | None = None,
) -> dict[str, Any]:
    layers = quantized_layers_from_weight_map(weight_map)
    if not layers:
        raise ValueError(
            "No NVFP4/FP8 layers found in the export weight map; "
            "cannot write SGLang MIXED_PRECISION metadata."
        )
    exclude = list(exclude_modules) if exclude_modules else ["mtp*", "mtp.layers.0*"]
    section: dict[str, Any] = {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": kv_cache_quant_algo,
        "quantized_layers": layers,
        "exclude_modules": exclude,
    }
    return section


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def rewrite_sglang_mixed_export(export_dir: str | Path) -> dict[str, Any]:
    """Patch ``hf_quant_config.json`` and ``config.json`` for SGLang mixed NVFP4.

    Returns the written ``quantized_layers`` map.
    """
    root = Path(export_dir)
    weight_map = load_weight_map(root)
    if not weight_map:
        raise FileNotFoundError(f"No safetensors weight map under {root}")

    existing = _read_json(root / "hf_quant_config.json")
    producer = {}
    if isinstance(existing.get("producer"), dict):
        producer = dict(existing["producer"])
    raw_quant = existing.get("quantization")
    old_quant = raw_quant if isinstance(raw_quant, dict) else {}
    exclude = old_quant.get("exclude_modules") if isinstance(old_quant, dict) else None
    kv = old_quant.get("kv_cache_quant_algo", "FP8") if isinstance(old_quant, dict) else "FP8"
    if not kv:
        kv = "FP8"

    section = build_mixed_quant_section(
        weight_map=weight_map,
        producer=producer,
        kv_cache_quant_algo=kv,
        exclude_modules=list(exclude) if isinstance(exclude, list) else None,
    )
    hf_payload = {
        "producer": producer or {"name": "modelopt", "version": "unknown"},
        "quantization": section,
    }
    (root / "hf_quant_config.json").write_text(
        json.dumps(hf_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    config_path = root / "config.json"
    config = _read_json(config_path)
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        quant_config = {}
    quant_config["quant_method"] = "modelopt"
    quant_config["quant_algo"] = "MIXED_PRECISION"
    quant_config["kv_cache_quant_algo"] = kv
    quant_config["quantized_layers"] = section["quantized_layers"]
    if "exclude_modules" in section:
        quant_config["exclude_modules"] = section["exclude_modules"]
        quant_config.setdefault("ignore", section["exclude_modules"])
    config["quantization_config"] = quant_config
    if config_path.is_file() or config:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return section["quantized_layers"]
