"""Rewrite ModelOpt HF exports so SGLang loads mixed NVFP4/FP8.

SGLang's ModelOpt loader accepts:

* ``quant_algo: NVFP4`` (uniform W4A4, group_size 16)
* ``quant_algo: MIXED_PRECISION`` plus a non-empty ``quantized_layers`` map
  (NVFP4 gs16 on MLP + ``lm_head``, FP8 on attention). Current SGLang main
  routes that map to ``modelopt_mixed``.

It **rejects** ``quant_algo: W4A8_NVFP4_FP8`` (uniform NVFP4 block 32 +
FP8 activations). This repo does not quantize to that tag.

ModelOpt 0.46 ``export_hf_checkpoint`` of a mixed quant_cfg often writes a
single ``NVFP4`` / ``W4A8_NVFP4_FP8`` tag. This module rebuilds the NVIDIA
card layout from the exported weight names.
"""

from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

SGLANG_MIXED_ALGOS = frozenset({"nvfp4_w4a8", "nvfp4_mixed", "nvfp4_w4a16_mixed"})

_NVFP4_LAYER = {"quant_algo": "NVFP4", "group_size": 16}
_W4A16_NVFP4_LAYER = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
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


def _normalize_quant_algo(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sglang_rejects_quant_algo(quant_algo: str | None) -> bool:
    """True for TRT-LLM uniform W4A8 tags (``W4A8_NVFP4_FP8`` / W4AFP8-style)."""
    if not quant_algo:
        return False
    key = quant_algo.strip().upper().replace("-", "_")
    compact = key.replace("_", "")
    if compact in {"W4A8NVFP4FP8", "W4AFP8"}:
        return True
    if "W4AFP8" in compact:
        return True
    return "W4A8" in compact and "NVFP4" in compact and "FP8" in compact


def _is_nvfp4_or_mixed_algo(quant_algo: str | None) -> bool:
    if not quant_algo or _sglang_rejects_quant_algo(quant_algo):
        return False
    key = quant_algo.strip().upper().replace("-", "_")
    compact = key.replace("_", "")
    return compact in {"NVFP4", "NVFP4AWQ", "MIXEDPRECISION", "MIXED"} or compact.startswith(
        "MIXED"
    )


def _quant_section(payload: dict[str, Any]) -> dict[str, Any]:
    quant = payload.get("quantization")
    if isinstance(quant, dict):
        return quant
    return payload


def inspect_sglang_quant_config(export_dir: str | Path) -> dict[str, Any]:
    """Read ``hf_quant_config.json`` and report whether SGLang can load it.

    Never downloads weights. Missing / unreadable config does not raise.

    Returns ``quant_algo``, ``has_quantized_layers``, ``sglang_ok``, ``warning``.
    ``sglang_ok`` is False only for TRT-LLM ``W4A8_NVFP4_FP8`` / W4AFP8-style tags.
    A bare ``NVFP4`` tag (uniform W4A4) stays ``sglang_ok`` True with a warning
    that mixed exports should run ``megaquant rewrite-sglang``.
    """
    path = Path(export_dir) / "hf_quant_config.json"
    quant_algo: str | None = None
    has_layers = False
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            raw = {}
        payload = raw if isinstance(raw, dict) else {}
        quant = _quant_section(payload)
        quant_algo = _normalize_quant_algo(quant.get("quant_algo"))
        if quant_algo is None:
            quant_algo = _normalize_quant_algo(payload.get("quant_algo"))
        layers = quant.get("quantized_layers")
        has_layers = isinstance(layers, dict) and bool(layers)

    sglang_ok = not _sglang_rejects_quant_algo(quant_algo)
    warning: str | None = None
    if not sglang_ok:
        warning = (
            f"SGLang rejects quant_algo={quant_algo} "
            "(uniform NVFP4 block 32 + FP8 activations). "
            "Do not rewrite these weights in place."
        )
    elif _is_nvfp4_or_mixed_algo(quant_algo) and not has_layers:
        warning = (
            f"{quant_algo} has no quantized_layers; run `megaquant rewrite-sglang` "
            "if this is mixed nvfp4_w4a8. Uniform NVFP4 W4A4 is OK for SGLang."
        )
    return {
        "quant_algo": quant_algo,
        "has_quantized_layers": has_layers,
        "sglang_ok": sglang_ok,
        "warning": warning,
    }


def sglang_quant_snapshot(model: str | None) -> dict[str, Any] | None:
    """Inspect an existing local export dir; None if ``model`` is missing / Hub-like."""
    if not model:
        return None
    path = Path(model)
    try:
        if not path.is_dir():
            return None
    except OSError:
        return None
    try:
        return inspect_sglang_quant_config(path)
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return None


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


def layer_entry_for_module(
    module: str, *, mlp_quant_algo: str = "NVFP4"
) -> dict[str, Any] | None:
    if _skipped(module):
        return None
    if any(_ends_with_tail(module, tail) for tail in _NVFP4_TAILS):
        if mlp_quant_algo == "W4A16_NVFP4":
            return dict(_W4A16_NVFP4_LAYER)
        return dict(_NVFP4_LAYER)
    if any(_ends_with_tail(module, tail) for tail in _FP8_TAILS):
        return dict(_FP8_LAYER)
    return None


def quantized_layers_from_weight_map(
    weight_map: dict[str, Any],
    *,
    mlp_quant_algo: str = "NVFP4",
    exported_layers: dict[str, Any] | None = None,
    tensor_metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Keep exported precision, including opted-in vision/MTP modules.

    The name-based map is only a fallback for older language-only exports.
    Actual BF16 weights must not acquire a quantized entry merely because a
    source config or a familiar module name says otherwise.
    """
    layers: dict[str, dict[str, Any]] = {}
    exported_layers = exported_layers or {}
    tensor_metadata = tensor_metadata or {}
    for tensor_name in weight_map:
        module = _module_from_tensor(str(tensor_name))
        if module is None:
            continue
        weight_name = module + ".weight"
        if weight_name not in weight_map:
            continue
        weight_info = tensor_metadata.get(weight_name, {})
        dtype = weight_info.get("dtype")
        if dtype in {"BF16", "F16", "F32", "F64"}:
            continue
        exported = exported_layers.get(module)
        if isinstance(exported, dict) and exported.get("quant_algo"):
            entry = dict(exported)
        else:
            entry = _entry_from_exported_tensors(module, weight_map, tensor_metadata)
            if (
                entry is not None
                and entry["quant_algo"] == "NVFP4"
                and mlp_quant_algo == "W4A16_NVFP4"
                and any(_ends_with_tail(module, tail) for tail in _NVFP4_TAILS)
            ):
                entry["quant_algo"] = mlp_quant_algo
            if entry is None:
                entry = layer_entry_for_module(module, mlp_quant_algo=mlp_quant_algo)
        if entry is None:
            continue
        layers[module] = entry
        if _ends_with_tail(module, "lm_head") and module != "lm_head":
            layers.setdefault("lm_head", dict(entry))
    return dict(sorted(layers.items()))


def _entry_from_exported_tensors(
    module: str, weight_map: dict[str, Any], tensor_metadata: dict[str, Any]
) -> dict[str, Any] | None:
    weight_info = tensor_metadata.get(module + ".weight", {})
    dtype = weight_info.get("dtype")
    if module + ".weight_scale" not in weight_map:
        return None
    if dtype in {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}:
        return dict(_FP8_LAYER)
    if module + ".weight_scale_2" not in weight_map or dtype not in {None, "U8"}:
        return None
    group_size = 16
    weight_shape = weight_info.get("shape", [])
    scale_shape = tensor_metadata.get(module + ".weight_scale", {}).get("shape", [])
    if weight_shape and scale_shape and scale_shape[-1]:
        # Packed FP4 stores two values per byte; one scale covers each group.
        width = int(weight_shape[-1]) * 2
        groups = int(scale_shape[-1])
        if width % groups:
            raise ValueError(f"Cannot infer NVFP4 group size for {module}")
        group_size = width // groups
    # Missing input_scale does not prove W4A16: an uncalibrated NVFP4 input
    # also has no scale. Preserve an explicit exported W4A16 entry above.
    return {"quant_algo": "NVFP4", "group_size": group_size}


def _read_tensor_metadata(root: Path, weight_map: dict[str, Any]) -> dict[str, Any]:
    """Read only safetensors headers, without loading multi-GB weight payloads."""
    tensors: dict[str, Any] = {}
    for filename in sorted({str(value) for value in weight_map.values()}):
        path = root / filename
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            header_size = int.from_bytes(prefix, "little")
            if header_size > 100_000_000 or header_size > path.stat().st_size - 8:
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(handle.read(header_size))
        for name, info in header.items():
            if name != "__metadata__" and isinstance(info, dict):
                tensors[name] = info
    return tensors


def _exclude_unquantized(patterns: list[str], layers: dict[str, Any]) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if not any(
            fnmatchcase(module, pattern) or module.startswith(pattern + ".")
            for module in layers
        )
    ]


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
    mlp_quant_algo: str = "NVFP4",
    exported_layers: dict[str, Any] | None = None,
    tensor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layers = quantized_layers_from_weight_map(
        weight_map,
        mlp_quant_algo=mlp_quant_algo,
        exported_layers=exported_layers,
        tensor_metadata=tensor_metadata,
    )
    if not layers:
        raise ValueError(
            "No NVFP4/FP8 layers found in the export weight map; "
            "cannot write SGLang MIXED_PRECISION metadata."
        )
    exclude = list(exclude_modules) if exclude_modules is not None else ["mtp*", "mtp.layers.0*"]
    exclude = _exclude_unquantized(exclude, layers)
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


def rewrite_sglang_mixed_export(
    export_dir: str | Path, *, mlp_quant_algo: str = "NVFP4"
) -> dict[str, Any]:
    """Patch ``hf_quant_config.json`` and ``config.json`` for SGLang mixed NVFP4.

    Returns the written ``quantized_layers`` map.
    """
    root = Path(export_dir)
    weight_map = load_weight_map(root)
    if not weight_map:
        raise FileNotFoundError(f"No safetensors weight map under {root}")
    config_quant = _read_json(root / "config.json").get("quantization_config")
    if isinstance(config_quant, dict):
        method = str(config_quant.get("quant_method") or "").replace("_", "-").lower()
        if method == "compressed-tensors":
            # Compressed-tensors stores NVFP4 as ``weight_packed``; a ModelOpt
            # layer map would drop every NVFP4 layer and relabel the loader.
            raise ValueError(
                f"{root} is a compressed-tensors export (llm-compressor). "
                "It is not a ModelOpt checkpoint and cannot be rewritten to "
                "MIXED_PRECISION. Re-quantize with backend: modelopt for SGLang "
                "modelopt_mixed."
            )

    existing = _read_json(root / "hf_quant_config.json")
    producer = {}
    if isinstance(existing.get("producer"), dict):
        producer = dict(existing["producer"])
    old_quant = _quant_section(existing)
    config_quant = config_quant if isinstance(config_quant, dict) else {}
    exclude = old_quant.get("exclude_modules", config_quant.get("exclude_modules"))
    kv = old_quant.get("kv_cache_quant_algo", config_quant.get("kv_cache_quant_algo", "FP8"))
    exported_layers: dict[str, Any] = {}
    for quant in (config_quant, old_quant):
        layers = quant.get("quantized_layers")
        if isinstance(layers, dict):
            exported_layers.update(layers)

    section = build_mixed_quant_section(
        weight_map=weight_map,
        producer=producer,
        kv_cache_quant_algo=kv,
        exclude_modules=list(exclude) if isinstance(exclude, list) else None,
        mlp_quant_algo=mlp_quant_algo,
        exported_layers=exported_layers,
        tensor_metadata=_read_tensor_metadata(root, weight_map),
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
        ignore = quant_config.get("ignore", section["exclude_modules"])
        quant_config["ignore"] = _exclude_unquantized(
            list(ignore) if isinstance(ignore, list) else section["exclude_modules"],
            section["quantized_layers"],
        )
    config["quantization_config"] = quant_config
    if config_path.is_file() or config:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return section["quantized_layers"]
