"""NVIDIA ModelOpt backend — first-class path for NVFP4 W4A8 (weights NVFP4, activations FP8).

This module is importable without ``nvidia-modelopt`` installed. All ``modelopt``
imports are lazy. ``describe_cfg()`` never needs ModelOpt. ``build_quant_cfg()``
falls back to documented dict construction when a preset attribute is missing.
``quantize()`` / ``export()`` raise ``BackendError`` if ModelOpt is absent.

Scheme → ModelOpt mapping
-------------------------
====================== ================================= ===================
Our scheme             ModelOpt object                   qformat
====================== ================================= ===================
``nvfp4_w4a8``         mixed (SGLang)                    ``mixed_nvfp4_fp8``
``nvfp4_mixed``        mixed (NVIDIA quality)            ``mixed_nvfp4_fp8``
``nvfp4_w4a4``         ``mtq.NVFP4_DEFAULT_CFG``         ``nvfp4``
``nvfp4``              alias of ``nvfp4_w4a4``
``nvfp4_w4a16``        ``mtq.W4A16_NVFP4_CFG``           ``w4a16_nvfp4``
``w4a16_nvfp4``        alias of ``nvfp4_w4a16``
``nvfp4_w4a16_mixed``  mixed W4A16 MLP + FP8 attn   ``mixed_w4a16_nvfp4``
``fp8_w8a8``           ``mtq.FP8_DEFAULT_CFG``           ``fp8``
``fp8``                alias of ``fp8_w8a8``
====================== ================================= ===================

Default ``nvfp4_w4a8`` is the **SGLang-serving** encoding: NVFP4 group_size 16
on MLP + ``lm_head``, FP8 on self-attn + linear-attn (same map as
``nvidia/Qwen3.8-27B-NVFP4``). Export rewrites ``hf_quant_config.json`` to
``quant_algo=MIXED_PRECISION`` plus ``quantized_layers``.

SGLang rejects ModelOpt ``quant_algo=W4A8_NVFP4_FP8`` (uniform NVFP4 block 32
+ FP8 activations). This backend does not build that config. A loaded
``W4A8_NVFP4_FP8_CFG`` is only a fallback base for the mixed schemes below,
and those paths immediately rewrite MLP + ``lm_head`` to NVFP4 group_size 16.

``nvfp4_mixed`` expression
--------------------------
Deepcopy ``NVFP4_DEFAULT_CFG`` (fallback: ``W4A8_NVFP4_FP8_CFG``) then append
later ``quant_cfg`` entries so they override earlier wildcards. Always force
NVIDIA ``nvidia/Qwen3.8-27B-NVFP4`` mixed precision — **never** leave W4A8
block-size 32 on MLP even if the loaded preset was ``W4A8_NVFP4_FP8_CFG``:

1. FP8 on ``*self_attn*weight_quantizer`` / ``*self_attn*input_quantizer``
2. FP8 on ``*linear_attn*weight_quantizer`` / ``*linear_attn*input_quantizer``
3. NVFP4 block-size **16** (``_NVFP4_BS16``) on ``*mlp*`` weight/input
4. Re-enable ``*lm_head*`` with the same NVFP4 bs16 weight+input —
   ModelOpt presets disable ``*lm_head*`` by default
5. Disable vision / mtp / conv1d / in_proj_a / in_proj_b / embed unless the
   recipe opts in via ``quantize_vision`` / ``quantize_mtp``

Hessian ``block_size`` is **16** (matches NVFP4 gs16), not 32.

List-style ``quant_cfg`` (ModelOpt YAML presets) and legacy dict-style
``quant_cfg`` (v0.48-era inline dicts) are both mutated after ``deepcopy``.

KV cache: ``recipe.kv_cache`` in ``{fp8, nvfp4}`` is merged as
``FP8_KV_CFG`` / ``NVFP4_KV_CFG`` targeting ``*[kv]_bmm_quantizer``. Other
values are left unchanged (not faked). Cast-mode ``use_constant_amax`` is
only used if the loaded KV preset already contains it.

Fallback when a named ``mtq.*_CFG`` attribute is missing (documented structure)::

    {
      "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": <nvfp4_bs32|nvfp4|fp8>},
        {"quantizer_name": "*input_quantizer", "cfg": <fp8|nvfp4|enable False>},
        {"quantizer_name": "*lm_head*", "enable": False},
      ],
      "algorithm": "max",
    }

Numeric attrs used in fallbacks (tuple form accepted by ModelOpt v0.48+)::

    nvfp4_bs32 = {"num_bits": (2, 1),
                  "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)}}
    nvfp4_bs16 = {"num_bits": (2, 1),
                  "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    fp8        = {"num_bits": (4, 3), "axis": None}
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from megaquant.backends.base import forward_loop_from_iter

try:
    from megaquant.exceptions import BackendError
except ImportError:  # Core exceptions.py may not exist yet.

    class BackendError(RuntimeError):  # type: ignore[no-redef]
        """Raised when the ModelOpt backend cannot run."""


SCHEME_ALIASES: dict[str, str] = {
    "nvfp4": "nvfp4_w4a4",
    "w4a16_nvfp4": "nvfp4_w4a16",
    "w4a16_nvfp4_fp8_attn": "nvfp4_w4a16_mixed",
    "fp8": "fp8_w8a8",
}

CANONICAL_SCHEMES: frozenset[str] = frozenset(
    {
        "nvfp4_w4a8",
        "nvfp4_mixed",
        "nvfp4_w4a4",
        "nvfp4_w4a16",
        "nvfp4_w4a16_mixed",
        "fp8_w8a8",
    }
)

_SCHEME_QFORMAT: dict[str, str] = {
    "nvfp4_w4a8": "mixed_nvfp4_fp8",
    "nvfp4_mixed": "mixed_nvfp4_fp8",
    "nvfp4_w4a4": "nvfp4",
    "nvfp4_w4a16": "w4a16_nvfp4",
    "nvfp4_w4a16_mixed": "mixed_w4a16_nvfp4",
    "fp8_w8a8": "fp8",
}

_SCHEME_CFG_NAME: dict[str, str] = {
    "nvfp4_w4a8": "NVFP4_DEFAULT_CFG+mixed_overrides",
    "nvfp4_mixed": "NVFP4_DEFAULT_CFG+mixed_overrides",
    "nvfp4_w4a4": "NVFP4_DEFAULT_CFG",
    "nvfp4_w4a16": "W4A16_NVFP4_CFG",
    "nvfp4_w4a16_mixed": "NVFP4_DEFAULT_CFG+w4a16_mixed_overrides",
    "fp8_w8a8": "FP8_DEFAULT_CFG",
}

_PRESET_ATTR: dict[str, tuple[str, ...]] = {
    "nvfp4_w4a8": ("NVFP4_DEFAULT_CFG", "W4A8_NVFP4_FP8_CFG"),
    "nvfp4_mixed": ("NVFP4_DEFAULT_CFG", "W4A8_NVFP4_FP8_CFG"),
    "nvfp4_w4a4": ("NVFP4_DEFAULT_CFG",),
    "nvfp4_w4a16": ("W4A16_NVFP4_CFG", "NVFP4_MLP_WEIGHT_ONLY_CFG"),
    "nvfp4_w4a16_mixed": ("NVFP4_DEFAULT_CFG", "W4A8_NVFP4_FP8_CFG"),
    "fp8_w8a8": ("FP8_DEFAULT_CFG",),
}

_SGLANG_MIXED = frozenset({"nvfp4_w4a8", "nvfp4_mixed"})
# MLP weights are NVFP4 group 16 with BF16 activations (W4A16_NVFP4).
_SGLANG_W4A16_MIXED = frozenset({"nvfp4_w4a16_mixed"})
_SGLANG_MIXED_EXPORT = _SGLANG_MIXED | _SGLANG_W4A16_MIXED

# ModelOpt numeric fallbacks (v0.48 tuple form). YAML presets use "e2m1"/"e4m3".
_NVFP4_BS16: dict[str, Any] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
}
_NVFP4_BS32: dict[str, Any] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (4, 3)},
}
_FP8_ATTR: dict[str, Any] = {"num_bits": (4, 3), "axis": None}

_VISION_IGNORE = ("*visual*", "*vision*")
_EMBED_IGNORE = ("*embed_tokens*", "*embed_positions*", "*embed*")
_GDN_IGNORE = (
    "*linear_attn.conv1d*",
    "*linear_attn.in_proj_a*",
    "*linear_attn.in_proj_b*",
)
_MTP_IGNORE = ("*mtp*",)

_MIXED_ATTN_PATTERNS = ("*self_attn*", "*linear_attn*")
_MIXED_NVFP4_PATTERNS = ("*mlp*", "*lm_head*")


@contextmanager
def _auxiliary_calibration_coverage(model: Any, recipe: Any):
    """Fail closed if enabled vision/MTP activation quantizers never execute."""
    model_spec = _attr(recipe, "model", None)
    required = []
    if _flag(model_spec, "quantize_vision", False):
        required.append("vision")
    if _flag(model_spec, "quantize_mtp", False):
        required.append("mtp")
    counts: dict[str, int] = {}
    groups: dict[str, list[str]] = {name: [] for name in required}
    handles = []

    def record(name: str):
        def hook(_module: Any, _args: Any, _output: Any) -> None:
            counts[name] += 1
        return hook

    try:
        if required:
            for name, module in model.named_modules():
                group = "mtp" if "mtp" in name else "vision" if any(
                    part in name for part in ("visual", "vision")
                ) else None
                if group not in groups or not name.endswith("input_quantizer"):
                    continue
                if not getattr(module, "is_enabled", False):
                    continue
                counts[name] = 0
                groups[group].append(name)
                handles.append(module.register_forward_hook(record(name)))
        yield
        for group, names in groups.items():
            missing = [name for name in names if counts[name] == 0]
            if not names or missing:
                detail = ", ".join(missing[:5]) or "no enabled input quantizers"
                raise BackendError(f"{group} activation calibration did not run: {detail}")
        if required:
            report = {group: {name: counts[name] for name in names}
                      for group, names in groups.items()}
            model._megaquant_calibration_coverage = report
            print("[megaquant] auxiliary calibration coverage: " + ", ".join(
                f"{group}={len(names)}" for group, names in groups.items()
            ), flush=True)
    finally:
        for handle in handles:
            handle.remove()


def canonicalize_scheme(scheme_name: str) -> str:
    key = (scheme_name or "").strip().lower().replace("-", "_")
    return SCHEME_ALIASES.get(key, key)


def describe_cfg(scheme_name: str, algorithm: str = "max") -> dict[str, Any]:
    """JSON-serializable scheme summary for dry-run. Does not import ModelOpt."""
    canonical = canonicalize_scheme(scheme_name)
    if canonical not in CANONICAL_SCHEMES:
        return {
            "scheme": canonical or scheme_name,
            "supported": False,
            "error": f"Unsupported ModelOpt scheme: {scheme_name!r}",
        }

    group_size: int | None = None if canonical == "fp8_w8a8" else 16

    summary: dict[str, Any] = {
        "scheme": canonical,
        "alias_of": scheme_name if canonicalize_scheme(scheme_name) != scheme_name else None,
        "supported": True,
        "modelopt_cfg": _SCHEME_CFG_NAME[canonical],
        "qformat": _SCHEME_QFORMAT[canonical],
        "group_size": group_size,
        "algorithm": algorithm or "max",
        "requires_modelopt": True,
    }
    if canonical in _SGLANG_W4A16_MIXED:
        summary.update(
            {
                "weight_format": "nvfp4+fp8",
                "activation_format": "bf16+fp8",
                "mlp_weight_format": "nvfp4",
                "mlp_activation_format": "bf16",
                "mlp_group_size": 16,
                "mlp_quant_algo": "W4A16_NVFP4",
                "attn_weight_format": "fp8",
                "attn_activation_format": "fp8",
                "sglang": "MIXED_PRECISION + quantized_layers (modelopt_mixed)",
                "targets": {
                    "nvfp4_w4a16": list(_MIXED_NVFP4_PATTERNS),
                    "fp8": list(_MIXED_ATTN_PATTERNS),
                    "disabled": [
                        "*visual*",
                        "*vision*",
                        "*embed*",
                        "*linear_attn.conv1d*",
                        "*linear_attn.in_proj_a*",
                        "*linear_attn.in_proj_b*",
                        "*mtp*",
                    ],
                },
                "notes": (
                    "Optional Marlin export, not the DGX Spark fast path. "
                    "NVFP4 group_size 16 weights and BF16 activations on *mlp* "
                    "and *lm_head* (quant_algo W4A16_NVFP4); FP8 weights and "
                    "activations on *self_attn* and *linear_attn*. Published "
                    "Spark tok/s use the mixed W4A4 checkpoint (nvfp4_mixed / "
                    "nvfp4_w4a8). Not uniform nvfp4_w4a16. Export writes "
                    "MIXED_PRECISION."
                ),
            }
        )
    elif canonical in _SGLANG_MIXED:
        summary.update(
            {
                "weight_format": "nvfp4+fp8",
                "activation_format": "nvfp4+fp8",
                "mlp_weight_format": "nvfp4",
                "mlp_activation_format": "nvfp4",
                "mlp_group_size": 16,
                "attn_weight_format": "fp8",
                "attn_activation_format": "fp8",
                "sglang": "MIXED_PRECISION + quantized_layers (modelopt_mixed)",
                "targets": {
                    "nvfp4": list(_MIXED_NVFP4_PATTERNS),
                    "fp8": list(_MIXED_ATTN_PATTERNS),
                    "disabled": [
                        "*visual*",
                        "*vision*",
                        "*embed*",
                        "*linear_attn.conv1d*",
                        "*linear_attn.in_proj_a*",
                        "*linear_attn.in_proj_b*",
                        "*mtp*",
                    ],
                },
                "notes": (
                    "SGLang-serving mixed NVFP4/FP8 (nvidia/Qwen3.8-27B-NVFP4 map): "
                    "NVFP4 group_size 16 on *mlp* and *lm_head*; FP8 on *self_attn* "
                    "and *linear_attn*. Export writes MIXED_PRECISION + "
                    "quantized_layers. SGLang rejects ModelOpt quant_algo "
                    "W4A8_NVFP4_FP8 (NVFP4 block 32)."
                ),
            }
        )
    elif canonical == "nvfp4_w4a4":
        summary.update(
            {
                "weight_format": "nvfp4",
                "activation_format": "nvfp4",
                "notes": "NVFP4_DEFAULT_CFG: NVFP4 W4A4, block size 16.",
            }
        )
    elif canonical == "nvfp4_w4a16":
        summary.update(
            {
                "weight_format": "nvfp4",
                "activation_format": "bf16",
                "notes": (
                    "W4A16_NVFP4_CFG if present, else NVFP4 weight-only "
                    "(enable *weight_quantizer, disable *input_quantizer), block 16."
                ),
            }
        )
    elif canonical == "fp8_w8a8":
        summary.update(
            {
                "weight_format": "fp8",
                "activation_format": "fp8",
                "notes": "FP8_DEFAULT_CFG: per-tensor FP8 E4M3 W8A8.",
            }
        )
    else:
        summary.update(
            {
                "weight_format": "nvfp4+fp8",
                "activation_format": "nvfp4+fp8",
                "mlp_weight_format": "nvfp4",
                "mlp_activation_format": "nvfp4",
                "mlp_group_size": 16,
                "attn_weight_format": "fp8",
                "attn_activation_format": "fp8",
                "targets": {
                    "nvfp4": list(_MIXED_NVFP4_PATTERNS),
                    "fp8": list(_MIXED_ATTN_PATTERNS),
                    "disabled": [
                        "*visual*",
                        "*vision*",
                        "*embed*",
                        "*linear_attn.conv1d*",
                        "*linear_attn.in_proj_a*",
                        "*linear_attn.in_proj_b*",
                        "*mtp*",
                    ],
                },
                "notes": (
                    "Deepcopy NVFP4_DEFAULT_CFG (fallback W4A8_NVFP4_FP8_CFG). "
                    "Always force NVFP4 group_size 16 on *mlp* and *lm_head* "
                    "(weights+inputs); FP8 on *self_attn* and *linear_attn*. "
                    "nvidia/Qwen3.8-27B-NVFP4 uses NVFP4 group_size 16 on "
                    "MLP+lm_head and FP8 on attention. Later disable "
                    "vision/mtp/conv1d/in_proj_a/in_proj_b/embed. lm_head is "
                    "re-enabled after ModelOpt's default disable."
                ),
            }
        )
    return summary


def _recipe_of(plan: Any) -> Any:
    return getattr(plan, "recipe", plan)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _flag(obj: Any, name: str, default: bool = False) -> bool:
    value = _attr(obj, name, default)
    return bool(value) if value is not None else default


def _scheme_of(plan: Any, recipe: Any) -> str:
    return str(_attr(recipe, "scheme", None) or _attr(plan, "scheme", "") or "")


def _algorithm_of(plan: Any, recipe: Any) -> str:
    raw = _attr(recipe, "algorithm", None) or _attr(plan, "algorithm", None) or "max"
    return str(raw)


def _import_mtq() -> Any | None:
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        return None
    return mtq


def _mtq_preset(mtq: Any, *names: str) -> tuple[dict[str, Any], str] | None:
    if mtq is None:
        return None
    for name in names:
        value = getattr(mtq, name, None)
        if isinstance(value, dict) and "quant_cfg" in value:
            return copy.deepcopy(value), name
    return None


def _append_entry(cfg: dict[str, Any], entry: dict[str, Any]) -> None:
    quant_cfg = cfg.setdefault("quant_cfg", [])
    if isinstance(quant_cfg, list):
        quant_cfg.append(copy.deepcopy(entry))
        return
    if isinstance(quant_cfg, dict):
        pattern = entry.get("quantizer_name") or "*"
        payload: dict[str, Any] = {}
        if entry.get("cfg") is not None:
            payload.update(copy.deepcopy(entry["cfg"]))
        if "enable" in entry:
            payload["enable"] = entry["enable"]
        elif "cfg" in entry:
            payload.setdefault("enable", True)
        quant_cfg[pattern] = payload
        return
    raise BackendError(f"Unsupported ModelOpt quant_cfg type: {type(quant_cfg)!r}")


def _disable_pattern(cfg: dict[str, Any], pattern: str) -> None:
    _append_entry(cfg, {"quantizer_name": pattern, "enable": False})


def _enable_pattern(cfg: dict[str, Any], pattern: str, attr: Any) -> None:
    # QuantizerCfgEntry.enable defaults True, but later disable entries must be
    # overridden by an explicit enable=True (list order, last match wins).
    if isinstance(attr, str):
        _append_entry(cfg, {"quantizer_name": pattern, "cfg": attr, "enable": True})
        return
    if isinstance(attr, list):
        _append_entry(
            cfg, {"quantizer_name": pattern, "cfg": copy.deepcopy(attr), "enable": True}
        )
        return
    _append_entry(
        cfg,
        {
            "quantizer_name": pattern,
            "cfg": copy.deepcopy(dict(attr)),
            "enable": True,
        },
    )


def _find_list_cfg(quant_cfg: list[Any], name: str) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for entry in quant_cfg:
        if isinstance(entry, Mapping) and entry.get("quantizer_name") == name:
            found = dict(entry)
    return found


def _weight_input_attrs(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Extract the effective global *weight_quantizer / *input_quantizer attrs."""
    quant_cfg = cfg.get("quant_cfg")
    weight: dict[str, Any] | None = None
    input_attr: dict[str, Any] | None = None
    input_disabled = False
    if isinstance(quant_cfg, list):
        w_entry = _find_list_cfg(quant_cfg, "*weight_quantizer")
        i_entry = _find_list_cfg(quant_cfg, "*input_quantizer")
        if w_entry and w_entry.get("cfg"):
            raw = w_entry["cfg"]
            weight = copy.deepcopy(raw[0] if isinstance(raw, list) else raw)
        if i_entry:
            if i_entry.get("enable") is False:
                input_disabled = True
            elif i_entry.get("cfg"):
                raw = i_entry["cfg"]
                input_attr = copy.deepcopy(raw[0] if isinstance(raw, list) else raw)
    elif isinstance(quant_cfg, dict):
        w_raw = quant_cfg.get("*weight_quantizer")
        i_raw = quant_cfg.get("*input_quantizer")
        if isinstance(w_raw, list) and w_raw:
            weight = copy.deepcopy(w_raw[0])
        elif isinstance(w_raw, dict):
            weight = copy.deepcopy(w_raw)
        if isinstance(i_raw, dict):
            if i_raw.get("enable") is False and set(i_raw) <= {"enable"}:
                input_disabled = True
            else:
                input_attr = copy.deepcopy(i_raw)
    if weight is None:
        weight = copy.deepcopy(_NVFP4_BS32)
    if isinstance(weight, dict):
        weight = {k: v for k, v in weight.items() if k != "enable"}
    if isinstance(input_attr, dict):
        input_attr = {k: v for k, v in input_attr.items() if k != "enable"}
    if input_disabled:
        return weight, None
    return weight, input_attr


def _fallback_uniform(
    weight_attr: dict[str, Any],
    input_attr: dict[str, Any] | None,
    *,
    algorithm: str | dict[str, Any] = "max",
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": copy.deepcopy(weight_attr)},
    ]
    if input_attr is None:
        entries.append({"quantizer_name": "*input_quantizer", "enable": False})
    else:
        entries.append({"quantizer_name": "*input_quantizer", "cfg": copy.deepcopy(input_attr)})
    for pattern in (
        "*lm_head*",
        "*linear_attn.conv1d*",
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
        "*block_sparse_moe.gate*",
        "*router*",
        "*mlp.gate.*",
        "*visual*",
        "*vision*",
        "*mtp*",
    ):
        entries.append({"quantizer_name": pattern, "enable": False})
    return {"quant_cfg": entries, "algorithm": algorithm}


def _fallback_for_scheme(canonical: str) -> dict[str, Any]:
    if canonical == "nvfp4_w4a4":
        return _fallback_uniform(_NVFP4_BS16, _NVFP4_BS16)
    if canonical == "nvfp4_w4a16":
        return _fallback_uniform(_NVFP4_BS16, None)
    if canonical == "nvfp4_w4a16_mixed":
        return _fallback_uniform(_NVFP4_BS16, _NVFP4_BS16)
    if canonical == "fp8_w8a8":
        return _fallback_uniform(_FP8_ATTR, _FP8_ATTR)
    # nvfp4_w4a8 / nvfp4_mixed (and any unknown canonical): NVFP4 W4A4
    # bs16, then _apply_mixed overlays FP8 attention + NVFP4 gs16 MLP.
    return _fallback_uniform(_NVFP4_BS16, _NVFP4_BS16)


def _load_base_cfg(mtq: Any, canonical: str) -> tuple[dict[str, Any], str]:
    names = _PRESET_ATTR[canonical]
    loaded = _mtq_preset(mtq, *names)
    if loaded is not None:
        cfg, name = loaded
        if canonical == "nvfp4_w4a16" and name != "W4A16_NVFP4_CFG":
            weight, _ = _weight_input_attrs(cfg)
            return (
                _fallback_uniform(weight, None),
                f"W4A16_NVFP4_CFG(fallback_from_{name})",
            )
        return cfg, name
    return _fallback_for_scheme(canonical), _SCHEME_CFG_NAME[canonical] + "(constructed)"


def _hessian_block_size(_canonical: str) -> int:
    # LocalHessianCalibConfig.block_size matches NVFP4 group_size 16.
    return 16


def _apply_algorithm(cfg: dict[str, Any], algorithm: str, canonical: str = "") -> None:
    key = algorithm.strip().lower().replace("-", "_")
    if key in {"", "max", "max_calib"}:
        cfg["algorithm"] = "max"
        return
    if key == "local_hessian":
        cfg["algorithm"] = {
            "method": "local_hessian",
            "fp8_scale_sweep": True,
            "block_size": _hessian_block_size(canonical),
        }
        return
    if key == "mse":
        cfg["algorithm"] = {"method": "mse"}
        if canonical.startswith("nvfp4"):
            # Use NVIDIA's NVFP4 MSE preset's FP8 block-scale sweep instead of
            # generic amax multiplier search; leave other formats' MSE intact.
            cfg["algorithm"]["fp8_scale_sweep"] = True
        return
    cfg["algorithm"] = algorithm


def _collect_ignore(plan: Any, recipe: Any) -> list[str]:
    patterns: list[str] = []

    def add(items: Any) -> None:
        if not items:
            return
        for pattern in items:
            if pattern and pattern not in patterns:
                patterns.append(str(pattern))

    add(_attr(plan, "ignore", None))
    add(_attr(recipe, "extra_ignore", None))
    add(_attr(recipe, "ignore", None))

    model = _attr(recipe, "model", None)
    if not _flag(model, "quantize_vision", False):
        add(_VISION_IGNORE)
    add(_EMBED_IGNORE)
    add(_GDN_IGNORE)
    if not _flag(model, "quantize_mtp", False):
        add(_MTP_IGNORE)
    return patterns


def _lm_head_ignored(ignore: list[str]) -> bool:
    return any("lm_head" in pattern for pattern in ignore)


def _apply_ignores(cfg: dict[str, Any], ignore: list[str]) -> None:
    for pattern in ignore:
        _disable_pattern(cfg, pattern)


def _apply_opt_in_enables(cfg: dict[str, Any], recipe: Any) -> None:
    """Opt in projection weights/inputs without enabling KV/output quantizers."""
    model = _attr(recipe, "model", None)
    patterns: list[str] = []
    if _flag(model, "quantize_vision", False):
        patterns.extend(("*visual*", "*vision*", "*vision_tower*", "*vision_model*"))
    if _flag(model, "quantize_mtp", False):
        patterns.append("*mtp*")
    for pattern in patterns:
        for suffix in ("weight_quantizer", "input_quantizer"):
            _append_entry(cfg, {"quantizer_name": pattern + suffix, "enable": True})


def _reenable_lm_head(
    cfg: dict[str, Any],
    weight_attr: Mapping[str, Any],
    input_attr: Mapping[str, Any] | None,
) -> None:
    _enable_pattern(cfg, "*lm_head*weight_quantizer", weight_attr)
    if input_attr is None:
        _disable_pattern(cfg, "*lm_head*input_quantizer")
    else:
        _enable_pattern(cfg, "*lm_head*input_quantizer", input_attr)


def _apply_mixed(cfg: dict[str, Any], used_cfg_name: str = "") -> None:
    """Force NVIDIA mixed precision: NVFP4 gs16 on MLP+lm_head, FP8 on attn.

    ``used_cfg_name`` is ignored for quantizer attrs. Even if the base preset
    was ``W4A8_NVFP4_FP8_CFG`` (bs32 weights + FP8 inputs), MLP and lm_head
    are rewritten to ``_NVFP4_BS16`` so the cfg matches
    ``nvidia/Qwen3.8-27B-NVFP4``.
    """
    del used_cfg_name
    nvfp4 = copy.deepcopy(_NVFP4_BS16)
    fp8 = copy.deepcopy(_FP8_ATTR)
    for pattern in _MIXED_ATTN_PATTERNS:
        _enable_pattern(cfg, f"{pattern}weight_quantizer", fp8)
        _enable_pattern(cfg, f"{pattern}input_quantizer", fp8)
    _enable_pattern(cfg, "*mlp*weight_quantizer", nvfp4)
    _enable_pattern(cfg, "*mlp*input_quantizer", nvfp4)
    _reenable_lm_head(cfg, nvfp4, nvfp4)


def _apply_mixed_w4a16(cfg: dict[str, Any]) -> None:
    """NVFP4 weight-only MLP + lm_head, FP8 attention.

    Activations on the MLP stay BF16 so decode does not pay an FP4 activation
    quant on the wide GEMMs. SGLang reads those layers as ``W4A16_NVFP4``.
    """
    nvfp4 = copy.deepcopy(_NVFP4_BS16)
    fp8 = copy.deepcopy(_FP8_ATTR)
    for pattern in _MIXED_ATTN_PATTERNS:
        _enable_pattern(cfg, f"{pattern}weight_quantizer", fp8)
        _enable_pattern(cfg, f"{pattern}input_quantizer", fp8)
    _enable_pattern(cfg, "*mlp*weight_quantizer", nvfp4)
    _disable_pattern(cfg, "*mlp*input_quantizer")
    _reenable_lm_head(cfg, nvfp4, None)


def _precision_to_attr(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    fmt = _attr(spec, "format", None)
    if fmt in {None, "bf16", "fp16", "none"}:
        return None
    group_size = _attr(spec, "group_size", None)
    if fmt == "fp8":
        return copy.deepcopy(_FP8_ATTR)
    if fmt == "nvfp4":
        return copy.deepcopy(_NVFP4_BS32 if group_size == 32 else _NVFP4_BS16)
    if fmt == "mxfp4":
        return {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        }
    if fmt == "int4":
        return {"num_bits": 4, "block_sizes": {-1: group_size or 128}}
    return None


def _apply_groups(cfg: dict[str, Any], recipe: Any) -> None:
    groups = _attr(recipe, "groups", None)
    if not groups:
        return
    for group in groups:
        targets = _attr(group, "targets", None) or []
        weight_attr = _precision_to_attr(_attr(group, "weights", None))
        act_spec = _attr(group, "activations", None)
        act_attr = _precision_to_attr(act_spec) if act_spec is not None else None
        for target in targets:
            target_s = str(target)
            if weight_attr is None:
                _disable_pattern(cfg, f"{target_s}*weight_quantizer")
            else:
                _enable_pattern(cfg, f"{target_s}*weight_quantizer", weight_attr)
            if act_spec is None:
                continue
            if act_attr is None:
                _disable_pattern(cfg, f"{target_s}*input_quantizer")
            else:
                _enable_pattern(cfg, f"{target_s}*input_quantizer", act_attr)
        for pattern in _attr(group, "ignore", None) or []:
            _disable_pattern(cfg, str(pattern))


def _kv_entries_from_fragment(fragment: Any) -> list[dict[str, Any]]:
    if fragment is None:
        return []
    if isinstance(fragment, list):
        return [copy.deepcopy(e) for e in fragment if isinstance(e, Mapping)]
    if not isinstance(fragment, Mapping):
        return []
    inner = fragment.get("quant_cfg", fragment)
    if isinstance(inner, list):
        return [copy.deepcopy(e) for e in inner if isinstance(e, Mapping)]
    if isinstance(inner, dict):
        entries: list[dict[str, Any]] = []
        for key, value in inner.items():
            if key == "default":
                continue
            payload = copy.deepcopy(value) if isinstance(value, dict) else {"enable": bool(value)}
            cfg_body = {k: v for k, v in payload.items() if k != "enable"}
            entry: dict[str, Any] = {"quantizer_name": key}
            if cfg_body:
                entry["cfg"] = cfg_body
            if "enable" in payload:
                entry["enable"] = payload["enable"]
            entries.append(entry)
        return entries
    return []


def _merge_kv_fragment(cfg: dict[str, Any], fragment: Any) -> None:
    for entry in _kv_entries_from_fragment(fragment):
        _append_entry(cfg, entry)


def _apply_kv_cache(cfg: dict[str, Any], recipe: Any, mtq: Any) -> None:
    kv = _attr(recipe, "kv_cache", None)
    if kv in {None, "", "none", "bf16", "fp16", False}:
        return
    kv_s = str(kv).strip().lower()
    if kv_s not in {"fp8", "nvfp4"}:
        return

    preset_name = "FP8_KV_CFG" if kv_s == "fp8" else "NVFP4_KV_CFG"
    fragment: Any = None
    if mtq is not None:
        fragment = getattr(mtq, preset_name, None)
        updater = getattr(mtq, "update_quant_cfg_with_kv_cache_quant", None)
        if updater is None:
            utils = getattr(mtq, "utils", None)
            updater = getattr(utils, "update_quant_cfg_with_kv_cache_quant", None)
        if updater is not None and fragment is not None:
            kv_inner = copy.deepcopy(fragment)
            if isinstance(kv_inner, dict) and "quant_cfg" in kv_inner:
                kv_payload = kv_inner["quant_cfg"]
            else:
                kv_payload = kv_inner
            try:
                merged = updater(cfg, kv_payload)
            except (TypeError, AttributeError, ValueError):
                merged = None
            if merged is cfg:
                return
            if isinstance(merged, dict) and "quant_cfg" in merged:
                cfg.clear()
                cfg.update(merged)
                return

    if fragment is not None:
        _merge_kv_fragment(cfg, copy.deepcopy(fragment))
        return

    attr = copy.deepcopy(_FP8_ATTR) if kv_s == "fp8" else copy.deepcopy(_NVFP4_BS16)
    _enable_pattern(cfg, "*[kv]_bmm_quantizer", attr)


def _output_dir(recipe: Any) -> Path:
    export = _attr(recipe, "export", None)
    raw = _attr(export, "output_dir", None)
    if raw is None:
        raw = _attr(recipe, "output_dir", None)
    if not raw:
        raise BackendError("Recipe has no export.output_dir")
    return Path(str(raw))


def _call_export_hf(export_fn: Any, model: Any, output_dir: Path) -> None:
    """Call ``export_hf_checkpoint`` without passing the path as ``dtype``.

    Current ModelOpt signature is ``(model, dtype=None, export_dir=...)``.
    A positional ``export_fn(model, path)`` would bind the directory to
    ``dtype`` and write to ``tempfile.gettempdir()``.
    """
    path = str(output_dir)
    try:
        export_fn(model, export_dir=path)
        return
    except TypeError:
        pass
    try:
        params = inspect.signature(export_fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "export_dir" in params:
        export_fn(model, export_dir=path)
        return
    export_fn(model, dtype=None, export_dir=path)


EXPORT_MIN_FREE_GIB = 8


def _is_cuda_oom(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    compact = blob.replace("_", "").replace(" ", "")
    return "outofmemory" in compact or "cudaoom" in compact


def _cuda_free_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return None
    try:
        if not cuda.is_available():
            return None
        free, _total = cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def _clear_partial_export(output_dir: Path) -> None:
    """Drop incomplete ModelOpt shard parts so a retry does not mix files."""
    for path in output_dir.glob("__shard_part_*"):
        try:
            path.unlink()
        except OSError:
            continue


def _module_to_cpu(module: Any) -> bool:
    to_fn = getattr(module, "to", None)
    if callable(to_fn):
        try:
            to_fn("cpu")
            return True
        except Exception:
            pass
    moved = False
    params = getattr(module, "parameters", None)
    if callable(params):
        for tensor in params(recurse=False):
            data = getattr(tensor, "data", None)
            if data is None:
                continue
            try:
                tensor.data = data.detach().to("cpu")
                moved = True
            except Exception:
                continue
    return moved


def _has_accelerate_offload(model: Any) -> bool:
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for _name, param in named_parameters():
            if getattr(getattr(param, "device", None), "type", None) == "meta":
                return True
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for _name, module in named_modules():
            hook = getattr(module, "_hf_hook", None)
            if getattr(hook, "offload", False):
                return True
            if any(getattr(item, "offload", False) for item in getattr(hook, "hooks", ())):
                return True
    return False


def _prepare_qwen35_export(model: Any) -> bool:
    if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
        return False
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    except (ImportError, AttributeError) as exc:
        raise BackendError(
            "Qwen3.5 export requires the Transformers 5.8-5.14 Qwen3.5 API; "
            "install 'transformers>=5.8,<5.15'"
        ) from exc

    gdn_cls = getattr(qwen35, "Qwen3_5GatedDeltaNet", None)
    if not isinstance(gdn_cls, type):
        raise BackendError(
            "Qwen3.5 export cannot find Qwen3_5GatedDeltaNet in Transformers; "
            "install 'transformers>=5.8,<5.15'"
        )

    def torch_fallback(name: str) -> Any:
        function = getattr(qwen35, name, None)
        if not callable(function):
            raise BackendError(
                f"Qwen3.5 export requires transformers.models.qwen3_5.{name}; "
                "this Transformers version has an incompatible GDN API. "
                "Install 'transformers>=5.8,<5.15'."
            )
        return function

    chunk_fallback = torch_fallback("torch_chunk_gated_delta_rule")
    recurrent_fallback = torch_fallback("torch_recurrent_gated_delta_rule")
    gdn_modules = [module for module in model.modules() if isinstance(module, gdn_cls)]
    for module in gdn_modules:
        # Transformers >= 5.15 dispatches through module-level functions.
        # Adding the old instance attributes there would silently do nothing.
        # Validate every layer before changing any layer or config.
        missing = [name for name in (
            "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
            "causal_conv1d_fn", "causal_conv1d_update",
        ) if not hasattr(module, name)]
        if missing:
            raise BackendError(
                "Qwen3.5 export needs the Transformers 5.8-5.14 instance GDN API; "
                f"{type(module).__name__} is missing {missing}. "
                "Install 'transformers>=5.8,<5.15'."
            )

    inner = getattr(model, "model", None)
    for module in (model, inner, getattr(inner, "language_model", None)):
        config = getattr(module, "config", None)
        if config is not None and getattr(config, "architectures", None) is None:
            config.architectures = [type(module).__name__]
    for module in gdn_modules:
        # ModelOpt's export tracing must use the deterministic PyTorch GDN
        # path.  The default may be a FLA/Hub kernel whose API differs across
        # Transformers releases and whose Triton kernels cannot be traced.
        module.chunk_gated_delta_rule = chunk_fallback
        module.recurrent_gated_delta_rule = recurrent_fallback
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = None
    return True


@contextmanager
def _qwen35_export_l2norm(enabled: bool):
    if not enabled:
        yield
        return
    try:
        import fla.modules.l2norm as l2norm
        import torch
    except ImportError:
        yield
        return

    original = l2norm.l2norm_fwd

    def torch_l2norm(x: Any, eps: float = 1e-6, output_dtype: Any = None) -> Any:
        # FLA's L2NormFunction unpacks ``(y, rstd)`` and its backward path
        # consumes the saved reciprocal norm.  Keep the calculation in
        # fp32, matching the Triton implementation, and restore the requested
        # output dtype for y while retaining fp32 rstd.
        x_float = x.float()
        rstd = torch.rsqrt((x_float * x_float).sum(dim=-1, keepdim=True) + eps)
        result = x_float * rstd
        result = result.to(output_dtype) if output_dtype is not None else result.to(x.dtype)
        return result, rstd.squeeze(-1)

    l2norm.l2norm_fwd = torch_l2norm
    try:
        yield
    finally:
        l2norm.l2norm_fwd = original


def _prepare_export_memory(model: Any, min_free_gib: int = EXPORT_MIN_FREE_GIB) -> str:
    """Free GPU workspace so NVFP4 pack (``_cast_fp4``) can allocate ~5 GiB.

    Preserve Accelerate's offload hooks: its streaming exporter materializes
    layers on demand, while moving those modules to CPU breaks dummy forward.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return "cpu-only"
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return "cpu-only"
    try:
        if not cuda.is_available():
            return "cpu-only"
        cuda.synchronize()
        cuda.empty_cache()
    except Exception:
        return "cpu-only"
    gc.collect()
    if _has_accelerate_offload(model) or type(model).__name__ == "Qwen3_5ForConditionalGeneration":
        return "preserve-offload"
    free = _cuda_free_bytes()
    need = max(1, int(min_free_gib)) * 1024**3
    if free is None or free >= need:
        return "cuda"
    free_before = free
    named = getattr(model, "named_modules", None)
    modules: list[Any] = []
    if callable(named):
        try:
            modules = [mod for _name, mod in named()]
        except Exception:
            modules = []
    if not modules:
        modules = [model]
    moved = False
    for module in reversed(modules):
        free = _cuda_free_bytes()
        if free is None or free >= need:
            break
        moved = _module_to_cpu(module) or moved
        try:
            cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    free = _cuda_free_bytes()
    if free is not None and free >= need:
        return "cuda-freed"
    if moved and free is not None and free > free_before:
        return "partial-cpu"
    return "cuda-unchanged"


def _restore_mtp_export(output_dir: Path, model: Any, recipe: Any) -> dict[str, Any] | None:
    """Copy ignored BF16 ``mtp.*`` tensors back into the Hugging Face export."""
    from megaquant.mtp_export import restore_bf16_mtp_from_recipe

    return restore_bf16_mtp_from_recipe(output_dir, recipe, model=model)


def _rewrite_sglang_export(
    output_dir: Path, canonical: str
) -> dict[str, dict[str, Any]] | None:
    """Patch ModelOpt HF metadata so SGLang ``modelopt_mixed`` can load it."""
    if canonical not in _SGLANG_MIXED_EXPORT:
        return None
    try:
        from megaquant.sglang_export import rewrite_sglang_mixed_export
    except ImportError as exc:
        raise BackendError(
            "megaquant.sglang_export is required to write SGLang MIXED_PRECISION metadata"
        ) from exc
    mlp_quant_algo = "W4A16_NVFP4" if canonical in _SGLANG_W4A16_MIXED else "NVFP4"
    try:
        return rewrite_sglang_mixed_export(output_dir, mlp_quant_algo=mlp_quant_algo)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise BackendError(
            f"SGLang MIXED_PRECISION rewrite failed under {output_dir}: {exc}"
        ) from exc


class ModelOptBackend:
    """NVIDIA ModelOpt PTQ backend (NVFP4 W4A8 first-class path)."""

    name = "modelopt"

    def available(self) -> bool:
        try:
            import modelopt.torch.quantization  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, scheme_name: str) -> bool:
        return canonicalize_scheme(scheme_name) in CANONICAL_SCHEMES

    def describe_cfg(self, scheme_name: str, algorithm: str = "max") -> dict[str, Any]:
        return describe_cfg(scheme_name, algorithm=algorithm)

    def build_quant_cfg(self, plan: Any) -> dict[str, Any]:
        recipe = _recipe_of(plan)
        canonical = canonicalize_scheme(_scheme_of(plan, recipe))
        if canonical not in CANONICAL_SCHEMES:
            raise BackendError(
                f"ModelOpt backend does not support scheme {_scheme_of(plan, recipe)!r}"
            )

        mtq = _import_mtq()
        cfg, used_name = _load_base_cfg(mtq, canonical)
        cfg = copy.deepcopy(cfg)

        if canonical in _SGLANG_MIXED:
            _apply_mixed(cfg, used_name)
        elif canonical in _SGLANG_W4A16_MIXED:
            _apply_mixed_w4a16(cfg)

        ignore = _collect_ignore(plan, recipe)
        _apply_opt_in_enables(cfg, recipe)
        # Embeddings, patch embedding convolutions and explicit user ignores
        # must still win when the containing vision/MTP block is opted in.
        _apply_ignores(cfg, ignore)

        if canonical not in _SGLANG_MIXED_EXPORT and not _lm_head_ignored(ignore):
            weight_attr, input_attr = _weight_input_attrs(cfg)
            _reenable_lm_head(cfg, weight_attr, input_attr)

        _apply_groups(cfg, recipe)
        _apply_kv_cache(cfg, recipe, mtq)
        # Algorithm after KV merge so a replacement quant_cfg cannot drop it.
        _apply_algorithm(cfg, _algorithm_of(plan, recipe), canonical)
        return cfg

    def quantize(self, model: Any, plan: Any, calib_iter: Any) -> Any:
        mtq = _import_mtq()
        if mtq is None:
            raise BackendError(
                "nvidia-modelopt is not installed; cannot quantize with the ModelOpt backend. "
                "Install with: pip install megaquant[modelopt]"
            )
        cfg = copy.deepcopy(self.build_quant_cfg(plan))
        recipe = _recipe_of(plan)
        run_batches = forward_loop_from_iter(calib_iter)

        def forward_loop(calibration_model: Any) -> None:
            from contextlib import nullcontext

            from megaquant.mtp_model import mtp_calibration

            mtp_context = mtp_calibration(calibration_model) if _flag(
                _attr(recipe, "model", None), "quantize_mtp", False
            ) else nullcontext()
            with _auxiliary_calibration_coverage(calibration_model, recipe), mtp_context:
                run_batches(calibration_model)

        return mtq.quantize(model, cfg, forward_loop=forward_loop)

    def export(self, model: Any, recipe: Any, tokenizer: Any = None) -> Path:
        try:
            from modelopt.torch.export import export_hf_checkpoint
        except ImportError as exc:
            raise BackendError(
                "nvidia-modelopt is not installed; cannot export with the ModelOpt backend. "
                "Install with: pip install megaquant[modelopt]"
            ) from exc

        output_dir = _output_dir(recipe)
        output_dir.mkdir(parents=True, exist_ok=True)
        coverage = getattr(model, "_megaquant_calibration_coverage", None)
        if coverage is not None:
            (output_dir / "calibration_coverage.json").write_text(
                json.dumps(coverage, indent=2) + "\n"
            )
        qwen35 = _prepare_qwen35_export(model)
        export_memory_kind = _prepare_export_memory(model)
        import torch

        export_oom_message = None
        try:
            with _qwen35_export_l2norm(qwen35), torch.inference_mode():
                _call_export_hf(export_hf_checkpoint, model, output_dir)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            # Only retain text. Leaving the except block releases the failed
            # export traceback and its temporary tensors before recovery.
            export_oom_message = f"{type(exc).__name__}: {exc}"

        if export_oom_message is not None:
            _clear_partial_export(output_dir)
            if export_memory_kind == "preserve-offload":
                raise BackendError(
                    "ModelOpt HF export ran out of CUDA memory while preserving "
                    "Qwen3.5/Accelerate model placement; refusing to repeat the same export. "
                    "Increase MEGAQUANT_GPU_HEADROOM_GIB or lower the GPU budget in "
                    "MEGAQUANT_MAX_MEMORY before loading the model. "
                    f"Original error: {export_oom_message}"
                )

            retry_memory_kind = _prepare_export_memory(
                model, min_free_gib=max(EXPORT_MIN_FREE_GIB, 12)
            )
            if retry_memory_kind not in {"cuda-freed", "partial-cpu"}:
                raise BackendError(
                    "ModelOpt HF export ran out of CUDA memory and no additional GPU "
                    f"memory could be released (preparation={retry_memory_kind}); "
                    "refusing to repeat the same export. Increase MEGAQUANT_GPU_HEADROOM_GIB "
                    "or lower the GPU budget in MEGAQUANT_MAX_MEMORY before loading "
                    f"the model. Original error: {export_oom_message}"
                )
            with _qwen35_export_l2norm(qwen35), torch.inference_mode():
                _call_export_hf(export_hf_checkpoint, model, output_dir)

        if tokenizer is not None:
            save = getattr(tokenizer, "save_pretrained", None)
            if callable(save):
                save(str(output_dir))

        from megaquant.vision_export import restore_vision_from_recipe

        vision_note = restore_vision_from_recipe(output_dir, recipe)
        mtp_note = _restore_mtp_export(output_dir, model, recipe)
        canonical = canonicalize_scheme(str(_attr(recipe, "scheme", "") or ""))
        sglang_layers = _rewrite_sglang_export(output_dir, canonical)
        meta = {
            "backend": self.name,
            "qformat": _SCHEME_QFORMAT.get(canonical, canonical),
            "algorithm": _algorithm_of(None, recipe),
            "modelopt_cfg": _SCHEME_CFG_NAME.get(canonical, canonical),
            "scheme": canonical or _attr(recipe, "scheme", None),
            "model_source": _attr(_attr(recipe, "model", None), "source", None),
        }
        if sglang_layers is not None:
            meta["sglang_quant_algo"] = "MIXED_PRECISION"
            meta["sglang_quantized_layers"] = len(sglang_layers)
        if vision_note is not None:
            meta["vision_tensors"] = vision_note["vision_tensors"]
        if mtp_note is not None:
            meta["mtp_tensors"] = mtp_note["mtp_tensors"]
        (output_dir / "backend_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_dir


def _self_register() -> None:
    try:
        from megaquant.registry import register_backend
    except ImportError:
        return
    backend = ModelOptBackend()
    try:
        params = list(inspect.signature(register_backend).parameters)
    except (TypeError, ValueError):
        params = ["name", "backend"]
    try:
        if len(params) >= 2:
            register_backend(backend.name, backend)
        else:
            register_backend(backend)
    except TypeError:
        register_backend(backend)


_self_register()
