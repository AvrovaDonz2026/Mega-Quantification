"""JSON-serializable NVFP4 / FP8 precision specs.

These dicts mirror ``megaquant.config.PrecisionSpec`` and are consumed by the
named scheme catalog and the llm-compressor backend. This module must **not**
import ``compressed_tensors`` (or llm-compressor) at import time.

Precision facts
---------------
NVFP4 W4A4 (stock ``NVFP4``)
    Weights: FP4 ``tensor_group`` ``group_size=16``, static, FP8 E4M3 scales.
    Activations: FP4 ``tensor_group`` dynamic-local ``group_size=16``, FP8 scales.

NVFP4 W4A16 (stock ``NVFP4A16``)
    Weights only: FP4 ``tensor_group`` ``group_size=16``. Activations stay BF16.

NVIDIA mixed / SGLang W4A8 (``nvfp4_mixed``, default ``nvfp4_w4a8``)
    MLP + ``lm_head``: NVFP4 W4A4 ``group_size=16`` (weights and activations).
    Attention: FP8 on ``self_attn.{q,k,v,o}_proj`` and
    ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}``. Export metadata is
    ``MIXED_PRECISION`` + ``quantized_layers`` so SGLang ``modelopt_mixed``
    can load it. Not ModelOpt ``W4A8_NVFP4_FP8`` gs32.

TensorRT-LLM uniform W4A8 (``w4a8_nvfp4_fp8``; **no** stock ``NVFP4A8``)
    Weights: FP4 ``tensor_group`` **``group_size=32``** (ModelOpt
    ``w4a8_nvfp4_fp8`` / ``nvfp4_bs32``), FP8 E4M3 scales.
    Activations: FP8 E4M3. SGLang rejects ``quant_algo=W4A8_NVFP4_FP8``.

FP8 W8A8 (stock ``FP8_DYNAMIC``)
    Channel-wise FP8 weights + dynamic per-token FP8 activations, or static
    per-tensor FP8 (``FP8``).

Qwen3.8 module names
--------------------
Covered by default targets: ``mlp.{gate,up,down}_proj``,
``self_attn.{q,k,v,o}_proj``, and Gated DeltaNet ``linear_attn`` projections
``in_proj_qkv`` / ``in_proj_z`` / ``out_proj`` (plus a bare ``in_proj``).

Stay on the **ignore** list (not targets): ``linear_attn.conv1d``,
``linear_attn.in_proj_a``, ``linear_attn.in_proj_b``.
"""

from __future__ import annotations

from typing import Any

# Explicit proj regex PLUS Linear so the same scheme works on Qwen3.8 hybrid
# (Gated DeltaNet + Gated Attention + FFN) and vanilla Llama / Qwen3.
DEFAULT_LINEAR_TARGETS: list[str] = [
    "Linear",
    "re:.*(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
    "in_proj_qkv|in_proj_z|out_proj).*",
]

# NVIDIA mixed recipe (nvidia/Qwen3.8-27B-NVFP4): NVFP4 group_size 16 on
# MLP + lm_head, FP8 on attention. Not W4A8 gs32.
MIXED_NVFP4_TARGETS: list[str] = [
    r"re:.*mlp\.(?:gate_proj|up_proj|down_proj).*",
    r"re:.*lm_head.*",
]

MIXED_FP8_TARGETS: list[str] = [
    r"re:.*self_attn\.(?:q_proj|k_proj|v_proj|o_proj).*",
    r"re:.*linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj).*",
    r"re:.*linear_attn\.in_proj$",
]

# ---------------------------------------------------------------------------
# PrecisionSpec-like dicts (JSON-serializable; no torch dtypes)
# ---------------------------------------------------------------------------

NVFP4_W4A4_WEIGHTS: dict[str, Any] = {
    "format": "nvfp4",
    "bits": 4,
    "group_size": 16,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor_group",
}

NVFP4_W4A4_ACTIVATIONS: dict[str, Any] = {
    "format": "nvfp4",
    "bits": 4,
    "group_size": 16,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": "local",
    "strategy": "tensor_group",
    "observer": "static_minmax",
}

NVFP4_W4A16_WEIGHTS: dict[str, Any] = {
    "format": "nvfp4",
    "bits": 4,
    "group_size": 16,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor_group",
}

# ModelOpt W4A8_NVFP4_FP8 / nvfp4_bs32 — group_size 32, not the W4A4 size 16.
NVFP4_W4A8_WEIGHTS: dict[str, Any] = {
    "format": "nvfp4",
    "bits": 4,
    "group_size": 32,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor_group",
}

# Default W4A8 activations: dynamic per-token FP8 E4M3.
NVFP4_W4A8_ACTIVATIONS: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": True,
    "strategy": "token",
    "observer": None,
}

# Alternative W4A8 activations: static per-tensor FP8 with minmax observer.
NVFP4_W4A8_ACTIVATIONS_STATIC: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor",
    "observer": "minmax",
}

FP8_W8A8_WEIGHTS: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "channel",
}

FP8_W8A8_ACTIVATIONS: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": True,
    "strategy": "token",
    "observer": None,
}

FP8_W8A8_WEIGHTS_STATIC: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor",
}

FP8_W8A8_ACTIVATIONS_STATIC: dict[str, Any] = {
    "format": "fp8",
    "bits": 8,
    "group_size": None,
    "scale_dtype": "float8_e4m3fn",
    "dynamic": False,
    "strategy": "tensor",
    "observer": "minmax",
}


def layer_group(
    name: str,
    *,
    targets: list[str],
    weights: dict[str, Any],
    activations: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one JSON-serializable LayerGroup dict."""
    return {
        "name": name,
        "targets": list(targets),
        "weights": dict(weights),
        "activations": None if activations is None else dict(activations),
    }


def uniform_group(
    name: str,
    *,
    weights: dict[str, Any],
    activations: dict[str, Any] | None,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Single-group wrapper using :data:`DEFAULT_LINEAR_TARGETS`."""
    return layer_group(
        name,
        targets=list(DEFAULT_LINEAR_TARGETS if targets is None else targets),
        weights=weights,
        activations=activations,
    )
