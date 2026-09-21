"""Named quantization scheme catalog.

Entries are JSON-serializable :class:`SchemeDef` objects. This module must
**not** import ``compressed_tensors`` (or llm-compressor) at import time; the
llm-compressor backend translates these dicts into ``QuantizationScheme`` /
``QuantizationArgs`` lazily.

On import, each entry is registered with ``megaquant.registry.register_scheme``
when that helper is importable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from megaquant.schemes.nvfp4 import (
    DEFAULT_LINEAR_TARGETS,
    FP8_W8A8_ACTIVATIONS,
    FP8_W8A8_WEIGHTS,
    MIXED_FP8_TARGETS,
    MIXED_NVFP4_TARGETS,
    NVFP4_W4A4_ACTIVATIONS,
    NVFP4_W4A4_WEIGHTS,
    NVFP4_W4A8_ACTIVATIONS,
    NVFP4_W4A8_WEIGHTS,
    NVFP4_W4A16_WEIGHTS,
    layer_group,
    uniform_group,
)

__all__ = [
    "SCHEME_CATALOG",
    "SchemeDef",
    "get_scheme",
]


@dataclass(frozen=True)
class SchemeDef:
    """One named quantization scheme.

    ``groups`` items are LayerGroup-shaped dicts::

        {"name", "targets", "weights": PrecisionSpec-like, "activations": dict|None}
    """

    name: str
    groups: list[dict[str, Any]]
    kv_cache: str | None
    notes: str
    vllm_support: str
    trtllm_support: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping (groups already contain plain dicts)."""
        return asdict(self)


SCHEME_CATALOG: dict[str, SchemeDef] = {
    "nvfp4_w4a8": SchemeDef(
        name="nvfp4_w4a8",
        groups=[
            uniform_group(
                "language_linears",
                weights=NVFP4_W4A8_WEIGHTS,
                activations=NVFP4_W4A8_ACTIVATIONS,
                targets=DEFAULT_LINEAR_TARGETS,
            )
        ],
        kv_cache="fp8",
        notes=(
            "Uniform NVFP4 weights (tensor_group group_size=32, FP8 E4M3 scales) "
            "plus dynamic per-token FP8 E4M3 activations. This is W4A8, not W4A4. "
            "Matches ModelOpt W4A8_NVFP4_FP8 / nvfp4_bs32. compressed-tensors has "
            "no stock NVFP4A8 preset — the llm-compressor backend emits a custom "
            "QuantizationScheme. Alternative activations: static per-tensor FP8 "
            "with a minmax observer (see NVFP4_W4A8_ACTIVATIONS_STATIC). "
            "Gated DeltaNet extras (linear_attn.conv1d / in_proj_a / in_proj_b) "
            "belong on the family ignore list, not in targets."
        ),
        vllm_support="limited; prefer TensorRT-LLM/ModelOpt",
        trtllm_support="native W4A8_NVFP4_FP8",
    ),
    "nvfp4_w4a4": SchemeDef(
        name="nvfp4_w4a4",
        groups=[
            uniform_group(
                "language_linears",
                weights=NVFP4_W4A4_WEIGHTS,
                activations=NVFP4_W4A4_ACTIVATIONS,
                targets=DEFAULT_LINEAR_TARGETS,
            )
        ],
        kv_cache="fp8",
        notes=(
            "Stock NVFP4: FP4 tensor_group group_size=16 weights and dynamic-local "
            "FP4 tensor_group group_size=16 activations, both with FP8 E4M3 scales. "
            "llm-compressor maps this to scheme='NVFP4'."
        ),
        vllm_support="native NVFP4 W4A4",
        trtllm_support="native NVFP4",
    ),
    "nvfp4_w4a16": SchemeDef(
        name="nvfp4_w4a16",
        groups=[
            uniform_group(
                "language_linears",
                weights=NVFP4_W4A16_WEIGHTS,
                activations=None,
                targets=DEFAULT_LINEAR_TARGETS,
            )
        ],
        kv_cache="fp8",
        notes=(
            "Stock NVFP4A16: FP4 tensor_group group_size=16 weights only; "
            "activations remain BF16. llm-compressor maps this to scheme='NVFP4A16'."
        ),
        vllm_support="native NVFP4A16",
        trtllm_support="native W4A16 NVFP4",
    ),
    "fp8_w8a8": SchemeDef(
        name="fp8_w8a8",
        groups=[
            uniform_group(
                "language_linears",
                weights=FP8_W8A8_WEIGHTS,
                activations=FP8_W8A8_ACTIVATIONS,
                targets=DEFAULT_LINEAR_TARGETS,
            )
        ],
        kv_cache="fp8",
        notes=(
            "FP8 W8A8: channel-wise FP8 weights + dynamic per-token FP8 "
            "activations (stock FP8_DYNAMIC). Alternative is static per-tensor "
            "FP8 (stock FP8) with a minmax observer."
        ),
        vllm_support="native FP8 W8A8",
        trtllm_support="native FP8",
    ),
    "nvfp4_mixed": SchemeDef(
        name="nvfp4_mixed",
        groups=[
            layer_group(
                "mlp_lm_head",
                targets=MIXED_NVFP4_TARGETS,
                weights=NVFP4_W4A8_WEIGHTS,
                activations=NVFP4_W4A8_ACTIVATIONS,
            ),
            layer_group(
                "attn_fp8",
                targets=MIXED_FP8_TARGETS,
                weights=FP8_W8A8_WEIGHTS,
                activations=FP8_W8A8_ACTIVATIONS,
            ),
        ],
        kv_cache="fp8",
        notes=(
            "NVIDIA-style mixed recipe: NVFP4 W4A8 (group_size=32, dynamic token "
            "FP8 acts) on mlp.{gate,up,down}_proj + lm_head; FP8 W8A8 on "
            "self_attn.{q,k,v,o}_proj and linear_attn {in_proj_qkv, in_proj_z, "
            "out_proj}. conv1d / in_proj_a / in_proj_b stay on the ignore list. "
            "Prefer ModelOpt custom quant_cfg for production mixed checkpoints."
        ),
        vllm_support="limited mixed NVFP4/FP8; prefer TensorRT-LLM/ModelOpt",
        trtllm_support="native mixed NVFP4 MLP + FP8 attn",
    ),
}


def get_scheme(name: str) -> SchemeDef:
    """Look up a catalog entry by name.

    Raises
    ------
    KeyError
        If ``name`` is not a catalog key.
    """
    if name in SCHEME_CATALOG:
        return SCHEME_CATALOG[name]
    lowered = name.lower().replace("-", "_")
    if lowered in SCHEME_CATALOG:
        return SCHEME_CATALOG[lowered]
    available = ", ".join(sorted(SCHEME_CATALOG))
    raise KeyError(f"Unknown scheme {name!r}. Available: {available}")


def _register_catalog() -> None:
    try:
        from megaquant.registry import register_scheme
    except ImportError:
        return
    for scheme in SCHEME_CATALOG.values():
        try:
            register_scheme(scheme.name, scheme)
        except TypeError:
            try:
                register_scheme(scheme)
            except Exception:
                return
        except Exception:
            return


_register_catalog()
