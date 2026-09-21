"""Qwen3.5 / 3.6 / 3.8 hybrid VLMs (Gated DeltaNet + Gated Attention + FFN).

Production target: ``Qwen/Qwen3.8-27B``

- Hugging Face id: ``Qwen/Qwen3.8-27B``
- ``architectures``: ``Qwen3_5ForConditionalGeneration``
- ``model_type``: ``qwen3_5`` (``text_config.model_type``: ``qwen3_5_text``)
- 27B dense VLM, hidden 5120, 64 layers, FFN intermediate 17408
- Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
- Vision encoder + MTP present. Native context 262144.

Language-model W4A8 keeps vision, MTP, embeddings, and GDN extras
(``linear_attn.conv1d`` / ``in_proj_a`` / ``in_proj_b``) in BF16 by default.
``lm_head`` is **not** ignored (NVIDIA mixed NVFP4 quantizes it).
"""

from __future__ import annotations

from typing import Any

from megaquant.models.base import BaseFamily, glob_to_ignore, recipe_flag

__all__ = ["Qwen35Family"]

QWEN35_ALWAYS_IGNORE: tuple[str, ...] = (
    "*embed_tokens*",
    "*embed_positions*",
    "*linear_attn.conv1d*",
    "*linear_attn.in_proj_a*",
    "*linear_attn.in_proj_b*",
)

QWEN35_VISION_IGNORE: tuple[str, ...] = ("*visual*", "*vision*")
QWEN35_MTP_IGNORE: tuple[str, ...] = ("*mtp*",)


class Qwen35Family(BaseFamily):
    name = "qwen3_5"
    model_types = ("qwen3_5", "qwen3_5_text", "qwen3_6", "qwen3_8")
    architectures = (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    )

    def default_ignore(self, recipe: Any) -> list[str]:
        patterns: list[str] = list(QWEN35_ALWAYS_IGNORE)
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            patterns.extend(QWEN35_VISION_IGNORE)
        if not recipe_flag(recipe, "model", "quantize_mtp", default=False):
            patterns.extend(QWEN35_MTP_IGNORE)
        # Intentionally no *mlp* / *lm_head* — mixed NVFP4 targets those GEMMs.
        return glob_to_ignore(patterns)

    def load_kwargs(self, recipe: Any) -> dict[str, Any]:
        kwargs = super().load_kwargs(recipe)
        kwargs["trust_remote_code"] = True
        kwargs["torch_dtype"] = "bfloat16"
        # Transformers 5.8 (required by Qwen3.8) prefers ``dtype``; keep both.
        kwargs["dtype"] = "bfloat16"
        # ConditionalGeneration VLM, not AutoModelForCausalLM / Qwen3-8B.
        kwargs.setdefault("model_cls", "AutoModelForImageTextToText")
        # Language-model W4A8 does not need the ViT in VRAM/RAM. Skipping it
        # leaves more of the 27B LM on a 32 GB 5090.
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            kwargs["language_model_only"] = True
        return kwargs


Qwen35Family().register()
