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
MTP tensors are still written into the Hugging Face export as
``mtp.safetensors``; they are not listed in ``quantized_layers``.
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
        # Transformers 5.8 (required by Qwen3.8) prefers ``dtype``; keep both.
        # Honor recipe.model.dtype via BaseFamily (default remains bfloat16).
        kwargs.setdefault("dtype", kwargs.get("torch_dtype", "bfloat16"))
        # ConditionalGeneration VLM, not AutoModelForCausalLM / Qwen3-8B.
        kwargs.setdefault("model_cls", "AutoModelForImageTextToText")
        # Language-model W4A8 does not need the ViT in VRAM/RAM. Skipping it
        # leaves more of the 27B LM on a 32 GB 5090.
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            kwargs["language_model_only"] = True
        return kwargs

    def plan_notes(self, recipe: Any) -> list[str]:
        from megaquant.runtime import gdn_kernel_plan_note, skip_vision_init_enabled

        notes: list[str] = []
        if not recipe_flag(recipe, "model", "quantize_vision", default=False):
            if skip_vision_init_enabled():
                notes.append(
                    "language-only PTQ: skip Qwen3_5 visual constructor so "
                    "device_map=auto packs the 27B LM onto GPU first"
                )
            pins = self.cpu_pin_needles(recipe)
            if pins:
                notes.append(
                    "pin "
                    + "/".join(pins)
                    + " to CPU during auto device_map (ViT-first packing otherwise)"
                )
        kernel_note = gdn_kernel_plan_note()
        if kernel_note:
            notes.append(kernel_note)
        return notes

    def prepare_load(self, recipe: Any) -> str | None:
        if recipe_flag(recipe, "model", "quantize_vision", default=False):
            return None
        from megaquant.runtime import skip_vision_init_enabled

        if not skip_vision_init_enabled():
            return None
        return skip_qwen35_visual_constructor()


_SKIP_VISUAL_PATCHED = False


def skip_qwen35_visual_constructor() -> str | None:
    """Do not build ``Qwen3_5Model.visual``.

    transformers 5.14 always does ``self.visual = AutoModel.from_config(...)``
    *before* the 27B language model, so ``device_map='auto'`` fills a 32 GB
    GPU with a tower that text-only W4A8 never runs. Forward only touches
    ``visual`` when ``pixel_values`` is set.
    """
    global _SKIP_VISUAL_PATCHED
    note = "Qwen3_5Model visual constructor skipped (language-only PTQ)"
    if _SKIP_VISUAL_PATCHED:
        return note
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    except Exception:
        return None
    if getattr(Qwen3_5Model.__init__, "_megaquant_skip_visual", False):
        _SKIP_VISUAL_PATCHED = True
        return note

    orig = Qwen3_5Model.__init__

    def __init__(self, config, *args, **kwargs):  # noqa: ANN001
        super(Qwen3_5Model, self).__init__(config)
        self.visual = None
        from transformers.models.auto.modeling_auto import AutoModel

        text_config = getattr(config, "text_config", None) or config
        self.language_model = AutoModel.from_config(text_config)
        self.rope_deltas = None
        post = getattr(self, "post_init", None)
        if callable(post):
            post()

    __init__._megaquant_skip_visual = True  # type: ignore[attr-defined]
    __init__._megaquant_orig_init = orig  # type: ignore[attr-defined]
    Qwen3_5Model.__init__ = __init__
    _SKIP_VISUAL_PATCHED = True
    return note


Qwen35Family().register()
