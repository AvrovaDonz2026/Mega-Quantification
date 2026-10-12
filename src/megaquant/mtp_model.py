"""Attach Qwen3.5's standalone MTP block before ModelOpt quantization.

Qwen3.5 checkpoints store ``mtp.*`` tensors at the repository root, while the
Transformers conditional-generation class does not construct that optional
block.  ModelOpt can therefore never see MTP unless the pipeline attaches a
small state-dict compatible module first.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from megaquant.exceptions import BackendError
from megaquant.mtp_export import _open_source_shard, _source_index, is_mtp_tensor_name


def _load_source_tensors(source: str | Path) -> dict[str, Any]:
    """Load only the source shards that contain standalone MTP tensors."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - pipeline dependency
        raise BackendError("safetensors is required to attach Qwen3.5 MTP") from exc

    index, root = _source_index(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return {}
    by_shard: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if is_mtp_tensor_name(str(key)):
            by_shard.setdefault(str(filename), []).append(str(key))

    tensors: dict[str, Any] = {}
    for filename, expected in by_shard.items():
        shard = _open_source_shard(root, filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in expected:
                if key not in handle.keys():
                    raise BackendError(f"MTP tensor {key} missing from {filename}")
                tensors[key] = handle.get_tensor(key)
    return tensors


def _text_model(model: Any) -> Any:
    for path in ("model.language_model", "language_model", "model", ""):
        candidate = model
        for part in path.split(".") if path else ():
            candidate = getattr(candidate, part, None)
        if all(hasattr(candidate, attr) for attr in ("embed_tokens", "layers", "norm")):
            return candidate
    raise BackendError("MTP calibration requires a Qwen3.5 text model with token embeddings")


def _build_mtp(config: Any, tensors: dict[str, Any]) -> Any:
    """Construct the dense Qwen3.5 MTP architecture without duplicating its weights."""
    import torch
    from torch import nn

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )
    except ImportError as exc:
        raise BackendError("Qwen3.5 MTP calibration requires Transformers Qwen3_5 modules") from exc

    config = deepcopy(config)
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    config._attn_implementation = "sdpa"

    class Qwen35MTP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Source tensors are assigned below: avoid allocating a second large
            # decoder in float32 while loading a 27B model on shared GPU memory.
            with torch.device("meta"):
                self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
                self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
                self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
                self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, layer_idx=0)])
                self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            # RoPE buffers are nonpersistent and must be materialized on CPU.
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
            self.calibration_batches = 0
            self.calibration_tokens = 0

        def forward(self, embeddings: Any, hidden_states: Any, position_ids: Any) -> Any:
            # Match SGLang Qwen3_5ForCausalLMMTP: normalized embedding first,
            # normalized target output second, then one full-attention block.
            hidden_states = self.fc(torch.cat([
                self.pre_fc_norm_embedding(embeddings),
                self.pre_fc_norm_hidden(hidden_states),
            ], dim=-1))
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            length = hidden_states.shape[1]
            mask = torch.full(
                (length, length), torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device, dtype=hidden_states.dtype,
            ).triu(diagonal=1)[None, None]
            hidden_states = self.layers[0](
                hidden_states, position_embeddings=position_embeddings, attention_mask=mask,
            )
            return self.norm(hidden_states)

    mtp = Qwen35MTP()
    state = {name.removeprefix("mtp."): value for name, value in tensors.items()}
    try:
        mtp.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise BackendError(
            f"Unsupported or incomplete standalone Qwen3.5 MTP weights: {exc}"
        ) from exc
    return mtp.eval()


def attach_mtp(model: Any, source: str | Path) -> int:
    """Attach source MTP weights under ``model.mtp`` and return tensor count.

    The source's 15 names are preserved, including its 8 Linear projections.
    ``mtp_calibration`` connects this block to the target's real forward pass
    while ModelOpt collects activation ranges.
    """
    if getattr(model, "mtp", None) is not None:
        return sum(1 for name, _ in model.named_parameters() if is_mtp_tensor_name(name))

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - pipeline dependency
        raise BackendError("torch is required to attach Qwen3.5 MTP") from exc

    tensors = _load_source_tensors(source)
    if not tensors:
        raise BackendError(f"source checkpoint has no standalone mtp.* tensors: {source}")

    text_model = _text_model(model)
    model.add_module("mtp", _build_mtp(text_model.config, tensors))
    return len(tensors)


def _move_mtp_calibration_state(mtp: Any, device: Any) -> None:
    """Move the branch and already-collected ModelOpt max statistics together."""
    import torch

    mtp.to(device=device)
    # ModelOpt initializes weight ranges before the first target forward.
    # MaxCalibrator is not an nn.Module, so its ordinary tensor attribute is
    # left on CPU by mtp.to(), unlike the quantizers' registered buffers.
    # Preserve those ranges when following the actual target output device.
    for module in mtp.modules():
        calibrator = getattr(module, "_calibrator", None)
        amax = getattr(calibrator, "_calib_amax", None)
        if torch.is_tensor(amax) and amax.device != device:
            calibrator._calib_amax = amax.to(device=device)


@contextmanager
def mtp_calibration(model: Any):
    """Run teacher-forced MTP from target forwards; remove the hook afterwards.

    Pair target output at token t with the actual input embedding at t+1. For
    multimodal batches these are the merged vision/text embeddings received by
    the language model. Padding and the final token (no next-token embedding)
    are excluded. No hidden-state history or extra full-model pass is needed.
    """
    mtp = getattr(model, "mtp", None)
    if mtp is None:
        yield
        return

    import torch

    text_model = _text_model(model)
    calls_before = mtp.calibration_batches

    def calibrate(module: Any, args: Any, kwargs: Any, output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        if hidden.ndim != 3:
            raise BackendError("MTP calibration requires batched sequence hidden states")
        inputs = dict(zip(("input_ids", "attention_mask", "position_ids"), args, strict=False))
        inputs.update(kwargs)
        embeddings = inputs.get("inputs_embeds")
        if embeddings is None:
            ids = inputs.get("input_ids")
            if ids is None:
                raise BackendError("MTP calibration received no token IDs or input embeddings")
            embeddings = module.embed_tokens(ids)
        embeddings = embeddings.detach().to(device=hidden.device, dtype=hidden.dtype)
        mask = inputs.get("attention_mask")
        if mask is not None and (not torch.is_tensor(mask) or mask.ndim != 2):
            raise BackendError("MTP calibration requires a 2D padding mask")
        positions = inputs.get("position_ids")
        if positions is None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
            positions = positions.expand(hidden.shape[0], -1)
        positions = positions.to(hidden.device)
        if positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[1:]
        elif positions.ndim == 2:
            positions = positions[None].expand(3, -1, -1)
        # The attached branch is initially on CPU to keep model loading cheap.
        # Calibration follows the target output's execution device (including
        # Accelerate offload), preserving quantizer buffers and any weight
        # ranges collected by ModelOpt before the target's first forward.
        _move_mtp_calibration_state(mtp, hidden.device)
        for row in range(hidden.shape[0]):
            valid = torch.ones(hidden.shape[1] - 1, dtype=torch.bool, device=hidden.device)
            if mask is not None:
                row_mask = mask[row].to(device=hidden.device, dtype=torch.bool)
                valid &= row_mask[:-1] & row_mask[1:]
            if not valid.any():
                continue
            next_positions = positions[..., row:row + 1, 1:][..., valid]
            mtp(
                embeddings[row:row + 1, 1:][:, valid],
                hidden[row:row + 1, :-1][:, valid].detach(),
                next_positions,
            )
            mtp.calibration_batches += 1
            mtp.calibration_tokens += int(valid.sum())

    handle = text_model.register_forward_hook(calibrate, with_kwargs=True)
    try:
        yield
        if mtp.calibration_batches == calls_before:
            raise BackendError("MTP calibration saw no usable next-token pairs; export is unsafe")
    finally:
        handle.remove()
