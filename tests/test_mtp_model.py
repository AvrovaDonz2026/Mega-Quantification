"""Exercise real Qwen3.5 MTP forwards and their target-model calibration hook."""

from __future__ import annotations

import pytest

from megaquant.exceptions import BackendError
from megaquant.mtp_model import attach_mtp, mtp_calibration

torch = pytest.importorskip("torch")
qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


@pytest.fixture
def target(monkeypatch):
    torch.manual_seed(7)
    config = qwen.Qwen3_5TextConfig(
        vocab_size=48, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "sdpa"

    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = qwen.Qwen3_5TextModel(config).eval()

        def forward(self, **kwargs):
            return self.language_model(**kwargs, use_cache=False)

    source = torch.nn.Module()
    source.fc = torch.nn.Linear(64, 32, bias=False)
    source.pre_fc_norm_embedding = qwen.Qwen3_5RMSNorm(32, config.rms_norm_eps)
    source.pre_fc_norm_hidden = qwen.Qwen3_5RMSNorm(32, config.rms_norm_eps)
    source.layers = torch.nn.ModuleList([qwen.Qwen3_5DecoderLayer(config, 0)])
    source.norm = qwen.Qwen3_5RMSNorm(32, config.rms_norm_eps)
    weights = {f"mtp.{key}": value.clone() for key, value in source.state_dict().items()}
    monkeypatch.setattr("megaquant.mtp_model._load_source_tensors", lambda _: weights)
    model = Target().eval()
    assert attach_mtp(model, "unused") == 15
    assert set(model.mtp.state_dict()) == set(source.state_dict())
    return model


def test_calibration_executes_all_eight_mtp_projections(target):
    ranges = {}
    handles = []
    for name, layer in target.mtp.named_modules():
        if isinstance(layer, torch.nn.Linear):
            def record(module, args, output, name=name):
                ranges[name] = args[0].abs().max().item()
                assert torch.isfinite(output).all()
            handles.append(layer.register_forward_hook(record))
    ids = torch.tensor([[2, 7, 3, 8, 12]])
    with torch.no_grad():
        expected = target(input_ids=ids).last_hidden_state
        assert ranges == {}  # Standalone target forwards must not run the draft.
        with mtp_calibration(target):
            actual = target(input_ids=ids).last_hidden_state
        torch.testing.assert_close(actual, expected)
        assert len(ranges) == 8
        assert all(value > 0 for value in ranges.values())
        assert target.mtp.calibration_tokens == 4
        assert target.mtp.calibration_batches == 1
        ranges.clear()
        target(input_ids=ids)
        assert ranges == {}  # The calibration-only hook was removed.
    for handle in handles:
        handle.remove()


def test_teacher_forcing_uses_merged_embeddings_next_positions_and_excludes_padding(target):
    # Different row padding and multimodal-style positions exercise pairing,
    # rather than merely asserting that a hook was invoked.
    embeddings = torch.randn(2, 5, 32)
    embeddings[0, 2] *= 4  # Stand in for an already merged vision embedding.
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    positions = torch.arange(5).expand(4, 2, -1).clone()
    positions[2] += 10
    captured = []
    handle = target.mtp.register_forward_pre_hook(
        lambda module, args: captured.append(tuple(value.clone() for value in args))
    )
    with torch.no_grad():
        expected = target(inputs_embeds=embeddings, attention_mask=mask,
                          position_ids=positions).last_hidden_state
        with mtp_calibration(target):
            target(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions)
    handle.remove()
    assert len(captured) == 2
    torch.testing.assert_close(captured[0][0], embeddings[0:1, 2:])
    torch.testing.assert_close(captured[0][1], expected[0:1, 1:4])
    torch.testing.assert_close(captured[0][2], positions[1:, 0:1, 2:])
    torch.testing.assert_close(captured[1][0], embeddings[1:2, 1:3])
    torch.testing.assert_close(captured[1][1], expected[1:2, :2])
    assert target.mtp.calibration_tokens == 5


def test_no_usable_pairs_fails_and_removes_hook(target):
    with torch.no_grad(), pytest.raises(BackendError, match="no usable next-token pairs"):
        with mtp_calibration(target):
            target(input_ids=torch.tensor([[2]]))
    assert not target.language_model._forward_hooks


def test_context_removes_hook_when_target_fails(target):
    with pytest.raises(RuntimeError, match="target failed"):
        with mtp_calibration(target):
            raise RuntimeError("target failed")
    assert not target.language_model._forward_hooks


def test_modelopt_collects_real_mtp_activation_ranges(target):
    mtq = pytest.importorskip("modelopt.torch.quantization")
    from megaquant.backends.modelopt import ModelOptBackend

    cfg = ModelOptBackend().build_quant_cfg({
        "scheme": "fp8_w8a8", "algorithm": "max", "kv_cache": None,
        "model": {"quantize_mtp": True, "quantize_vision": False},
    })

    def forward_loop(model):
        with torch.no_grad(), mtp_calibration(model):
            model(input_ids=torch.tensor([[2, 7, 3, 8, 12]]))

    mtq.quantize(target, cfg, forward_loop=forward_loop)
    ranges = {
        name: layer.input_quantizer.amax
        for name, layer in target.mtp.named_modules()
        if isinstance(layer, torch.nn.Linear)
    }
    assert len(ranges) == 8
    for name, amax in ranges.items():
        assert amax is not None, f"MTP input activation range missing: {name}"
        assert torch.isfinite(amax).all(), name
        assert (amax > 0).all(), name
