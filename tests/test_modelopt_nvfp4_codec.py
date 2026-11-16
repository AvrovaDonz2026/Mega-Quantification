import importlib.util
import inspect
import sys
import types

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("codec", "src/megaquant/modelopt_nvfp4_codec.py")
codec = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = codec
SPEC.loader.exec_module(codec)


def test_code_rejects_cpu_and_bad_scale():
    with pytest.raises(ValueError):
        codec._codes(torch.zeros((1, 16)), torch.ones((1, 1)))


def test_pack_is_low_even_high_odd_and_contiguous():
    codes = torch.tensor([[1, 2, 8, 15]], dtype=torch.uint8)
    packed = codec._pack(codes)
    assert packed.tolist() == [[0x21, 0xF8]]
    assert packed.is_contiguous() and packed.dtype == torch.uint8


def test_version_gate_is_fail_closed(monkeypatch):
    monkeypatch.setattr(codec.importlib.metadata, "version", lambda _: "0.46.1")
    with pytest.raises(RuntimeError, match="0.47.0"):
        codec._gate(
            {
                "nvfp4": types.SimpleNamespace(NVFP4QTensor=object()),
                "quant_utils": object(),
                "resident": object(),
                "streaming": object(),
            }
        )


def test_audit_defaults_empty_and_non_nested():
    audit = codec.CodecAudit()
    assert audit.candidate_calls == audit.delegated_calls == 0
    audit.delegate("unsupported")
    assert audit.reasons["unsupported"] == 1


def test_install_restores_descriptors_and_validates_before_callback(monkeypatch):
    class FakeQ:
        @classmethod
        def _cast_fp4(cls, weight):
            return weight

        @classmethod
        def quantize(
            cls,
            input,
            block_size,
            weights_scaling_factor=None,
            weights_scaling_factor_2=None,
            keep_high_precision=False,
            try_tensorrt=False,
        ):
            return (
                cls(input.shape, input.dtype, input),
                weights_scaling_factor,
                weights_scaling_factor_2,
            )

        def __init__(self, shape, dtype, data):
            self.metadata = {"shape": shape, "dtype": dtype}
            self.data = data

    def dispatch(name, module, ctx):
        return ctx()

    uq = types.SimpleNamespace(
        get_quantization_format=lambda module: module.quantization_format,
        QUANTIZATION_NVFP4="nvfp4",
        QUANTIZATION_NVFP4_AWQ="nvfp4_awq",
        QUANTIZATION_NVFP4_SVDQUANT="nvfp4_svdquant",
        QUANTIZATION_W4A16_NVFP4="w4a16_nvfp4",
        QUANTIZATION_W4A8_NVFP4_FP8="w4a8_nvfp4_fp8",
    )
    FakeQ._is_static_quantizer = classmethod(
        lambda cls, q: getattr(q, "global_amax", None) is not None
    )
    mods = {
        "nvfp4": types.SimpleNamespace(NVFP4QTensor=FakeQ),
        "resident": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "streaming": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "quant_utils": uq,
    }
    monkeypatch.setattr(codec, "_gate", lambda _: None)
    monkeypatch.setattr(codec, "_validate_inputs", lambda *_: None)
    monkeypatch.setattr(
        codec.importlib,
        "import_module",
        lambda name: mods[next(k for k, v in codec.MODULES.items() if v == name)],
    )
    original = inspect.getattr_static(FakeQ, "quantize")
    seen = {}

    def packed(weight, scale):
        seen["shape"] = tuple(weight.shape)
        return torch.zeros((1, 8), dtype=torch.uint8)

    with codec.install(codec=packed) as audit:
        module = types.SimpleNamespace(
            quantization_format="nvfp4", weight_quantizer=types.SimpleNamespace(global_amax=1)
        )
        out = mods["resident"]._dispatch_export_handler(
            "x",
            module,
            lambda: FakeQ.quantize(torch.ones((1, 16)), 16, torch.ones((1, 1)), torch.ones(())),
        )
        assert out[0].data.shape == (1, 8) and seen["shape"] == (1, 16)
        assert audit.candidate_calls == 1 and audit.delegated_calls == 0
        with pytest.raises(RuntimeError, match="nested"):
            with codec.install(codec=packed):
                pass
    assert inspect.getattr_static(FakeQ, "quantize") is original
    assert mods["resident"]._dispatch_export_handler is dispatch
