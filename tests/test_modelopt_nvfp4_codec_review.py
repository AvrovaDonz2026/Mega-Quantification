"""Independent behavioral checks for the optional ModelOpt NVFP4 hook."""

import hashlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "codec_review", Path(__file__).parents[1] / "src/megaquant/modelopt_nvfp4_codec.py"
)
assert SPEC and SPEC.loader
codec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = codec
SPEC.loader.exec_module(codec)


class FakeQTensor:
    vendor_calls = 0

    @classmethod
    def _cast_fp4(cls, weight):
        return weight

    @classmethod
    def _is_static_quantizer(cls, quantizer):
        return getattr(quantizer, "global_amax", None) is not None

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
        cls.vendor_calls += 1
        return (
            cls(input.shape, input.dtype, input),
            weights_scaling_factor,
            weights_scaling_factor_2,
        )

    def __init__(self, shape, dtype, data):
        self.shape, self.dtype, self.data = shape, dtype, data


@pytest.fixture
def runtime(monkeypatch):
    FakeQTensor.vendor_calls = 0

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
    mods = {
        "nvfp4": types.SimpleNamespace(NVFP4QTensor=FakeQTensor),
        "resident": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "streaming": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "quant_utils": uq,
    }
    monkeypatch.setattr(codec, "_gate", lambda _: None)
    monkeypatch.setattr(
        codec.importlib,
        "import_module",
        lambda name: mods[
            next(key for key, module_name in codec.MODULES.items() if module_name == name)
        ],
    )
    return mods, dispatch


def _invoke(mods, module, weight=None, block=16, block_scale=None, global_scale=None):
    weight = torch.ones((1, 16)) if weight is None else weight
    block_scale = torch.ones((1, 1)) if block_scale is None else block_scale
    global_scale = torch.ones(()) if global_scale is None else global_scale
    return mods["resident"]._dispatch_export_handler(
        "layer",
        module,
        lambda: FakeQTensor.quantize(weight, block, block_scale, global_scale),
    )


def test_non_nvfp4_and_dynamic_modules_delegate(runtime):
    mods, _ = runtime
    callbacks = []

    def callback(weight, scale):
        callbacks.append((weight, scale))
        return torch.zeros((1, 8), dtype=torch.uint8)

    with codec.install(codec=callback) as audit:
        _invoke(
            mods,
            types.SimpleNamespace(
                quantization_format="fp8", weight_quantizer=types.SimpleNamespace(global_amax=1)
            ),
        )
        _invoke(
            mods,
            types.SimpleNamespace(
                quantization_format="nvfp4",
                weight_quantizer=types.SimpleNamespace(global_amax=None),
            ),
        )
    assert callbacks == []
    assert FakeQTensor.vendor_calls == 2
    assert audit.candidate_calls == 0
    assert audit.delegated_calls == 2


def test_valid_static_dispatch_uses_callback_with_fp32_effective_scale(runtime, monkeypatch):
    mods, _ = runtime
    # Permit the CUDA-only candidate seam on CPU while retaining actual dispatch,
    # scale multiplication, callback result validation, and packed tensor wrapping.
    monkeypatch.setattr(codec, "_validate_inputs", lambda weight, scale: None)
    seen = {}

    def callback(weight, scale):
        seen["weight"] = weight
        seen["scale"] = scale
        return torch.tensor([[0x21] * 8], dtype=torch.uint8)

    module = types.SimpleNamespace(
        quantization_format="nvfp4", weight_quantizer=types.SimpleNamespace(global_amax=1)
    )
    block_scale = torch.tensor([[2.0]])
    global_scale = torch.tensor(0.5)
    with codec.install(codec=callback) as audit:
        result = _invoke(mods, module, block_scale=block_scale, global_scale=global_scale)
    assert seen["weight"].shape == (1, 16)
    assert seen["scale"].dtype == torch.float32
    torch.testing.assert_close(seen["scale"], torch.tensor([[1.0]]))
    assert result[0].data.tolist() == [[0x21] * 8]
    assert audit.candidate_attempts == audit.candidate_calls == 1
    assert audit.failures == audit.delegated_calls == 0


def test_cpu_input_delegates_without_calling_candidate(runtime):
    mods, _ = runtime
    callbacks = []
    module = types.SimpleNamespace(
        quantization_format="nvfp4", weight_quantizer=types.SimpleNamespace(global_amax=1)
    )
    with codec.install(codec=lambda weight, scale: callbacks.append(weight)) as audit:
        _invoke(mods, module)
    assert callbacks == []
    assert FakeQTensor.vendor_calls == 1
    assert audit.candidate_attempts == audit.candidate_calls == 0
    assert audit.delegated_calls == 1


def test_invalid_callback_output_fails_and_restores_bindings(runtime, monkeypatch):
    mods, dispatch = runtime
    monkeypatch.setattr(codec, "_validate_inputs", lambda weight, scale: None)
    original = inspect.getattr_static(FakeQTensor, "quantize")
    module = types.SimpleNamespace(
        quantization_format="nvfp4", weight_quantizer=types.SimpleNamespace(global_amax=1)
    )
    with pytest.raises(ValueError, match="invalid packed"):
        with codec.install(
            codec=lambda weight, scale: torch.zeros((1, 8), dtype=torch.uint8)[:, ::2]
        ) as audit:
            _invoke(mods, module)
    assert audit.failures == 1
    assert inspect.getattr_static(FakeQTensor, "quantize") is original
    assert mods["resident"]._dispatch_export_handler is dispatch
    assert mods["streaming"]._dispatch_export_handler is dispatch
    assert codec._DEPTH == 0


def test_gate_rejects_replaced_live_quantize_binding(monkeypatch):
    class ReplacedQ:
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
            return None

        @classmethod
        def _cast_fp4(cls, weight):
            return weight

    uq = types.SimpleNamespace()
    for name in (
        "QUANTIZATION_NVFP4",
        "QUANTIZATION_NVFP4_AWQ",
        "QUANTIZATION_NVFP4_SVDQUANT",
        "QUANTIZATION_W4A16_NVFP4",
        "QUANTIZATION_W4A8_NVFP4_FP8",
    ):
        setattr(uq, name, name)
    def dispatch(*args):
        return None

    nv_module = types.SimpleNamespace(NVFP4QTensor=ReplacedQ, __name__="fake.nvfp4")
    modules = {
        "nvfp4": nv_module,
        "resident": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "streaming": types.SimpleNamespace(_dispatch_export_handler=dispatch),
        "quant_utils": uq,
    }
    monkeypatch.setattr(codec.importlib.metadata, "version", lambda _: codec.VERSION)
    module_keys = {id(modules[key]): key for key in modules}
    def module_sha(module):
        return codec.SOURCE_SHA256[module_keys[id(module)]]

    monkeypatch.setattr(codec, "_module_sha", module_sha)
    # The validated private context rejects this replacement by checking the
    # live function's module, qualname, source file, and source digest.
    with pytest.raises(RuntimeError, match="quantize"):
        codec._gate(modules)


def test_synthetic_binding_passes_live_function_gate():
    source = (
        "def quantize(cls, input, block_size, weights_scaling_factor, "
        "weights_scaling_factor_2, keep_high_precision, try_tensorrt): pass\n"
    )

    def quantize(
        cls,
        input,
        block_size,
        weights_scaling_factor,
        weights_scaling_factor_2,
        keep_high_precision,
        try_tensorrt,
    ):
        return None

    module = types.SimpleNamespace(__name__=__name__, __file__=__file__)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(codec.inspect, "getsourcefile", lambda fn: __file__)
        monkeypatch.setattr(codec.inspect, "getsource", lambda fn: source)
        monkeypatch.setitem(
            codec.FUNCTION_SHA256, "quantize", hashlib.sha256(source.encode()).hexdigest()
        )
        codec._verify_function(
            quantize,
            module,
            "quantize",
            (
                "cls",
                "input",
                "block_size",
                "weights_scaling_factor",
                "weights_scaling_factor_2",
                "keep_high_precision",
                "try_tensorrt",
            ),
        )
    finally:
        monkeypatch.undo()
