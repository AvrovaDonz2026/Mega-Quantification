"""Calibration input device must follow the LM, not a CPU-pinned ViT."""

from __future__ import annotations

from types import SimpleNamespace

from megaquant.backends.base import preferred_forward_device


class _Dev:
    def __init__(self, kind: str):
        self.type = kind


class _Param:
    def __init__(self, kind: str):
        self.device = _Dev(kind)


class _Module:
    def __init__(self, params: list[_Param], device_map: dict | None = None):
        self._params = params
        if device_map is not None:
            self.hf_device_map = device_map

    def parameters(self):
        return iter(self._params)


def test_preferred_device_skips_cpu_visual_for_cuda_lm() -> None:
    visual = _Module([_Param("cpu")])
    lm = _Module([_Param("cuda")])
    model = _Module(
        [*visual._params, *lm._params],
        device_map={"model.visual": "cpu", "model.language_model": 0},
    )
    model.get_input_embeddings = lambda: lm  # noqa: E731
    device = preferred_forward_device(model)
    assert device is not None
    kind = getattr(device, "type", None)
    if kind is None:
        # torch missing: hf_device_map still yields an accelerator placement
        assert str(device) in {"0", "cuda:0"} or device == 0
    else:
        assert kind == "cuda"


def test_preferred_device_uses_hf_device_map_when_no_embeddings() -> None:
    model = _Module(
        [_Param("cpu")],
        device_map={"model.visual": "cpu", "model.language_model.layers.0": "cuda:0"},
    )
    device = preferred_forward_device(model)
    assert device is not None
    text = str(device)
    assert "cuda" in text or text == "0" or getattr(device, "type", None) == "cuda"


def test_preferred_device_cpu_fallback() -> None:
    model = SimpleNamespace()
    device = preferred_forward_device(_Module([_Param("cpu")]))
    assert device is not None
    assert getattr(device, "type", "cpu") == "cpu"
    assert preferred_forward_device(model) is None
