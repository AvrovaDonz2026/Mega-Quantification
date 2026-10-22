"""ModelOpt export integration checks for opted-in auxiliary precision."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from megaquant.backends import modelopt
from megaquant.exceptions import BackendError


class _Quantizer(torch.nn.Module):
    def __init__(self, bits: tuple[int, int] = (2, 1), group_size: int = 16) -> None:
        super().__init__()
        self.num_bits = bits
        self.block_sizes = {-1: group_size}
        self.is_enabled = True


class _AuxiliaryLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(32, 32))
        self.weight_quantizer = _Quantizer()
        self.input_quantizer = _Quantizer()


class _AuxiliaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Module()
        self.model.visual.proj = _AuxiliaryLinear()


def _recipe(output_dir: Path) -> dict[str, Any]:
    return {
        "scheme": "nvfp4_w4a4",
        "model": {"quantize_vision": True, "quantize_mtp": False},
        "export": {"output_dir": str(output_dir)},
    }


def _install_export_stub(monkeypatch: pytest.MonkeyPatch, export_fn: Any) -> None:
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.export"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["modelopt.torch.export"].export_hf_checkpoint = export_fn
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr("megaquant.vision_export.restore_vision_from_recipe", lambda *_: None)
    monkeypatch.setattr(modelopt, "_restore_mtp_export", lambda *_: None)
    monkeypatch.setattr(modelopt, "_rewrite_sglang_export", lambda *_: None)


def test_opted_in_bf16_aux_export_fails_without_success_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_export(_model: Any, *, export_dir: str) -> None:
        root = Path(export_dir)
        root.mkdir(parents=True, exist_ok=True)
        save_file(
            {"model.visual.proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16)},
            str(root / "model.safetensors"),
        )

    _install_export_stub(monkeypatch, fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", lambda *_: "preserve-offload")

    with pytest.raises(BackendError):
        modelopt.ModelOptBackend().export(_AuxiliaryModel(), _recipe(tmp_path))

    assert not (tmp_path / "backend_meta.json").exists()


def test_valid_packed_aux_export_records_precision_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_export(_model: Any, *, export_dir: str) -> None:
        root = Path(export_dir)
        root.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "model.visual.proj.weight": torch.zeros(32, 16, dtype=torch.uint8),
                "model.visual.proj.weight_scale": torch.ones(
                    32, 2, dtype=torch.float8_e4m3fn
                ),
                "model.visual.proj.weight_scale_2": torch.ones((), dtype=torch.float32),
                "model.visual.proj.input_scale": torch.ones(1, dtype=torch.float32),
            },
            str(root / "model.safetensors"),
        )

    _install_export_stub(monkeypatch, fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", lambda *_: "preserve-offload")

    output = modelopt.ModelOptBackend().export(_AuxiliaryModel(), _recipe(tmp_path))

    assert output == tmp_path
    meta = json.loads((tmp_path / "backend_meta.json").read_text())
    assert meta["auxiliary_precision"] == {
        "checked": 1,
        "nvfp4": 1,
        "fp8": 0,
        "input_scaled": 1,
    }
