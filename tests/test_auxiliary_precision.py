from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from megaquant.auxiliary_precision import (
    snapshot_auxiliary_quantizers,
    validate_auxiliary_precision,
)
from megaquant.exceptions import BackendError


class FakeQ:
    def __init__(self, bits=(2, 1), group=16, enabled=True):
        self.num_bits = bits
        self.block_sizes = {-1: group}
        self.is_enabled = enabled


class FakeParent:
    def __init__(self, shape=(32, 32), bits=(2, 1), group=16, input_enabled=True):
        self.weight = torch.zeros(shape)
        self.weight_quantizer = FakeQ(bits, group)
        self.input_quantizer = FakeQ(bits, group, input_enabled)


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        yield "", self
        yield from self._modules.items()


def recipe(vision=True, mtp=False):
    return SimpleNamespace(model=SimpleNamespace(quantize_vision=vision, quantize_mtp=mtp))


def write_export(
    root: Path,
    module="model.visual.proj",
    *,
    fp8=False,
    input_scale=True,
    shape=(32, 32),
    group=16,
):
    tensors = {}
    if fp8:
        tensors[module + ".weight"] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
        tensors[module + ".weight_scale"] = torch.ones(1, dtype=torch.float32)
    else:
        tensors[module + ".weight"] = torch.zeros((shape[0], shape[1] // 2), dtype=torch.uint8)
        tensors[module + ".weight_scale"] = torch.ones(
            (shape[0], shape[1] // group), dtype=torch.float8_e4m3fn
        )
        tensors[module + ".weight_scale_2"] = torch.ones((), dtype=torch.float32)
    if input_scale:
        tensors[module + ".input_scale"] = torch.ones(1, dtype=torch.float32)
    save_file(tensors, str(root / "model.safetensors"))


def test_snapshot_and_validate_nvfp4(tmp_path):
    model = FakeModel({"model.visual.proj": FakeParent()})
    expected = snapshot_auxiliary_quantizers(model, recipe())
    assert expected["model.visual.proj"]["format"] == "NVFP4"
    write_export(tmp_path)
    assert validate_auxiliary_precision(tmp_path, expected) == {
        "checked": 1,
        "nvfp4": 1,
        "fp8": 0,
        "input_scaled": 1,
    }


def test_snapshot_optout_and_missing_enabled_branch(tmp_path):
    model = FakeModel({"model.visual.proj": FakeParent()})
    assert snapshot_auxiliary_quantizers(model, recipe(vision=False)) == {}
    with pytest.raises(BackendError, match="no enabled auxiliary"):
        snapshot_auxiliary_quantizers(FakeModel({}), recipe())


def test_bf16_fails_closed(tmp_path):
    expected = {
        "model.visual.proj": {
            "format": "NVFP4",
            "group_size": 16,
            "weight_shape": [32, 32],
            "input_enabled": False,
        }
    }
    save_file(
        {"model.visual.proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(BackendError, match="BF16|missing.*weight_scale|NVFP4"):
        validate_auxiliary_precision(tmp_path, expected)


def test_packed_shape_error(tmp_path):
    expected = {
        "model.visual.proj": {
            "format": "NVFP4",
            "group_size": 16,
            "weight_shape": [32, 32],
            "input_enabled": False,
        }
    }
    write_export(tmp_path, shape=(32, 48), input_scale=False)
    with pytest.raises(BackendError, match="shape"):
        validate_auxiliary_precision(tmp_path, expected)


def test_fp8_and_alias_ambiguity(tmp_path):
    expected = {
        "model.mtp.proj": {
            "format": "FP8",
            "weight_shape": [8, 8],
            "input_enabled": True,
        }
    }
    write_export(tmp_path, module="mtp.proj", fp8=True, shape=(8, 8))
    assert validate_auxiliary_precision(tmp_path, expected)["fp8"] == 1
    save_file(
        {
            "mtp.proj.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            "mtp.proj.weight_scale": torch.ones(1),
            "model.mtp.proj.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
            "model.mtp.proj.weight_scale": torch.ones(1),
        },
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(BackendError, match="Ambiguous"):
        validate_auxiliary_precision(tmp_path, expected)


def test_quantized_layer_map_mismatch(tmp_path):
    expected = {
        "model.visual.proj": {
            "format": "NVFP4",
            "group_size": 16,
            "weight_shape": [32, 32],
            "input_enabled": False,
        }
    }
    write_export(tmp_path, input_scale=False)
    with pytest.raises(BackendError, match="metadata mismatch"):
        validate_auxiliary_precision(
            tmp_path, expected, {"model.visual.proj": {"quant_algo": "FP8"}}
        )


@pytest.mark.parametrize("suffix", ["weight_scale", "weight_scale_2", "input_scale"])
def test_missing_required_scale_fails(tmp_path, suffix):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent()}), recipe()
    )
    write_export(tmp_path)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    del tensors["model.visual.proj." + suffix]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(BackendError, match="scale missing|tensor missing"):
        validate_auxiliary_precision(tmp_path, expected)


def test_disabled_input_allows_explicit_w4a16(tmp_path):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent(input_enabled=False)}), recipe()
    )
    write_export(tmp_path, input_scale=False)
    layers = {"model.visual.proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}
    assert validate_auxiliary_precision(tmp_path, expected, layers)["input_scaled"] == 0


def test_group_metadata_matches_actual_quantizer(tmp_path):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent(group=32)}), recipe()
    )
    write_export(tmp_path, group=32)
    layers = {"model.visual.proj": {"quant_algo": "NVFP4", "group_size": 32}}
    assert validate_auxiliary_precision(tmp_path, expected, layers)["checked"] == 1
    layers["model.visual.proj"]["group_size"] = 16
    with pytest.raises(BackendError, match="metadata mismatch"):
        validate_auxiliary_precision(tmp_path, expected, layers)


def test_empty_scale_shape_is_backend_error(tmp_path):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent()}), recipe()
    )
    write_export(tmp_path)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    tensors["model.visual.proj.weight_scale"] = torch.ones((), dtype=torch.float8_e4m3fn)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(BackendError, match="weight_scale shape"):
        validate_auxiliary_precision(tmp_path, expected)


def test_index_controls_tensor_source_not_unindexed_copies(tmp_path):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent()}), recipe()
    )
    write_export(tmp_path)
    packed = load_file(str(tmp_path / "model.safetensors"))
    save_file(packed, str(tmp_path / "z-packed.safetensors"))
    key = "model.visual.proj.weight"
    save_file({key: torch.zeros(32, 32, dtype=torch.bfloat16)},
              str(tmp_path / "a-bf16.safetensors"))
    mapping = {key: "a-bf16.safetensors"}
    mapping.update({k: "z-packed.safetensors" for k in packed if k != key})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(BackendError, match="BF16"):
        validate_auxiliary_precision(tmp_path, expected)
    # Even a valid packed copy cannot supply an unindexed required weight.
    del mapping[key]
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(BackendError, match="module missing"):
        validate_auxiliary_precision(tmp_path, expected)


@pytest.mark.parametrize("shard", ["../escape.safetensors", "/tmp/escape.safetensors", 3])
def test_invalid_index_shard_rejected(tmp_path, shard):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent()}), recipe()
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.visual.proj.weight": shard}})
    )
    with pytest.raises(BackendError, match="Invalid safetensors"):
        validate_auxiliary_precision(tmp_path, expected)


def test_index_mapping_cannot_use_unindexed_shard_copy(tmp_path):
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": FakeParent()}), recipe()
    )
    write_export(tmp_path)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    save_file(tensors, str(tmp_path / "b.safetensors"))
    key = "model.visual.proj.weight"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "a.safetensors"}})
    )
    with pytest.raises(BackendError, match="Indexed safetensors shard missing"):
        validate_auxiliary_precision(tmp_path, expected)


def test_snapshot_survives_export_mutating_quantizers():
    parent = FakeParent()
    expected = snapshot_auxiliary_quantizers(
        FakeModel({"model.visual.proj": parent}), recipe()
    )
    parent.weight_quantizer.num_bits = (4, 3)
    parent.weight_quantizer.block_sizes[-1] = 32
    parent.input_quantizer.is_enabled = False
    spec = expected["model.visual.proj"]
    assert (spec["format"], spec["group_size"], spec["input_enabled"]) == ("NVFP4", 16, True)


def test_unsupported_input_precision_is_not_silently_accepted():
    parent = FakeParent()
    parent.input_quantizer.num_bits = (4, 3)
    with pytest.raises(BackendError, match="weight/input quantizer combination"):
        snapshot_auxiliary_quantizers(FakeModel({"model.visual.proj": parent}), recipe())


@pytest.mark.parametrize("branch", ["visual", "vision", "mtp"])
def test_supported_auxiliary_prefix_aliases(tmp_path, branch):
    model = FakeModel({f"model.{branch}.proj": FakeParent()})
    expected = snapshot_auxiliary_quantizers(
        model, recipe(vision=branch != "mtp", mtp=branch == "mtp")
    )
    write_export(tmp_path, module=f"{branch}.proj")
    assert validate_auxiliary_precision(tmp_path, expected)["checked"] == 1


def test_empty_expectations_need_no_checkpoint(tmp_path):
    assert validate_auxiliary_precision(tmp_path, {})["checked"] == 0
