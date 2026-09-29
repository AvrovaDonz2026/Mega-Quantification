"""CPU checks for the guarded SGLang Spark build-time compatibility patch."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_sglang_spark.py"
SPEC = importlib.util.spec_from_file_location("spark_patch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)

VISION_SOURCE = '''class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        self.visual = Qwen3VLMoeVisionModel(
                config.vision_config,
                quant_config=None,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                prefix=add_prefix("model.visual", prefix),
        )
'''

MTP_SOURCE = '''from sglang.srt.layers.logits_processor import LogitsProcessor

def _mtp_quant_config(quant_config):
    if quant_config and quant_config.get_name() == "modelopt_mixed":
        # MIXED_PRECISION lists mtp.* layers only when the MTP head is quantized.
        if any(name.startswith("mtp.") for name in quant_config.quantized_layers):
            return quant_config
        return None
    if quant_config and (
        quant_config.get_name() == "modelopt_fp4"
        and quant_config.is_checkpoint_nvfp4_serialized
    ):
        return None
    return quant_config

class Qwen3_5ForCausalLMMTP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        quant_config = _mtp_quant_config(quant_config)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        self.config = mtp_config

    def forward(self, hidden_states):
        if hidden_states is not None:
            hidden_states = self.fc(hidden_states)
        return hidden_states
'''

QUANT_SOURCE = '''class ModelOptFp4Config:
    @classmethod
    def from_config(cls, config):
        quant_config = cls()
        quant_method = config.get("quant_algo")
        quant_config.is_w4a16 = quant_method == "W4A16_NVFP4"
        return quant_config
'''

CONFIG_SOURCE = '''class ModelConfig:
    def _config_draft_model(self):
        if self.is_draft_model:
            self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1
            self.hf_text_config.num_nextn_predict_layers = 1
'''

FIXTURES = {
    "srt/models/qwen3_vl.py": VISION_SOURCE,
    "srt/models/qwen3_5_mtp.py": MTP_SOURCE,
    "srt/layers/quantization/modelopt_quant.py": QUANT_SOURCE,
    "srt/configs/model_config.py": CONFIG_SOURCE,
}


class QuantConfig:
    def __init__(self, name, layers=None, *, serialized=True, uniform=False):
        self.name = name
        self.is_checkpoint_nvfp4_serialized = serialized
        if uniform:
            self._megaquant_quantized_layers = layers or {}
        else:
            self.quantized_layers = layers or {}

    def is_layer_excluded(self, prefix):
        return prefix.startswith(("visual.", "mtp."))

    def get_name(self):
        return self.name


def _namespace(relative_path, **extra):
    source = patch.transform(relative_path, FIXTURES[relative_path])
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
    namespace = {
        "nn": SimpleNamespace(Module=object),
        "copy": copy,
        "Qwen3_5ForCausalLM": SimpleNamespace(
            packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        ),
        **extra,
    }
    exec(compile(tree, "patched.py", "exec"), namespace)
    return namespace


@pytest.fixture
def upstream_tree(tmp_path, monkeypatch):
    hashes = {}
    for relative_path, source in FIXTURES.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        hashes[relative_path] = hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setattr(patch, "UPSTREAM_SHA256", hashes)
    return tmp_path


def test_patch_preflights_entire_tree_and_is_idempotent(upstream_tree):
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert patch.patch_tree(upstream_tree, "0.5.20", check=True)["status"] == "checked"
    assert not (upstream_tree / patch.MANIFEST).exists()
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    result = patch.patch_tree(upstream_tree, "0.5.20")
    assert result["status"] == "patched"
    for name, details in result["files"].items():
        raw = (upstream_tree / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == details["patched_sha256"]
        ast.parse(raw)
    second = patch.patch_tree(upstream_tree, "0.5.20")
    assert second["status"] == "already-patched"
    assert second["files"] == result["files"]


def test_unexpected_source_fails_before_any_write(upstream_tree):
    last = upstream_tree / "srt/configs/model_config.py"
    last.write_text(last.read_text() + "# changed upstream\n")
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="SHA256 mismatch"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert not (upstream_tree / patch.MANIFEST).exists()


@pytest.mark.parametrize("version", ["0.5.19", "0.5.20.post1", "0.5.21", "0.0.0.dev1"])
def test_other_versions_are_rejected(upstream_tree, version):
    with pytest.raises(patch.PatchError, match="expected SGLang"):
        patch.patch_tree(upstream_tree, version)
    assert not (upstream_tree / patch.MANIFEST).exists()


def test_modified_patched_source_is_rejected(upstream_tree):
    patch.patch_tree(upstream_tree, "0.5.20")
    target = upstream_tree / next(iter(FIXTURES))
    target.write_text(target.read_text() + "# modified after patch\n")
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20")


def test_manifest_cannot_omit_a_patch_target(upstream_tree):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    del manifest["files"][next(iter(FIXTURES))]
    path.write_text(json.dumps(manifest))
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20")


def test_vision_uses_explicit_quantization_without_mutating_text_config():
    helper = _namespace("srt/models/qwen3_vl.py")["_megaquant_vision_quant_config"]
    key = "model.visual.blocks.0.attn.qkv"
    cfg = QuantConfig("modelopt_mixed", {key: {"quant_algo": "FP8"}})
    visual = helper(cfg)
    assert visual is not cfg
    assert visual.quantized_layers[key + "_proj"] == {"quant_algo": "FP8"}
    assert key + "_proj" not in cfg.quantized_layers
    assert helper(QuantConfig("modelopt_mixed", {"model.layers.0": {}})) is None
    assert helper(QuantConfig("gptq", {key: {}})) is None
    assert helper(None) is None
    uniform = QuantConfig("modelopt_fp4", {key: {}}, uniform=True)
    assert helper(uniform) is uniform
    assert helper(QuantConfig("modelopt_fp4", uniform=True)) is None

    uniform_spark = QuantConfig("modelopt_fp4", uniform=True)
    uniform_spark.is_layer_excluded = lambda prefix: False
    assert helper(uniform_spark) is uniform_spark


def test_mtp_gate_distinguishes_serialized_bf16_and_quantized_heads():
    gate = _namespace("srt/models/qwen3_5_mtp.py")["_mtp_quant_config"]
    for name, uniform in [("modelopt_mixed", False), ("modelopt_fp4", True)]:
        quantized = QuantConfig(name, {"mtp.fc": {"quant_algo": "NVFP4"}}, uniform=uniform)
        assert gate(quantized) is quantized
        assert gate(QuantConfig(name, {"lm_head": {}}, uniform=uniform)) is None
    uniform_spark = QuantConfig("modelopt_fp4", uniform=True)
    uniform_spark.is_layer_excluded = lambda prefix: False
    assert gate(uniform_spark) is uniform_spark
    online = QuantConfig("modelopt_fp4", serialized=False)
    assert gate(online) is online
    assert gate(None) is None


def test_mtp_fc_receives_quant_config_and_returns_only_tensor():
    class FakeLinear:
        def __init__(self, input_size, output_size, **kwargs):
            self.shape = (input_size, output_size)
            self.kwargs = kwargs

        def __call__(self, value):
            return value + 1, None

    namespace = _namespace(
        "srt/models/qwen3_5_mtp.py",
        ReplicatedLinear=FakeLinear,
        add_prefix=lambda name, prefix: f"{prefix}.{name}" if prefix else name,
    )
    cls = namespace["Qwen3_5ForCausalLMMTP"]
    cfg = QuantConfig("modelopt_mixed", {"mtp.fc": {"quant_algo": "NVFP4"}})
    text = SimpleNamespace(hidden_size=8, num_hidden_layers=64, layer_types=["linear_attention"])
    model = cls(text, cfg)
    assert model.fc.shape == (16, 8)
    assert model.fc.kwargs == {"bias": False, "quant_config": cfg, "prefix": "mtp.fc"}
    assert model.forward(3) == 4
    assert model.config.layer_types == ["full_attention"]
    assert text.layer_types == ["linear_attention"]
    assert cls.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]


@pytest.mark.parametrize("nested", [False, True])
def test_uniform_config_preserves_exported_layer_evidence(nested):
    cls = _namespace("srt/layers/quantization/modelopt_quant.py")["ModelOptFp4Config"]
    quant = {"quant_algo": "NVFP4", "quantized_layers": {"mtp.fc": {"quant_algo": "NVFP4"}}}
    config = cls.from_config({"quantization": quant} if nested else quant)
    assert config._megaquant_quantized_layers == quant["quantized_layers"]
    assert cls.from_config({"quant_algo": "NVFP4"})._megaquant_quantized_layers == {}


def test_draft_cache_metadata_is_one_layer_and_target_config_is_unchanged():
    cls = _namespace(
        "srt/configs/model_config.py",
        get_hf_text_config=lambda config: config.text_config,
    )["ModelConfig"]
    target = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(num_hidden_layers=64, layer_types=["linear_attention"] * 64),
    )
    draft = cls()
    draft.hf_config = target
    draft.hf_text_config = target.text_config
    draft.is_draft_model = True
    draft._config_draft_model()
    assert target.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert target.text_config.num_hidden_layers == 64
    assert draft.hf_config is not target
    assert draft.hf_text_config is draft.hf_config.text_config
    assert draft.hf_text_config.num_hidden_layers == 1
    assert draft.hf_text_config.layer_types == ["full_attention"]
    assert draft.hf_text_config.full_attention_interval == 1
