"""Exercise the streaming traversal with real CPU tensors and a packing handler."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import itertools
import sys
import types
import zipfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch import nn

from megaquant import modelopt_streaming as compat
from megaquant.exceptions import BackendError

# Small executable contract for CI without ModelOpt. The optional wheel cases
# execute the complete pinned vendor function using exactly the same harness.
_CONTRACT = """def _export_transformers_checkpoint_streaming(model, export_dir="."):
    ctx = ExportContext(model=model)
    writer = _StreamingShardWriter(export_dir, 100)
    seen_keys: set[str] = set()
    decoder_owned_ids = {id(m) for layer in model.decoders for m in layer.modules()}
    name_to_module = dict(model.named_modules())
    for layer_name, layer_module in model.named_modules():
        if layer_module not in model.decoders:
            continue
        with enable_weight_access_and_writeback(
            layer_module, model, name_to_module, writeback=False
        ):
            for sub_name, sub_mod in layer_module.named_modules():
                full_name = f"{layer_name}.{sub_name}" if sub_name else layer_name
                _dispatch_export_handler(full_name, sub_mod, ctx)
            for key, tensor in layer_module.state_dict().items():
                full_key = layer_name + "." + key
                if full_key not in seen_keys:
                    seen_keys.add(full_key)
                    writer.add(full_key, tensor)
    for name, module in model.named_modules():
        if id(module) in decoder_owned_ids:
            continue
        if not requires_weight_materialization(module, model, name_to_module):
            continue
        with enable_weight_access_and_writeback(module, model, name_to_module, writeback=False):
            for sub_name, sub_mod in module.named_modules():
                full_name = f"{name}.{sub_name}" if sub_name else name
                _dispatch_export_handler(full_name, sub_mod, ctx)
            for key, tensor in module.state_dict().items():
                full_key = name + "." + key
                if full_key not in seen_keys and not tensor.is_meta:
                    seen_keys.add(full_key)
                    writer.add(full_key, tensor)
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if name not in seen_keys and not tensor.is_meta:
            seen_keys.add(name)
            writer.add(name, tensor)
    writer.finalize()
    return None, {}
"""


class QuantLinear(nn.Linear):
    def __init__(self):
        super().__init__(16, 16, bias=False, dtype=torch.bfloat16)


class Config:
    tie_word_embeddings = False
    torch_dtype = torch.bfloat16
    _name_or_path = "/__nonexistent_model_source__"

    def save_pretrained(self, path):
        pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleList([nn.Sequential(QuantLinear())])
        self.visual = nn.Sequential(QuantLinear())
        self.mtp = nn.Sequential(QuantLinear())
        self.config = Config()
        self.generation_config = None


@pytest.fixture(params=["contract", "wheel"])
def exporter_source(request):
    if request.param == "contract":
        return _CONTRACT
    wheel = Path(__file__).parents[1] / "docker/wheelhouse/nvidia_modelopt-0.47.0-py3-none-any.whl"
    if not wheel.exists():
        pytest.skip("optional pinned ModelOpt wheel unavailable")
    with zipfile.ZipFile(wheel) as archive:
        source = archive.read("modelopt/torch/export/unified_export_hf_streaming.py").decode()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_export_transformers_checkpoint_streaming"
    )
    source = ast.get_source_segment(source, node) + "\n"
    assert hashlib.sha256(source.encode()).hexdigest() == compat._SOURCE_SHA256
    return source


@pytest.fixture
def harness(monkeypatch):
    def build(source):
        calls, writes, windows, restored = [], {}, [], []
        fail = {"name": None}

        def dispatch(name, module, ctx):
            calls.append(name)
            if name == fail["name"]:
                raise ValueError("packing failed")
            if isinstance(module, QuantLinear):
                assert module.weight.dtype == torch.bfloat16, "module packed twice"
                module.weight = nn.Parameter(
                    torch.ones(16, 8, dtype=torch.uint8), requires_grad=False
                )
                module.register_buffer("weight_scale", torch.ones(16, 1))

        @contextlib.contextmanager
        def access(module, *args, writeback):
            windows.append(writeback)
            module._materialized = True
            try:
                yield
            finally:
                module._materialized = False
                restored.append(module)

        def requires(module, *args):
            return getattr(module, "_offloaded", False)

        class Writer:
            def __init__(self, *args):
                pass

            def add(self, key, value):
                assert key not in writes, "tensor serialized twice"
                writes[key] = value.clone()

            def finalize(self):
                pass

        imports = {
            "modelopt.torch.quantization.plugins.huggingface": {
                "_reconstruct_fused_moe_linear": lambda *a: None
            },
            "modelopt.torch.quantization.utils.core_utils": {
                "enable_weight_access_and_writeback": access,
                "requires_weight_materialization": requires,
            },
            "modelopt.torch.quantization.utils.layerwise_calib": {
                "LayerActivationCollector": types.SimpleNamespace(
                    get_decoder_layers=lambda model: list(model.decoders)
                )
            },
            "modelopt.torch.export.registry": {
                "ExportModuleRegistry": types.SimpleNamespace(
                    match=lambda module: dispatch if isinstance(module, QuantLinear) else None
                )
            },
        }
        for name, attrs in imports.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, module)

        def no_op(*a, **kw):
            return None

        namespace = {
            "torch": torch,
            "nn": nn,
            "Path": Path,
            "itertools": itertools,
            "contextlib": contextlib,
            "Any": object,
            "_resolve_export_dtype": lambda *a: torch.bfloat16,
            "_prepare_moe_inputs": no_op,
            "requantize_resmooth_fused_llm_layers": no_op,
            "get_quant_config": lambda *a, **kw: {"quantization": {"kv_cache_quant_algo": None}},
            "_add_mtp_exclusions": no_op,
            "_warn_on_unsynced_moe_gate_up": no_op,
            "_assert_no_split_rules": no_op,
            "build_reverse_name_mapper": no_op,
            "_parse_shard_size": lambda *a: 100,
            "_StreamingShardWriter": Writer,
            "ExportContext": lambda **kw: types.SimpleNamespace(**kw),
            "_postprocess_single_tensor": lambda key, value, *a: (key, value),
            "_dispatch_export_handler": dispatch,
            "_sanitize_generation_config_for_save": no_op,
            "_patch_revert_weight_conversion": lambda: [],
            "_unpatch_revert_weight_conversion": no_op,
            "enable_weight_access_and_writeback": access,
            "requires_weight_materialization": requires,
        }
        exec(compile(source, "<vendor-export-test>", "exec"), namespace)
        original = namespace["_export_transformers_checkpoint_streaming"]
        streaming = types.ModuleType(compat._MODULE)
        streaming._export_transformers_checkpoint_streaming = original
        monkeypatch.setattr(compat, "_modelopt_version", lambda: "0.47.0")
        monkeypatch.setattr(compat.importlib, "import_module", lambda name: streaming)
        monkeypatch.setattr(compat.inspect, "getsource", lambda fn: source)
        monkeypatch.setattr(compat, "_SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
        return types.SimpleNamespace(
            module=streaming,
            original=original,
            calls=calls,
            writes=writes,
            windows=windows,
            restored=restored,
            fail=fail,
            dispatch=dispatch,
        )

    return build


def test_vendor_bf16_counterexample_and_repaired_packing(exporter_source, harness, tmp_path):
    original = harness(exporter_source)
    original.original(Model(), export_dir=tmp_path)
    assert original.writes["visual.0.weight"].dtype == torch.bfloat16
    assert original.writes["mtp.0.weight"].dtype == torch.bfloat16
    fixed = harness(exporter_source)
    with compat.streaming_export_compat(offloaded=True):
        fixed.module._export_transformers_checkpoint_streaming(Model(), export_dir=tmp_path)
    for prefix in ("visual.0", "mtp.0", "decoders.0.0"):
        assert fixed.writes[prefix + ".weight"].dtype == torch.uint8
        assert fixed.writes[prefix + ".weight"].shape == (16, 8)
        assert prefix + ".weight_scale" in fixed.writes
    assert fixed.module._export_transformers_checkpoint_streaming is fixed.original
    assert all(count == 1 for count in Counter(fixed.calls).values())
    assert fixed.windows and not any(fixed.windows)


def test_offloaded_parent_and_child_never_pack_twice(exporter_source, harness, tmp_path):
    state = harness(exporter_source)
    model = Model()
    model.visual._offloaded = model.visual[0]._offloaded = True
    with compat.streaming_export_compat(offloaded=True):
        state.module._export_transformers_checkpoint_streaming(model, export_dir=tmp_path)
    assert Counter(state.calls)["visual.0"] == 1
    assert Counter(state.calls)["decoders.0.0"] == 1
    assert state.writes["visual.0.weight"].dtype == torch.uint8
    assert "visual.0.weight_scale" in state.writes
    assert not any(state.windows)
    assert all(not module._materialized for module in state.restored)


def test_handler_failure_restores_context_and_exporter(exporter_source, harness, tmp_path):
    state = harness(exporter_source)
    model = Model()
    model.visual._offloaded = True
    state.fail["name"] = "visual.0"
    with pytest.raises(ValueError, match="packing failed"):
        with compat.streaming_export_compat(offloaded=True):
            state.module._export_transformers_checkpoint_streaming(model, export_dir=tmp_path)
    assert state.module._export_transformers_checkpoint_streaming is state.original
    assert not model.visual._materialized
    assert not any(state.windows)


def test_nested_context_and_live_dispatch_globals(harness, tmp_path):
    state = harness(_CONTRACT)
    before_globals = dict(state.original.__globals__)
    with compat.streaming_export_compat(offloaded=True):
        patched = state.module._export_transformers_checkpoint_streaming
        assert patched.__globals__ is state.original.__globals__
        with compat.streaming_export_compat(offloaded=True):
            assert state.module._export_transformers_checkpoint_streaming is patched
        seen = []

        def diagnostic(*args):
            seen.append(args[0])
            return state.dispatch(*args)

        patched.__globals__["_dispatch_export_handler"] = diagnostic
        patched(Model(), export_dir=tmp_path)
        assert seen
        patched.__globals__["_dispatch_export_handler"] = state.dispatch
    assert state.original.__globals__ == before_globals
    assert state.module._export_transformers_checkpoint_streaming is state.original


@pytest.mark.parametrize("version", [None, "0.46.1", "0.48.0"])
def test_other_versions_preserve_vendor(version, monkeypatch):
    monkeypatch.setattr(compat, "_modelopt_version", lambda: version)
    monkeypatch.setattr(
        compat.importlib, "import_module", lambda name: pytest.fail("unexpected import")
    )
    with compat.streaming_export_compat(offloaded=True):
        pass


def test_nonoffloaded_does_not_import_or_inspect_version(monkeypatch):
    monkeypatch.setattr(
        compat, "_modelopt_version", lambda: pytest.fail("unexpected version lookup")
    )
    with compat.streaming_export_compat(offloaded=False):
        pass


def test_modified_source_rejected_without_mutation(harness, monkeypatch):
    state = harness(_CONTRACT)
    monkeypatch.setattr(compat, "_SOURCE_SHA256", "0" * 64)
    with pytest.raises(BackendError, match="source"):
        with compat.streaming_export_compat(offloaded=True):
            pytest.fail("export must not start")
    assert state.module._export_transformers_checkpoint_streaming is state.original
