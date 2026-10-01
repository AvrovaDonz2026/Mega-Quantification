"""CPU checks for the guarded SGLang Spark build-time compatibility patch."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

QUANT_SOURCE = '''class ModelOptFp8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.enable_flashinfer_bmm = flashinfer_per_tensor_fp8_supported()
        self.use_marlin = False
        if is_cuda():
            self.use_marlin = (
                envs.SGLANG_FORCE_FP8_MARLIN.get() or can_auto_enable_marlin_fp8()
            )
        # The SM12x facade selects the best qualified small-M FP8 kernel.
        cuda_capability = torch.cuda.get_device_capability() if is_cuda() else None
        self.use_sm120_fp8 = cuda_capability is not None and cuda_capability[0] == 12

class ModelOptFp4Config:
    @classmethod
    def from_config(cls, config):
        quant_config = cls()
        quant_method = config.get("quant_algo")
        quant_config.is_w4a16 = quant_method == "W4A16_NVFP4"
        return quant_config

class ModelOptFp4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def apply(
        self, layer, x, bias=None,
    ):
        x_m, _ = x.shape
        layer.last_input = x
        output = x @ layer.weight.T
        if bias is not None:
            output = output + bias
        return output.reshape(x_m, layer.weight.shape[0])

class ModelOptNvFp4A16LinearMethod(LinearMethodBase):
    pass
'''

CONFIG_SOURCE = '''class ModelConfig:
    def _config_draft_model(self):
        if self.is_draft_model:
            self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"
            self.hf_config.num_nextn_predict_layers = 1
            self.hf_text_config.num_nextn_predict_layers = 1
'''

UNQUANT_SOURCE = '''class UnquantizedLinearMethod(LinearMethodBase):
    def apply(
        self, layer, x, bias=None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        elif (
            get_bf16_gemm_backend().is_cutedsl()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and (bias is None or bias.dtype == torch.bfloat16)
            and not layer.weight.requires_grad
            and (bias is None or not bias.requires_grad)
        ):
            if torch.compiler.is_compiling():
                # The m-dependent kernel heuristic would guard on the symbolic
                # token dim under Dynamo and recompile per shape bucket; the
                # opaque op resolves it at runtime with concrete shapes,
                # keeping the per-shape kernel choice.
                return bf16_gemm_dispatch(x, layer.weight, bias)
            return _bf16_gemm_dispatch_impl(x, layer.weight, bias)

        return F.linear(x, layer.weight, bias)
'''

FLASHINFER_PATH = "srt/layers/attention/flashinfer_backend.py"
FLASHINFER_SOURCE = '''class FlashInferAttnBackend:
    def __init__(self, model_runner):
        self.__dict__.update(model_runner.backend_options)
        fmha_backend = "auto"

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if forward_batch.forward_mode.is_decode_or_idle():
            self.forward_metadata = "original-decode"
        elif forward_batch.forward_mode.is_target_verify():
            self.forward_metadata = "original-verify"
        else:
            self.forward_metadata = "original-prefill"

    def init_cuda_graph_state(
        self,
    ):
        return "original-graph"

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if k is not None and save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        return self.prefill_result(q)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        return decode_wrapper.forward(q, k, v, layer, forward_batch, save_kv_cache)

    def _get_wrapper_idx(
        self, layer,
    ):
        return 0

class FlashInferIndicesUpdaterPrefill:
    def call_begin_forward(
        self, wrapper_ragged, wrapper_paged, req_pool_indices,
        paged_kernel_lens, paged_kernel_lens_sum, seq_lens, prefix_lens,
        kv_start_idx, kv_indptr, qo_indptr, use_ragged, spec_info,
        cross_attention_custom_mask=None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert prefix_lens is not None
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            self.attn_backend.kv_index_translator.fill_packed_read_stream(
                req_pool_indices=req_pool_indices, seq_lens=paged_kernel_lens,
                indptr=kv_indptr, total_tokens=paged_kernel_lens_sum,
                out=kv_indices, kv_start_idx=kv_start_idx,
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = cross_attention_custom_mask
        else:
            return "original-spec"
        return qo_indptr, kv_indptr, kv_indices, custom_mask

class FlashInferMultiStepDraftBackend:
    pass
'''

FIXTURES = {
    "srt/models/qwen3_vl.py": VISION_SOURCE,
    "srt/models/qwen3_5_mtp.py": MTP_SOURCE,
    "srt/layers/quantization/modelopt_quant.py": QUANT_SOURCE,
    "srt/configs/model_config.py": CONFIG_SOURCE,
    "kernels/ops/attention/fla/fused_recurrent.py": (
        "def packed_decode(b_val, b):\n"
        "    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)\n"
        "    return beta_val\n"
    ),
    "srt/layers/quantization/unquant.py": UNQUANT_SOURCE,
    FLASHINFER_PATH: FLASHINFER_SOURCE,
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
        "LinearMethodBase": object,
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


def test_old_patch_manifest_is_rejected_before_any_write(upstream_tree):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["patch"] = "megaquant-spark-v1"
    path.write_text(json.dumps(manifest))
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}


def test_old_v2_manifest_cannot_skip_gdn_precision_fix(upstream_tree):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["patch"] = "megaquant-spark-v2"
    del manifest["files"]["kernels/ops/attention/fla/fused_recurrent.py"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20", check=True)


def test_old_v3_manifest_cannot_skip_sm121_fp8_alignment(upstream_tree):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["patch"] = "megaquant-spark-v3"
    path.write_text(json.dumps(manifest))
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20", check=True)
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}


@pytest.mark.parametrize("check", [False, True])
def test_old_v4_manifest_cannot_skip_sm121_ba_alignment(upstream_tree, check):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["patch"] = "megaquant-spark-v4"
    del manifest["files"]["srt/layers/quantization/unquant.py"]
    path.write_text(json.dumps(manifest))
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    before_manifest = path.read_bytes()
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20", check=check)
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert path.read_bytes() == before_manifest


@pytest.mark.parametrize("check", [False, True])
def test_old_v5_manifest_cannot_skip_fp8_attention_alignment(upstream_tree, check):
    patch.patch_tree(upstream_tree, "0.5.20")
    path = upstream_tree / patch.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["patch"] = "megaquant-spark-v5"
    del manifest["files"][FLASHINFER_PATH]
    path.write_text(json.dumps(manifest))
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    before_manifest = path.read_bytes()
    with pytest.raises(patch.PatchError, match="manifest or installed files"):
        patch.patch_tree(upstream_tree, "0.5.20", check=check)
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert path.read_bytes() == before_manifest


def test_seventh_attention_target_hash_drift_fails_before_any_write(upstream_tree):
    target = upstream_tree / FLASHINFER_PATH
    target.write_text(target.read_text() + "# upstream attention changed\n")
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="flashinfer_backend.py: upstream SHA256 mismatch"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert not (upstream_tree / patch.MANIFEST).exists()


@pytest.mark.parametrize(
    ("old", "new", "label"),
    [
        ('fmha_backend = "auto"', 'fmha_backend = "changed"', "scope"),
        ("if forward_batch.forward_mode.is_decode_or_idle():", "if changed():", "plan"),
        ("decode_wrapper = self.forward_metadata.decode_wrappers[", "changed = [", "forward"),
        ("custom_mask = cross_attention_custom_mask", "custom_mask = changed", "mask"),
    ],
)
def test_attention_fragment_drift_fails_before_any_write(
    upstream_tree, monkeypatch, old, new, label
):
    target = upstream_tree / FLASHINFER_PATH
    target.write_text(target.read_text().replace(old, new))
    monkeypatch.setitem(
        patch.UPSTREAM_SHA256, FLASHINFER_PATH, hashlib.sha256(target.read_bytes()).hexdigest()
    )
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match=f"attention decode {label}.*expected one upstream"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert not (upstream_tree / patch.MANIFEST).exists()


class MockSequence:
    def __init__(self, values):
        self.values = list(values)
        self.device = "cuda:0"

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        value = self.values[key]
        return MockSequence(value) if isinstance(key, slice) else value

    def __setitem__(self, key, value):
        self.values[key] = value.values if isinstance(value, MockSequence) else value

    def __sub__(self, other):
        if isinstance(other, MockSequence):
            return MockSequence(a - b for a, b in zip(self.values, other.values, strict=True))
        return MockSequence(value - other for value in self.values)


class MockAttentionTorch:
    float8_e4m3fn, bool, int32 = "fp8", "bool", "int32"

    def __init__(self):
        self.capability, self.available = (12, 1), True
        self.cuda = SimpleNamespace(
            is_available=lambda: self.available,
            get_device_capability=lambda device: self.capability,
        )

    def cumsum(self, values, dim):
        assert dim == 0
        total, result = 0, []
        for value in values.values:
            total += value
            result.append(total)
        return MockSequence(result)

    def empty(self, count, dtype, device):
        assert dtype == self.int32 and device == "cuda:0"
        return MockSequence([0] * count)

    def ones(self, count, dtype, device):
        assert dtype == self.bool and device == "cuda:0"
        return MockSequence([True] * count)


class MockPrefillMetadata:
    def __init__(self, wrappers, use_ragged, extend_no_prefix, *, swa_out_cache_loc):
        self.prefill_wrappers = wrappers
        self.use_ragged = use_ragged
        self.extend_no_prefix = extend_no_prefix
        self.swa_out_cache_loc = swa_out_cache_loc


def _attention_fixture():
    torch = MockAttentionTorch()
    parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5"),
        num_attention_heads=24,
        kv_heads=4,
        head_dim=256,
    )
    config.get_num_kv_heads = lambda tp, dcp: config.kv_heads
    runner = SimpleNamespace(
        model_config=config,
        server_args=SimpleNamespace(disable_cuda_graph=True),
        is_draft_worker=False,
        device="cuda:0",
        backend_options={
            "flashinfer_kv_cache_dtype": "fp8",
            "page_size": 1,
            "num_wrappers": 1,
            "skip_prefill": False,
            "prefill_uses_dequant_workspace": False,
            "decode_uses_dequant_workspace": False,
        },
    )
    namespace = _namespace(
        FLASHINFER_PATH,
        torch=torch,
        get_parallel=lambda: parallel,
        ForwardBatch=object,
        PrefillMetadata=MockPrefillMetadata,
    )
    return namespace, runner, torch, parallel


@pytest.mark.parametrize(
    ("owner", "attribute", "value"),
    [
        pytest.param(None, None, None, id="supported-sm121"),
        pytest.param("torch", "capability", (12, 0), id="sm120"),
        pytest.param("torch", "capability", (12, 2), id="sm122"),
        pytest.param("torch", "capability", (9, 0), id="sm90"),
        pytest.param("torch", "available", False, id="no-cuda"),
        pytest.param("options", "flashinfer_kv_cache_dtype", "bf16", id="bf16-kv"),
        pytest.param("options", "page_size", 16, id="paged-kv"),
        pytest.param("options", "num_wrappers", 2, id="swa-or-cross-attention"),
        pytest.param("options", "skip_prefill", True, id="skip-prefill"),
        pytest.param("options", "prefill_uses_dequant_workspace", True, id="prefill-fp4"),
        pytest.param("options", "decode_uses_dequant_workspace", True, id="decode-fp4"),
        pytest.param("runner", "is_draft_worker", True, id="draft"),
        pytest.param("config", "num_attention_heads", 32, id="different-q-heads"),
        pytest.param("config", "kv_heads", 8, id="different-kv-heads"),
        pytest.param("config", "head_dim", 128, id="different-head-dim"),
        pytest.param("parallel", "attn_tp_size", 2, id="different-tp-heads"),
        pytest.param("model", "model_type", "qwen3_5_moe", id="different-model"),
        pytest.param("server", "disable_cuda_graph", False, id="cuda-graph"),
    ],
)
def test_attention_decode_guard_is_limited_to_verified_shape(owner, attribute, value):
    namespace, runner, torch, parallel = _attention_fixture()
    objects = {
        "runner": runner,
        "torch": torch,
        "parallel": parallel,
        "config": runner.model_config,
        "model": runner.model_config.hf_config,
        "server": runner.server_args,
    }
    if owner == "options":
        runner.backend_options[attribute] = value
    elif owner is not None:
        setattr(objects[owner], attribute, value)
    backend = namespace["FlashInferAttnBackend"](runner)
    assert backend._megaquant_sm121_fp8_prefill_decode is (owner is None)


def test_attention_decode_guard_preserves_subclasses():
    namespace, runner, _, _ = _attention_fixture()

    class Derived(namespace["FlashInferAttnBackend"]):
        pass

    assert Derived(runner)._megaquant_sm121_fp8_prefill_decode is False


@pytest.mark.parametrize("lengths", [[53], [53, 79], [1], [1, 2]])
@pytest.mark.parametrize("decode", [False, True])
def test_attention_query_and_kv_indices_preserve_request_boundaries(lengths, decode):
    namespace, runner, _, _ = _attention_fixture()
    backend = namespace["FlashInferAttnBackend"](runner)
    wrapper = object()
    backend.prefill_wrappers_verify = [wrapper]

    def fill(**arguments):
        offset = 0
        for request, length in zip(
            arguments["req_pool_indices"].values, arguments["seq_lens"].values, strict=True
        ):
            arguments["out"].values[offset : offset + length] = [
                1000 * request + token for token in range(length)
            ]
            offset += length

    backend.kv_index_translator = SimpleNamespace(fill_packed_read_stream=fill)
    updater = namespace["FlashInferIndicesUpdaterPrefill"]()
    updater.attn_backend = backend
    count = len(lengths)
    seq_lens = MockSequence(lengths)
    query_count = 3 if decode else 1
    original_mask = object()
    qo, kv, indices, mask = updater.call_begin_forward(
        None, wrapper if decode else object(), MockSequence(range(count)),
        seq_lens, sum(lengths), seq_lens, seq_lens - query_count, None,
        MockSequence([0] * (count + 1)), MockSequence([0] * (count + 1)),
        False, None, cross_attention_custom_mask=original_mask,
    )
    assert qo.values == [query_count * request for request in range(count + 1)]
    assert kv.values == [sum(lengths[:request]) for request in range(count + 1)]
    assert indices.values[:sum(lengths)] == [
        1000 * request + token for request, length in enumerate(lengths) for token in range(length)
    ]
    if decode:
        assert mask.values == [True] * (3 * sum(lengths))
    else:
        assert mask is original_mask


@pytest.mark.parametrize("phase", ["decode", "verify", "prefill", "idle"])
@pytest.mark.parametrize("spec_info", [None, "draft-info"])
def test_attention_metadata_changes_only_normal_decode(phase, spec_info):
    namespace, runner, _, _ = _attention_fixture()
    backend = namespace["FlashInferAttnBackend"](runner)
    calls = []
    backend.prefill_wrappers_verify = [object()]
    backend.indices_updater_prefill = SimpleNamespace(update=lambda *a, **kw: calls.append((a, kw)))
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: phase == "decode",
            is_decode_or_idle=lambda: phase in {"decode", "idle"},
            is_target_verify=lambda: phase == "verify",
        ),
        spec_info=spec_info,
        req_pool_indices=MockSequence([0, 1]),
        seq_lens=MockSequence([1, 53]),
        seq_lens_cpu=[1, 53],
        seq_lens_sum=54,
        encoder_lens=None,
    )
    backend.init_forward_metadata(batch)
    if phase == "decode" and spec_info is None:
        assert isinstance(backend.forward_metadata, MockPrefillMetadata)
        assert backend.forward_metadata.prefill_wrappers is backend.prefill_wrappers_verify
        assert backend.forward_metadata.use_ragged is False
        assert backend.forward_metadata.extend_no_prefix is False
        assert len(calls) == 1
        assert calls[0][0] == (
            batch.req_pool_indices, batch.seq_lens, batch.seq_lens_cpu, batch.seq_lens_sum,
        )
        assert calls[0][1]["prefix_lens"].values == [-2, 50]
        assert calls[0][1]["spec_info"] is None
        assert calls[0][1]["fixed_split_size"] is None
    else:
        expected = "decode" if phase == "idle" else phase
        assert backend.forward_metadata == "original-" + expected
        assert calls == []


class MockAttentionRows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows), 24, 256)

    def repeat_interleave(self, count, dim):
        assert count == 3 and dim == 0
        return MockAttentionRows(row for row in self.rows for _ in range(count))

    def reshape(self, count, clones, width):
        assert clones == 3 and width == -1 and len(self.rows) == count * clones
        return self

    def __getitem__(self, key):
        assert key == (slice(None), 0)
        return MockAttentionRows(self.rows[::3])

    def contiguous(self):
        return self


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("save_kv_cache", [False, True])
def test_attention_decode_pads_only_queries_and_writes_original_kv_once(count, save_kv_cache):
    namespace, runner, _, _ = _attention_fixture()
    backend = namespace["FlashInferAttnBackend"](runner)
    backend.forward_metadata = MockPrefillMetadata([], False, False, swa_out_cache_loc=None)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        spec_info=None,
        out_cache_loc=object(),
    )
    q = MockAttentionRows(range(count))
    k, v, layer = object(), object(), object()
    calls, writes = [], []

    def result(queries):
        calls.append(queries.rows)
        return MockAttentionRows(range(len(queries.rows)))

    backend.prefill_result = result
    backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=lambda *a: writes.append(a))
    output = backend.forward_decode(q, k, v, layer, batch, save_kv_cache=save_kv_cache)
    assert calls == [[row for row in range(count) for _ in range(3)]]
    assert writes == ([(layer, batch.out_cache_loc, k, v)] if save_kv_cache else [])
    assert output.rows == [3 * row for row in range(count)]


@pytest.mark.parametrize("case", ["draft", "cpu", "sm120", "verify", "spec-decode"])
def test_attention_decode_preserves_original_fallback(case):
    namespace, runner, torch, _ = _attention_fixture()
    if case == "draft":
        runner.is_draft_worker = True
    elif case == "cpu":
        torch.available = False
    elif case == "sm120":
        torch.capability = (12, 0)
    backend = namespace["FlashInferAttnBackend"](runner)
    fallback, calls = object(), []

    def forward(*arguments):
        calls.append(arguments)
        return fallback

    backend.forward_metadata = SimpleNamespace(decode_wrappers=[SimpleNamespace(forward=forward)])
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: case != "verify"),
        spec_info=object() if case == "spec-decode" else None,
    )
    arguments = (object(), object(), object(), object(), batch, False)
    assert backend.forward_decode(*arguments) is fallback
    assert calls == [arguments]


def test_attention_patch_preserves_all_original_statements():
    source = patch.transform(FLASHINFER_PATH, FLASHINFER_SOURCE)
    source = source.removeprefix(patch.MARKER)
    for addition in (
        patch.FLASHINFER_DECODE_GUARD, patch.FLASHINFER_DECODE_PLAN,
        patch.FLASHINFER_DECODE_FORWARD, patch.FLASHINFER_DECODE_MASK,
    ):
        assert source.count(addition) == 1
        source = source.replace(addition, "", 1)
    assert source == FLASHINFER_SOURCE
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(FLASHINFER_SOURCE))


def test_sixth_ba_target_hash_drift_fails_before_any_write(upstream_tree):
    target = upstream_tree / "srt/layers/quantization/unquant.py"
    target.write_text(target.read_text() + "# upstream BA implementation changed\n")
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="unquant.py: upstream SHA256 mismatch"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert not (upstream_tree / patch.MANIFEST).exists()


def test_ba_fragment_drift_fails_before_any_write(upstream_tree, monkeypatch):
    name = "srt/layers/quantization/unquant.py"
    target = upstream_tree / name
    target.write_text(
        target.read_text().replace("if use_intel_amx_backend(layer):", "if changed(layer):")
    )
    monkeypatch.setitem(
        patch.UPSTREAM_SHA256, name, hashlib.sha256(target.read_bytes()).hexdigest()
    )
    before = {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    with pytest.raises(patch.PatchError, match="BF16 GDN BA.*expected one upstream fragment"):
        patch.patch_tree(upstream_tree, "0.5.20")
    assert before == {name: (upstream_tree / name).read_bytes() for name in FIXTURES}
    assert not (upstream_tree / patch.MANIFEST).exists()


@pytest.mark.parametrize(
    ("capability", "is_cuda_device", "use_facade"),
    [
        ((12, 0), True, True),
        ((12, 1), True, False),
        ((12, 2), True, True),
        ((9, 0), True, False),
        (None, False, False),
    ],
)
def test_fp8_constructor_uses_common_backend_only_on_sm121(capability, is_cuda_device, use_facade):
    capability_queries = []

    def get_device_capability():
        assert is_cuda_device
        capability_queries.append(capability)
        return capability

    namespace = _namespace(
        "srt/layers/quantization/modelopt_quant.py",
        torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=get_device_capability)),
        is_cuda=lambda: is_cuda_device,
        cutlass_fp8_supported=lambda: True,
        flashinfer_per_tensor_fp8_supported=lambda: True,
        envs=SimpleNamespace(SGLANG_FORCE_FP8_MARLIN=SimpleNamespace(get=lambda: False)),
        can_auto_enable_marlin_fp8=lambda: False,
    )
    config = object()
    method = namespace["ModelOptFp8LinearMethod"](config)
    assert method.use_sm120_fp8 is use_facade
    assert method.quant_config is config
    assert method.cutlass_fp8_supported is True
    assert method.enable_flashinfer_bmm is True
    assert method.use_marlin is False
    assert capability_queries == ([capability] if is_cuda_device else [])


@pytest.mark.parametrize(
    ("changes", "use_persistent"),
    [
        pytest.param({}, True, id="sm121-m1"),
        pytest.param({"rows": 3}, True, id="sm121-m3"),
        pytest.param({"capability": (12, 0)}, False, id="sm120"),
        pytest.param({"capability": (12, 2)}, False, id="sm122"),
        pytest.param({"capability": (9, 0)}, False, id="sm90"),
        pytest.param({"is_cuda": False}, False, id="cpu"),
        pytest.param(
            {"prefix": "model.layers.2.linear_attn.in_proj_a"}, False, id="other-projection"
        ),
        pytest.param({"prefix": "in_proj_ba"}, False, id="bare-prefix"),
        pytest.param({"dtype": "fp16"}, False, id="fp16-input"),
        pytest.param({"weight_dtype": "fp16"}, False, id="fp16-weight"),
        pytest.param({"ndim": 3}, False, id="rank3-input"),
        pytest.param({"bias": True}, False, id="bias"),
        pytest.param({"requires_grad": True}, False, id="input-grad"),
        pytest.param({"weight_requires_grad": True}, False, id="weight-grad"),
    ],
)
def test_ba_apply_uses_fixed_triton_only_for_sm121_inference(monkeypatch, changes, use_persistent):
    arguments = {
        "rows": 1,
        "capability": (12, 1),
        "is_cuda": True,
        "prefix": "model.layers.2.linear_attn.in_proj_ba",
        "dtype": "bf16",
        "weight_dtype": "bf16",
        "ndim": 2,
        "bias": False,
        "requires_grad": False,
        "weight_requires_grad": False,
        **changes,
    }
    calls, capability_queries = [], []
    persistent_output, fallback_output, transposed_weight = object(), object(), object()

    def persistent(**kwargs):
        calls.append(("persistent", kwargs))
        return persistent_output

    def linear(*args):
        calls.append(("fallback", args))
        return fallback_output

    def capability(device):
        capability_queries.append(device)
        return arguments["capability"]

    module_name = "sglang.srt.batch_invariant_ops.batch_invariant_ops"
    module = ModuleType(module_name)
    module._matmul_persistent_triton = persistent
    monkeypatch.setitem(sys.modules, module_name, module)
    namespace = _namespace(
        "srt/layers/quantization/unquant.py",
        torch=SimpleNamespace(
            Tensor=object,
            bfloat16="bf16",
            cuda=SimpleNamespace(get_device_capability=capability),
        ),
        _is_cuda=True,
        _use_aiter=False,
        F=SimpleNamespace(linear=linear),
        use_intel_amx_backend=lambda layer: False,
        get_bf16_gemm_backend=lambda: SimpleNamespace(is_cutedsl=lambda: False),
    )
    x = SimpleNamespace(
        is_cuda=arguments["is_cuda"],
        ndim=arguments["ndim"],
        shape=(arguments["rows"], 5120),
        dtype=arguments["dtype"],
        requires_grad=arguments["requires_grad"],
        device="cuda:0",
    )
    weight = SimpleNamespace(
        is_cuda=arguments["is_cuda"],
        ndim=2,
        dtype=arguments["weight_dtype"],
        requires_grad=arguments["weight_requires_grad"],
        t=lambda: transposed_weight,
    )
    layer = SimpleNamespace(prefix=arguments["prefix"], weight=weight)
    bias = object() if arguments["bias"] else None
    result = namespace["UnquantizedLinearMethod"]().apply(layer, x, bias)
    if use_persistent:
        assert result is persistent_output
        assert calls == [("persistent", {"a": x, "b": transposed_weight, "out_dtype": x.dtype})]
    else:
        assert result is fallback_output
        assert calls == [("fallback", (x, weight, bias))]
    assert capability_queries == ([x.device] if use_persistent or "capability" in changes else [])


def test_ba_patch_preserves_all_original_fallback_statements():
    original = ast.parse(UNQUANT_SOURCE).body[0].body[0]
    source = patch.transform("srt/layers/quantization/unquant.py", UNQUANT_SOURCE)
    patched = ast.parse(source).body[0].body[0]
    # Only the new guarded dispatch precedes the original apply implementation.
    assert isinstance(patched.body[0], ast.If)
    assert ast.dump(ast.Module(body=patched.body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=original.body, type_ignores=[])
    )


def test_vision_uses_explicit_quantization_without_mutating_text_config():
    helper = _namespace("srt/models/qwen3_vl.py")["_megaquant_vision_quant_config"]
    key = "model.visual.blocks.0.attn.qkv"
    cfg = QuantConfig("modelopt_mixed", {key: {"quant_algo": "FP8"}})
    visual = helper(cfg)
    assert visual is not cfg
    assert visual.quantized_layers[key + "_proj"] == {"quant_algo": "FP8"}
    assert visual.quantized_layers["visual.blocks.0.attn.qkv_proj"] == {
        "quant_algo": "FP8"
    }
    assert visual.quantized_layers["model.visual.blocks.0.attn.qkv_proj"] == {
        "quant_algo": "FP8"
    }
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


@pytest.mark.parametrize("name", ["modelopt_mixed", "modelopt_fp4"])
@pytest.mark.parametrize("shape", [(2, 3, 8), (2, 3, 4, 8), (0, 3, 8), (8,), (6, 8)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_nvfp4_visual_linear_preserves_leading_dimensions(name, shape, use_bias):
    torch = pytest.importorskip("torch")
    helper = _namespace("srt/models/qwen3_vl.py")["_megaquant_vision_quant_config"]
    config = helper(
        QuantConfig(
            name,
            {"model.visual.blocks.0.attn.qkv": {"quant_algo": "NVFP4"}},
            uniform=name == "modelopt_fp4",
        )
    )
    cls = _namespace("srt/layers/quantization/modelopt_quant.py", torch=torch)[
        "ModelOptFp4LinearMethod"
    ]
    method = cls(config)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    if x.ndim > 2 and x.numel():
        x = x.transpose(0, 1)
        assert not x.is_contiguous()
    layer = SimpleNamespace(weight=torch.arange(40, dtype=torch.float64).reshape(5, 8))
    bias = torch.arange(5, dtype=torch.float64) if use_bias else None
    output = method.apply(layer, x, bias)
    torch.testing.assert_close(output, torch.nn.functional.linear(x, layer.weight, bias))
    assert output.shape == (*x.shape[:-1], 5)
    assert layer.last_input.ndim == 2
    if x.ndim == 2:
        assert layer.last_input is x


def test_nvfp4_shape_adapter_preserves_prequantized_tuple_path():
    torch = pytest.importorskip("torch")
    decorate = _namespace("srt/layers/quantization/modelopt_quant.py", torch=torch)[
        "_megaquant_fp4_apply_nd"
    ]
    packed = (torch.ones(3, 4, dtype=torch.uint8), torch.ones(3, 1))
    layer = object()
    bias = object()
    seen = []

    def apply(self, received_layer, x, received_bias):
        seen.append((self, received_layer, x, received_bias))
        return x

    method = object()
    assert decorate(apply)(method, layer, packed, bias) is packed
    assert seen == [(method, layer, packed, bias)]


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
