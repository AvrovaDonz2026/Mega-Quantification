"""Qwen3.5 export compatibility checks that do not load model weights."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from megaquant.backends.modelopt import (
    _prepare_qwen35_export,
    _qwen35_export_l2norm,
)
from megaquant.exceptions import BackendError


def test_quantization_installs_bound_transformers_to_supported_gdn_api() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    project = tomllib.loads((root / "pyproject.toml").read_text())
    project_requirements = project["project"]["optional-dependencies"]["hf"]
    project_transformers = [
        Requirement(item) for item in project_requirements
        if Requirement(item).name == "transformers"
    ]
    assert len(project_transformers) == 1
    assert project_transformers[0].specifier == SpecifierSet(">=5.8,<5.15")

    for filename in ("requirements-gpu.txt", "requirements-gpu-spark.txt"):
        requirements = [
            Requirement(item.strip())
            for item in (root / "docker" / filename).read_text().splitlines()
            if item.strip() and not item.lstrip().startswith("#")
        ]
        transformers = [item for item in requirements if item.name == "transformers"]
        assert len(transformers) == 1
        assert transformers[0].specifier == SpecifierSet(">=5.8,<5.15")


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, *, missing: str | None = None):
    """Install the small Qwen3.5 API surface used by the export preparation."""

    transformers = types.ModuleType("transformers")
    models = types.ModuleType("transformers.models")
    qwen_pkg = types.ModuleType("transformers.models.qwen3_5")
    modeling = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")

    class Qwen3_5GatedDeltaNet:
        def __init__(self):
            self.chunk_gated_delta_rule = lambda *args: ("fla", None)
            self.recurrent_gated_delta_rule = self.chunk_gated_delta_rule
            self.causal_conv1d_fn = lambda *args: "fla-conv"
            self.causal_conv1d_update = self.causal_conv1d_fn

        def forward(self, value):
            return self.chunk_gated_delta_rule(value, value, value, value, value)

    def torch_chunk_gated_delta_rule(query, key, value, g, beta, **kwargs):
        return query, None

    def torch_recurrent_gated_delta_rule(query, key, value, g, beta, **kwargs):
        return query, None

    modeling.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNet
    if missing != "chunk":
        modeling.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    if missing != "recurrent":
        modeling.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    qwen_pkg.modeling_qwen3_5 = modeling
    models.qwen3_5 = qwen_pkg
    transformers.models = models
    for name, module in (
        ("transformers", transformers),
        ("transformers.models", models),
        ("transformers.models.qwen3_5", qwen_pkg),
        ("transformers.models.qwen3_5.modeling_qwen3_5", modeling),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return Qwen3_5GatedDeltaNet


def _fake_qwen35_model(gdn_cls):
    class Qwen3_5ForConditionalGeneration:
        def __init__(self):
            self.gdn = gdn_cls()

        def modules(self):
            return iter((self, self.gdn))

    return Qwen3_5ForConditionalGeneration()


def test_qwen35_export_installs_cpu_gdn_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    gdn_cls = _install_fake_transformers(monkeypatch)
    model = _fake_qwen35_model(gdn_cls)

    assert model.gdn.forward("input") == ("fla", None)
    assert _prepare_qwen35_export(model) is True
    assert model.gdn.forward("input") == ("input", None)
    assert model.gdn.recurrent_gated_delta_rule(1, 2, 3, 4, 5) == (1, None)
    assert model.gdn.causal_conv1d_fn is None
    assert model.gdn.causal_conv1d_update is None


def test_qwen35_export_fails_on_incompatible_gdn_api(monkeypatch: pytest.MonkeyPatch) -> None:
    gdn_cls = _install_fake_transformers(monkeypatch, missing="chunk")
    with pytest.raises(BackendError, match="torch_chunk_gated_delta_rule"):
        _prepare_qwen35_export(_fake_qwen35_model(gdn_cls))


@pytest.mark.parametrize("missing", (
    "chunk_gated_delta_rule", "recurrent_gated_delta_rule",
    "causal_conv1d_fn", "causal_conv1d_update",
))
def test_qwen35_export_rejects_module_level_gdn_api_before_modification(
    monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    gdn_cls = _install_fake_transformers(monkeypatch)
    model = _fake_qwen35_model(gdn_cls)
    unsupported = gdn_cls()
    delattr(unsupported, missing)
    model.modules = lambda: iter((model, model.gdn, unsupported))
    model.config = types.SimpleNamespace(architectures=None)

    with pytest.raises(BackendError, match=f"instance GDN API.*{missing}"):
        _prepare_qwen35_export(model)
    assert model.gdn.forward("input") == ("fla", None)
    assert not hasattr(unsupported, missing)
    assert model.config.architectures is None


def test_l2norm_export_patch_returns_output_and_rstd(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    fla = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    l2norm = types.ModuleType("fla.modules.l2norm")

    def original(*args, **kwargs):  # pragma: no cover - only restoration sentinel
        return "original"

    l2norm.l2norm_fwd = original
    fla_modules.l2norm = l2norm
    fla.modules = fla_modules
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.modules", fla_modules)
    monkeypatch.setitem(sys.modules, "fla.modules.l2norm", l2norm)

    x = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]], dtype=torch.bfloat16)
    with _qwen35_export_l2norm(True):
        y, rstd = l2norm.l2norm_fwd(x)
        assert y.dtype is torch.bfloat16
        assert torch.equal(y, torch.tensor(
            [[0.6, 0.8], [0.0, 1.0], [0.0, 0.0]], dtype=torch.bfloat16,
        ))
        assert rstd.dtype is torch.float32
        assert torch.allclose(rstd, torch.tensor([0.2, 0.5, 1000.0]))
        assert l2norm.l2norm_fwd(x, output_dtype=torch.float32)[0].dtype is torch.float32
    assert l2norm.l2norm_fwd is original

    with pytest.raises(RuntimeError, match="restore"):
        with _qwen35_export_l2norm(True):
            assert l2norm.l2norm_fwd is not original
            raise RuntimeError("restore")
    assert l2norm.l2norm_fwd is original


def test_real_qwen35_gdn_cpu_forward_uses_export_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    original_fallback = qwen35.torch_chunk_gated_delta_rule
    calls = []

    def chunk_fallback(query, key, value, g, beta, **kwargs):
        calls.append(query.shape)
        return original_fallback(query, key, value, g, beta, **kwargs)

    monkeypatch.setattr(qwen35, "torch_chunk_gated_delta_rule", chunk_fallback)
    config = qwen35.Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention"],
    )
    gdn = qwen35.Qwen3_5GatedDeltaNet(config, layer_idx=0)

    def unsupported(*args, **kwargs):
        raise AssertionError("export fallback was not installed")

    gdn.chunk_gated_delta_rule = unsupported

    class Qwen3_5ForConditionalGeneration:
        def modules(self):
            return iter((self, gdn))

    model = Qwen3_5ForConditionalGeneration()
    assert _prepare_qwen35_export(model) is True
    output = gdn(torch.randn(1, 4, 32))
    assert calls == [(1, 4, 2, 8)]
    assert output.shape == (1, 4, 32)
    assert torch.isfinite(output).all()
