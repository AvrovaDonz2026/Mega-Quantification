"""Export recovery policy without a CUDA or ModelOpt installation."""

from __future__ import annotations

import gc
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from megaquant.backends import modelopt
from megaquant.exceptions import BackendError


@pytest.fixture
def export_stub(monkeypatch: pytest.MonkeyPatch):
    def install(export_fn: Any) -> None:
        for name in ("modelopt", "modelopt.torch", "modelopt.torch.export"):
            module = ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
        sys.modules["modelopt.torch.export"].export_hf_checkpoint = export_fn
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
        monkeypatch.setattr("megaquant.vision_export.restore_vision_from_recipe", lambda *_: None)
        monkeypatch.setattr(modelopt, "_restore_mtp_export", lambda *_: None)
        monkeypatch.setattr(modelopt, "_rewrite_sglang_export", lambda *_: None)

    return install


def _recipe(output_dir: Path) -> dict[str, Any]:
    return {"scheme": "nvfp4_w4a4", "export": {"output_dir": str(output_dir)}}


@pytest.mark.parametrize("preparation", ["preserve-offload", "cuda", "cuda-unchanged", "cpu-only"])
def test_export_oom_does_not_retry_without_more_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_stub: Any, preparation: str
) -> None:
    export_calls = []
    prepare_calls = []

    def fake_export(_model: Any, *, export_dir: str) -> None:
        export_calls.append(export_dir)
        (Path(export_dir) / "__shard_part_00000.safetensors").write_bytes(b"partial")
        raise RuntimeError("CUDA out of memory. Tried to allocate 4.74 GiB")

    def prepare(_model: Any, **kwargs: Any) -> str:
        prepare_calls.append(kwargs)
        return preparation

    export_stub(fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", prepare)
    with pytest.raises(BackendError, match="refusing to repeat the same export") as caught:
        modelopt.ModelOptBackend().export(object(), _recipe(tmp_path))
    assert "MEGAQUANT_GPU_HEADROOM_GIB" in str(caught.value)
    assert "Tried to allocate 4.74 GiB" in str(caught.value)
    assert len(export_calls) == 1
    assert len(prepare_calls) == (1 if preparation == "preserve-offload" else 2)
    assert not list(tmp_path.glob("__shard_part_*"))
    assert not (tmp_path / "backend_meta.json").exists()
    assert caught.value.__context__ is None


@pytest.mark.parametrize("retry_preparation", ["cuda-freed", "partial-cpu"])
def test_export_retries_once_after_freeing_memory_and_releasing_failed_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_stub: Any, retry_preparation: str
) -> None:
    export_calls = []
    prepare_calls = []
    resources = []

    class TemporaryTensor:
        pass

    def fake_export(_model: Any, *, export_dir: str) -> None:
        export_calls.append(export_dir)
        if len(export_calls) == 1:
            tensor = TemporaryTensor()
            resources.append(weakref.ref(tensor))
            (Path(export_dir) / "__shard_part_00000.safetensors").write_bytes(b"partial")
            raise RuntimeError("CUDA out of memory")
        assert not list(Path(export_dir).glob("__shard_part_*"))

    def prepare(_model: Any, **kwargs: Any) -> str:
        prepare_calls.append(kwargs)
        if len(prepare_calls) == 1:
            return "cuda"
        gc.collect()
        assert resources[0]() is None, "failed export traceback still retains a tensor"
        assert kwargs == {"min_free_gib": 12}
        return retry_preparation

    export_stub(fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", prepare)
    assert modelopt.ModelOptBackend().export(object(), _recipe(tmp_path)) == tmp_path
    assert len(export_calls) == 2
    assert len(prepare_calls) == 2
    assert (tmp_path / "backend_meta.json").is_file()


def test_non_oom_export_error_is_propagated_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_stub: Any
) -> None:
    failure = ValueError("invalid export metadata")
    export_calls = []

    def fake_export(_model: Any, *, export_dir: str) -> None:
        export_calls.append(export_dir)
        raise failure

    export_stub(fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", lambda *_: "preserve-offload")
    with pytest.raises(ValueError) as caught:
        modelopt.ModelOptBackend().export(object(), _recipe(tmp_path))
    assert caught.value is failure
    assert len(export_calls) == 1


def test_preserved_placement_can_export_successfully_first_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_stub: Any
) -> None:
    export_calls = []

    def fake_export(_model: Any, *, export_dir: str) -> None:
        export_calls.append(export_dir)

    export_stub(fake_export)
    monkeypatch.setattr(modelopt, "_prepare_export_memory", lambda *_: "preserve-offload")
    assert modelopt.ModelOptBackend().export(object(), _recipe(tmp_path)) == tmp_path
    assert len(export_calls) == 1


@pytest.mark.parametrize(
    ("moved", "free_after", "expected"),
    [(False, 1, "cuda-unchanged"), (True, 1, "cuda-unchanged"),
     (False, 2, "cuda-unchanged"), (True, 2, "partial-cpu"), (True, 8, "cuda-freed")],
)
def test_memory_preparation_requires_measured_progress(
    monkeypatch: pytest.MonkeyPatch, moved: bool, free_after: int, expected: str
) -> None:
    cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                           empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    free_values = iter([1024**3, 1024**3, free_after * 1024**3])
    monkeypatch.setattr(modelopt, "_cuda_free_bytes", lambda: next(free_values))
    monkeypatch.setattr(modelopt, "_module_to_cpu", lambda _: moved)
    assert modelopt._prepare_export_memory(object()) == expected
