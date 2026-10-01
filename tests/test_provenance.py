"""Keep export provenance local to each artifact and identify baked code."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from megaquant import pipeline
from megaquant.config import load_recipe

REPO = Path(__file__).resolve().parents[1]
IMAGE_SHA = "a" * 40
CHECKOUT_SHA = "b" * 40


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> pipeline.ResolvedPlan:
    monkeypatch.setattr(pipeline, "_git_sha", lambda: IMAGE_SHA)
    recipe = load_recipe(REPO / "recipes" / "qwen3.8-27b-nvfp4-w4a8.spark.yaml")
    return pipeline.ResolvedPlan(
        recipe=recipe,
        backend_name="modelopt",
        family_name="qwen3_5",
        ignore=[],
        groups=[],
    )


def test_existing_export_directory_with_dot_keeps_own_provenance(tmp_path, plan) -> None:
    export = tmp_path / "Qwen3.8-27B-NVFP4-W4A8-spark"
    export.mkdir()
    pipeline._write_provenance(plan, export)
    report = json.loads((export / "provenance.json").read_text())
    assert report["scheme"] == "nvfp4_w4a8"
    assert report["git_sha"] == IMAGE_SHA
    assert not (tmp_path / "provenance.json").exists()


@pytest.mark.parametrize("filename", ["model.safetensors", "model"])
def test_actual_file_exports_write_provenance_in_parent(tmp_path, plan, filename) -> None:
    export = tmp_path / filename
    export.write_bytes(b"export")
    pipeline._write_provenance(plan, export)
    assert (tmp_path / "provenance.json").is_file()
    assert export.read_bytes() == b"export"


@pytest.mark.parametrize("filename", ["not-created-yet", "not-created-yet.bin"])
def test_nonexistent_exports_preserve_path_heuristic(tmp_path, plan, filename) -> None:
    export = tmp_path / filename
    pipeline._write_provenance(plan, export)
    expected = export.parent if export.suffix else export
    assert (expected / "provenance.json").is_file()


def test_two_existing_exports_have_independent_provenance(tmp_path, plan) -> None:
    w4a8 = tmp_path / "Qwen3.8-27B-NVFP4-W4A8-spark"
    w4a4 = tmp_path / "Qwen3.8-27B-NVFP4-W4A4-spark"
    w4a8.mkdir()
    w4a4.mkdir()
    pipeline._write_provenance(plan, w4a8)
    first_bytes = (w4a8 / "provenance.json").read_bytes()
    plan.recipe.scheme = "nvfp4_w4a4"
    pipeline._write_provenance(plan, w4a4)
    assert (w4a8 / "provenance.json").read_bytes() == first_bytes
    assert json.loads(first_bytes)["scheme"] == "nvfp4_w4a8"
    assert json.loads((w4a4 / "provenance.json").read_text())["scheme"] == "nvfp4_w4a4"
    assert not (tmp_path / "provenance.json").exists()


@pytest.fixture
def image_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("MEGAQUANT_ROOT", str(tmp_path))
    return tmp_path


def test_valid_manifest_revision_takes_precedence_over_checkout(image_root, monkeypatch) -> None:
    (image_root / "container-manifest.json").write_text(json.dumps({"git_revision": IMAGE_SHA}))

    def unexpected_git(*args, **kwargs):
        pytest.fail("Git must not override the revision baked into the image")

    monkeypatch.setattr(pipeline.subprocess, "run", unexpected_git)
    assert pipeline._git_sha() == IMAGE_SHA


@pytest.mark.parametrize(
    "manifest_bytes",
    [
        None,
        b"not JSON",
        b"\xff",
        b"[]",
        b"null",
        b"{}",
        b'{"git_revision": "unknown"}',
        b'{"git_revision": 123}',
        b'{"git_revision": "1ae52c7"}',
        json.dumps({"git_revision": "g" * 40}).encode(),
    ],
)
def test_missing_or_invalid_manifest_falls_back_to_git(
    image_root, monkeypatch, manifest_bytes
) -> None:
    if manifest_bytes is not None:
        (image_root / "container-manifest.json").write_bytes(manifest_bytes)
    calls = []

    def git_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=CHECKOUT_SHA + "\n")

    monkeypatch.setattr(pipeline.subprocess, "run", git_run)
    assert pipeline._git_sha() == CHECKOUT_SHA
    assert len(calls) == 1
    assert calls[0][0][0] == ["git", "rev-parse", "HEAD"]
    assert calls[0][1]["timeout"] == 5


def test_default_container_root_is_used_when_environment_unset(monkeypatch) -> None:
    monkeypatch.delenv("MEGAQUANT_ROOT", raising=False)
    paths = []

    def read_manifest(path):
        paths.append(path)
        return json.dumps({"git_revision": IMAGE_SHA})

    monkeypatch.setattr(Path, "read_text", read_manifest)
    assert pipeline._git_sha() == IMAGE_SHA
    assert paths == [Path("/opt/megaquant/container-manifest.json")]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout=""),
        SimpleNamespace(returncode=0, stdout="\n"),
        OSError("Git missing"),
        subprocess.TimeoutExpired("git", 5),
    ],
)
def test_unavailable_git_returns_none_without_a_manifest(image_root, monkeypatch, result) -> None:
    def git_run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pipeline.subprocess, "run", git_run)
    assert pipeline._git_sha() is None
