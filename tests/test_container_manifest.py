"""Exercise image provenance checks against actual file and package drift."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "container_manifest.py"
SPEC = importlib.util.spec_from_file_location("container_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "image"
    files = {
        "src/megaquant/__init__.py": '__version__ = "0.1.1"\n',
        "recipes/spark.yaml": "mamba_ssm_dtype: float32\n",
        "scripts/check.py": "print('checked')\n",
        "docker/requirements-gpu-spark.txt": "transformers==5.12.1\n",
        "pyproject.toml": '[project]\nname = "megaquant"\n',
        "LICENSE": "AGPL-3.0-or-later\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    installed = tmp_path / "site-packages" / "megaquant"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text('__version__ = "0.1.1"\n')
    (installed / "cli.py").write_text("def main(): return 0\n")
    packages = [
        SimpleNamespace(metadata={"Name": "MegaQuant"}, version="0.1.1"),
        SimpleNamespace(metadata={"Name": "nvidia_modelopt"}, version="0.47.0"),
    ]
    monkeypatch.setattr(
        manifest.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(installed / "__init__.py"))
        if name == "megaquant"
        else None,
    )
    monkeypatch.setattr(manifest.importlib.metadata, "distributions", lambda: packages)
    monkeypatch.setattr(manifest.platform, "machine", lambda: "aarch64")
    return SimpleNamespace(root=root, installed=installed, packages=packages)


def invoke(
    image: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    action: str,
) -> tuple[int, dict]:
    argv = [str(SCRIPT), action, "--root", str(image.root)]
    if action == "record":
        argv.extend([
            "--kind", "ptq",
            "--base-image", "example/public-image@sha256:" + "a" * 64,
            "--revision", "b" * 40,
        ])
    monkeypatch.setattr(sys, "argv", argv)
    code = manifest.main()
    return code, json.loads(capsys.readouterr().out)


def test_record_and_verify_unchanged_image(image, monkeypatch, capsys) -> None:
    code, report = invoke(image, monkeypatch, capsys, "record")
    assert code == 0
    assert report["status"] == "recorded"
    recorded = json.loads((image.root / "container-manifest.json").read_text())
    assert recorded["git_revision"] == "b" * 40
    assert recorded["kind"] == "ptq"
    assert recorded["architecture"] == "aarch64"
    assert recorded["packages"]["nvidia-modelopt"] == "0.47.0"
    assert set(recorded["installed_megaquant_sha256"]) == {"__init__.py", "cli.py"}
    code, report = invoke(image, monkeypatch, capsys, "verify")
    assert code == 0
    assert report["status"] == "verified"
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("location", "change", "expected"),
    [
        ("src/megaquant/__init__.py", "edit", "files_sha256: src/megaquant/__init__.py changed"),
        ("recipes/spark.yaml", "edit", "files_sha256: recipes/spark.yaml changed"),
        ("recipes/extra.yaml", "add", "files_sha256: recipes/extra.yaml changed"),
        ("recipes/spark.yaml", "delete", "files_sha256: recipes/spark.yaml changed"),
        ("scripts/check.py", "delete", "files_sha256: scripts/check.py changed"),
        (
            "docker/requirements-gpu-spark.txt",
            "edit",
            "files_sha256: docker/requirements-gpu-spark.txt changed",
        ),
        ("pyproject.toml", "edit", "files_sha256: pyproject.toml changed"),
        ("installed/cli.py", "edit", "installed_megaquant_sha256: cli.py changed"),
        ("installed/extra.py", "add", "installed_megaquant_sha256: extra.py changed"),
        ("installed/cli.py", "delete", "installed_megaquant_sha256: cli.py changed"),
    ],
)
def test_verification_fails_on_code_or_recipe_drift(
    image, monkeypatch, capsys, location: str, change: str, expected: str
) -> None:
    invoke(image, monkeypatch, capsys, "record")
    path = (
        image.installed / location.removeprefix("installed/")
        if location.startswith("installed/")
        else image.root / location
    )
    if change == "delete":
        path.unlink()
    else:
        path.write_text("tampered content\n")
    code, report = invoke(image, monkeypatch, capsys, "verify")
    assert code == 1
    assert report["status"] == "failed"
    assert report["errors"] == [expected]


@pytest.mark.parametrize("change", ["version", "add", "delete"])
def test_verification_fails_on_dependency_drift(image, monkeypatch, capsys, change) -> None:
    invoke(image, monkeypatch, capsys, "record")
    if change == "version":
        image.packages[1].version = "0.48.0"
    elif change == "delete":
        image.packages.pop()
    else:
        image.packages.append(SimpleNamespace(metadata={"Name": "rogue-package"}, version="1"))
    code, report = invoke(image, monkeypatch, capsys, "verify")
    assert code == 1
    assert report["status"] == "failed"
    expected = "rogue-package" if change == "add" else "nvidia-modelopt"
    assert report["errors"] == [f"packages: {expected} changed"]


def test_verification_fails_on_wrong_architecture(image, monkeypatch, capsys) -> None:
    invoke(image, monkeypatch, capsys, "record")
    monkeypatch.setattr(manifest.platform, "machine", lambda: "x86_64")
    code, report = invoke(image, monkeypatch, capsys, "verify")
    assert code == 1
    assert report["errors"] == ["architecture changed"]


def test_runtime_outputs_and_bytecode_do_not_change_provenance(image, monkeypatch, capsys) -> None:
    invoke(image, monkeypatch, capsys, "record")
    for relative in (
        "outputs/model.safetensors",
        "src/megaquant/__pycache__/__init__.cpython-312.pyc",
        "src/megaquant/cli.pyc",
    ):
        path = image.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime content")
    code, report = invoke(image, monkeypatch, capsys, "verify")
    assert code == 0
    assert report["errors"] == []


def test_missing_installed_package_fails_snapshot(image, monkeypatch) -> None:
    monkeypatch.setattr(manifest.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="installed megaquant package is missing"):
        manifest.snapshot(image.root)
