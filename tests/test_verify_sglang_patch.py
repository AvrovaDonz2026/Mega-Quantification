"""Runtime gate for the numerics-critical Spark SGLang patch."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_sglang_patch.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("verify_sglang_patch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def serving_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ModuleType]:
    package = tmp_path / "site-packages" / "sglang"
    package.mkdir(parents=True)
    manifest = package / verifier.MANIFEST
    files = {}
    from patch_sglang_spark import PATCH_ID, UPSTREAM_SHA256, _sha256

    for relative, expected in UPSTREAM_SHA256.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("patched source\n")
        files[relative] = {
            "upstream_sha256": expected,
            "patched_sha256": _sha256(path.read_bytes()),
        }
    manifest.write_text(json.dumps({"patch": PATCH_ID, "sglang": "0.5.20", "files": files}))
    module = ModuleType("sglang")
    module.__file__ = str(package / "__init__.py")
    return package, module


def test_valid_serving_patch_passes(tmp_path, monkeypatch) -> None:
    package, module = serving_package(tmp_path, monkeypatch)
    assert verifier.verify("sglang", module, "0.5.20") == "megaquant-spark-v8"


def test_tampered_serving_patch_fails(tmp_path, monkeypatch) -> None:
    package, module = serving_package(tmp_path, monkeypatch)
    from patch_sglang_spark import UPSTREAM_SHA256

    (package / next(iter(UPSTREAM_SHA256))).write_text("tampered")
    with pytest.raises(verifier.PatchError, match="manifest or installed files were changed"):
        verifier.verify("sglang", module, "0.5.20")


def test_missing_serving_manifest_fails(tmp_path) -> None:
    module = ModuleType("sglang")
    module.__file__ = str(tmp_path / "sglang" / "__init__.py")
    Path(module.__file__).parent.mkdir()
    with pytest.raises(verifier.PatchError, match="missing .megaquant-spark-patch.json"):
        verifier.verify("sglang", module, "0.5.20")


def test_non_serving_image_does_not_require_sglang() -> None:
    assert verifier.verify("ptq", None, None) == "not-applicable"


def test_unknown_image_kind_fails_closed() -> None:
    with pytest.raises(verifier.PatchError, match="unknown container kind"):
        verifier.verify("", None, None)


def test_main_skips_ptq_manifest_without_importing_sglang(
    tmp_path, monkeypatch, capsys
) -> None:
    record = tmp_path / "container-manifest.json"
    record.write_text('{"kind":"ptq"}')
    monkeypatch.setattr(verifier, "RECORD_PATH", record)
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    assert verifier.main() == 0
    assert "not applicable" in capsys.readouterr().out


def test_main_rejects_unknown_kind_before_sglang_import(tmp_path, monkeypatch, capsys) -> None:
    record = tmp_path / "container-manifest.json"
    record.write_text('{"kind":"other"}')
    monkeypatch.setattr(verifier, "RECORD_PATH", record)
    assert verifier.main() == 1
    assert "unknown container kind" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["[]", "null", '{"kind":null}', "{}"])
def test_main_rejects_invalid_container_record(tmp_path, monkeypatch, capsys, content) -> None:
    record = tmp_path / "container-manifest.json"
    record.write_text(content)
    monkeypatch.setattr(verifier, "RECORD_PATH", record)
    assert verifier.main() == 1
    assert capsys.readouterr().err


def test_main_requires_serving_patch_manifest(tmp_path, monkeypatch, capsys) -> None:
    record = tmp_path / "container-manifest.json"
    record.write_text('{"kind":"sglang"}')
    monkeypatch.setattr(verifier, "RECORD_PATH", record)
    package = tmp_path / "site-packages" / "sglang"
    package.mkdir(parents=True)
    module = ModuleType("sglang")
    module.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", module)
    monkeypatch.setattr(verifier.importlib.metadata, "version", lambda name: "0.5.20")
    assert verifier.main() == 1
    assert "missing .megaquant-spark-patch.json" in capsys.readouterr().err
