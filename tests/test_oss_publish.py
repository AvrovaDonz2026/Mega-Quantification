"""OSS publish helpers — no network, no oss2 required."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "oss_publish.py"


def _load():
    spec = importlib.util.spec_from_file_location("oss_publish", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_content_hash_is_sha256_of_sorted_safetensors_lines(tmp_path: Path) -> None:
    oss = _load()
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"bbb")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"aaa")
    (tmp_path / "config.json").write_text("{}\n")
    digest, lines = oss.content_hash(tmp_path)
    expected_lines = [
        f"model-00001-of-00002.safetensors {hashlib.sha256(b'aaa').hexdigest()}\n",
        f"model-00002-of-00002.safetensors {hashlib.sha256(b'bbb').hexdigest()}\n",
    ]
    assert lines == expected_lines
    assert digest == hashlib.sha256("".join(expected_lines).encode()).hexdigest()


def test_canonical_scheme_from_name_and_alias() -> None:
    oss = _load()
    assert oss.canonical_scheme("nvfp4_w4a4", Path("/tmp")) == "w4a4"
    assert oss.canonical_scheme("mixed", Path("/tmp")) == "mixed"
    fake = Path("/tmp/Qwen3.8-27B-NVFP4-W4A8")
    assert oss.canonical_scheme(None, fake) == "w4a8"


def test_dry_run_writes_sidecars(tmp_path: Path) -> None:
    oss = _load()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}\n")
    manifest = oss.publish(
        tmp_path,
        scheme="w4a4",
        bucket_name="example-bucket",
        endpoint="https://oss.example.com",
        dry_run=True,
    )
    assert manifest["scheme"] == "w4a4"
    assert manifest["bucket"] == "example-bucket"
    assert (tmp_path / "SHA256SUMS.txt").is_file()
    assert (tmp_path / "oss_manifest.json").is_file()
    assert "Mega-Quantification/w4a4/" in manifest["prefix"]
    assert manifest["content_hash"] in manifest["prefix"]
    with pytest.raises(ValueError):
        oss.canonical_scheme("fp16", tmp_path)


def test_publish_requires_bucket_and_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oss = _load()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.delenv("OSS_BUCKET", raising=False)
    monkeypatch.delenv("OSS_BUCKET_NAME", raising=False)
    monkeypatch.delenv("OSS_ENDPOINT", raising=False)
    monkeypatch.delenv("MEGAQUANT_OSS_ENV", raising=False)
    with pytest.raises(ValueError, match="OSS_BUCKET"):
        oss.publish(tmp_path, scheme="w4a4", dry_run=True)
