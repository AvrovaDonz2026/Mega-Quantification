"""OSS publish helpers — no network, no oss2 required."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import quote

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
    with pytest.raises(ValueError):
        oss.canonical_scheme("w4a8_nvfp4_fp8", Path("/tmp"))
    with pytest.raises(ValueError):
        oss.canonical_scheme("nvfp4_w4a8_trtllm", Path("/tmp"))
    fake = Path("/tmp/Qwen3.8-27B-NVFP4-W4A8")
    assert oss.canonical_scheme(None, fake) == "w4a8"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    oss = _load()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}\n")
    before = set(tmp_path.iterdir())
    manifest = oss.publish(
        tmp_path,
        scheme="w4a4",
        bucket_name="example-bucket",
        endpoint="https://oss.example.com",
        dry_run=True,
    )
    assert manifest["scheme"] == "w4a4"
    assert manifest["bucket"] == "example-bucket"
    assert set(tmp_path.iterdir()) == before  # dry-run must not write sidecars
    assert "Mega-Quantification/w4a4/" in manifest["prefix"]
    assert manifest["content_hash"] in manifest["prefix"]
    # ...but the manifest still previews every file that would be uploaded.
    names = [entry["name"] for entry in manifest["files"]]
    assert "model.safetensors" in names
    assert "SHA256SUMS.txt" in names
    assert "oss_manifest.json" not in names
    sums_entry = next(entry for entry in manifest["files"] if entry["name"] == "SHA256SUMS.txt")
    config_hash = hashlib.sha256(b"{}\n").hexdigest()
    weights_hash = hashlib.sha256(b"weights").hexdigest()
    expected_sums = (
        f"{config_hash}  config.json\n"
        f"{weights_hash}  model.safetensors\n"
    ).encode()
    assert sums_entry["sha256"] == hashlib.sha256(expected_sums).hexdigest()
    assert sums_entry["size"] == len(expected_sums)
    with pytest.raises(ValueError):
        oss.canonical_scheme("fp16", tmp_path)


def test_publish_sidecars_and_reported_upload_status(tmp_path, monkeypatch):
    oss = _load()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}\n")
    calls = []

    class FakeBucket:
        pass

    monkeypatch.setattr(oss, "_bucket", lambda *_args: FakeBucket())

    def upload(_bucket, key, path, *, force):
        calls.append((Path(key).name, Path(path).read_bytes(), force))
        return "skip" if Path(key).name == "config.json" and not force else "put"

    monkeypatch.setattr(oss, "_upload_one", upload)
    manifest = oss.publish(
        tmp_path,
        scheme="mixed",
        bucket_name="example-bucket",
        endpoint="https://oss.example.com",
    )
    manifest_path = tmp_path / "oss_manifest.json"
    recorded = json.loads(manifest_path.read_text())
    assert recorded == manifest
    files = {entry["name"]: entry for entry in manifest["files"]}
    assert "oss_manifest.json" not in files
    sums = next(data for name, data, _ in calls if name == "SHA256SUMS.txt")
    assert hashlib.sha256(sums).hexdigest() == files["SHA256SUMS.txt"]["sha256"]
    statuses = {entry["name"]: entry["status"] for entry in manifest["upload"]}
    assert statuses["config.json"] == "skip"
    assert statuses["oss_manifest.json"] == "put"
    assert calls[-1][0] == "oss_manifest.json" and calls[-1][2] is True


def test_fetch_dry_run_prints_public_url(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "oss_fetch", REPO_ROOT / "scripts" / "oss_fetch.py"
    )
    assert spec is not None and spec.loader is not None
    fetch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_mod)
    digest = "a" * 64
    before = set(tmp_path.iterdir())
    urls = fetch_mod.fetch(
        scheme="mixed",
        content_hash=digest,
        dest=tmp_path,
        bucket_name="example-bucket",
        endpoint="https://oss.example.com",
        dry_run=True,
    )
    assert urls == [
        f"https://example-bucket.oss.example.com/Mega-Quantification/mixed/{digest}/SHA256SUMS.txt"
    ]
    assert set(tmp_path.iterdir()) == before
    with pytest.raises(ValueError, match="64-char"):
        fetch_mod.fetch(
            scheme="mixed",
            content_hash="short",
            dest=tmp_path,
            bucket_name="example-bucket",
            endpoint="https://oss.example.com",
            dry_run=True,
        )


def _load_fetch():
    spec = importlib.util.spec_from_file_location(
        "oss_fetch", REPO_ROOT / "scripts" / "oss_fetch.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _BoundedResponse(_Response):
    def read(self, size=-1):
        assert size <= 1024 * 1024
        return super().read(size)


def _fetch_fixture(monkeypatch, files: dict[str, bytes]):
    fetch_mod = _load_fetch()
    entries = [(hashlib.sha256(data).hexdigest(), name) for name, data in files.items()]
    sums = "".join(f"{digest}  {name}\n" for digest, name in entries)
    safetensors = sorted(
        (name, digest) for digest, name in entries if name.endswith(".safetensors")
    )
    content_hash = hashlib.sha256(
        "".join(f"{name} {digest}\n" for name, digest in safetensors).encode()
    ).hexdigest()
    base = f"https://example-bucket.oss.example.com/Mega-Quantification/mixed/{content_hash}"
    payloads = {f"{base}/SHA256SUMS.txt": sums.encode()}
    payloads.update({f"{base}/{quote(name, safe='')}": data for name, data in files.items()})
    def urlopen(url, timeout):
        assert timeout == 60
        if url not in payloads:
            raise FileNotFoundError(url)
        return _Response(payloads[url])
    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", urlopen)
    return fetch_mod, content_hash, payloads


def test_fetch_validates_manifest_and_downloads_atomically(tmp_path, monkeypatch):
    files = {"model part.safetensors": b"weights", "config.json": b"{}\n"}
    fetch_mod, content_hash, _ = _fetch_fixture(monkeypatch, files)
    dest = tmp_path / "model"
    names = fetch_mod.fetch(
        scheme="mixed", content_hash=content_hash, dest=dest,
        bucket_name="example-bucket", endpoint="https://oss.example.com",
    )
    assert set(names) == set(files)
    assert {name: (dest / name).read_bytes() for name in files} == files
    assert (dest / "SHA256SUMS.txt").is_file()


def test_fetch_rejects_mismatched_file_checksum_without_partial_files(tmp_path, monkeypatch):
    files = {"model.safetensors": b"weights"}
    fetch_mod, content_hash, payloads = _fetch_fixture(monkeypatch, files)
    url = next(url for url in payloads if url.endswith("/model.safetensors"))
    payloads[url] = b"corrupt"
    dest = tmp_path / "model"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_mod.fetch(
            scheme="mixed", content_hash=content_hash, dest=dest,
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    assert not dest.exists()


def test_fetch_streams_bounded_and_cleans_staging_on_second_failure(tmp_path, monkeypatch):
    files = {"a.safetensors": b"a" * (2 * 1024 * 1024), "b.bin": b"b"}
    fetch_mod, content_hash, payloads = _fetch_fixture(monkeypatch, files)
    original = fetch_mod.urllib.request.urlopen
    calls = 0

    def urlopen(url, timeout):
        nonlocal calls
        if url.endswith("/b.bin"):
            calls += 1
            raise OSError("second file failed")
        response = original(url, timeout)
        return _BoundedResponse(response.read())

    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "model"
    with pytest.raises(OSError, match="second file failed"):
        fetch_mod.fetch(
            scheme="mixed", content_hash=content_hash, dest=dest,
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    assert calls == 1
    assert not dest.exists()
    assert not list(tmp_path.glob(".model.*"))


def test_fetch_never_overwrites_existing_target(tmp_path, monkeypatch):
    files = {"model.safetensors": b"weights"}
    fetch_mod, content_hash, _ = _fetch_fixture(monkeypatch, files)
    dest = tmp_path / "model"
    dest.mkdir()
    target = dest / "model.safetensors"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        fetch_mod.fetch(
            scheme="mixed", content_hash=content_hash, dest=dest,
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("name", ["../escape", "folder/file", "folder\\file", "SHA256SUMS.txt"])
def test_fetch_rejects_unsafe_manifest_names_before_download(tmp_path, monkeypatch, name):
    fetch_mod = _load_fetch()
    body = f"{'a' * 64}  {name}\n".encode()
    monkeypatch.setattr(
        fetch_mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(body),
    )
    dest = tmp_path / "model"
    with pytest.raises(ValueError):
        fetch_mod.fetch(
            scheme="mixed", content_hash="a" * 64, dest=dest,
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    assert not dest.exists()


def test_fetch_rejects_content_hash_mismatch_and_existing_symlink(tmp_path, monkeypatch):
    files = {"model.safetensors": b"weights"}
    fetch_mod, content_hash, _ = _fetch_fixture(monkeypatch, files)
    sums_url = (
        f"https://example-bucket.oss.example.com/Mega-Quantification/mixed/"
        f"{'0' * 64}/SHA256SUMS.txt"
    )
    original_urlopen = fetch_mod.urllib.request.urlopen
    valid_sums_url = (
        f"https://example-bucket.oss.example.com/Mega-Quantification/mixed/"
        f"{content_hash}/SHA256SUMS.txt"
    )
    monkeypatch.setattr(
        fetch_mod.urllib.request,
        "urlopen",
        lambda url, timeout: original_urlopen(valid_sums_url if url == sums_url else url, timeout),
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        fetch_mod.fetch(
            scheme="mixed", content_hash="0" * 64, dest=tmp_path / "wrong",
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "model.safetensors").symlink_to(tmp_path / "victim")
    with pytest.raises(FileExistsError):
        fetch_mod.fetch(
            scheme="mixed", content_hash=content_hash, dest=dest,
            bucket_name="example-bucket", endpoint="https://oss.example.com",
        )
    assert not (tmp_path / "victim").exists()


def test_missing_oss2_explains_the_oss_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    oss = _load()
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "oss2":
            raise ImportError("No module named 'oss2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"megaquant\[oss\]"):
        oss._require_oss2()


def test_publish_requires_bucket_and_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oss = _load()
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.delenv("OSS_BUCKET", raising=False)
    monkeypatch.delenv("OSS_BUCKET_NAME", raising=False)
    monkeypatch.delenv("OSS_ENDPOINT", raising=False)
    monkeypatch.delenv("MEGAQUANT_OSS_ENV", raising=False)
    with pytest.raises(ValueError, match="OSS_BUCKET"):
        oss.publish(tmp_path, scheme="w4a4", dry_run=True)
