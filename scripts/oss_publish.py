#!/usr/bin/env python3
"""Publish a Mega-Quantification HF export to object storage.

Layout::

    <prefix>/<scheme>/<content-hash>/

``content-hash`` is SHA256 of sorted ``{filename} {sha256}\\n`` lines for every
``*.safetensors`` in the export directory. Files larger than 5 GiB use
multipart ``resumable_upload``. Bucket and endpoint come from ``--bucket`` /
``--endpoint`` or ``OSS_BUCKET`` / ``OSS_ENDPOINT`` (optional gitignored
``.oss.env``). Anonymous auth unless ``OSS_ACCESS_KEY_ID`` and
``OSS_ACCESS_KEY_SECRET`` are set.

Uploads require ``oss2`` (``pip install megaquant[oss]`` or ``pip install oss2``).
``--dry-run`` does not.

Usage after PTQ::

    python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-W4A4 --scheme w4a4
    python scripts/oss_publish.py outputs/Qwen3.8-27B-NVFP4-mixed --scheme mixed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_PREFIX = "Mega-Quantification"
SIMPLE_PUT_LIMIT = 5 * 1024**3
READ_CHUNK = 8 * 1024 * 1024

SCHEME_ALIASES: dict[str, str] = {
    "nvfp4_w4a8": "w4a8",
    "w4a8": "w4a8",
    "nvfp4_w4a4": "w4a4",
    "w4a4": "w4a4",
    "nvfp4_mixed": "mixed",
    "mixed": "mixed",
}

SKIP_NAMES = frozenset({"SHA256SUMS.txt", "oss_manifest.json"})


def _load_optional_env_file() -> None:
    """Load KEY=VAL from a gitignored .oss.env; never overrides the process env."""
    candidates = []
    explicit = os.environ.get("MEGAQUANT_OSS_ENV", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    candidates.extend(
        (
            Path.cwd() / ".oss.env",
            here.parent.parent / ".oss.env",
            here.parent / ".oss.env",
        )
    )
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        for raw in resolved.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        return


def _require_setting(cli: str | None, *env_names: str, flag: str) -> str:
    if cli and cli.strip():
        return cli.strip()
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    joined = " / ".join(env_names)
    raise ValueError(f"set {flag} or {joined}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_manifest_lines(export_dir: Path) -> list[str]:
    names = sorted(
        p.name for p in export_dir.iterdir() if p.suffix == ".safetensors" and p.is_file()
    )
    if not names:
        raise FileNotFoundError(f"no .safetensors in {export_dir}")
    lines: list[str] = []
    for name in names:
        digest = sha256_file(export_dir / name)
        lines.append(f"{name} {digest}\n")
    return lines


def content_hash(export_dir: Path) -> tuple[str, list[str]]:
    lines = safetensors_manifest_lines(export_dir)
    blob = "".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), lines


def canonical_scheme(raw: str | None, export_dir: Path) -> str:
    if raw:
        key = raw.strip()
        if key in SCHEME_ALIASES:
            return SCHEME_ALIASES[key]
        raise ValueError(f"unknown scheme '{raw}'; expected one of {sorted(set(SCHEME_ALIASES))}")
    provenance = export_dir / "provenance.json"
    if provenance.is_file():
        payload = json.loads(provenance.read_text())
        scheme = payload.get("scheme")
        if isinstance(scheme, str) and scheme in SCHEME_ALIASES:
            return SCHEME_ALIASES[scheme]
    name = export_dir.name.lower()
    for needle, mapped in (("w4a8", "w4a8"), ("w4a4", "w4a4"), ("mixed", "mixed")):
        if needle in name:
            return mapped
    raise ValueError(f"could not infer scheme from {export_dir}; pass --scheme")


def list_upload_files(export_dir: Path) -> list[Path]:
    files = [
        path
        for path in sorted(export_dir.iterdir())
        if path.is_file() and path.name not in SKIP_NAMES
    ]
    if not files:
        raise FileNotFoundError(f"no files to upload in {export_dir}")
    return files


def write_sha256sums(export_dir: Path, files: list[Path], hashes: dict[str, str]) -> Path:
    lines = [f"{hashes[path.name]}  {path.name}\n" for path in files]
    dest = export_dir / "SHA256SUMS.txt"
    dest.write_text("".join(lines), encoding="utf-8")
    return dest


def public_base_url(bucket: str, endpoint: str, key_prefix: str) -> str:
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://{bucket}.{host}/{key_prefix.rstrip('/')}"


def _require_oss2():
    try:
        import oss2
    except ImportError as exc:
        raise ImportError(
            "oss2 is required to upload. Install with: pip install megaquant[oss] "
            "(or pip install oss2)"
        ) from exc
    return oss2


def _auth():
    oss2 = _require_oss2()

    key_id = os.environ.get("OSS_ACCESS_KEY_ID", "")
    secret = os.environ.get("OSS_ACCESS_KEY_SECRET", "")
    if key_id and secret:
        return oss2.Auth(key_id, secret)
    return oss2.AnonymousAuth()


def _bucket(endpoint: str, bucket_name: str):
    oss2 = _require_oss2()

    return oss2.Bucket(_auth(), endpoint, bucket_name)


def _object_exists(bucket, key: str, size: int) -> bool:
    try:
        meta = bucket.head_object(key)
    except Exception:
        return False
    remote = getattr(meta, "content_length", None)
    if remote is None:
        headers = getattr(meta, "headers", {}) or {}
        remote = headers.get("Content-Length") or headers.get("content-length")
    try:
        return int(remote) == int(size)
    except (TypeError, ValueError):
        return False


def _upload_one(bucket, key: str, path: Path, *, force: bool) -> str:
    oss2 = _require_oss2()

    size = path.stat().st_size
    if not force and _object_exists(bucket, key, size):
        return "skip"
    if size > SIMPLE_PUT_LIMIT:
        oss2.resumable_upload(
            bucket,
            key,
            str(path),
            multipart_threshold=SIMPLE_PUT_LIMIT,
            part_size=64 * 1024 * 1024,
            num_threads=4,
        )
        return "multipart"
    bucket.put_object_from_file(key, str(path))
    return "put"


def publish(
    export_dir: Path,
    *,
    scheme: str | None = None,
    bucket_name: str | None = None,
    endpoint: str | None = None,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    _load_optional_env_file()
    bucket_name = _require_setting(bucket_name, "OSS_BUCKET", "OSS_BUCKET_NAME", flag="--bucket")
    endpoint = _require_setting(endpoint, "OSS_ENDPOINT", flag="--endpoint")
    export_dir = export_dir.resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError(export_dir)
    mapped = canonical_scheme(scheme, export_dir)
    files = list_upload_files(export_dir)
    hashes = {path.name: sha256_file(path) for path in files}
    st_files = sorted(
        (path for path in files if path.suffix == ".safetensors"),
        key=lambda path: path.name,
    )
    if not st_files:
        raise FileNotFoundError(f"no .safetensors in {export_dir}")
    st_lines = [f"{path.name} {hashes[path.name]}\n" for path in st_files]
    digest = hashlib.sha256("".join(st_lines).encode("utf-8")).hexdigest()
    key_prefix = f"{prefix.rstrip('/')}/{mapped}/{digest}"
    sha_path = write_sha256sums(export_dir, files, hashes)
    hashes[sha_path.name] = sha256_file(sha_path)
    upload_list = files + [sha_path]
    base = public_base_url(bucket_name, endpoint, key_prefix)

    def file_entry(path: Path) -> dict[str, Any]:
        return {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashes[path.name],
            "url": f"{base}/{path.name}",
        }

    manifest: dict[str, Any] = {
        "bucket": bucket_name,
        "endpoint": endpoint,
        "scheme": mapped,
        "content_hash": digest,
        "prefix": key_prefix + "/",
        "base_url": base,
        "source_dir": str(export_dir),
        "safetensors": [line.strip() for line in st_lines],
        "files": [file_entry(path) for path in upload_list],
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    man_path = export_dir / "oss_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes[man_path.name] = sha256_file(man_path)
    upload_list = upload_list + [man_path]
    manifest["files"].append(file_entry(man_path))
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        return manifest

    bucket = _bucket(endpoint, bucket_name)
    results: list[dict[str, str]] = []
    for path in upload_list:
        key = f"{key_prefix}/{path.name}"
        status = _upload_one(bucket, key, path, force=force)
        results.append({"name": path.name, "key": key, "status": status})
        print(f"[oss] {status:9} {key} ({path.stat().st_size} bytes)", flush=True)
    manifest["upload"] = results
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _upload_one(bucket, f"{key_prefix}/{man_path.name}", man_path, force=True)
    print(f"[oss] published {manifest['base_url']}/", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--scheme", help="w4a8 | w4a4 | mixed (or nvfp4_*)")
    parser.add_argument("--bucket", default=None, help="or OSS_BUCKET")
    parser.add_argument("--endpoint", default=None, help="or OSS_ENDPOINT")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-upload even if size matches")
    args = parser.parse_args(argv)
    try:
        publish(
            args.export_dir,
            scheme=args.scheme,
            bucket_name=args.bucket,
            endpoint=args.endpoint,
            prefix=args.prefix,
            dry_run=args.dry_run,
            force=args.force,
        )
    except Exception as exc:
        print(f"[oss] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
