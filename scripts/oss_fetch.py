#!/usr/bin/env python3
"""Download a published Mega-Quantification export over HTTPS.

Uses the same layout as ``oss_publish.py``::

    <prefix>/<scheme>/<content-hash>/SHA256SUMS.txt

Bucket and endpoint come from ``--bucket`` / ``--endpoint`` or ``OSS_BUCKET`` /
``OSS_ENDPOINT`` (optional gitignored ``.oss.env``). Public-read buckets need
no keys. Does not bake a bucket name into this script.

Usage::

    python scripts/oss_fetch.py --scheme mixed --hash <content-hash> \\
        --dest outputs/Qwen3.8-27B-NVFP4-mixed
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

_PUBLISH = Path(__file__).resolve().parent / "oss_publish.py"


def _load_publish():
    import importlib.util

    spec = importlib.util.spec_from_file_location("oss_publish", _PUBLISH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_PUBLISH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch(
    *,
    scheme: str,
    content_hash: str,
    dest: Path,
    bucket_name: str | None = None,
    endpoint: str | None = None,
    prefix: str = "Mega-Quantification",
    dry_run: bool = False,
) -> list[str]:
    oss = _load_publish()
    oss._load_optional_env_file()
    bucket_name = oss._require_setting(
        bucket_name, "OSS_BUCKET", "OSS_BUCKET_NAME", flag="--bucket"
    )
    endpoint = oss._require_setting(endpoint, "OSS_ENDPOINT", flag="--endpoint")
    mapped = oss.SCHEME_ALIASES.get(scheme, scheme)
    if mapped not in set(oss.SCHEME_ALIASES.values()):
        raise ValueError(f"unknown scheme '{scheme}'")
    requested_digest = content_hash.strip().lower()
    if len(requested_digest) != 64 or any(c not in "0123456789abcdef" for c in requested_digest):
        raise ValueError("--hash must be a 64-char hex content hash")
    key_prefix = f"{prefix.rstrip('/')}/{mapped}/{requested_digest}"
    base = oss.public_base_url(bucket_name, endpoint, key_prefix)
    dest = dest.absolute()
    sums_url = f"{base}/SHA256SUMS.txt"
    if dry_run:
        print(f"[oss-fetch] dry-run {sums_url} -> {dest}")
        return [sums_url]
    with urllib.request.urlopen(sums_url, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    entries: list[tuple[str, str]] = []  # (sha256, name)
    seen: set[str] = set()
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+(\*?)(.+)", line)
        if match is None:
            raise ValueError(f"malformed SHA256SUMS entry: {line!r}")
        file_digest, name = match.group(1), match.group(3)
        file_digest = file_digest.lower()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError(f"unsafe file name in SHA256SUMS: {name!r}")
        if name in {"SHA256SUMS.txt", "oss_manifest.json"}:
            raise ValueError(f"reserved file name in SHA256SUMS: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate file name in SHA256SUMS: {name!r}")
        seen.add(name)
        entries.append((file_digest, name))
    if not entries:
        raise FileNotFoundError(f"empty SHA256SUMS at {sums_url}")
    extra = "oss_manifest.json"
    st_entries = sorted((name, digest) for digest, name in entries if name.endswith(".safetensors"))
    if not st_entries:
        raise ValueError("SHA256SUMS has no .safetensors entries")
    actual_content_hash = hashlib.sha256(
        "".join(f"{name} {digest}\n" for name, digest in st_entries).encode("utf-8")
    ).hexdigest()
    if actual_content_hash != requested_digest:
        raise ValueError(
            f"content hash mismatch: requested {requested_digest}, "
            f"manifest describes {actual_content_hash}"
        )

    # Validate the whole destination before requesting any file bytes.
    if dest.is_symlink():
        raise FileExistsError(f"refusing to use symlink destination: {dest}")
    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(dest)
    for _, name in entries:
        target = dest / name
        if target.is_symlink() or target.exists():
            raise FileExistsError(f"refusing to overwrite existing destination: {target}")
    sums_path = dest / "SHA256SUMS.txt"
    if sums_path.is_symlink() or sums_path.exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {sums_path}")
    if (dest / extra).is_symlink() or (dest / extra).exists():
        raise FileExistsError(f"refusing to overwrite existing destination: {dest / extra}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    names: list[str] = []
    committed: list[Path] = []
    created_dest = False
    try:
        for expected, name in entries:
            url = f"{base}/{urllib.parse.quote(name, safe='')}"
            print(f"[oss-fetch] GET {url}", flush=True)
            temporary = staging / name
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as resp, temporary.open("wb") as handle:
                while chunk := resp.read(1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(f"sha256 mismatch for {name}: expected {expected}, got {actual}")
            names.append(name)

        try:
            extra_path = staging / extra
            with urllib.request.urlopen(f"{base}/{extra}", timeout=60) as resp, extra_path.open(
                "wb"
            ) as handle:
                while chunk := resp.read(1024 * 1024):
                    handle.write(chunk)
            names.append(extra)
        except Exception:
            pass

        (staging / "SHA256SUMS.txt").write_text(body, encoding="utf-8")
        commit_names = [*names, "SHA256SUMS.txt"]
        if not dest.exists():
            dest.mkdir(parents=True)
            created_dest = True
        for name in commit_names:
            target = dest / name
            os.link(staging / name, target)
            committed.append(target)
    except Exception:
        for target in committed:
            target.unlink(missing_ok=True)
        if created_dest:
            dest.rmdir()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[oss-fetch] wrote {len(names)} files to {dest}", flush=True)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--hash", required=True, dest="content_hash")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--prefix", default="Mega-Quantification")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        fetch(
            scheme=args.scheme,
            content_hash=args.content_hash,
            dest=args.dest,
            bucket_name=args.bucket,
            endpoint=args.endpoint,
            prefix=args.prefix,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[oss-fetch] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
