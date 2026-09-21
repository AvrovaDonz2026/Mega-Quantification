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
import sys
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
    digest = content_hash.strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("--hash must be a 64-char hex content hash")
    key_prefix = f"{prefix.rstrip('/')}/{mapped}/{digest}"
    base = oss.public_base_url(bucket_name, endpoint, key_prefix)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    sums_url = f"{base}/SHA256SUMS.txt"
    if dry_run:
        print(f"[oss-fetch] dry-run {sums_url} -> {dest}")
        return [sums_url]
    with urllib.request.urlopen(sums_url, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    names: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        names.append(parts[-1])
    if not names:
        raise FileNotFoundError(f"empty SHA256SUMS at {sums_url}")
    (dest / "SHA256SUMS.txt").write_text(body, encoding="utf-8")
    for name in names:
        url = f"{base}/{name}"
        out = dest / name
        print(f"[oss-fetch] GET {url}", flush=True)
        urllib.request.urlretrieve(url, out)
    extra = "oss_manifest.json"
    try:
        urllib.request.urlretrieve(f"{base}/{extra}", dest / extra)
        names.append(extra)
    except Exception:
        pass
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
