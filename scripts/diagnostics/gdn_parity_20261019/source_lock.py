"""Enforce the bundle's immutable source revision and archive allowlist."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
LOCK_PATH = BUNDLE_ROOT / "source-lock.json"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_locked_source(revision: str, archive: Path, variant: str | None = None) -> dict[str, str]:
    try:
        lock = json.loads(LOCK_PATH.read_text())
        variants = lock["variants"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid source lock: {LOCK_PATH}") from exc
    expected = {"original", "decode-beta", "all-beta", "shared-decode"}
    if not isinstance(variants, dict) or set(variants) != expected:
        raise RuntimeError("source lock must contain exactly the four pinned variants")
    actual_sha = _sha256(Path(archive))
    if variant is not None:
        item = variants.get(variant)
        if not isinstance(item, dict):
            raise RuntimeError(f"unknown locked source variant: {variant}")
        if item.get("commit") != revision:
            raise RuntimeError(f"source revision is not locked for variant {variant}")
        if item.get("archive_sha256") != actual_sha:
            raise RuntimeError(f"source archive hash is not locked for variant {variant}")
        return {"variant": variant, "commit": str(item["commit"]), "archive_sha256": actual_sha}
    matches = [(name, item) for name, item in variants.items() if isinstance(item, dict) and item.get("commit") == revision and item.get("archive_sha256") == actual_sha]
    if len(matches) != 1:
        raise RuntimeError("source revision/archive does not match a unique bundle lock entry")
    name, item = matches[0]
    return {"variant": name, "commit": str(item["commit"]), "archive_sha256": actual_sha}
