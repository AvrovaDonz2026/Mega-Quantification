#!/usr/bin/env python3
"""Require the recorded Spark SGLang patch in serving images."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_sglang_spark import MANIFEST, PatchError, patch_tree

RECORD_PATH = Path("/opt/megaquant/container-manifest.json")


def verify(kind: str, sglang: ModuleType | None, version: str | None) -> str:
    if kind == "ptq":
        return "not-applicable"
    if kind != "sglang":
        raise PatchError(f"unknown container kind: {kind!r}")
    if sglang is None or sglang.__file__ is None:
        raise PatchError("SGLang serving image has no importable sglang package")
    package_root = Path(sglang.__file__).parent
    if not (package_root / MANIFEST).is_file():
        raise PatchError(f"SGLang serving image is missing {MANIFEST}")
    if version is None:
        raise PatchError("SGLang serving image has no installed distribution version")
    result = patch_tree(package_root, version, check=True)
    return result["patch"]


def main() -> int:
    try:
        record = json.loads(RECORD_PATH.read_text())
        if not isinstance(record, dict):
            raise PatchError("container manifest must be a JSON object")
        kind = record.get("kind")
        if kind == "ptq":
            print("[megaquant] SGLang patch check not applicable")
            return 0
        if kind != "sglang":
            verify(kind, None, None)
        import sglang

        patch_id = verify(kind, sglang, importlib.metadata.version("sglang"))
    except (
        ImportError,
        KeyError,
        OSError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
        PatchError,
    ) as exc:
        print(f"[megaquant] SGLang patch verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"[megaquant] SGLang patch verified: {patch_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
