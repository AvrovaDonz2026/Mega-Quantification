#!/usr/bin/env python3
"""Record and verify the code, configuration and packages baked into an image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

MANIFEST = Path("container-manifest.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict:
    files = {}
    for directory in ("src", "recipes", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(root))] = sha256(path)
    for path in sorted((root / "docker").glob("requirements*.txt")):
        files[str(path.relative_to(root))] = sha256(path)
    for path in sorted((root / "docker").glob("*.sh")):
        files[str(path.relative_to(root))] = sha256(path)
    for name in ("pyproject.toml", "LICENSE"):
        files[name] = sha256(root / name)
    spec = importlib.util.find_spec("megaquant")
    if spec is None or spec.origin is None:
        raise RuntimeError("The installed megaquant package is missing")
    package = Path(spec.origin).parent
    package_files = {
        str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*.py"))
    }
    packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "architecture": platform.machine(),
        "files_sha256": files,
        "installed_megaquant_sha256": package_files,
        "packages": dict(sorted(packages.items())),
    }


def verify(record: dict, current: dict) -> list[str]:
    errors = []
    if record["architecture"] != current["architecture"]:
        errors.append("architecture changed")
    for group in ("files_sha256", "installed_megaquant_sha256", "packages"):
        expected, actual = record[group], current[group]
        for name in sorted(expected.keys() | actual.keys()):
            if expected.get(name) != actual.get(name):
                errors.append(f"{group}: {name} changed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--root", type=Path, default=Path("/opt/megaquant"))
    parser.add_argument("--kind", choices=("ptq", "sglang"))
    parser.add_argument("--base-image")
    parser.add_argument("--revision")
    args = parser.parse_args()
    path = args.root / MANIFEST
    current = snapshot(args.root)
    if args.action == "record":
        if not all((args.kind, args.base_image, args.revision)):
            parser.error("record requires --kind, --base-image and --revision")
        record = {
            "schema_version": 1,
            "kind": args.kind,
            "base_image": args.base_image,
            "git_revision": args.revision,
            **current,
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "recorded", "manifest_sha256": sha256(path)}))
        return 0
    record = json.loads(path.read_text())
    errors = verify(record, current)
    print(
        json.dumps(
            {
                "status": "failed" if errors else "verified",
                "kind": record["kind"],
                "git_revision": record["git_revision"],
                "manifest_sha256": sha256(path),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
