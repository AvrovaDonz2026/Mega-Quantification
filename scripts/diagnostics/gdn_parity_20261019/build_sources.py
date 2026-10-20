#!/usr/bin/env python3
"""Rebuild the pinned source archives and compact manifests from real commits."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

VARIANTS = ("original", "decode-beta", "all-beta", "shared-decode")


def run(*args: str, cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE)
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(repo: Path, commit: str, prefix: str, destination: Path) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(
            ("git", "archive", "--format=tar.gz", f"--prefix={prefix}/", commit),
            cwd=repo,
            check=True,
            stdout=handle,
        )
    members = files = 0
    with tarfile.open(destination, "r:gz") as bundle:
        for member in bundle:
            members += 1
            files += int(member.isfile())
    return members, files


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/Users/zxy/sglang"))
    parser.add_argument("--output", type=Path, default=here / "source-package")
    parser.add_argument("--lock", type=Path, default=here / "source-lock.json")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    lock = json.loads(args.lock.expanduser().resolve().read_text())
    variants = lock.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise SystemExit("source-lock.json must contain exactly the four pinned variants")
    if run("git", "rev-parse", "--is-inside-work-tree", cwd=repo) != "true":
        raise SystemExit(f"not a git worktree: {repo}")

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "sglang-source-package-summary-v1",
        "source_repository": str(repo),
        "revisions": [],
        "packages": [],
    }
    for variant in VARIANTS:
        entry = variants[variant]
        commit = str(entry["commit"])
        actual_type = run("git", "cat-file", "-t", f"{commit}^{{commit}}", cwd=repo)
        if actual_type != "commit":
            raise SystemExit(f"{variant}: {commit} is not a commit")
        prefix = f"sglang-{variant}"
        archive_path = output / f"{prefix}.tar.gz"
        member_count, file_count = archive(repo, commit, prefix, archive_path)
        archive_hash = sha256(archive_path)
        expected_hash = str(entry["archive_sha256"])
        if archive_hash != expected_hash:
            raise SystemExit(
                f"{variant}: archive SHA mismatch: generated {archive_hash}, lock {expected_hash}"
            )
        manifest = {
            "schema": "sglang-source-manifest-v1",
            "source_repository": str(repo),
            "variant": variant,
            "archive": archive_path.name,
            "archive_prefix": prefix,
            "archive_member_count": member_count,
            "file_count": file_count,
            "metadata": {"commit": commit, "git_cat_file_type": "commit"},
            "revisions": [
                {
                    "name": "vision-mtp",
                    "variant": variant,
                    "commit": commit,
                    "archive": archive_path.name,
                    "archive_prefix": prefix,
                    "archive_sha256": archive_hash,
                }
            ],
        }
        (output / f"{prefix}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        item = {
            "name": variant,
            "variant": variant,
            "commit": commit,
            "archive": archive_path.name,
            "archive_prefix": prefix,
            "archive_sha256": archive_hash,
            "manifest": f"{prefix}.manifest.json",
            "archive_member_count": member_count,
            "file_count": file_count,
        }
        summary["revisions"].append(item)
        summary["packages"].append(item)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "summary": str(output / "summary.json"), "status": "pass"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
