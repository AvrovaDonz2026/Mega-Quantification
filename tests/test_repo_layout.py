"""Public tree: license, recipe index, and files that must stay out of git."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_agpl_license_is_at_repo_root() -> None:
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("                    GNU AFFERO GENERAL PUBLIC LICENSE")
    assert "Version 3, 19 November 2007" in text
    assert "changing it is not allowed" in text
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "AGPL-3.0-or-later" in readme
    assert "Copyright 2026 Donz" in readme
    project = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "AGPL-3.0-or-later"' in project


def test_recipe_index_names_every_yaml() -> None:
    index = (REPO / "recipes" / "README.md").read_text(encoding="utf-8")
    yamls = sorted(p.name for p in (REPO / "recipes").glob("*.yaml"))
    assert yamls, "recipes/ should contain YAML"
    missing = [name for name in yamls if name not in index]
    assert missing == []


def test_gitignore_keeps_secrets_and_weights_out() -> None:
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for needle in (
        ".env",
        ".oss.env",
        "docker-compose.override.yml",
        "recipes/*.pod.yaml",
        "outputs/",
        "*.safetensors",
    ):
        assert needle in text.splitlines() or any(
            line.strip() == needle for line in text.splitlines()
        ), needle


def test_architecture_doc_is_public() -> None:
    text = (REPO / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "Repository layout" in text
    assert "Do not `git commit`" not in text
    assert "Agent Core" not in text
