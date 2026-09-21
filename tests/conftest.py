"""Shared fixtures. Tests must run on CPU with no model weights."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def recipes_dir(repo_root: Path) -> Path:
    return repo_root / "recipes"


@pytest.fixture(scope="session")
def w4a8_recipe_path(recipes_dir: Path) -> Path:
    return recipes_dir / "qwen3.8-27b-nvfp4-w4a8.yaml"


@pytest.fixture(scope="session")
def mixed_recipe_path(recipes_dir: Path) -> Path:
    return recipes_dir / "qwen3.8-27b-nvfp4-mixed.yaml"
