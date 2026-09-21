from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_w4a8_recipe_yaml_loads():
    from megaquant.config import load_recipe

    recipe = load_recipe(ROOT / "recipes/qwen3.8-27b-nvfp4-w4a8.yaml")
    assert recipe.model.source == "Qwen/Qwen3.8-27B"
    assert recipe.scheme == "nvfp4_w4a8"
    assert recipe.family == "qwen3_5"
    assert recipe.backend == "modelopt"


def test_qwen35_ignore_keeps_mlp():
    from megaquant.config import load_recipe
    from megaquant.models.qwen3_5 import Qwen35Family

    recipe = load_recipe(ROOT / "recipes/qwen3.8-27b-nvfp4-w4a8.yaml")
    ignore = Qwen35Family().default_ignore(recipe)
    joined = " ".join(ignore)
    assert "visual" in joined
    assert "mtp" in joined
    assert "conv1d" in joined
    assert "in_proj_a" in joined
    assert "mlp" not in joined


def test_nvfp4_w4a8_group_size_32():
    from megaquant.schemes.catalog import get_scheme

    scheme = get_scheme("nvfp4_w4a8")
    weights = scheme.groups[0]["weights"]
    acts = scheme.groups[0]["activations"]
    assert weights["group_size"] == 32
    assert acts["bits"] == 8
    w4a4 = get_scheme("nvfp4_w4a4")
    assert w4a4.groups[0]["weights"]["group_size"] == 16


def test_cli_plan_offline(tmp_path, monkeypatch):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "megaquant.cli",
            "plan",
            "-c",
            str(ROOT / "recipes/qwen3.8-27b-nvfp4-w4a8.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Qwen/Qwen3.8-27B" in proc.stdout
    assert "nvfp4_w4a8" in proc.stdout
    assert "group_size: 32" in proc.stdout
    assert "language_linears" in proc.stdout or "Linear" in proc.stdout
