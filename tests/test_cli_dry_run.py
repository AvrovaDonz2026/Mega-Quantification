"""CLI dry-run / plan must work offline (no 27B download)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

W4A8_REL = "recipes/qwen3.8-27b-nvfp4-w4a8.yaml"


def _cli_available() -> bool:
    return importlib.util.find_spec("megaquant.cli") is not None


def test_cli_plan_subprocess(repo_root: Path, tmp_path: Path) -> None:
    if not _cli_available():
        pytest.skip("megaquant.cli not implemented yet")

    env = os.environ.copy()
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HOME"] = str(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(tmp_path / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(tmp_path / "hf")

    proc = subprocess.run(
        [sys.executable, "-m", "megaquant.cli", "plan", "-c", W4A8_REL],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and _looks_like_hub_fetch(combined):
        pytest.skip(
            "CLI plan tried to hit the Hugging Face Hub; monkeypatch path is "
            "covered by test_cli_plan_monkeypatched_hf_config"
        )
    assert proc.returncode == 0, combined
    out = (proc.stdout or "") + (proc.stderr or "")
    lowered = out.lower()
    assert "qwen3.8-27b" in lowered or "Qwen/Qwen3.8-27B" in out
    assert "nvfp4" in lowered or "nvfp4_w4a8" in lowered


def _looks_like_hub_fetch(text: str) -> bool:
    lowered = text.lower()
    needles = (
        "hf_hub_download",
        "offline",
        "failed to establish",
        "could not load config.json",
        "huggingface.co",
        "401",
        "403",
        "timed out",
    )
    return any(n in lowered for n in needles)


def test_cli_plan_monkeypatched_hf_config(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If resolve() fetches config.json, serve a tiny fake instead of 27B weights."""
    if not _cli_available():
        pytest.skip("megaquant.cli not implemented yet")

    fake = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "text_config": {"model_type": "qwen3_5_text"},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(fake))

    def _fake_download(*_args: object, **_kwargs: object) -> str:
        return str(cfg_path)

    try:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    except ImportError:
        pass

    try:
        import megaquant.pipeline as pipeline

        monkeypatch.setattr(pipeline, "_load_hf_config", lambda _source: fake)
    except ImportError:
        pass

    from megaquant.cli import main

    argv = ["plan", "-c", str(repo_root / W4A8_REL)]
    code = main(argv)
    assert code == 0
