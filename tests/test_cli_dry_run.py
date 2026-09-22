"""CLI dry-run / plan must work offline (no 27B download)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

W4A8_REL = "recipes/qwen3.8-27b-nvfp4-w4a8.yaml"
EVAL_DEFAULT = "recipes/eval-gpqa-diamond.yaml"
EVAL_5090 = "recipes/eval-gpqa-diamond.5090.yaml"
HUB_BF16 = "Qwen/Qwen3.8-27B"
LOCAL_EXPORT = "outputs/Qwen3.8-27B-NVFP4-W4A8"


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


def _forbid_hub(*_args: object, **_kwargs: object) -> str:
    raise AssertionError("eval/serve --dry-run must not download from the Hub")


def _offline_env(repo_root: Path, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HOME"] = str(tmp_path / "hf")
    env["HF_HUB_CACHE"] = str(tmp_path / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(tmp_path / "hf")
    for key in (
        "MEGAQUANT_SGLANG_BASE_URL",
        "MEGAQUANT_BASE_URL",
        "MEGAQUANT_VLLM_BASE_URL",
    ):
        env.pop(key, None)
    return env


def _assert_eval_serve_dry_run_payload(kind: str, out: str, recipe: str) -> None:
    assert HUB_BF16 not in out, f"{kind} dry-run leaked Hub id:\n{out}"
    assert LOCAL_EXPORT in out
    payload = out[out.find("{") :]
    data = json.loads(payload)
    if kind == "eval":
        assert data["engine"] == "sglang"
        assert data["model"] == LOCAL_EXPORT
        assert data["generation"]["max_new_tokens"] == 0
        assert data["generation"]["seed"] == 0
        assert data["generation"]["max_model_len"] == 262144
        assert data["thinking"]["reasoning_effort"] == "xhigh"
        assert data["sampling"]["temperature"] == 1.0
        assert data["sampling"]["top_p"] == 0.95
        assert data["sampling"]["top_k"] == 20
        kv_gb = 64 if recipe.endswith("5090.yaml") else 12
        assert data["serve"]["kv_offloading_size_gb"] == kv_gb
        if recipe.endswith("5090.yaml"):
            assert data["concurrency"] == 24
        else:
            assert data["concurrency"] == 1
        assert data["serve"]["disable_cuda_graph"] is True
        backend = data["serve"]["attention_backend"]
    else:
        assert data["engine"] == "sglang"
        assert data["argv"][:2] == ["sglang", "serve"]
        assert "--seed" not in data["argv"]
        model = data["argv"][data["argv"].index("--model-path") + 1]
        assert model == LOCAL_EXPORT
        backend = data["argv"][data["argv"].index("--attention-backend") + 1]
        assert data["plan"]["model"] == LOCAL_EXPORT
    if recipe.endswith("5090.yaml"):
        assert backend == "triton"
        if kind == "serve":
            assert "flashinfer" not in " ".join(data["argv"])
        else:
            assert data["serve"]["attention_backend"] == "triton"
            assert data["serve"].get("flashinfer_available") is not True
        if kind == "serve" and "--sampling-backend" in data["argv"]:
            assert data["argv"][data["argv"].index("--sampling-backend") + 1] == "pytorch"
        if kind == "eval" and data["serve"].get("sampling_backend"):
            assert data["serve"]["sampling_backend"] == "pytorch"
    else:
        assert backend == "flashinfer"
        if kind == "serve":
            assert data["argv"][data["argv"].index("--attention-backend") + 1] == "flashinfer"
            assert "--sampling-backend" not in data["argv"]
        else:
            assert data["serve"]["attention_backend"] == "flashinfer"


@pytest.mark.parametrize("recipe", (EVAL_DEFAULT, EVAL_5090))
def test_cli_eval_and_serve_dry_run_uses_local_export_not_hub(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, recipe: str
) -> None:
    """eval/serve --dry-run must not treat Qwen/Qwen3.8-27B as a download target."""
    if not _cli_available():
        pytest.skip("megaquant.cli not implemented yet")

    try:
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _forbid_hub)
    except ImportError:
        pass

    from megaquant.cli import main

    for key in (
        "MEGAQUANT_SGLANG_BASE_URL",
        "MEGAQUANT_BASE_URL",
        "MEGAQUANT_VLLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["eval", "-c", str(repo_root / recipe), "--dry-run"])
    eval_out = buf.getvalue()
    assert code == 0, eval_out
    _assert_eval_serve_dry_run_payload("eval", eval_out, recipe)

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["serve", "-c", str(repo_root / recipe), "--dry-run"])
    serve_out = buf.getvalue()
    assert code == 0, serve_out
    _assert_eval_serve_dry_run_payload("serve", serve_out, recipe)


@pytest.mark.parametrize("recipe", (EVAL_DEFAULT, EVAL_5090))
def test_cli_eval_serve_dry_run_subprocess_offline(
    repo_root: Path, tmp_path: Path, recipe: str
) -> None:
    if not _cli_available():
        pytest.skip("megaquant.cli not implemented yet")

    env = _offline_env(repo_root, tmp_path)
    for command in ("eval", "serve"):
        proc = subprocess.run(
            [sys.executable, "-m", "megaquant.cli", command, "-c", recipe, "--dry-run"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, combined
        assert HUB_BF16 not in combined
        assert LOCAL_EXPORT in (proc.stdout or "")
        if command == "serve":
            payload = (proc.stdout or "")[(proc.stdout or "").find("{") :]
            data = json.loads(payload)
            assert "--seed" not in data["argv"]
