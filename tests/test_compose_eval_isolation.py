"""Compose GPQA eval must not steal the GPU; serve-sglang keeps it.

CPU-only. Parses docker-compose.yml; does not run Docker.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker-compose.yml"
HUB_BF16 = "Qwen/Qwen3.8-27B"
LOCAL_EXPORT = "outputs/Qwen3.8-27B-NVFP4-W4A8"


def _load() -> tuple[str, dict]:
    text = COMPOSE.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return text, data


def _nvidia_devices(svc: dict) -> list[dict]:
    deploy = svc.get("deploy") or {}
    resources = deploy.get("resources") or {}
    reservations = resources.get("reservations") or {}
    devices = reservations.get("devices") or []
    out: list[dict] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        caps = device.get("capabilities") or []
        if "gpu" in caps or device.get("driver") == "nvidia":
            out.append(device)
    return out


def _has_gpu_attach(svc: dict) -> bool:
    gpus = svc.get("gpus")
    if gpus not in (None, False, [], 0):
        return True
    return bool(_nvidia_devices(svc))


def test_x_common_env_does_not_inject_empty_cuda_visible_devices() -> None:
    text, data = _load()
    common = data["x-common-env"]
    assert "CUDA_VISIBLE_DEVICES" not in common
    assert "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" not in text
    assert "CUDA_VISIBLE_DEVICES: ''" not in text
    assert 'CUDA_VISIBLE_DEVICES: ""' not in text


def test_x_sglang_defaults_deepgemm_jit_off_and_has_no_gpu() -> None:
    _text, data = _load()
    sglang = data["x-sglang"]
    assert not _has_gpu_attach(sglang)
    env = sglang["environment"]
    assert env["SGLANG_ENABLE_JIT_DEEPGEMM"] == "${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_eval_gpqa_does_not_attach_a_gpu() -> None:
    _text, data = _load()
    eval_svc = data["services"]["eval-gpqa"]
    assert eval_svc.get("gpus") in (None, False, [])
    assert eval_svc.get("deploy") in (None, {}, False)
    assert _nvidia_devices(eval_svc) == []
    assert not _has_gpu_attach(eval_svc)
    env = eval_svc["environment"]
    assert env.get("NVIDIA_VISIBLE_DEVICES") == ""
    assert str(env.get("MEGAQUANT_SKIP_GPU_REPORT")) == "1"
    url = str(env.get("MEGAQUANT_SGLANG_BASE_URL") or "")
    assert "serve-sglang:30000" in url
    assert "CUDA_VISIBLE_DEVICES" not in env
    cmd = " ".join(str(x) for x in (eval_svc.get("command") or []))
    assert "eval-gpqa-diamond.yaml" in cmd
    assert "eval-gpqa-diamond.5090.yaml" not in cmd
    assert HUB_BF16 not in cmd
    assert LOCAL_EXPORT in cmd


def test_serve_sglang_attaches_the_gpu() -> None:
    _text, data = _load()
    serve = data["services"]["serve-sglang"]
    assert _has_gpu_attach(serve)
    assert serve.get("gpus") == "all"
    devices = _nvidia_devices(serve)
    assert devices
    assert any("gpu" in (d.get("capabilities") or []) for d in devices)
    env = serve["environment"]
    assert env.get("NVIDIA_VISIBLE_DEVICES") == "all"
    assert env["SGLANG_ENABLE_JIT_DEEPGEMM"] == "${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
    cmd = " ".join(str(x) for x in (serve.get("command") or []))
    assert "eval-gpqa-diamond.yaml" in cmd
    assert "eval-gpqa-diamond.5090.yaml" not in cmd
    assert HUB_BF16 not in cmd
    assert LOCAL_EXPORT in cmd
