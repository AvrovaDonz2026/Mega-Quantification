"""Docker packaging invariants. CPU-only; does not run Docker."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_dockerfile_does_not_copy_weights(repo_root: Path) -> None:
    dockerfile = (repo_root / "Dockerfile").read_text()
    copy_lines = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().startswith("COPY ")
    ]
    assert copy_lines
    for line in copy_lines:
        lowered = line.lower()
        assert "weight" not in lowered
        assert ".safetensors" not in lowered
        assert "huggingface" not in lowered
        # Do not bake a local 27B checkout into the image.
        assert " models" not in f" {line} ".lower()
        assert not line.lower().startswith("copy models")


def test_compose_services_profiles_and_volumes(repo_root: Path) -> None:
    compose_path = repo_root / "docker-compose.yml"
    text = compose_path.read_text()
    data = yaml.safe_load(text)
    services = data["services"]
    for name in (
        "megaquant",
        "quantize",
        "mixed",
        "w4a4",
        "serve-sglang",
        "serve-vllm",
        "eval-gpqa",
        "shell",
    ):
        assert name in services, f"missing compose service {name}"

    profiles = services["quantize"].get("profiles") or []
    assert "gpu" in profiles
    assert "gpu" in (services["w4a4"].get("profiles") or [])
    assert "gpu" in (services["eval-gpqa"].get("profiles") or [])
    assert "gpu" in (services["serve-sglang"].get("profiles") or [])
    assert "gpu" in (services["serve-vllm"].get("profiles") or [])
    mixed_cmd = " ".join(str(x) for x in (services["mixed"].get("command") or []))
    assert "mixed.5090.yaml" in mixed_cmd
    w4a4_cmd = " ".join(str(x) for x in (services["w4a4"].get("command") or []))
    assert "w4a4.5090.yaml" in w4a4_cmd
    eval_cmd = " ".join(str(x) for x in (services["eval-gpqa"].get("command") or []))
    assert "eval-gpqa-diamond.yaml" in eval_cmd
    serve_cmd = " ".join(str(x) for x in (services["serve-sglang"].get("command") or []))
    assert "serve" in serve_cmd
    assert "eval-gpqa-diamond.yaml" in serve_cmd
    assert "MEGAQUANT_SGLANG_BASE_URL" in text
    assert "30000" in text
    assert "serve-sglang:30000" in text
    assert "SGLANG_ENABLE_JIT_DEEPGEMM" in text
    serve_build = services["serve-sglang"].get("build") or {}
    if isinstance(serve_build, dict):
        assert serve_build.get("dockerfile") == "Dockerfile.sglang"
        serve_args = serve_build.get("args") or {}
        assert serve_args.get("BASE_IMAGE") == (
            "${SGLANG_BASE_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
        )
    eval_svc = services["eval-gpqa"]
    assert eval_svc.get("gpus") in (None, False, [])
    eval_deploy = eval_svc.get("deploy") or {}
    eval_devices = (
        (eval_deploy.get("resources") or {}).get("reservations") or {}
    ).get("devices")
    assert not eval_devices
    serve_image = str(services["serve-sglang"].get("image") or "")
    assert "megaquant:sglang" in serve_image
    assert "fetch-export" in services
    assert "fetch-gpqa" in services

    volume_blob = text
    megaquant_vols = services["megaquant"].get("volumes") or []
    if megaquant_vols:
        volume_blob = "\n".join(str(item) for item in megaquant_vols)
    assert "/cache/huggingface" in volume_blob
    assert "/models" in volume_blob
    assert "/opt/megaquant/outputs" in volume_blob
    assert "/opt/megaquant/offload" in volume_blob
    assert "./src:/opt/megaquant/src" in volume_blob
    assert "offload" in text


def test_gpu_pod_packs_host_ram_threads_and_batch(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "gpu-pod.sh").read_text()
    assert "MEGAQUANT_GPU_HEADROOM_GIB" in script
    assert "MEGAQUANT_CPU_RESERVE_GIB" in script
    assert "MEGAQUANT_BATCH_SIZE" in script
    assert "OMP_NUM_THREADS" in script
    assert "nproc" in script
    assert "0:26GiB,cpu:40GiB" not in script
    assert "batch_size: 4" in script
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in script
    assert "w4a4" in script
    assert "mixed" in script
    assert "qwen3.8-27b-nvfp4-${SCHEME}.5090.yaml" in script
    packed = "qwen3.8-27b-nvfp4-${SCHEME}.pod.yaml"
    assert packed in script or "qwen3.8-27b-nvfp4-w4a4.pod.yaml" in script
    assert "nvfp4_mixed" in script
    assert "nvfp4_w4a4" in script
    assert 'MEGAQUANT_GPU_HEADROOM_GIB:-1' in script or 'MEGAQUANT_GPU_HEADROOM_GIB:-"1"' in script
    assert "eval-gpqa-diamond.yaml" in script
    assert "serve|eval" in script
    assert "HiCache" in script or "KV CPU offload" in script
    assert "${EVAL_MODEL}/gpqa_diamond" in script
    assert "MEGAQUANT_SGLANG_BASE_URL" in script
    assert "http://127.0.0.1:30000/v1" in script
    assert "publish|rewrite-sglang" in script
    assert "oss_publish.py" in script
    assert "publish|rewrite-sglang|schemes|families" in script
    assert "CUDAHOSTCXX" in script
    assert "cc1plus" in script


def test_gpu_pod_publish_and_rewrite_sglang_skip_cuda_probe(repo_root: Path) -> None:
    script = (repo_root / "scripts" / "gpu-pod.sh").read_text()
    assert "plan|quantize|serve|eval|publish|rewrite-sglang" in script
    assert "publish|rewrite-sglang" in script
    assert "oss_publish.py" in script
    assert "--scheme" in script
    assert "megaquant.cli rewrite-sglang" in script
    skip_idx = script.find("publish|rewrite-sglang|schemes|families")
    probe_idx = script.find("torch.cuda.is_available")
    smi_idx = script.find("nvidia-smi")
    publish_exec = script.find("oss_publish.py")
    assert skip_idx != -1
    assert probe_idx != -1
    assert smi_idx != -1
    assert skip_idx < probe_idx
    assert skip_idx < smi_idx
    assert skip_idx < publish_exec
    # Probe remains for GPU commands (plan/quantize/serve/eval).
    assert "sys.exit(" in script[probe_idx : probe_idx + 400] or "CUDA not available" in script


def test_ngc_compose_covers_eval_and_serve(repo_root: Path) -> None:
    data = yaml.safe_load((repo_root / "docker-compose.ngc.yml").read_text())
    services = data["services"]
    for name in ("eval-gpqa", "serve-sglang", "serve-vllm", "w4a4", "mixed"):
        assert name in services, name
    assert services["serve-sglang"]["image"] == "megaquant:sglang"
    assert services["eval-gpqa"]["image"] == "megaquant:sglang"
    assert services["quantize"]["image"] == "megaquant:nvfp4-ngc"


def test_dockerfile_copies_scripts(repo_root: Path) -> None:
    dockerfile = (repo_root / "Dockerfile").read_text()
    assert "COPY scripts ./scripts" in dockerfile


def test_dockerfile_sglang_does_not_copy_weights(repo_root: Path) -> None:
    dockerfile = (repo_root / "Dockerfile.sglang").read_text()
    assert "COPY src ./src" in dockerfile
    assert "requirements-sglang.txt" in dockerfile
    assert "sglang" in dockerfile.lower()
    assert "cu130" in dockerfile.lower() or "580" in dockerfile
    for line in dockerfile.splitlines():
        if line.strip().startswith("COPY "):
            lowered = line.lower()
            assert "safetensors" not in lowered
            assert "huggingface" not in lowered


def test_install_host_does_not_bake_tenant_dns(repo_root: Path) -> None:
    script = (repo_root / "docker" / "install-host.sh").read_text()
    readme = (repo_root / "docker" / "README.md").read_text()
    assert "docker.io" in script
    assert "nvidia-container-toolkit" in script
    assert "MEGAQUANT_DOCKER_DNS" in script
    assert "systemd-resolved" in script
    assert "gai.conf" in script
    assert "MEGAQUANT_PREFER_IPV4" in script
    assert "--dns-only" in script
    assert "megaquant-docker-hub-pin" in script
    assert "MEGAQUANT_DOCKER_PIN_CANARY" in script
    assert "registry-1.docker.io" in script
    assert "get.docker.com" not in script
    assert "100.90.90.90" not in script
    assert "100.90.90.100" not in script
    assert "MEGAQUANT_DOCKER_DNS" in readme
    assert "megaquant-stack.tar" in readme
    text_globs = (
        "*.sh",
        "*.yml",
        "*.yaml",
        "*.md",
        "*.py",
        "*.txt",
        "*.example",
        "Dockerfile*",
        "Makefile",
    )
    hits: list[str] = []
    for glob in text_globs:
        for path in repo_root.rglob(glob):
            if ".git" in path.parts or path == repo_root / "tests" / "test_docker_packaging.py":
                continue
            blob = path.read_text(encoding="utf-8", errors="ignore")
            if "100.90.90.90" in blob or "100.90.90.100" in blob:
                hits.append(str(path.relative_to(repo_root)))
    assert hits == []


def test_entrypoint_bare_plan_uses_recipe_default(repo_root: Path) -> None:
    script = (repo_root / "docker" / "entrypoint.sh").read_text()
    assert "RECIPE:-recipes/qwen3.8-27b-nvfp4-w4a8.yaml" in script
    assert '== "plan"' in script or "== 'plan'" in script
    assert "-c" in script
    assert "eval-gpqa-diamond.yaml" in script
    assert '"serve"' in script
    assert "CUDAHOSTCXX" in script
    assert "cc1plus" in script
    assert "MEGAQUANT_SKIP_GPU_REPORT" in script


def test_sglang_requirements_pin_engine(repo_root: Path) -> None:
    req = (repo_root / "docker" / "requirements-sglang.txt").read_text()
    assert "sglang==0.5.20" in req
    assert "580" in req or "CUDA 13" in req


def test_compose_does_not_export_empty_cuda_visible_devices(repo_root: Path) -> None:
    text = (repo_root / "docker-compose.yml").read_text()
    assert "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" not in text
    # Empty CUDA_VISIBLE_DEVICES hides every GPU from torch.
    text = (repo_root / "docker-compose.yml").read_text()
    assert "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-16}" in text
    assert "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-16}" in text


def test_serve_and_eval_scripts_are_executable_helpers(repo_root: Path) -> None:
    serve = (repo_root / "scripts" / "serve_sglang.sh").read_text()
    eval_sh = (repo_root / "scripts" / "eval_gpqa.sh").read_text()
    assert "megaquant.cli serve" in serve
    assert "KV" in serve or "HiCache" in serve
    assert "cc1plus" in serve
    assert "CUDAHOSTCXX" in serve
    assert "megaquant.cli eval" in eval_sh
    assert "eval-gpqa-diamond.yaml" in eval_sh
    wrapper = (repo_root / "scripts" / "serve_vllm.sh").read_text()
    assert "serve_sglang.sh" in wrapper


def test_sglang_default_base_is_cuda_128_optional_129_rebuild(repo_root: Path) -> None:
    """Default megaquant:sglang stays CUDA 12.8.1; 12.9 rebuild is documented."""
    dockerfile = (repo_root / "Dockerfile.sglang").read_text()
    compose_text = (repo_root / "docker-compose.yml").read_text()
    readme = (repo_root / "docker" / "README.md").read_text()
    env_example = (repo_root / ".env.example").read_text()
    host_check = (repo_root / "docker" / "host-check.sh").read_text()
    data = yaml.safe_load(compose_text)
    services = data["services"]

    assert "ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04" in dockerfile
    assert "${SGLANG_BASE_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}" in compose_text
    assert "SGLANG_ENABLE_JIT_DEEPGEMM: ${SGLANG_ENABLE_JIT_DEEPGEMM:-0}" in compose_text
    assert "${EVAL_RECIPE:-recipes/eval-gpqa-diamond.yaml}" in compose_text

    serve_cmd = " ".join(str(x) for x in (services["serve-sglang"].get("command") or []))
    eval_cmd = " ".join(str(x) for x in (services["eval-gpqa"].get("command") or []))
    assert "eval-gpqa-diamond.yaml" in serve_cmd
    assert "eval-gpqa-diamond.5090.yaml" not in serve_cmd
    assert "eval-gpqa-diamond.5090.yaml" not in eval_cmd

    # Two-image invariant: PTQ vs SGLang.
    assert "megaquant:nvfp4" in str(services["megaquant"].get("image"))
    assert "megaquant:sglang" in str(services["serve-sglang"].get("image"))
    assert (services["serve-sglang"].get("build") or {}).get("dockerfile") == "Dockerfile.sglang"
    assert (services["quantize"].get("build") or {}).get("dockerfile") in (None, "Dockerfile")
    ptq_dockerfile = (repo_root / "Dockerfile").read_text()
    assert "nvidia/cuda:12.8.1-devel-ubuntu24.04" in ptq_dockerfile

    documented = f"{dockerfile}\n{compose_text}\n{readme}\n{env_example}"
    assert "nvidia/cuda:12.9.1-devel-ubuntu24.04" in documented
    assert "SGLANG_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04" in documented
    assert "12.9" in readme
    assert "eval-gpqa-diamond.5090.yaml" in readme
    assert "SGLANG_BASE_IMAGE" in env_example
    assert "SGLANG_ENABLE_JIT_DEEPGEMM" in env_example
    assert "12.9" in host_check
    assert "FlashInfer" in host_check
    assert "580" in host_check
    assert "nvcc --version" in host_check
    # Operator override, not a runtime probe.
    assert "do not probe nvcc" in dockerfile.lower() or "Do not probe nvcc" in dockerfile
