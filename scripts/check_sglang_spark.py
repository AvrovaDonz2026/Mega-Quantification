#!/usr/bin/env python3
"""Check the Spark serving container before loading a checkpoint (GPU required)."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

from check_sglang_ba import check_ba_batch_invariance
from check_sglang_fp8 import check_fp8_batch_invariance
from check_sglang_gdn import check_gdn_decode_verify
from container_manifest import MANIFEST, snapshot, verify
from patch_sglang_spark import patch_tree


def main() -> None:
    import sgl_kernel
    import torch

    root = Path("/opt/megaquant")
    manifest = json.loads((root / MANIFEST).read_text())
    errors = verify(manifest, snapshot(root))
    if errors:
        raise SystemExit(f"Container manifest failed: {errors}")
    import sglang

    patch = patch_tree(
        Path(sglang.__file__).parent, importlib.metadata.version("sglang"), check=True
    )
    report = {
        "machine": platform.machine(),
        "sglang": importlib.metadata.version("sglang"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sgl_kernel": sgl_kernel.__file__,
        "git_revision": manifest["git_revision"],
        "container_manifest": "verified",
        "sglang_patch": patch,
    }
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: run the Spark image with --gpus all")
    report["device"] = torch.cuda.get_device_name(0)
    report["capability"] = torch.cuda.get_device_capability(0)
    if report["machine"] != "aarch64" or report["capability"] != (12, 1):
        raise SystemExit(f"Expected ARM64 GB10 / SM121, got {report}")
    matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    if not torch.all(matrix @ matrix == 32).item():
        raise SystemExit("CUDA matrix multiplication check failed")
    torch.cuda.synchronize()
    report["cuda_matmul"] = "passed"
    report["gdn_decode_verify"] = check_gdn_decode_verify()
    report["fp8_batch_invariance"] = check_fp8_batch_invariance()
    report["ba_batch_invariance"] = check_ba_batch_invariance()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
