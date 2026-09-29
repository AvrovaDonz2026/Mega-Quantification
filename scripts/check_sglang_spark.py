#!/usr/bin/env python3
"""Check the Spark serving container before loading a checkpoint (GPU required)."""

from __future__ import annotations

import importlib.metadata
import json
import platform


def main() -> None:
    import sgl_kernel
    import torch

    report = {
        "machine": platform.machine(),
        "sglang": importlib.metadata.version("sglang"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sgl_kernel": sgl_kernel.__file__,
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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
