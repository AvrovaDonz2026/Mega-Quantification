#!/usr/bin/env python3
"""Validate Spark PTQ runtime, full checkpoint and multimodal calibration inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from container_manifest import MANIFEST, sha256, snapshot, verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.8-27B"))
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    from megaquant.backends.modelopt import _prepare_qwen35_export

    root = Path("/opt/megaquant")
    manifest = json.loads((root / MANIFEST).read_text())
    errors = verify(manifest, snapshot(root))
    if errors:
        raise RuntimeError(f"Container manifest failed: {errors}")
    if platform.machine() != "aarch64" or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Expected ARM64 GB10 / SM121")
    matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    if not torch.all(matrix @ matrix == 32).item():
        raise RuntimeError("CUDA matrix multiplication failed")

    config = qwen.Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention"],
    )
    gdn = qwen.Qwen3_5GatedDeltaNet(config, layer_idx=0).to("cuda", dtype=torch.bfloat16)

    class Qwen3_5ForConditionalGeneration:
        def modules(self):
            return iter((self, gdn))

    if not _prepare_qwen35_export(Qwen3_5ForConditionalGeneration()):
        raise RuntimeError("GDN export preparation failed")
    with torch.inference_mode():
        result = gdn(torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16))
    if result.shape != (1, 4, 32) or not torch.isfinite(result).all().item():
        raise RuntimeError("GDN CUDA forward failed")

    calibration = args.data / "calibration.jsonl"
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    if sha256(calibration) != data_manifest["calibration"]["sha256"]:
        raise RuntimeError("Calibration checksum mismatch")
    rows = [json.loads(line) for line in calibration.read_text().splitlines() if line.strip()]
    if len(rows) != 256 or len(data_manifest["images"]) != 128:
        raise RuntimeError("Expected 256 calibration rows and 128 images")
    for item in data_manifest["images"]:
        path = args.data / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Image checksum mismatch: {item['path']}")
        with Image.open(path) as image:
            image.verify()
    index_path = args.model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not all((args.model / shard).is_file() for shard in shards):
        raise RuntimeError("Full model shards are missing")
    report = {
        "passed": True,
        "git_revision": manifest["git_revision"],
        "container_manifest_sha256": sha256(root / MANIFEST),
        "architecture": platform.machine(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "nvidia-modelopt", "accelerate", "Pillow")
        },
        "cuda_matmul": "passed",
        "gdn_export_fallback_cuda_forward": "passed",
        "calibration_rows": len(rows),
        "image_files_verified": len(data_manifest["images"]),
        "calibration_sha256": sha256(calibration),
        "source_index_sha256": sha256(index_path),
        "source_shards": shards,
        "source_tensors": len(index["weight_map"]),
        "source_mtp_tensors": sum(key.startswith("mtp.") for key in index["weight_map"]),
        "source_vision_tensors": sum(
            key.startswith("model.visual.") for key in index["weight_map"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
