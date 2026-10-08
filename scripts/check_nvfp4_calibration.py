#!/usr/bin/env python3
"""Check NVFP4 scale calibration on synthetic CUDA inputs without model downloads.

This operator check compares max, MSE and Local-Hessian using identical weights
and calibration inputs. Its errors are not a model quality benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import modelopt.torch.quantization as mtq
    import torch
    from torch import nn

    from megaquant.backends.modelopt import ModelOptBackend
    from megaquant.config import load_recipe
    from megaquant.pipeline import QuantPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("This check requires CUDA and the PTQ dependencies")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(256, 128, bias=False)

        def forward(self, value):
            return self.mlp(value)

    original = Projection().to("cuda", dtype=torch.bfloat16).eval()
    # Unequal channel variances exercise the input-aware Hessian objective.
    variance = torch.logspace(-1, 1, 256, device="cuda")
    calibration = (torch.randn(512, 256, device="cuda") * variance).to(torch.bfloat16)
    heldout = (torch.randn(256, 256, device="cuda") * variance).to(torch.bfloat16)
    weight = original.mlp.weight.detach().float()
    reference = heldout.float() @ weight.T
    reports = {}
    quantized_weights = {}
    input_scales = {}
    recipe = load_recipe(Path("recipes/qwen3.8-27b-nvfp4-w4a4.spark.yaml"))

    for algorithm in ("max", "mse", "local_hessian"):
        model = copy.deepcopy(original)
        candidate = recipe.model_copy(update={"algorithm": algorithm})
        cfg = ModelOptBackend().build_quant_cfg(QuantPipeline(candidate).resolve())
        calls = 0

        def forward_loop(calibration_model):
            nonlocal calls
            calls += 1
            for batch in calibration.split(64):
                calibration_model(batch)

        with torch.inference_mode():
            model = mtq.quantize(model, cfg, forward_loop=forward_loop)
            quantizer = model.mlp.weight_quantizer
            if quantizer.num_bits != (2, 1) or quantizer.block_sizes.get(-1) != 16:
                raise RuntimeError("Calibration changed the NVFP4 block-16 format")
            input_quantizer = model.mlp.input_quantizer
            if input_quantizer.num_bits != (2, 1) or input_quantizer.block_sizes.get(-1) != 16:
                raise RuntimeError("Calibration changed the NVFP4 activation format")
            quantized_weight = quantizer(model.mlp.weight).float()
            output = model(heldout).float()
            if not torch.isfinite(output).all() or not torch.isfinite(quantized_weight).all():
                raise RuntimeError(f"Non-finite quantized values: {algorithm}")
            amax = quantizer.amax.detach().float().cpu()
            if not torch.isfinite(amax).all() or not (amax > 0).all():
                raise RuntimeError(f"Invalid weight scales: {algorithm}")
            if algorithm != "max":
                global_amax = quantizer.global_amax.detach().float()
                if not torch.isfinite(global_amax).all() or not (global_amax > 0).all():
                    raise RuntimeError(f"Invalid global weight scale: {algorithm}")
            quantized_weights[algorithm] = quantized_weight.detach().cpu()
            input_scales[algorithm] = input_quantizer.amax.detach().float().cpu()
            if not torch.isfinite(input_scales[algorithm]).all() or not (
                input_scales[algorithm] > 0
            ).all():
                raise RuntimeError(f"Invalid activation scales: {algorithm}")
            weight_mse = (quantized_weight - weight).square().mean().item()
            weight_output = heldout.float() @ quantized_weight.T
            reports[algorithm] = {
                "algorithm_config": cfg["algorithm"],
                "forward_loop_calls": calls,
                "weight_scale_count": amax.numel(),
                "weight_block_type": quantizer.block_sizes.get("type", "static"),
                "weight_calibrator": type(quantizer._calibrator).__name__,
                "weight_scale_sha256": hashlib.sha256(amax.numpy().tobytes()).hexdigest(),
                "weight_mse": weight_mse,
                "heldout_weight_output_relative_mse": (
                    (weight_output - reference).square().sum() / reference.square().sum()
                ).item(),
                "heldout_w4a4_output_relative_mse": (
                    (output - reference).square().sum() / reference.square().sum()
                ).item(),
            }
        del model

    for algorithm in ("mse", "local_hessian"):
        reports[algorithm]["weight_values_changed_from_max"] = int(
            (quantized_weights[algorithm] != quantized_weights["max"]).sum().item()
        )
        if reports[algorithm]["weight_calibrator"] != "NVFP4MSECalibrator":
            raise RuntimeError(f"NVFP4 scale search was skipped: {algorithm}")
        if reports[algorithm]["weight_values_changed_from_max"] == 0:
            raise RuntimeError(f"Weight scale refinement had no effect: {algorithm}")
        if not torch.equal(input_scales[algorithm], input_scales["max"]):
            raise RuntimeError(f"Input calibration changed unexpectedly: {algorithm}")
    if reports["mse"]["weight_mse"] > reports["max"]["weight_mse"] * 1.001:
        raise RuntimeError("MSE scale search worsened its measured weight objective")
    if reports["local_hessian"]["forward_loop_calls"] < 2:
        raise RuntimeError("Local-Hessian did not collect an additional input pass")

    report = {
        "passed": True,
        "scope": "synthetic NVFP4 operator calibration; not GPQA or full-model accuracy",
        "seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "nvidia-modelopt")
        },
        "calibration_shape": list(calibration.shape),
        "heldout_shape": list(heldout.shape),
        "weight_shape": list(weight.shape),
        "activation_max_scales_equal": True,
        "algorithms": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
