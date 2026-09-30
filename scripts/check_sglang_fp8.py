#!/usr/bin/env python3
"""Require batch-invariant ModelOpt static FP8 linear results on Spark / SM121."""

from __future__ import annotations

import json


def check_fp8_batch_invariance() -> list[dict]:
    """Exercise the installed weight loader and dispatch without overriding a backend."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FP8 batch invariance validation requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"FP8 Spark validation requires SM121, got {capability}")

    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptFp8Config,
        ModelOptFp8LinearMethod,
    )

    def compare(first, batched) -> dict:
        if first.shape != batched.shape or first.dtype != batched.dtype:
            raise RuntimeError(
                "FP8 comparison shape/dtype mismatch: "
                f"{first.shape}/{first.dtype} vs {batched.shape}/{batched.dtype}"
            )
        # Compare storage bits as well as values: float equality hides signed
        # zeros and is not implemented for every CUDA FP8 tensor operation.
        bits_dtype = torch.uint8 if first.element_size() == 1 else torch.int16
        if first.element_size() == 4:
            bits_dtype = torch.int32
        bitwise_nonzero = int(
            torch.count_nonzero(first.view(bits_dtype) != batched.view(bits_dtype)).item()
        )
        difference = (first.float() - batched.float()).abs()
        nonfinite = int(torch.count_nonzero(~torch.isfinite(difference)).item())
        return {
            "equal": bitwise_nonzero == 0,
            "bitwise_nonzero": bitwise_nonzero,
            "nonzero": int(torch.count_nonzero(difference).item()),
            "max_abs": None if nonfinite else difference.max().item(),
            "nonfinite": nonfinite,
        }

    reports = []
    # Qwen3.8 GDN qkv and output projection dimensions, with checkpoint-like
    # fixed scales. The M=1 and M=3 calls must use the same serialized weights.
    for seed, n, k in ((42, 10240, 5120), (11, 5120, 6144)):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode(), torch.device("cuda"):
            method = ModelOptFp8LinearMethod(ModelOptFp8Config(is_checkpoint_fp8_serialized=True))
            layer = torch.nn.Module()
            method.create_weights(
                layer,
                input_size_per_partition=k,
                output_partition_sizes=[n],
                input_size=k,
                output_size=n,
                params_dtype=torch.bfloat16,
            )
            layer.weight_scale.fill_(0.004713)
            layer.input_scale.fill_(0.1529017985)
            weight = torch.randn((n, k), dtype=torch.bfloat16, generator=generator).mul_(0.1)
            serialized_weight, _ = static_quant_fp8(weight, layer.weight_scale, repeat_scale=False)
            layer.weight.copy_(serialized_weight)
            del weight, serialized_weight
            method.process_weights_after_loading(layer)
            if method.use_marlin or layer.weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("FP8 validation requires static FP8 weights and activations")

            inputs = torch.randn((3, k), dtype=torch.bfloat16, generator=generator)
            batch_quantized, batch_scale = static_quant_fp8(
                inputs, layer.input_scale, repeat_scale=False
            )
            batch_output = method.apply(layer, inputs)
            if batch_output.shape != (3, n) or batch_output.dtype != torch.bfloat16:
                raise RuntimeError(
                    "Expected BF16 FP8 linear output of shape "
                    f"{(3, n)}, got {batch_output.shape}/{batch_output.dtype}"
                )
            rows = []
            for row in range(3):
                single_input = inputs[row : row + 1]
                single_quantized, single_scale = static_quant_fp8(
                    single_input, layer.input_scale, repeat_scale=False
                )
                single_output = method.apply(layer, single_input)
                rows.append(
                    {
                        "row": row,
                        "quantized_bytes": compare(
                            single_quantized, batch_quantized[row : row + 1]
                        ),
                        "quantization_scale": compare(
                            single_scale.reshape(-1), batch_scale.reshape(-1)
                        ),
                        "output": compare(single_output, batch_output[row : row + 1]),
                    }
                )
            torch.cuda.synchronize()
            report = {
                "seed": seed,
                "n": n,
                "k": k,
                "batch": 3,
                "input_dtype": str(inputs.dtype),
                "weight_dtype": str(layer.weight.dtype),
                "processed_weight_shape": list(layer.weight.shape),
                "input_scale": layer.input_scale.item(),
                "weight_scale": layer.weight_scale.reshape(-1)[0].item(),
                "weight_scale_numel": layer.weight_scale.numel(),
                "sm120_fp8_facade": method.use_sm120_fp8,
                "flashinfer_bmm": layer.use_flashinfer_bmm,
                "marlin": method.use_marlin,
                "cutlass_supported": method.cutlass_fp8_supported,
                "rows": rows,
            }
            failures = [
                {"row": row["row"], "check": name, **row[name]}
                for row in rows
                for name in ("quantized_bytes", "quantization_scale", "output")
                if not row[name]["equal"] or row[name]["nonfinite"]
            ]
            if failures:
                report["status"] = "failed"
                report["failures"] = failures
                raise RuntimeError(
                    "ModelOpt FP8 batch invariance mismatch: " + json.dumps(report, allow_nan=False)
                )
            report["status"] = "passed"
            reports.append(report)
    return reports


if __name__ == "__main__":
    print(json.dumps(check_fp8_batch_invariance(), indent=2, allow_nan=False))
