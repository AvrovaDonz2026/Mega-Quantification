#!/usr/bin/env python3
"""Require batch-invariant BF16 GDN BA projection results on Spark / SM121."""

from __future__ import annotations

import json


def check_ba_batch_invariance() -> list[dict]:
    """Exercise the installed unquantized linear loader and dispatch unchanged."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("BA batch invariance validation requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"BA Spark validation requires SM121, got {capability}")

    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    n, k = 96, 5120
    prefix = "model.language_model.layers.2.linear_attn.in_proj_ba"
    # BF16 rounding contributes at most about 1/256 relative error. Allow a
    # small margin and an absolute budget for FP32 reduction near cancellation.
    # Evaluate against FP64 dot products of the actual BF16 inputs and weights.
    rtol, atol = 1 / 256 + 0.0001, 0.001

    def check_output(output, batch: int) -> None:
        if output.shape != (batch, n) or output.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Expected BF16 BA output of shape {(batch, n)}, got {output.shape}/{output.dtype}"
            )

    def compare(single, batched) -> dict:
        difference = (single.float() - batched.float()).abs()
        nonfinite = int(
            torch.count_nonzero(~torch.isfinite(single)).item()
            + torch.count_nonzero(~torch.isfinite(batched)).item()
        )
        bitwise_nonzero = int(
            torch.count_nonzero(single.view(torch.int16) != batched.view(torch.int16)).item()
        )
        return {
            "equal": bitwise_nonzero == 0,
            "bitwise_nonzero": bitwise_nonzero,
            "nonzero": int(torch.count_nonzero(difference).item()),
            "max_abs": None if nonfinite else difference.max().item(),
            "nonfinite": nonfinite,
        }

    def compare_reference(output, reference) -> dict:
        # The CPU FP64 reference remains independent of CUDA dispatch overrides.
        actual = output.cpu().double()
        difference = (actual - reference).abs()
        nonfinite = int(
            torch.count_nonzero(~torch.isfinite(actual)).item()
            + torch.count_nonzero(~torch.isfinite(reference)).item()
        )
        outside_tolerance = int(
            torch.count_nonzero(difference > atol + rtol * reference.abs()).item()
        )
        return {
            "equal_within_tolerance": nonfinite == 0 and outside_tolerance == 0,
            "outside_tolerance": outside_tolerance,
            "max_abs": None if nonfinite else difference.max().item(),
            "nonfinite": nonfinite,
        }

    reports = []
    for seed in (42, 11):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode(), torch.device("cuda"):
            method = UnquantizedLinearMethod()
            layer = torch.nn.Module()
            layer.prefix = prefix
            method.create_weights(
                layer,
                input_size_per_partition=k,
                output_partition_sizes=[n // 2, n // 2],
                input_size=k,
                output_size=n,
                params_dtype=torch.bfloat16,
            )
            weight = torch.empty((n, k), dtype=torch.float32).uniform_(
                -0.125, 0.125, generator=generator
            )
            layer.weight.copy_(weight.to(torch.bfloat16))
            del weight
            method.process_weights_after_loading(layer)
            inputs = torch.empty((12, k), dtype=torch.float32).uniform_(-2, 2, generator=generator)
            inputs = inputs.to(torch.bfloat16)
            reference = inputs.cpu().double() @ layer.weight.cpu().double().t()
            single_outputs = []
            for row in range(12):
                output = method.apply(layer, inputs[row : row + 1])
                check_output(output, 1)
                single_outputs.append(output)
            single_outputs = torch.cat(single_outputs)
            single_reference = compare_reference(single_outputs, reference)
            batches = []
            for batch in (3, 12):
                output = method.apply(layer, inputs[:batch])
                check_output(output, batch)
                rows = [
                    {
                        "row": row,
                        **compare(single_outputs[row : row + 1], output[row : row + 1]),
                    }
                    for row in range(batch)
                ]
                batches.append(
                    {
                        "batch": batch,
                        "rows": rows,
                        "fp64_reference": compare_reference(output, reference[:batch]),
                    }
                )
            torch.cuda.synchronize()
            report = {
                "seed": seed,
                "n": n,
                "k": k,
                "prefix": prefix,
                "input_dtype": str(inputs.dtype),
                "weight_dtype": str(layer.weight.dtype),
                "input_bound": 2,
                "weight_bound": 0.125,
                "reference_dtype": "torch.float64",
                "reference_device": "cpu",
                "reference_rtol": rtol,
                "reference_atol": atol,
                "single_fp64_reference": single_reference,
                "batches": batches,
            }
            failures = [
                {"batch": batch["batch"], **row}
                for batch in batches
                for row in batch["rows"]
                if not row["equal"] or row["nonfinite"]
            ]
            if (
                failures
                or not single_reference["equal_within_tolerance"]
                or any(not batch["fp64_reference"]["equal_within_tolerance"] for batch in batches)
            ):
                report["status"] = "failed"
                report["failures"] = failures
                raise RuntimeError(
                    "GDN BA batch invariance or accuracy mismatch: "
                    + json.dumps(report, allow_nan=False)
                )
            report["status"] = "passed"
            reports.append(report)
    return reports


if __name__ == "__main__":
    print(json.dumps(check_ba_batch_invariance(), indent=2, allow_nan=False))
