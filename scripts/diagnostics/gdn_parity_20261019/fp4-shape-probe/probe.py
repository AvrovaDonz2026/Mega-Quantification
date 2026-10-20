#!/usr/bin/env python3
"""Synthetic NVFP4 row-shape probe for the FlashInfer mm_fp4 path.

This is deliberately a small kernel probe, not a model accuracy test.  It
quantizes one fixed BF16 input row and one fixed BF16 weight matrix, repeats
the input row to M=1/2/3/4/8, and invokes the same transposed packed layout
used by ``ModelOptFp4LinearMethod.apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable


def _parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            k_text, n_text = item.split("x", 1)
            k, n = int(k_text), int(n_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid shape {item!r}; expected KxN, e.g. 5120x12288"
            ) from exc
        if k <= 0 or n <= 0 or k % 16:
            raise argparse.ArgumentTypeError(
                f"shape {item!r} must have positive K,N and K divisible by 16"
            )
        shapes.append((k, n))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one KxN shape is required")
    return shapes


def _parse_rows(text: str) -> list[int]:
    try:
        rows = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from exc
    if not rows or any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("rows must contain positive integers")
    return rows


def _as_float8(scale, torch):
    # FlashInfer versions may return E4M3 values as raw uint8 bytes.  The
    # ModelOpt path passes a float8 view to CUTLASS, matching this conversion.
    if scale.dtype != torch.float8_e4m3fn:
        scale = scale.view(torch.float8_e4m3fn)
    return scale


def _sha256(tensor, torch) -> str:
    # NumPy builds commonly lack a bfloat16 dtype; hash the raw contiguous
    # storage through a uint8 view so summaries work for BF16 and FP8 alike.
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _scalar_summary(tensor, torch) -> dict[str, object]:
    flat = tensor.detach().to(torch.float32).flatten()
    finite = bool(torch.isfinite(flat).all().item())
    if flat.numel() == 0:
        return {
            "finite": finite,
            "sum": 0.0,
            "absmax": 0.0,
            "l2": 0.0,
            "sha256": _sha256(tensor, torch),
        }
    return {
        "finite": finite,
        "sum": float(flat.sum().item()),
        "absmax": float(flat.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "sha256": _sha256(tensor, torch),
    }


def _custom_shape_supported(m: int, k: int, n: int) -> bool:
    # Keep this in sync with qwen3x_nvfp4_gemm.py.  The probe calls FlashInfer
    # directly, but reporting the gate prevents a synthetic result being read
    # as evidence that the SM120 custom kernel ran.
    return (m, k, n) in {
        (rows, kk, nn)
        for rows in (1, 2, 4, 8)
        for kk, nn in (
            (2560, 18432),
            (9216, 2560),
            (4096, 24576),
            (12288, 4096),
        )
    } or (m, k, n) == (9, 17408, 5120)


def _run_shape(k: int, n: int, rows: Iterable[int], seed: int, backend: str, torch, flashinfer):
    # Keep one fixed BF16 vector and one fixed BF16 weight matrix for every M.
    generator = torch.Generator(device="cuda").manual_seed(seed + k * 17 + n * 31)
    x_one = torch.randn((1, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)

    # This follows the public FlashInfer benchmark and ModelOpt's scalar alpha
    # contract: alpha is the product of the activation and weight global scales.
    f4_max = 6.0
    e4m3_max = 448.0
    x_scale = (e4m3_max * f4_max) / torch.amax(x_one.to(torch.float32).abs())
    w_scale = (e4m3_max * f4_max) / torch.amax(w_bf16.to(torch.float32).abs())
    alpha = (1.0 / (x_scale * w_scale)).to(torch.float32).reshape(1)

    # SGLang's fp4_utils selects the CUDA quantizer on SM121 (CuTe DSL is only
    # selected on SM100), so keep that backend explicit in this reproduction.
    x_one_fp4, x_one_sf = flashinfer.fp4_quantize(x_one, x_scale, backend="cuda")
    w_fp4, w_sf = flashinfer.fp4_quantize(w_bf16, w_scale, backend="cuda")
    x_one_sf = _as_float8(x_one_sf, torch)
    w_sf = _as_float8(w_sf, torch)

    # ModelOpt apply() passes weight.T and weight_scale_interleaved.T to mm_fp4.
    w_fp4_t = w_fp4.T.contiguous()
    w_sf_t = w_sf.T.contiguous()
    # Always establish the comparison row with M=1, even when a caller asks
    # for a subset/order such as ``--rows 3,4``.
    rows = sorted(set(rows))
    if 1 not in rows:
        rows = [1, *rows]
    reference = None
    result_rows: list[dict[str, object]] = []
    for m in rows:
        x = x_one.expand(m, -1).contiguous()
        x_fp4, x_sf = flashinfer.fp4_quantize(x, x_scale, backend="cuda")
        x_sf = _as_float8(x_sf, torch)
        packed_first_exact = bool(torch.equal(x_fp4[0], x_one_fp4[0]))
        scale_first_exact = bool(torch.equal(x_sf[0], x_one_sf[0]))
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        flashinfer.mm_fp4(
            x_fp4,
            w_fp4_t,
            x_sf,
            w_sf_t,
            alpha,
            torch.bfloat16,
            out,
            backend=backend,
        )
        first = out[0]
        if reference is None:
            reference = first.detach().clone()
        delta = (first.to(torch.float32) - reference.to(torch.float32)).abs()
        row_delta = (out.to(torch.float32) - first.to(torch.float32)).abs()
        result_rows.append(
            {
                "m": m,
                "input_fp4_shape": list(x_fp4.shape),
                "input_scale_shape": list(x_sf.shape),
                "input_first_fp4_exact_m1": packed_first_exact,
                "input_first_scale_exact_m1": scale_first_exact,
                "output": _scalar_summary(out, torch),
                "first_row_sha256": _sha256(first, torch),
                "first_row_exact_m1": bool(torch.equal(first, reference)),
                "first_row_max_abs_vs_m1": float(delta.max().item()),
                "rows_exact_first": bool(torch.equal(out, first.expand_as(out))),
                "rows_max_abs_vs_first": float(row_delta.max().item()),
            }
        )
    return {
        "k": k,
        "n": n,
        "weight_fp4_shape": list(w_fp4.shape),
        "weight_scale_shape": list(w_sf.shape),
        "weight_transposed_stride": list(w_fp4_t.stride()),
        "weight_scale_transposed_stride": list(w_sf_t.stride()),
        "custom_shape_supported_by_sm120_gate": {
            str(m): _custom_shape_supported(m, k, n) for m in rows
        },
        "rows": result_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=_parse_shapes,
        default=_parse_shapes("5120x12288,5120x64,4096x5120"),
        help="comma-separated KxN shapes",
    )
    parser.add_argument("--rows", type=_parse_rows, default=_parse_rows("1,2,3,4,8"))
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument(
        "--backend",
        choices=("cutlass", "cudnn", "cute-dsl", "auto"),
        default="cutlass",
        help="FlashInfer mm_fp4 backend; cutlass mirrors SM121 ModelOpt auto",
    )
    args = parser.parse_args()

    try:
        import flashinfer
        import torch
    except ImportError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}))
        return 2
    if not torch.cuda.is_available():
        print(json.dumps({"status": "unavailable", "error": "CUDA is unavailable"}))
        return 2

    capability = tuple(torch.cuda.get_device_capability())
    custom_runtime_gate = capability == (12, 0)
    report: dict[str, object] = {
        "status": "ok",
        "probe": "synthetic_flashinfer_nvfp4_row_shape",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(),
        "capability": list(capability),
        "dispatch": "flashinfer.mm_fp4",
        "backend": args.backend,
        "qwen3x_custom_runtime_gate": custom_runtime_gate,
        "qwen3x_custom_runtime_gate_note": (
            "SM121 intentionally false; qwen3x gate requires capability==(12,0)"
        ),
        "warning": "Synthetic fixed-weight probe; does not prove full-model parity.",
        "shapes": [],
    }
    for k, n in args.shapes:
        try:
            report["shapes"].append(
                _run_shape(k, n, args.rows, args.seed, args.backend, torch, flashinfer)
            )
        except Exception as exc:  # Keep one JSON result useful for CI logs.
            report["shapes"].append({"k": k, "n": n, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if all("error" not in item for item in report["shapes"]) else 1


if __name__ == "__main__":
    sys.exit(main())
