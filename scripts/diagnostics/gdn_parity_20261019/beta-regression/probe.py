#!/usr/bin/env python3
"""Scalar-only GPU regression for packed GDN beta precision.

Run this file with ``PYTHONPATH`` pointing at the SGLang source worktree.  The
fixture is loaded from this directory so every variant uses the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixture.py"
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_git_revision(path: Path) -> str | None:
    """Return a local checkout revision when available; archives have none."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def load_fixture():
    spec = importlib.util.spec_from_file_location("beta_regression_fixture", FIXTURE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixture {FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_variant(packed_path: Path, gating_path: Path) -> dict[str, object]:
    packed = packed_path.read_text()
    gating = gating_path.read_text()
    # Restrict the packed check to the ordinary GDN packed kernel, not KDA.
    packed_body = packed.split("def fused_recurrent_gated_delta_rule_packed_decode_kernel", 1)[1]
    packed_body = packed_body.split("def fused_recurrent_gated_delta_rule_packed_decode(", 1)[0]
    packed_fp32 = not bool(re.search(r"beta_val\s*=.*\.to\(b\.dtype\.element_ty\)", packed_body))
    gating_body = gating.split("def fused_gdn_gating_kernel", 1)[1]
    gating_fp32 = not bool(re.search(r"blk_beta_output.*\.to\(b\.dtype\.element_ty\)", gating_body))
    if packed_fp32 and gating_fp32:
        observed = "all-beta"
    elif packed_fp32:
        observed = "decode-beta"
    elif not gating_fp32:
        observed = "original"
    else:
        observed = "unknown"
    return {
        "observed_variant": observed,
        "packed_beta_fp32_source": packed_fp32,
        "gating_beta_fp32_source": gating_fp32,
    }


def scalar_diff(lhs, rhs) -> dict[str, object]:
    import torch

    difference = (lhs.float() - rhs.float()).abs()
    return {
        "nonzero": int(torch.count_nonzero(difference).item()),
        "max_abs": float(difference.max().item()),
        "finite": bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()),
    }


def reference_gdn_gating(torch, triton, tl, A_log, a, b, dt_bias):
    """Mirror fused_gdn_gating with FP32 math and no beta dtype round-trip."""

    @triton.jit
    def reference_kernel(
        g,
        beta_output,
        A_log_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        stride_a,
        stride_b,
        num_heads: tl.constexpr,
        beta_scale: tl.constexpr,
        threshold: tl.constexpr,
        block_heads: tl.constexpr,
    ):
        batch_idx, head_block = tl.program_id(0), tl.program_id(1)
        heads = head_block * block_heads + tl.arange(0, block_heads)
        mask = heads < num_heads
        a_value = tl.load(a_ptr + batch_idx * stride_a + heads, mask=mask, other=0).to(tl.float32)
        b_value = tl.load(b_ptr + batch_idx * stride_b + heads, mask=mask, other=0).to(tl.float32)
        bias_value = tl.load(dt_bias_ptr + heads, mask=mask, other=0).to(tl.float32)
        log_value = tl.load(A_log_ptr + heads, mask=mask, other=0).to(tl.float32)
        x = a_value + bias_value
        beta_x = beta_scale * x
        softplus_x = tl.where(
            beta_x <= threshold,
            (1.0 / beta_scale) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        g_value = -tl.exp(log_value) * softplus_x
        # Match Triton's fused sigmoid implementation while retaining FP32.
        beta_value = tl.sigmoid(b_value)
        g_offset = batch_idx * num_heads + heads
        tl.store(g + g_offset, g_value, mask=mask)
        tl.store(beta_output + g_offset, beta_value, mask=mask)

    g = torch.empty((a.shape[0], 1, a.shape[1]), device=a.device, dtype=torch.float32)
    beta_output = torch.empty_like(g)
    reference_kernel[(a.shape[0], triton.cdiv(a.shape[1], 8))](
        g[:, 0],
        beta_output[:, 0],
        A_log,
        a,
        b,
        dt_bias,
        a.stride(0),
        b.stride(0),
        a.shape[1],
        1.0,
        20.0,
        8,
        num_warps=1,
    )
    return g, beta_output


def run_gating_case(torch, gating_fn) -> dict[str, object]:
    """Compare fused prefill gating with the explicit FP32 reference."""
    heads = 8
    # These BF16 logits deliberately have sigmoid values that are not BF16
    # representable; a projection-dtype round trip is therefore observable.
    b = torch.tensor(
        [-3.0, -1.25, -0.5, 0.25, 0.75, 1.375, 2.5, 3.25],
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(1, heads)
    a = torch.tensor(
        [-2.0, -1.0, -0.25, 0.5, 1.0, 1.75, 2.25, 3.0],
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(1, heads)
    A_log = torch.linspace(-2.0, -0.25, heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(-0.75, 0.75, heads, device="cuda", dtype=torch.float32)
    g, beta = gating_fn(A_log, a, b, dt_bias)
    import triton
    import triton.language as tl

    expected_g, expected_beta = reference_gdn_gating(torch, triton, tl, A_log, a, b, dt_bias)
    g_diff = scalar_diff(g, expected_g)
    beta_diff = scalar_diff(beta, expected_beta)
    g_tolerance = 1e-5
    return {
        "heads": heads,
        "beta_input_bf16": b.dtype == torch.bfloat16,
        "beta_non_bf16_representable": bool(
            torch.any(expected_beta != expected_beta.to(torch.bfloat16).to(torch.float32))
        ),
        "finite_outputs": bool(torch.isfinite(g).all() and torch.isfinite(beta).all()),
        "g_tolerance": g_tolerance,
        "g_within_tolerance": g_diff["max_abs"] <= g_tolerance,
        "g": g_diff,
        "beta": beta_diff,
    }


def expected_gate(variant: str, fast_ops: str) -> tuple[bool, bool, bool]:
    """Return (decode_gate, prefill_gate, validation_gate)."""
    if fast_ops != "0":
        return False, False, False
    if variant == "original":
        return False, False, True
    if variant == "decode-beta":
        return True, False, True
    if variant == "all-beta":
        return True, True, True
    raise ValueError(variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "decode-beta", "all-beta"), required=True)
    parser.add_argument("--fla-use-fast-ops", choices=("0", "1"), default="0")
    parser.add_argument(
        "--source-revision",
        default=None,
        help="source archive or checkout revision for metadata; never a validation gate",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="-",
        default=None,
        type=Path,
        help="write JSON to PATH; with no PATH, print the report only",
    )
    args = parser.parse_args()
    if args.source_revision is not None and not REVISION_RE.fullmatch(args.source_revision):
        parser.error("--source-revision must be exactly 40 hexadecimal characters")
    os.environ["FLA_USE_FAST_OPS"] = args.fla_use_fast_ops

    # Imports happen only after FLA_USE_FAST_OPS is set.
    import torch
    import triton

    fixture = load_fixture()
    from sglang.kernels.ops.attention.fla import fused_recurrent, fused_sigmoid_gating_recurrent
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    packed_path = Path(fused_recurrent.__file__).resolve()
    gating_path = Path(
        __import__(
            "sglang.kernels.ops.attention.fla.fused_gdn_gating", fromlist=["__file__"]
        ).__file__
    ).resolve()
    verify_path = Path(fused_sigmoid_gating_recurrent.__file__).resolve()
    source_info = source_variant(packed_path, gating_path)
    source_match = source_info["observed_variant"] == args.variant
    if not torch.cuda.is_available():
        raise RuntimeError("GDN beta regression requires CUDA")

    cases = []
    for isolate_beta in (True, False):
        fixture_name = "isolated-beta" if isolate_beta else "random"
        for seed, heads, value_heads in ((42, 4, 8), (11, 16, 48)):
            row = fixture.run_case(seed, heads, value_heads, isolate_beta=isolate_beta)
            row["fixture"] = fixture_name
            cases.append(row)

    isolated = [row for row in cases if row["isolate_beta"]]
    random_cases = [row for row in cases if not row["isolate_beta"]]
    decode_match = all(row["output_nonzero"] == 0 and row["state_nonzero"] == 0 for row in isolated)
    decode_mismatch = all(row["state_nonzero"] > 0 for row in isolated)
    random_match = all(
        row["output_nonzero"] == 0 and row["state_nonzero"] == 0 for row in random_cases
    )
    random_mismatch = all(row["state_nonzero"] > 0 for row in random_cases)
    decode_guards = all(
        row[key]
        for row in cases
        for key in (
            "finite_outputs_and_states",
            "verify_source_unchanged",
            "unrelated_packed_state_unchanged",
            "intermediate_guards_unchanged",
        )
    )
    prefill = run_gating_case(torch, fused_gdn_gating)
    prefill_match = (
        prefill["finite_outputs"]
        and prefill["g_within_tolerance"]
        and prefill["beta"]["nonzero"] == 0
    )
    prefill_valid = prefill["finite_outputs"] and prefill["g_within_tolerance"]
    prefill_mismatch = prefill_valid and prefill["beta"]["nonzero"] > 0
    guards = decode_guards and prefill_valid
    want_decode, want_prefill, validation_gate = expected_gate(args.variant, args.fla_use_fast_ops)
    behavior_pass = (
        (decode_match and random_match) if want_decode else (decode_mismatch and random_mismatch)
    ) and (prefill_match if want_prefill else prefill_mismatch)
    # Fast-ops runs are observational: retain their scalar results but do not
    # turn a different exponential implementation into a validation failure.
    passed = bool(guards and (source_match and behavior_pass if validation_gate else True))
    local_revision = optional_git_revision(packed_path.parent)
    report = {
        "schema_version": 2,
        "source_revision": args.source_revision or local_revision or "unavailable",
        "source_revision_source": (
            "explicit" if args.source_revision else "local-git" if local_revision else "unavailable"
        ),
        "variant": args.variant,
        "observed_variant": source_info["observed_variant"],
        "expectation": {
            "decode": "match" if want_decode else "mismatch",
            "prefill": "match" if want_prefill else "mismatch",
        },
        "status": "passed" if passed else "failed",
        "validation_gate": validation_gate,
        "source_variant_match": source_match,
        "guards": guards,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "fla_use_fast_ops": args.fla_use_fast_ops,
        "packed_source_sha256": sha256(packed_path),
        "verify_source_sha256": sha256(verify_path),
        "gating_source_sha256": sha256(gating_path),
        "imported_source_hashes": {
            "packed": sha256(packed_path),
            "verify": sha256(verify_path),
            "gating": sha256(gating_path),
        },
        "fixture_sha256": sha256(FIXTURE),
        "decode_match": decode_match,
        "decode_mismatch": decode_mismatch,
        "random_match": random_match,
        "random_mismatch": random_mismatch,
        "prefill_match": prefill_match,
        "prefill_mismatch": prefill_mismatch,
        "prefill_valid": prefill_valid,
        "prefill": prefill,
        "cases": cases,
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(payload, end="")
    if args.report is not None and str(args.report) != "-":
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
