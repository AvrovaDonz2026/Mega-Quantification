#!/usr/bin/env python3
"""Require identical GDN packed decode and speculative verify on a CUDA GPU."""

from __future__ import annotations

import json


def check_gdn_decode_verify() -> list[dict]:
    import torch
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as target_verify,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GDN parity validation requires CUDA")
    reports = []
    # Exercise grouped heads, multiple value tiles, and an envelope-strided
    # state pool. A verify chain must reproduce each sequential decode state.
    for seed, heads, value_heads in ((42, 4, 8), (11, 16, 48)):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        steps, key_dim, value_dim = 32, 128, 128

        def random(*shape, dtype=torch.bfloat16, generator=generator):
            return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

        q = random(1, steps, heads, key_dim)
        k = random(1, steps, heads, key_dim)
        v = random(1, steps, value_heads, value_dim)
        a, b = random(steps, value_heads), random(steps, value_heads)
        decay, bias = (
            random(value_heads, dtype=torch.float32),
            random(value_heads, dtype=torch.float32),
        )
        # Select one layer from an envelope that contains two layers per slot.
        initial = random(2, 2, value_heads, value_dim, key_dim, dtype=torch.float32) * 0.1
        packed_envelope, verify_envelope = initial.clone(), initial.clone()
        packed_state, verify_state = packed_envelope[:, 0], verify_envelope[:, 0]
        indices = torch.ones(1, device="cuda", dtype=torch.int32)
        outputs, states = [], []
        for step in range(steps):
            mixed = torch.cat((q[0, step].flatten(), k[0, step].flatten(), v[0, step].flatten()))
            output = torch.empty(1, 1, value_heads, value_dim, device="cuda", dtype=q.dtype)
            packed_decode(
                mixed.unsqueeze(0),
                a[step : step + 1],
                b[step : step + 1],
                decay,
                bias,
                key_dim**-0.5,
                packed_state,
                output,
                indices,
                True,
            )
            outputs.append(output)
            states.append(packed_state[1].clone())
        intermediate = torch.empty(
            1, steps, value_heads, value_dim, key_dim, device="cuda", dtype=torch.float32
        )
        reference = target_verify(
            A_log=decay,
            a=a,
            dt_bias=bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=verify_state,
            initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate,
            intermediate_state_indices=torch.zeros_like(indices),
            cache_steps=steps,
            retrieve_parent_token=(
                torch.arange(steps, device="cuda", dtype=torch.int32) - 1
            ).unsqueeze(0),
        )
        packed_output, packed_states = torch.cat(outputs, dim=1), torch.stack(states)
        differences = {
            "output": (packed_output.float() - reference.float()).abs(),
            "state": (packed_states - intermediate[0]).abs(),
        }
        report = {"seed": seed, "heads": heads, "value_heads": value_heads, "steps": steps}
        for name, difference in differences.items():
            report[name + "_nonzero"] = int(torch.count_nonzero(difference).item())
            report[name + "_max_abs"] = difference.max().item()
        if any(report[name + "_nonzero"] for name in differences):
            raise RuntimeError(f"GDN decode/verify mismatch: {report}")
        if not torch.equal(verify_envelope, initial) or not torch.equal(
            packed_envelope[:, 1], initial[:, 1]
        ):
            raise RuntimeError("GDN validation changed an unrelated or read-only state")
        reports.append(report)
    return reports


if __name__ == "__main__":
    print(json.dumps(check_gdn_decode_verify(), indent=2))
