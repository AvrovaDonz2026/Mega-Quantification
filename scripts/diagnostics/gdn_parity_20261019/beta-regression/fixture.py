"""Packed GDN decode must retain the FP32 beta used by target verification.

One-hot Q/K isolate the beta precision boundary from normalization and
reduction-order differences. All comparisons are exact, including the full
nonzero FP32 state after every token of the verify chain.
"""

import unittest

import torch
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as target_verify,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def run_case(seed, heads, value_heads, *, isolate_beta=True):
    """Return scalar differences; no tensors or token data leave this function."""
    if not torch.cuda.is_available():
        raise RuntimeError("GDN regression requires CUDA")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    steps, key_dim, value_dim = 32, 128, 128

    def random(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    if isolate_beta:
        q = torch.zeros(1, steps, heads, key_dim, device="cuda", dtype=torch.bfloat16)
        for step in range(steps):
            for head in range(heads):
                q[0, step, head, (3 * step + 7 * head) % key_dim] = 1
        k = q.clone()
    else:
        q = random(1, steps, heads, key_dim)
        k = random(1, steps, heads, key_dim)
    v = random(1, steps, value_heads, value_dim)
    # Noncontiguous a/b reproduce the split projection's token pitch.
    b, a = random(steps, 2 * value_heads).split(value_heads, dim=-1)
    decay = random(value_heads, dtype=torch.float32)
    bias = random(value_heads, dtype=torch.float32)
    initial = (
        torch.rand(
            3,
            2,
            value_heads,
            value_dim,
            key_dim,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
        + 0.05
    )
    packed_envelope, verify_envelope = initial.clone(), initial.clone()
    packed_state, verify_state = packed_envelope[:, 0], verify_envelope[:, 0]
    indices = torch.ones(1, device="cuda", dtype=torch.int32)
    assert packed_state.stride(0) == 2 * value_heads * value_dim * key_dim
    assert not packed_state.is_contiguous()
    assert b.dtype == torch.bfloat16 and packed_state.dtype == torch.float32
    assert b.stride(0) == 2 * value_heads and a.stride(0) == 2 * value_heads

    scale = 1.0 if isolate_beta else key_dim**-0.5
    outputs, states = [], []
    for step in range(steps):
        mixed = torch.cat(
            (q[0, step].flatten(), k[0, step].flatten(), v[0, step].flatten())
        ).unsqueeze(0)
        output = torch.empty(1, 1, value_heads, value_dim, device="cuda", dtype=q.dtype)
        packed_decode(
            mixed,
            a[step : step + 1],
            b[step : step + 1],
            decay,
            bias,
            scale,
            packed_state,
            output,
            indices,
            use_qk_l2norm_in_kernel=not isolate_beta,
        )
        outputs.append(output)
        states.append(packed_state[1].clone())

    # Keep the intermediate cache contiguous and independently indexed. The
    # initial state above is the only intentionally envelope-strided buffer.
    cache_envelope = random(3, steps, value_heads, value_dim, key_dim, dtype=torch.float32)
    original_cache = cache_envelope.clone()
    intermediate = cache_envelope
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
        scale=scale,
        initial_state_source=verify_state,
        initial_state_indices=indices,
        cu_seqlens=torch.tensor([0, steps], device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=not isolate_beta,
        disable_state_update=True,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=torch.ones_like(indices),
        cache_steps=steps,
        retrieve_parent_token=(torch.arange(steps, device="cuda", dtype=torch.int32) - 1).unsqueeze(
            0
        ),
    )
    packed_output, packed_states = torch.cat(outputs, dim=1), torch.stack(states)
    differences = {
        "output": (packed_output.float() - reference.float()).abs(),
        "state": (packed_states - intermediate[1]).abs(),
    }
    report = {
        "seed": seed,
        "heads": heads,
        "value_heads": value_heads,
        "steps": steps,
        "beta_input_bf16": b.dtype == torch.bfloat16,
        "state_fp32": packed_state.dtype == torch.float32,
        "state_slot_stride": packed_state.stride(0),
        "normalize_qk": not isolate_beta,
        "isolate_beta": isolate_beta,
        "finite_outputs_and_states": all(
            bool(torch.isfinite(tensor).all())
            for tensor in (packed_output, reference, packed_states, intermediate[1])
        ),
        "verify_source_unchanged": torch.equal(verify_envelope, initial),
        "unrelated_packed_state_unchanged": (
            torch.equal(packed_envelope[:, 1], initial[:, 1])
            and torch.equal(packed_envelope[0], initial[0])
            and torch.equal(packed_envelope[2], initial[2])
        ),
        "intermediate_guards_unchanged": (
            torch.equal(cache_envelope[0], original_cache[0])
            and torch.equal(cache_envelope[2], original_cache[2])
        ),
    }
    for name, difference in differences.items():
        report[name + "_nonzero"] = int(torch.count_nonzero(difference).item())
        report[name + "_max_abs"] = float(difference.max().item())
        differing_steps = torch.any(difference.reshape(steps, -1) != 0, dim=1)
        nonzero_steps = torch.nonzero(differing_steps, as_tuple=False).flatten()
        report[name + "_first_differing_step"] = (
            int(nonzero_steps[0].item()) if nonzero_steps.numel() else -1
        )
    return report


class TestGdnPackedBetaFp32(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise RuntimeError("Registered GDN GPU regression requires CUDA")

    def test_bf16_beta_fp32_strided_chain(self):
        for seed, heads, value_heads in ((42, 4, 8), (11, 16, 48)):
            with self.subTest(seed=seed, heads=heads, value_heads=value_heads):
                report = run_case(seed, heads, value_heads)
                self.assertTrue(report["finite_outputs_and_states"], report)
                self.assertTrue(report["verify_source_unchanged"], report)
                self.assertTrue(report["unrelated_packed_state_unchanged"], report)
                self.assertTrue(report["intermediate_guards_unchanged"], report)
                self.assertEqual(report["output_nonzero"], 0, report)
                self.assertEqual(report["state_nonzero"], 0, report)


if __name__ == "__main__":
    unittest.main()
