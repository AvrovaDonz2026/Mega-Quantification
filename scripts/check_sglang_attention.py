#!/usr/bin/env python3
"""Require identical SM121 FP8 attention decode and three-token verification."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _summarize(left, right) -> dict:
    import torch

    left, right = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Attention output shape or dtype differs")
    byte_differences = int(
        torch.count_nonzero(left.view(torch.uint8) != right.view(torch.uint8)).item()
    )
    difference = (left.float() - right.float()).abs()
    return {
        "identical_bytes": byte_differences == 0,
        "byte_difference_count": byte_differences,
        "numeric_difference_count": int(torch.count_nonzero(left != right).item()),
        "max_abs": float(difference.max().item()),
        "numel": left.numel(),
        "dtype": str(left.dtype),
    }


def _run_case(wrapper_class, workspace, lengths, seed) -> dict:
    import torch
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferAttnBackend,
        PrefillMetadata,
    )

    device = workspace.device
    generator = torch.Generator(device=device).manual_seed(seed)
    batch, q_heads, kv_heads, dim = len(lengths), 24, 4, 256

    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)

    first_queries = random((batch, q_heads, dim))
    baseline_q = first_queries.repeat_interleave(3, dim=0)
    verify_q = torch.cat(
        [
            torch.cat((first_queries[request : request + 1], random((2, q_heads, dim))))
            for request in range(batch)
        ]
    )
    keys, values, verify_keys, verify_values = [], [], [], []
    masks, baseline_masks = [], []
    for length in lengths:
        k, v = random((length, kv_heads, dim)), random((length, kv_heads, dim))
        keys.append(k)
        values.append(v)
        # Saturating finite FP8 values must remain invisible to the first query.
        future_k = torch.full((2, kv_heads, dim), 448.0, dtype=k.dtype, device=device)
        future_v = torch.full((2, kv_heads, dim), 448.0, dtype=v.dtype, device=device)
        future_k[1].neg_()
        future_v[1].neg_()
        verify_keys.append(torch.cat((k, future_k)))
        verify_values.append(torch.cat((v, future_v)))
        masks.append(
            (
                torch.arange(length + 2, device=device)[None, :]
                < length + torch.arange(3, device=device)[:, None]
            ).reshape(-1)
        )
        baseline_masks.append(torch.ones(3 * length, dtype=torch.bool, device=device))

    def cache(k, v):
        return (
            torch.cat(k).to(torch.float8_e4m3fn).unsqueeze(1).contiguous(),
            torch.cat(v).to(torch.float8_e4m3fn).unsqueeze(1).contiguous(),
        )

    baseline_cache, verify_cache = cache(keys, values), cache(verify_keys, verify_values)
    qo_indptr = torch.arange(0, 3 * (batch + 1), 3, dtype=torch.int32, device=device)

    def attention(query, kv, sequence_lengths, mask):
        lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)
        kv_indptr = torch.cat(
            (torch.zeros(1, dtype=torch.int32, device=device), lengths_tensor.cumsum(0))
        ).to(torch.int32)
        wrapper = wrapper_class(workspace, "NHD", backend="fa2")
        wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            torch.arange(sum(sequence_lengths), dtype=torch.int32, device=device),
            torch.ones(batch, dtype=torch.int32, device=device),
            q_heads,
            kv_heads,
            dim,
            1,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            custom_mask=mask,
            fixed_split_size=None,
            non_blocking=False,
        )
        output = wrapper.forward(
            query, kv, causal=True, sm_scale=dim**-0.5, k_scale=1.0, v_scale=1.0
        )
        return wrapper, output

    baseline_wrapper, baseline = attention(
        baseline_q, baseline_cache, lengths, torch.cat(baseline_masks)
    )
    _, verified = attention(
        verify_q, verify_cache, [length + 2 for length in lengths], torch.cat(masks)
    )
    first_baseline = baseline.reshape(batch, 3, q_heads, dim)[:, 0]
    first_verified = verified.reshape(batch, 3, q_heads, dim)[:, 0]
    metrics = _summarize(first_baseline, first_verified)
    finite = bool(torch.isfinite(baseline).all().item() and torch.isfinite(verified).all().item())
    report = {
        "seed": seed,
        "batch_size": batch,
        "decode_kv_lengths": list(lengths),
        "verify_kv_lengths": [length + 2 for length in lengths],
        "cloned_queries_per_request": 3,
        "first_query_ignores_two_poisoned_future_tokens": True,
        "all_outputs_finite": finite,
        "metrics": metrics,
        "passed": finite and metrics["identical_bytes"],
    }
    locations, offset = [], 0
    for length in lengths:
        locations.append(offset + length - 1)
        offset += length
    locations = torch.tensor(locations, dtype=torch.int64, device=device)
    expected_k = torch.stack([k[-1] for k in keys])
    expected_v = torch.stack([v[-1] for v in values])

    class SyntheticCache:
        # Exercise the actual backend route with a small byte-addressed KV fixture.
        def __init__(self):
            self.cache = tuple(value.clone() for value in baseline_cache)
            for value in self.cache:
                value.view(torch.uint8)[locations] = 0
            self.write_count = 0

        def get_kv_buffer(self, layer_id):
            if layer_id != 3:
                raise AssertionError("Attention validation accessed an unrelated layer")
            return self.cache

        def set_kv_buffer(self, layer, write_location, k, v, k_scale, v_scale):
            if not torch.equal(write_location.loc, locations):
                raise AssertionError("Backend changed the real current-token cache locations")
            if (
                k.shape[0] != batch
                or not torch.equal(k, expected_k)
                or not torch.equal(v, expected_v)
            ):
                raise AssertionError("Backend padded or changed K/V instead of only Q")
            self.write_count += 1
            for destination, source, scale in zip(
                self.cache, (k, v), (k_scale, v_scale), strict=True
            ):
                source = source.clone()
                if scale is not None:
                    source.div_(scale)
                destination.view(torch.uint8)[locations] = (
                    source.to(torch.float8_e4m3fn).unsqueeze(1).view(torch.uint8)
                )

    pool = SyntheticCache()
    backend = object.__new__(FlashInferAttnBackend)
    backend._megaquant_sm121_fp8_prefill_decode = True
    backend.forward_metadata = PrefillMetadata([baseline_wrapper], False, False)
    backend.num_wrappers = 1
    backend.prefill_uses_dequant_workspace = False
    backend.decode_uses_dequant_workspace = False
    backend.token_to_kv_pool = pool
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: False)
    layer = SimpleNamespace(
        layer_id=3,
        is_cross_attention=False,
        attn_type=None,
        logit_cap=0.0,
        scaling=dim**-0.5,
        tp_q_head_num=q_heads,
        tp_k_head_num=kv_heads,
        head_dim=dim,
        sliding_window_size=-1,
        k_scale=None,
        v_scale=None,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        spec_info=None,
        out_cache_loc=locations,
    )
    output = backend.forward_decode(
        first_queries, expected_k, expected_v, layer, forward_batch, save_kv_cache=True
    ).reshape(batch, q_heads, dim)
    backend_metrics = _summarize(output, first_verified)
    backend_finite = bool(torch.isfinite(output).all().item())
    report.update(
        {
            "backend_forward_decode_exercised": True,
            "backend_kv_write_count": pool.write_count,
            "backend_real_kv_rows_written": batch,
            "backend_all_outputs_finite": backend_finite,
            "backend_vs_verify": backend_metrics,
            "passed": report["passed"]
            and pool.write_count == 1
            and backend_finite
            and backend_metrics["identical_bytes"],
        }
    )
    return report


def check_attention_decode_verify() -> dict:
    import flashinfer
    import torch
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    if flashinfer.__version__ != "0.6.18":
        raise RuntimeError("Attention validation requires FlashInfer 0.6.18")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Attention validation requires a CUDA SM121 GPU")
    workspace_mib = 384
    with torch.inference_mode():
        workspace = torch.empty(workspace_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cases = [
            _run_case(BatchPrefillWithPagedKVCacheWrapper, workspace, lengths, seed)
            for lengths, seed in (([1], 13), ([2], 29), ([53], 13), ([2, 127], 29))
        ]
    report = {
        "schema": "flashinfer_sm121_fp8_three_query_decode_preflight_v1",
        "passed": all(case["passed"] for case in cases),
        "case_count": len(cases),
        "device_capability": [12, 1],
        "flashinfer_version": flashinfer.__version__,
        "torch_version": torch.__version__,
        "workspace_mib": workspace_mib,
        "actual_backend_forward_decode_exercised": True,
        "kv_dtype": "torch.float8_e4m3fn",
        "query_dtype": "torch.bfloat16",
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "page_size": 1,
        "cases": cases,
        "limitations": [
            "Synthetic kernel regression; no checkpoint weights or raw activations are used.",
            "Four cases do not establish invariance for all context lengths or model prompts.",
        ],
    }
    if not report["passed"]:
        raise RuntimeError(f"Attention decode/verify mismatch: {report}")
    return report


if __name__ == "__main__":
    print(json.dumps(check_attention_decode_verify(), indent=2))
