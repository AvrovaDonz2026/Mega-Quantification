# DGX Spark robo-dog-ab93 environment readiness - 2026-10-13

## Status

The original host is back online and the exact frozen runtime is available on
`aitopatom-aacd`: `megaquant:gpqa256k-79b5dfe` resolves to
`sha256:ea5a6262a6ad57e0fd2d99102916a4194a9b7a2ead8fc5c4d69f7a348ac8ad25`.
The original BF16 model is present there, and the five frozen metadata hashes
match. All 18 source shards also match the frozen source record on
`robo-dog-ab93` (55,563,006,776 bytes, zero mismatches).

The shard count, byte total, and zero-mismatch result come from the serial
read-only record `artifacts/robo-env-20261013/weights/bf16-sha256-record.json`.
The original-image and metadata observations were made on `aitopatom-aacd`; the
rebuilt-image details are recorded in
`artifacts/robo-env-20261013/image/build-record.md`.

`robo-dog-ab93` has a separately rebuilt ARM64 image from source revision
`79b5dfe` and patch v6. Its digest is
`sha256:491c17774a3ad0371f3795a47e3d773802a90121cc7bccd6520bd600c40385fc`;
it is a new runtime and is not claimed byte-identical to the frozen image.

The latest read-only operator check found active RoCE on both hosts on
`192.0.2.0/24` and `192.0.3.0/24`, with point-to-point ping succeeding from
aitopatom to robo. This records host readiness at that check; it does not claim
that the earlier disconnected-interface snapshot was wrong or that the links
remain active after the check. TP2 inference has not yet run: the remaining
gates are verbs/NCCL collective checks, matching image and runtime identity on
both hosts, and a short distributed SGLang smoke test.

## Gates to resume

1. Select one image identity and verify it on both hosts. The frozen image is
   currently only attested on aitopatom; the rebuilt image is currently only
   attested on robo.
2. Run a matching verbs test and an `all_reduce_perf` binary compatible with
   the selected CUDA/NCCL runtime; record that NCCL uses RoCE rather than a
   socket fallback.
3. Confirm both hosts use the same verified image and weights, then pass a
   short two-rank SGLang text smoke test before adding multimodal or MTP work.

These are readiness gates, not completed checks. PR #43990's CPU test is a
separate code-level result and does not establish image readiness, network
availability, distributed initialization, or TP2 inference.
