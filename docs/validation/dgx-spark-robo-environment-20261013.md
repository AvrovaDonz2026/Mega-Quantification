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

## Fresh Robo PTQ run

The dedicated PTQ run on `robo-dog-ab93` used the public MegaQuant revision
`11407cdc7bdd79e28e57c82a43f0f519260f0ff2` and a dedicated PTQ image derived
from base image
`sha256:20f41d6ea8897ac17a3a2c43edbcc1ff051da71607297a0d107d24f9715bacd0`.
The resulting image ID was
`sha256:ab4831056abce468f1e96917c7968c5dd4fce38b0072533fc3a6a0e8d978e876`.
The GB10/SM121 preflight passed with CUDA 13.0, PyTorch `2.13.0+cu130`,
Transformers `5.12.1`, and ModelOpt `0.47.0`.

Calibration was freshly prepared from 128 COCO images and 128 WikiText rows
(seed 42, maximum image side 448, text limit 600 characters). The manifest is at
`/home/zhiman/megaquant-robo-env-20261013/auxiliary-e2e/data-parent/calibration/manifest.json`
and has SHA256
`d421c63ff95f844a03d95a1747d3db24363f5e7f8e2fd674e9af3042ee2d96ed`. Its
256-row `calibration.jsonl` has SHA256
`2e36b6bf0b775aba5601a5c5ce7e702654d44e8f8eb674798d09b55fafbf49cd`.
The calibration is a new upstream-integration fixture and differs from the
historical frozen calibration.

Both full quantization stages and their strict export audits returned code 0:

| Format | Quantized output | Quantization log | Audit log | Calibration coverage |
|---|---|---|---|---|
| W4A4 | `/home/zhiman/megaquant-robo-env-20261013/auxiliary-e2e/outputs/w4a4` | `validation/w4a4-quantize.log` | `validation/w4a4-export-audit.log` | 165 vision quantizers × 128 calls; 8 MTP quantizers × 256 calls |
| W4A8 | `/home/zhiman/megaquant-robo-env-20261013/auxiliary-e2e/outputs/w4a8` | `validation/w4a8-quantize.log` | `validation/w4a8-export-audit.log` | 165 vision quantizers × 128 calls; 8 MTP quantizers × 256 calls |

The stage return codes and timestamps are recorded in
`validation/quant-job-status.json`; the preflight result is in
`validation/spark-preflight.json`, and the inherited package baseline is in
`validation/base-pip-check.log` under the same remote run root.

The audit verifies export structure, source/provenance coverage, calibration
execution evidence, packed shapes, and finite scale payloads. It does not
measure generation accuracy, multimodal quality, MTP acceptance quality, or
serving performance. The base container has 14 pre-existing `pip check`
conflicts; the dependency-delta check found no new conflicts introduced by the
PTQ image.

The remaining gate is a separate full serving run from each fresh output,
covering text, image/OCR, and embedded MTP requests with the final serving
runtime. The benchmark client also needs an explicit review of its
`python_code` output budget and length/truncation gate before any score is
reported. TP2 and broad generation accuracy remain unverified. A separate
natural KV A/B investigation is still incomplete: the FP8-KV arm is the only
arm observed so far and the BF16-KV recovery arm is still running, so it does
not support a conclusion about repetition or stopping behavior.

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
