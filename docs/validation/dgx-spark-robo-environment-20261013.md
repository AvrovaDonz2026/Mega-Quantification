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

The separate full serving run is prepared with the exact `f22670d` SGLang
checkout, immutable PTQ image ID, and fresh export manifests. Its first
W4A4-baseline launch reached container import and failed before model loading:
the image's `sglang-kernel 0.4.9` wheel is built against `torch==2.14.1`, while
the Spark base has `torch 2.13.0+cu130`; the missing
`CUDAErrorLogCapture` symbol is an ABI mismatch. The runner preserved the full
container ID and log, then removed only that owned container. The selected host
ports `31000/31001` belong to existing nginx/Python services and were not
touched; retries use `31100/31101`.

The dependency closure was rebuilt locally on `robo-dog-ab93`: the image keeps
`flashinfer-python/cubin/jit-cache 0.7.1`, restores the base's ABI-compatible
`sglang-kernel 0.4.6.post1`, and sets
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` while the base remains on Torch 2.13.
The derived image is
`megaquant-aux-ptq:11407cd-20261013-fi071-kernel046` with local ID
`sha256:7d9ba02b9968e61435174e541141e4f4dd6df03fcb9a5f99108bee67531294be`.
It passed a GPU import smoke with the exact checkout. The four sequential
baseline/MTP runs, vision requests, and the independent quantized-execution
probe remain pending; no serving quality claim is made yet. TP2 and broad
generation accuracy remain unverified.

The first compatible-image launch then exposed a metadata bug in the W4A4
export. Its serialized visual tensors were correctly packed (`proj`
`[1152,576]`, `qkv` `[3456,576]`), but `hf_quant_config.json` had no
`quantized_layers` map. SGLang consequently constructed BF16 visual matrices
and rejected the packed weights (`[1152,1152]` versus `[1152,576]`). The
exporter fix now derives auxiliary entries from safetensors headers and writes
110 visual plus 8 MTP NVFP4 entries while preserving `quant_algo=NVFP4`; BF16
restored tensors are excluded. A metadata-only repair of the remote W4A4
validation export reproduced the expected 110/8 coverage, and a new serving
job was started. Its final serving and MTP/vision request results remain
pending because the Teleport resource lease expired while the model was
loading.

The natural KV A/B recovery is complete. Under the same seed, 262144 context,
24576 output budget, `cached_tokens=0`, and 385 stream events, both arms ended
by the length budget without EOS or a thinking-end marker. The FP8-KV arm
entered an exact 4-token cycle at native output index 13135 and stayed in it
for 11441 tokens (46.55% of output). The BF16-KV arm did not meet the exact
cycle threshold (at least four cycles over 512 tokens), but also exhausted its
budget. The arms shared three initial tokens and first diverged at zero-based
index 3. This single non-deterministic pair supports KV-dtype sensitivity for
this case; it does not establish stopping recovery, general quality, or a
unique root cause. The private statistics record is
`natural-kv-ab-statistics-20261013.json` (SHA256
`b0404f297f6d758ed2bc44d258954002c782bb79c42da39fd9f93e706ccc88df`); source
text and token IDs were not downloaded.

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
