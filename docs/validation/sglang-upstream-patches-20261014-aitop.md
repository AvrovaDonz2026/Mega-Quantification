# Aitop follow-up validation — 2026-10-14

This is an in-progress follow-up to
[`sglang-upstream-patches-20261013.md`](sglang-upstream-patches-20261013.md).
It records the exact newer source heads and the checks attempted after moving
to a second DGX Spark. No new model-serving or GPQA result is established here.

## Pinned inputs

| Input | Value |
|---|---|
| Logprob source | `875522a1b2239f906b8da2067e818feacc56fbee` |
| Vision/MTP source | `4a667614d810f7b5570be23eb75956c603a85c3a` |
| Shared SGLang base | `5bcaf0f8ede9fccbe63ffa1e29f51239e384460a` |
| CPU validation image | `sha256:20f41d6ea8897ac17a3a2c43edbcc1ff051da71607297a0d107d24f9715bacd0` |
| Python / PyTorch in image | 3.12 / 2.13.0+cu130 |

Both source archives were transferred with their pinned SHA-256 hashes and
staged successfully. The detached CPU container used `runc`, no network, and
no GPU device request. It installed the ARM64 coverage tools from a local
wheelhouse. The first source identity probe failed before any test suite ran:
PyTorch attempted to create `/root/.cache/sglang/inductor` on the read-only
container root. This is a cache-path setup failure, not a test failure or a
source mismatch. No CPU regression or coverage result is claimed. A retry
needs `SGLANG_CACHE_DIR` and the Inductor/Triton caches redirected to writable
temporary storage, then a fresh uniquely named container and result directory.

## Historical model inventory

The available full checkpoints are the scored 2026-10-01 exports, quantized
from source revision `ac53833dc5af3f03a10087449ca80e9484deba48` with ModelOpt
`max`, 256 image-enabled calibration samples, sequence length 1024, batch size
1, and seed 42. They are not the fresh 2026-10-13 Robo exports. Their existing
GPQA results remain attached to their original run and runtime; no score is
transferred to the newer SGLang source.

The original W4A4 `hf_quant_config.json` has `quant_algo=NVFP4` but no
`quantized_layers` map. Its safetensors headers contain 110 packed visual
projection weights and 8 packed MTP projection weights, so the missing map is
metadata, not absent auxiliary tensors. The original W4A8 map contains 519
layers, including 110 visual and 8 MTP entries. Header inspection found the
expected packed U8 weights for visual projections; W4A8 MTP has four U8 and
four FP8 projection weights. These observations do not establish that the
newer source serves either checkpoint.

The W4A4 checkpoint must only be tested through a separate metadata view. If
created, that view must retain the original files and clearly record the
original metadata hashes and the exact metadata-fix revision. The scored
checkpoint itself must not be edited. No complete view audit, serving request,
vision/MTP execution probe, or GPQA run has completed yet.

## Current status and next gate

Initial Aitop inspection found the GB10 idle, with no SGLang inference process.
The GPU regression container and full-model serving job were prepared locally
but were not started. A subsequent `tsh ssh` attempt entered Teleport's
single-use MFA callback path; the local callback listener could not bind under
the current execution environment. Further remote commands require a working
Teleport login/session so the prepared jobs can be staged and observed.

Remaining gates are: rerun the two exact-head CPU suites and changed-line
coverage with writable caches; build and verify the pinned FlashInfer 0.7.1
GPU derivative while retaining the image's Torch/kernel ABI; run the public
SM121 NVFP4 regression; validate the immutable model inventory and isolated
metadata views; then run the fixed baseline/MTP text and image checks and the
separate quantized-layer execution probe. GPQA is a later, separate run and
requires a complete journal plus the repository's formal checker.
