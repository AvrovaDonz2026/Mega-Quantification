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

The retry with writable caches passed the exact source identity checks for both
heads and ran all four direct registered-test commands successfully: logprob
5 tests, and vision/MTP 13, 18, and 12 tests. Its optional `pytest-cov`
collection still failed inside the fixed image before test execution: importing
Torch under coverage raised `SystemError: bad call flags`; coverage also could
not map its native tracer from the temporary target directory. Those failures
are isolated to the coverage invocation. The direct test evidence is retained
in the private Aitop result directory; changed-line coverage remains unclaimed.

A coverage-only retry was then run with `COVERAGE_CORE=pytrace`, plugin
autoload disabled, an explicit `pytest_cov` plugin, and fresh writable paths.
Both source identity probes and offline tool installation passed, but the
coverage commands still did not produce coverage data: the logprob collection
raised `ValueError: module functions cannot set METH_CLASS or METH_STATIC`
while importing the image's Torch, and the vision/MTP command terminated with
return code `-11` (segmentation fault) while rendering the pytest failure.
This is a fixed-image Python/Torch/coverage runtime failure; it is not evidence
of a source regression. The direct retry2 passes remain the valid CPU test
evidence, and changed-line coverage is still unclaimed.

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
The first GPU environment attempt stopped at dependency resolution because
the CUDA 13 index has no `flashinfer-cubin==0.7.1` package entry. The official
v0.7.1 release does publish that cubin wheel and the AArch64 CUDA-13 generic
and SM121 JIT wheels, so a second detached build now downloads those exact
release assets over host networking and records their hashes. The Python wheel
download step is active; the 1.5-GB cubin asset is still downloading, so no
GPU image, regression result, or serving result exists yet. The model identity
gate has passed for both historical metadata views: W4A4 and W4A8 each have
immutable file hashes, the metadata-fix revision `a0c6163d15601529a424384124471f44ea579310`,
110 visual entries, and 8 MTP entries. This validates inventory/provenance only;
it does not establish runtime serving.

Remaining gates are: build and verify the pinned FlashInfer 0.7.1 GPU
derivative while retaining the image's Torch/kernel ABI; run the public SM121
NVFP4 regression; run the fixed baseline/MTP text and image checks and the
separate quantized-layer execution probe; then assess whether a compatible
coverage environment can be supplied. GPQA is a later, separate run and
requires a complete journal plus the repository's formal checker.
