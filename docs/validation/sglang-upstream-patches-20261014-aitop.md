# Aitop follow-up validation — 2026-10-14

This is an in-progress follow-up to
[`sglang-upstream-patches-20261013.md`](sglang-upstream-patches-20261013.md).
It records the exact newer source heads and the checks attempted after moving
to a second DGX Spark. No new model-serving or GPQA result is established here.

| Gate | Latest established result |
|---|---|
| Direct CPU suites | 48 tests passed across both exact heads |
| Pytest under coverage | Logprob 5 passed; vision/MTP 43 passed, with 34 subtests |
| Changed-line coverage threshold | Passed: logprob 100% (2/2 executable changed lines), vision/MTP 87% (71/81); minimum 60% |
| Historical model identities | Both isolated metadata views passed |
| Public SM121 GPU regression | Dependency download failed; verified-cache recovery being prepared |
| Full-model baseline/MTP and auxiliary apply probe | Dependent continuation stopped before serving |

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
in the private Aitop result directory; that attempt produced no changed-line
coverage result.

A coverage-only retry was then run with `COVERAGE_CORE=pytrace`, plugin
autoload disabled, an explicit `pytest_cov` plugin, and fresh writable paths.
Both source identity probes and offline tool installation passed, but the
coverage commands still did not produce coverage data: the logprob collection
raised `ValueError: module functions cannot set METH_CLASS or METH_STATIC`
while importing the image's Torch, and the vision/MTP command terminated with
return code `-11` (segmentation fault) while rendering the pytest failure.
This is a fixed-image Python/Torch/coverage runtime failure; it is not evidence
of a source regression. The direct retry2 passes remain the valid CPU test
evidence; that coverage-only attempt produced no changed-line coverage result.

A fourth isolated run resolved the test/coverage import failure by initializing
only NumPy, Torch, YAML and PIL without a tracer, then starting Python coverage
through its public API before any SGLang module or test collection. It invoked
the real `pytest.main()` on the unchanged registered tests: logprob **5 passed**
and vision/MTP **43 passed** (**34 subtests passed**). Both source identity probes
passed, the measured production files were verified, and coverage XML/data were
produced. Only the subsequent diff-cover aggregation failed: its native chardet
extension could not map from the noexec temporary mount. A fifth isolated run
reused these immutable coverage files with SHA256 checks and changed only the
aggregation environment: executable temporary storage and a shadow Git root
containing byte-identical production files, required by diff-cover even with a
precomputed diff. Both formal `diff-cover --fail-under=60` commands passed;
the result container exited 0 with no OOM. Logprob covered **2/2 executable
changed lines (100%)**, and vision/MTP covered **71/81 (87%, as reported by
diff-cover)**. All five changed production files appeared in the reports, so
the passing percentage is not an empty path match. No test was repeated and
the threshold was not lowered.

Coverage report SHA256 hashes:

| Source | XML | Formal diff report |
|---|---|---|
| Logprob | `afa47c322477f906a27a8e1ea058c320c0a4b579c2fe0b399a4c8bec619a0b92` | `5986d922b5124d701acfc7a10a441055a46c2e2e3b3b7b5d46ee5ddb44a07084` |
| Vision/MTP | `6b3d448f15df731925726ee47d890e112ba8ca1aecdab91950027fcad686f356` | `86d023c4398784bd74730a6560caaabd8a663c111cf963dfd6607e9515b28949` |

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

Both checkpoints are tested through separate metadata views that retain the
original packed weights and record the original metadata hashes plus exact
metadata-fix revision. The original scored checkpoints remain intact. The
model identity gate passed after hashing all model files and verifying both
views; its manifest SHA256 is
`27a9e5c83ec796bd589f5078a82a7f9ad69b83e0dcd838144fa64c840fd5fb1c`.
No serving request, vision/MTP execution probe, or new GPQA run has completed.

## Current status and next gate

Initial Aitop inspection found the GB10 idle, with no SGLang inference process.
The first GPU environment attempt stopped at dependency resolution because
the CUDA 13 index has no `flashinfer-cubin==0.7.1` package entry. The official
v0.7.1 release does publish that cubin wheel and the AArch64 CUDA-13 generic
and SM121 JIT wheels, so a second detached build attempted those exact
release assets over host networking. No GPU image, regression result, or
serving result exists yet. The original host
supervisor exceeded its 3600-second download timeout while the owned download
container kept running. A new detached continuation observes that same full
container ID and validates its labels, source archive and all 10,577 source
files. It was configured to wait for a clean download exit, verify the four
wheel filenames, versions and hashes, then build offline and run the public
GPU test. At 07:20:50 UTC the downloader exited 1 with `incomplete-download`:
only 547.4 MB of the 1551.4-MB cubin wheel had arrived after six attempts,
with repeated GitHub connection/read timeouts. The continuation correctly
failed without building or running the GPU test. The model identity
gate has passed for both historical metadata views: W4A4 and W4A8 each have
immutable file hashes, the metadata-fix revision `a0c6163d15601529a424384124471f44ea579310`,
110 visual entries, and 8 MTP entries. This validates inventory/provenance only;
it does not establish runtime serving.

The complete serving continuation was also detached. It waits for
the GPU resume record's final successful exit and strict one-test/no-skip proof,
then reuses that exact image ID for W4A4 baseline/MTP and W4A8 baseline/MTP.
Only after all four ordinary modes pass does it start the separate four-mode
auxiliary apply probe. Each result is bound to this run's metadata, full
container ID and image/source hashes; MTP reports also bind their current
baseline hash. Serving uses host networking with dedicated loopback ports
31200/31201 so initial FlashInfer JIT asset access follows the working network
route. It stopped at the failed GPU dependency gate before any model launched;
all evidence was retained. Recovery uses a separate wheel cache and requires
all four assets to match their official release SHA256 digests before an
offline build. Neither a launched supervisor nor a downloaded artifact is a
serving pass.

Remaining gates are: build and verify the pinned FlashInfer 0.7.1 GPU
derivative while retaining the image's Torch/kernel ABI; run the public SM121
NVFP4 regression; run the fixed baseline/MTP text and image checks and the
separate quantized-layer execution probe. GPQA is a later, separate run and
requires a complete journal plus the repository's formal checker.
