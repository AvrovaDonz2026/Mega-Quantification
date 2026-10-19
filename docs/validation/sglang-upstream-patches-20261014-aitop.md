# Aitop follow-up validation — 2026-10-14

This is an in-progress follow-up to
[`sglang-upstream-patches-20261013.md`](sglang-upstream-patches-20261013.md).
It records the exact newer source heads and the checks attempted after moving
to a second DGX Spark. The latest v6 matrix used the beta-only GDN precision
patch with the retained v4 image and completed all four serving modes and all
four independent execution probes. The fixed-fixture MTP code parity gate still
fails; no new GPQA result is established here.

| Gate | Latest established result |
|---|---|
| Direct CPU suites | 48 tests passed across both exact heads |
| Pytest under coverage | Logprob 5 passed; vision/MTP 43 passed, with 34 subtests |
| Changed-line coverage threshold | Passed: logprob 100% (2/2 executable changed lines), vision/MTP 87% (71/81); minimum 60% |
| Historical model identities | Both isolated metadata views passed |
| Public SM121 GPU regression | v4 passed: exactly 1 test, no skips, container exit 0 |
| v6 W4A4/W4A8 baseline | Both passed: 5 fixtures, each with 2 warmup + 5 measured trials |
| v6 W4A4/W4A8 MTP | Functional, stop, AST and vision checks passed; code baseline parity failed 7/7 trials per mode |
| v6 auxiliary apply probes | All four passed; 110 visual runtime layers per mode and 5 MTP runtime layers per MTP mode |
| v6 cleanup | All 8 serving/probe cleanup gates passed (`removed=true`, return code 0) |
| New GPQA / release gate | Not established: this is a 32K fixed-fixture experiment, and source identity is not a verifiable commit |

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
At completion of this inventory gate, no serving request, vision/MTP execution
probe, or new GPQA run had completed. Later v4 serving results are recorded below.

## Earlier download attempts and continuation plan

Initial Aitop inspection found the GB10 idle, with no SGLang inference process.
The first GPU environment attempt stopped at dependency resolution because
the CUDA 13 index has no `flashinfer-cubin==0.7.1` package entry. The official
v0.7.1 release does publish that cubin wheel and the AArch64 CUDA-13 generic
and SM121 JIT wheels, so a second detached build attempted those exact
release assets over host networking. At that stage, no GPU image, regression
result, or serving result existed. The original host
supervisor exceeded its 3600-second download timeout while the owned download
container kept running. A new detached continuation observed that same full
container ID and validated its labels, source archive and all 10,577 source
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

The complete serving continuation was also detached. It was configured to wait for
the GPU resume record's final successful exit and strict one-test/no-skip proof,
then reuse that exact image ID for W4A4 baseline/MTP and W4A8 baseline/MTP.
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

At the end of the failed download attempt, the remaining gates were: build and
verify the pinned FlashInfer 0.7.1 GPU
derivative while retaining the image's Torch/kernel ABI; run the public SM121
NVFP4 regression; run the fixed baseline/MTP text and image checks and the
separate quantized-layer execution probe. GPQA is a later, separate run and
requires a complete journal plus the repository's formal checker.

## Current v4 results — 2026-10-14

A separate verified-cache run recovered the GPU environment without changing
the pinned SGLang source or restarting the failed download attempts. All four
FlashInfer wheels matched the SHA256 digests in the official
[v0.7.1 release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.1)
(release ID `408296015`), both locally and after transfer. The derivative built
with `--pull=false --network=none`, installing the wheels with
`--no-index --no-deps`; all 73 base image layers, Torch `2.13.0+cu130`, and
`sglang-kernel==0.4.6.post1` were retained. Source verification compared all
10,577 archived files. The wrapper also verified source imports and SM121
capability before running the unchanged public test:

```bash
python test/registered/kernels/ops/quantization/test_modelopt_nvfp4_vision.py -f
```

The test reported exactly **1 test, OK, no skips**, and the retained container
exited 0 without OOM. The GPU gate completed at **10:41:57 UTC**.

| v4 identity | Value |
|---|---|
| GPU job | `a00f1a83b1624d61b243b32907465cbc` |
| Source archive SHA256 | `ac27ebc4ada7c0e895ad3366501bd8213ca7d0c92abed98f77057ac9ab24ef9e` |
| Derivative image ID | `sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf` |
| Dockerfile SHA256 | `84cf9bef0779a173b99105fa648c0df04ee791fe16c38bff4c9706fdd41a25ee` |
| Official release manifest SHA256 | `3236d374e7c567dadcf9126d93438fddc996233ca0f36254220f8fc97b2385ea` |
| GPU record SHA256 | `c9fabf8e39ae4c65da81d64266b00e181697b355dfd6b0e76a80110b751b2055` |
| Public GPU log SHA256 | `0a7b74d649869b926b41edad5b8104315f9451e5c41f34568953ec9ca6f064d6` |

| Verified official wheel | SHA256 |
|---|---|
| `flashinfer_cubin-0.7.1-py3-none-any.whl` | `502eaa5f71e768950ada429618481a886388fcb34a5da6a2258979713cbaa327` |
| `flashinfer_jit_cache-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl` | `759f35213fca2aca68d9afecf324888fdf95f20628e8a944dde031bcbbed25de` |
| `flashinfer_jit_cache_sm121a-0.7.1+cu130-cp39-abi3-manylinux_2_28_aarch64.whl` | `712231a38b301cfdb507045f125ce07906bbb6b97ec05f87a3a2d353fea0aaa5` |
| `flashinfer_python-0.7.1-py3-none-any.whl` | `69302ed6e91314a85a4392324becc194594a782ba181646d25d5c305e1b7e8c4` |

The serving continuation verified the completed GPU record and reused its exact
derivative image with the historical W4A4 metadata view. It used TP1, one running
request, context length 32,768, FP8 E4M3 KV cache, FlashInfer attention and
`flashinfer_cutlass` FP4 GEMM, PyTorch sampling, Triton linear attention/Mamba,
and disabled prefill/decode CUDA graphs. MTP used EAGLE with 2 steps, top-k 1,
and 3 draft tokens. The context limit belongs to these fixed smoke tests;
256K GPQA validation is a separate gate.

Each of the five fixtures ran 2 warmup and 5 measured trials with temperature 0,
thinking disabled, and a confirmed prefix-cache flush before every request.
W4A4 baseline passed all 35 trials, including output/logprob alignment,
successful stopping without length termination, code AST checks, the red/blue
image counterfactual, and OCR. Code checks do not execute the generated code
or establish algorithmic correctness.

W4A4 MTP executed and passed all functional checks, but the `python_code`
fixture failed exact token-output parity against this run's baseline in all
7 trials: baseline produced **277 tokens**, MTP produced **272 tokens**. Both
outputs passed the code structure and successful-stop checks. Arithmetic and
all three vision fixtures had identical baseline/MTP token-output digests in
all 7 trials each. The MTP report therefore has `validation_passed=false`;
its failed code trials are excluded from valid measured-trial summaries.
Passing visual responses do not establish that all quantized visual/MTP layers
executed: the independent auxiliary apply probe has not run.

| Serving evidence | SHA256 |
|---|---|
| Fixture source | `f0e8ff85f40b4aaa9847ab95dd9098591845f137faab574ca6d146f0be3f04da` |
| W4A4 baseline report | `79ef2ba4cfbf64c2ebc3dd2d59910356ce935ca4d0e9ecf2ab629f51579dc59b` |
| W4A4 MTP report, bound to that baseline | `4ff6ab53f3295a362e626363371dd81a635bd4e3f3f0e0f93388629413493a72` |
| Serving continuation terminal status | `2b50cb309d65d90e2bbafda49c3373cc00106316945a92387020d50bfb953b83` |

The continuation correctly stopped at **10:53:19 UTC**. W4A8 baseline/MTP and
the four-mode auxiliary apply probe were not started. The next gate is to
isolate the reproducible W4A4 MTP parity difference while retaining these failed
reports, then rerun the required serving modes and execution probe. There is
no new GPQA score or release validation result.

## Latest v6 beta-only matrix — 2026-10-14

The detached v6 job ran from `2026-10-14T13:24:39Z` through
`2026-10-14T14:01:42Z`:

| Matrix identity | Value |
|---|---|
| Job | `5074b1bec1db47bbbc5ad891a8f338ca` |
| Source input | `4a667614d810f7b5570be23eb75956c603a85c3a` plus `beta-only.patch` |
| Source archive SHA256 | `cf1a72861bcd59735a38f2050b6876ef2a009bd4f989e0d998c845e66c853b2e` |
| Source manifest SHA256 | `d1d747980444f814cd4aaf4773394e4bdbdc8baa2b58f82fc6c7b0d1cf247ff4` |
| Serving image | Retained v4 image `sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf` |
| Deployment detail | Read-only source bind mount replaced the image source; no beta derivative image was built |
| Model generation | Historical 2026-10-01 W4A4 and W4A8 exports |
| Protocol | TP1, 32,768 context, temperature 0, thinking disabled |

The ordinary matrix completed all four modes. Each mode ran five fixtures with
two warmups and five measured trials, so each mode had **35 total requests and
25 measured requests**. Both baseline modes passed all functional checks,
including output/logprob alignment, successful stop, AST structure, and the
red/blue image counterfactual. Both MTP modes passed functional, stop, AST, and
vision checks. Arithmetic and all three vision fixtures had exact baseline/MTP
token parity.

The code fixture did not meet the MTP baseline parity gate in either scheme:

| Scheme | Baseline code tokens | MTP code tokens | Exact parity |
|---|---:|---:|---|
| W4A4 | 271 | 272 | Failed in 7/7 trials |
| W4A8 | 271 | 277 | Failed in 7/7 trials |

The code outputs still stopped normally and passed the AST checks. Failed code
parity trials are excluded from valid timing summaries, so the MTP reports are
not passing validation reports. MTP draft metrics are reported for observation,
including trials that fail parity: W4A4 accepted `960/980` drafts (97.959%)
over `490` verify calls; W4A8 accepted `965/1010` (95.545%) over `505` verify
calls. These acceptance figures do not establish output correctness.

All four independent metadata-only execution probes passed. Each scheme
reported 110 visual quantized runtime layers. The packed MTP path reported 5
runtime layers (this is not the same count as the 8 MTP checkpoint entries).
W4A4 used `ModelOptFp4LinearMethod` for all 5 MTP runtime layers; W4A8 used
3 FP4 and 2 FP8 MTP runtime layers. All eight serving/probe cleanup gates
reported `removed=true` with return code 0.

The baseline code fixture's client timing was:

| Scheme | Median end-to-end throughput | Median wall time | Output tokens |
|---|---:|---:|---:|
| W4A4 | 14.06686 tokens/s | 19.26513 s | 271 |
| W4A8 | 12.22726 tokens/s | 22.1636 s | 271 |

These are single-request timings for the fixed fixture, including request
preprocessing, prefill, decode, and transport. They are not a representative
throughput benchmark; the 256K GPQA evaluation uses a separate protocol.

### v6 evidence hashes

The detached evidence was downloaded into the private audit bundle and checked
against its manifest. The matrix status and four ordinary mode reports have
these SHA-256 digests:

| Evidence | SHA256 |
|---|---|
| Matrix status | `630be91a29679abceae22332bca688aae8ce8c1277591f68cfe1a421cd7ab30d` |
| W4A4 baseline report | `32201d14fb7a45751800f450407450e17e58fdaca6d83e8a1a65e6536c7a61e8` |
| W4A4 MTP report | `b74c24c5f87fd0de3c09d603bfb58e8b85ae1489495a6f2df322e4596736cd60` |
| W4A8 baseline report | `f0e02eb4c316cc2ee7b36613b378c0dcf7dbda1546213d674895eddfb5b29299` |
| W4A8 MTP report | `c787b233ff7bc8733cd844d81252144bd17b54b1ef0a051ffa33d998abd4d4e2` |

### Beta-only GDN ablation

The isolated packed-GDN ablation supports the suspected precision mechanism. With
`FLA_USE_FAST_OPS=0`, FP32 strided state, BF16 `a/b`, and 32 steps, the
unpatched variant produced the expected mismatch for both GQA fixtures:
`(4,8)` had output/state difference counts `8630/540565`, and `(16,48)` had
`50676/3206618`; both first diverged at step 0. The beta-only FP32 sigmoid
variant produced zero output and state differences on those fixtures. A random
Q/K check also produced zero differences, but is observational evidence only.

The fixture file hashes differ between the unpatched and patched runs
(`b144de711b7261f32a5d1ad236fe38ac16aee190de974fd32469039a42b5658b` versus
`acc1aa2a94a7a8162d19ef7672cf7674d15be60bc471fd1072b5533fb848d8e9`), due to an
additional cache guard. They are therefore not a single identical-fixture
paired gate. The unpatched report hash is
`91ae00c58e2645f02edd7cd9003ee58942379f01b902425ff2966387e6050887`; the
beta-only report hash is
`6cb9fcdfd3fa7bd4180be46f61496b1860d307fdbdc9b2cc5f64d67244a6601a`.
The ablation is local kernel evidence and does not by itself prove that the
full-model parity issue is fixed. The full-model v6 matrix above used this
beta-only source patch through a read-only source mount.

### Reproducibility boundary

The matrix records `sglang_revision=73d91d5216455ede7eb33633cb2ccfe5ddf35076`,
but this value is a file blob object ID rather than a verifiable Git commit.
The source archive, manifest, patch, image ID, and result hashes pin the
experiment's contents, and all eight runtime source-hash gates observed the
patched kernel SHA256
`4664a9202cbd528a2647804bc96d0ab92eacbe06c1c97a742c139591e93d4470`.
However, the revision field does not identify a patched commit that can be
checked out. This is experimental evidence, not validation of a release pinned
to a real patched Git commit. A formal rerun should commit the beta patch,
regenerate the source manifest with that commit ID, and rerun the required
gates with the exact image and source deployment recorded. The original v4
reports and JSON evidence remain unchanged.
