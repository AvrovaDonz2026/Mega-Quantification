# DGX Spark release validation - 2026-10-01

**Status: passed within the tested scope.** This new end-to-end rerun for
release `v0.1.2` rebuilt both images twice, performed both complete PTQ runs,
served the new exports with vision/OCR and quantized MTP, and reloaded both
images into separate empty Docker stores. All required runtime gates passed.
The earlier
[clean-container validation](dgx-spark-clean-container-20260930.md) is background
evidence and does not substitute for this rerun.

The frozen runtime revision is
`ac53833dc5af3f03a10087449ca80e9484deba48`. Its Git source archive SHA256 is
`cdc83ea227a753ee5950c63fe2ea5d6b225935537006ebacd76edbedd64aa97e`.
Both quantization and serving use this revision. The companion
[JSON record](dgx-spark-release-20261001.json) records
`complete: true` and `status: passed`.

## Tested scope

The rerun uses one physical ARM64 NVIDIA GB10 / SM121 DGX Spark with TP1.
It rebuilds both runtime images twice with no cached build steps and build
networking disabled, then runs full W4A8 and W4A4 quantization into new output
directories. The original BF16 27B checkpoint and calibration dataset are
reused as inputs; existing quantized weights are not used as rerun outputs.

Serving uses the newly exported weights, baked repository code and recipes,
with a normal baseline run and an embedded quantized MTP run for each format.
The fixed requests cover arithmetic, a 192-token Python response, red and blue
visual counterfactuals, and OCR. The Python request checks a nonempty response
and token parity; it is not a code-correctness benchmark.

Both final images also undergo save/checksum/load validation in separate empty
Docker stores with private mount and network namespaces and private containerd.
That portability check is CPU-only on the same physical host and kernel.

This record does not verify TP2, a second physical machine, arbitrary-prompt
determinism, quantization quality against the BF16 model, GPQA accuracy, speedup,
or public container-registry publication. Client elapsed times, when retained,
are diagnostic timings rather than performance measurements.

## Required gates

| Gate | Current result |
|---|---|
| Two offline clean PTQ builds and matching manifests | Passed |
| Two offline clean serving builds and matching manifests | Passed |
| PTQ GPU preflight | Passed |
| Serving GPU preflight | Passed |
| Full W4A8 PTQ and strict source/provenance/calibration audit | Passed |
| Full W4A4 PTQ and strict source/provenance/calibration audit | Passed |
| W4A8 baseline, vision/OCR and embedded MTP token parity | Passed |
| W4A4 baseline, vision/OCR and embedded MTP token parity | Passed |
| PTQ image empty-store archive roundtrip | Passed |
| Serving image empty-store archive roundtrip | Passed |
| Required CI at the frozen runtime | Passed; 408 passed, 1 skipped |

Completion of this record refers to the frozen runtime rerun. The subsequent
documentation PR CI and merged-main CI are publication gates to check on
GitHub before tagging; they are not claimed as completed in this record.

## Completed early gates

Both PTQ builds and both serving builds exited with code 0. Each pair produced
identical recorded container manifests. All four manifests, installed-package
version checks and image version labels identify MegaQuant `0.1.2` and frozen
revision `ac53833`. Both serving builds also passed the seven-file loader patch
check. The image indexes differ between builds; this verifies matching software
inputs and installed files, without claiming byte-identical OCI images.

| Image | First clean OCI index | Repeat clean OCI index | Container manifest SHA256 |
|---|---|---|---|
| PTQ | `sha256:cb9c47110cbe9a74e6a0495e662fc36400710048fe909e5beca63f510b1647f1` | `sha256:792cb90f415fa447e772a7e4f0165484a79a35a953f75ddb3eafd142763dbe97` | `0f01e93cbf473eb7a989644a83fd5d9ba624ade2067914502b9270956b6a0620` |
| Serving | `sha256:e1a9ab63032c17d6be209e601bc28ca7e5d4d075c6ff5e703ec8ed7a5d82a29a` | `sha256:000a0756e62cbd32484e84fb498e403896408f8b42fb6092d5bb7a21d741f6aa` | `6df2201b86d6753b561d98dde39513e03c3c303b77cb5ad9cd3d52f647dbd410` |

The normal PTQ GPU preflight passed CUDA matrix multiplication and the GDN
export fallback forward. It recorded CUDA `13.4`, PyTorch
`2.14.0a0+4fdf77b940.nv26.8.63802676`, ModelOpt `0.47.0` and Transformers
`5.12.1`. It verified all 256 calibration rows, decoded and checksum-verified
128 images, and matched the source-index and calibration hashes below.
All 18 full BF16 source-shard checksums matched the previous input manifest.

Required push and PR CI passed at the frozen revision. The
[PR CI run](https://github.com/AvrovaDonz2026/Mega-Quantification/actions/runs/36803695110)
reported **408 passed, 1 skipped** in 24.67 seconds. The skipped test requires
ModelOpt, which is absent from the GitHub CPU test environment.

Frozen-source wheel and sdist validation also passed. The rebuilt wheel from
the sdist was byte-identical to the directly built wheel. Package import,
`pip check`, CLI help and both baked quantization plans passed in a disposable
CPU environment; these checks do not substitute for full GPU runs.

The serving GPU preflight exited with code 0 in 24.295 seconds. It verified the
installed manifest and all seven v6 patch files, CUDA matrix multiplication,
GDN output/state agreement, static FP8 batch invariance, and BF16 GDN gate
batch invariance with a separate CPU FP64 reference. All six FP8 comparison
rows and 30 BF16 gate comparison rows were byte-identical, with no nonfinite
values or out-of-tolerance reference elements.

Its four synthetic attention cases exercised the actual patched backend at
KV lengths 1, 2, 53, and a two-request batch with lengths 2 and 127. All output
bytes matched the first verification query, two poisoned masked future tokens
had no effect, and each case wrote only its real KV rows once. This kernel
regression does not establish invariance for arbitrary contexts or prompts.

## Fresh W4A8 quantization

The full W4A8 run used the frozen PTQ image and normal Compose recipe, with
offline model loading and a new export directory. It exited with code 0 in
886.567 seconds. The baked public auditor, `scripts/audit_spark_export.py`,
then exited with code 0 in 25.197 seconds, with no errors or warnings. Export
provenance exactly matched the supplied frozen revision `ac53833`.

The audit checked safetensors layout and HF index references, every source
language/vision/MTP tensor and logical element count, packed NVFP4 shapes,
weight and activation scale presence, calibration execution evidence, and
configuration/quantization metadata agreement. It mapped all 333 vision and
15 MTP source tensors into the export. Streaming scans of 1,345 scale tensors,
covering 1,197,496,120 bytes, found no NaN or infinity.

| W4A8 export branch | Quantized projections | Calibration execution |
|---|---|---|
| Language | 193 NVFP4 + 208 FP8 | Full configured multimodal PTQ run |
| Vision | 110 NVFP4 | 165 observed quantizers; 128 calls each |
| MTP | 4 NVFP4 + 4 FP8 | 8 observed quantizers; 256 calls each |

The export contains 2,544 tensors in three weight shards. The structural audit
does not verify generation correctness or numerical quality.

## Fresh W4A4 quantization

The full W4A4 run used the same frozen PTQ image and normal Compose recipe,
with offline model loading and a new export directory. It exited with code 0
in 757.809 seconds. The baked public auditor then exited with code 0 in 34.743
seconds, with no errors or warnings and exact `ac53833` provenance.

The audit checked the same structure, source coverage, calibration evidence
and scale payloads as W4A8. All 333 vision and 15 MTP source tensors mapped
into the export. Streaming scans of 1,557 scale tensors, covering 1,654,937,400
bytes, found no NaN or infinity.

| W4A4 export branch | Quantized projections | Calibration execution |
|---|---|---|
| Language | 401 NVFP4 | Full configured multimodal PTQ run |
| Vision | 110 NVFP4 | 165 observed quantizers; 128 calls each |
| MTP | 8 NVFP4 | 8 observed quantizers; 256 calls each |

The export contains 2,756 tensors in two weight shards. Both formats' complete
quantization and strict audits have passed.

## Serving the newly quantized weights

All four baseline/MTP client runs passed using the newly exported weights.
Container inspections verified the normal baked entrypoint and recipe, with
no live source or recipe mounts and no `PYTHONPATH` override. Server logs
confirmed TP1, multimodal loading, FP32 Mamba state, FP8 KV cache and disabled
CUDA graphs. MTP loaded `Qwen3_5ForCausalLMMTP` from the same quantized checkpoint
with ModelOpt quantization and EAGLE (two steps, top-k 1, three draft tokens);
no external draft checkpoint was supplied.

Each format passed all five fixed cases: arithmetic returned `391`, the
Python response emitted 192 tokens, the image counterfactuals returned red and
blue, and OCR returned `3729`. Baseline and MTP received identical requests,
and all 205 output token IDs per format were exactly equal. All 820 emitted
token logprobs across the four modes were finite and aligned with their token
IDs. Every MTP case recorded positive accepted drafts and verification steps.

| Format | Matching output IDs | Accepted/proposed drafts | Verification steps | Baseline/MTP client seconds |
|---|---:|---:|---:|---:|
| W4A8 | 205/205 | 136/146 | 73 | 27.095 / 16.781 |
| W4A4 | 205/205 | 137/142 | 71 | 23.996 / 14.557 |

All four start, check, stop and remove stages exited with code 0. These fixed
functional requests and diagnostic client times do not establish arbitrary
prompt parity, code correctness, model quality or performance improvement.

## Image archive portability

The new PTQ image passed its archive roundtrip in 524.043 seconds. A separate
Docker store started with zero images and containers, in private mount and
network namespaces with private containerd. The saved archive contained
12,038,999,040 bytes, SHA256
`f1c70f755cd4b10574e46a382be1432a8b5d7d45bcd24052409999dd0842a50e`.
After loading and running the checks, the selected image config remained
`7b6cd2985762f83b1819c0defaba3b12b43159934cf3cca31245bde5d86199d6` and
all 79 layer diff IDs were unchanged.

With networking disabled and no GPU or host source/recipe mounts, the loaded
image verified its installed manifest, imported ModelOpt/Transformers/Pillow/
PyArrow and installed MegaQuant `0.1.2`, and produced both baked quantization
plans. The host bridge and addresses were unchanged. All three preexisting
containers retained their identities, PIDs, start times, running states and
networks, and their gateway remained reachable.

The new serving image also passed a separate empty-store roundtrip, in
690.127 seconds. Its archive contained 14,752,754,688 bytes, SHA256
`a45baf968ccf038c26622d5309d6586e35c4df68a233c1d2d9c411fd60a4fd42`.
The loaded image retained config
`9a48dfb3dcb697e70f49bf92419af32c849bb5b3878a9f698404f2c80f32d55a` and
all 80 layer diff IDs. The installed manifest and every one of the seven v6
loader patch files matched the GPU-tested serving image. Installed imports,
the multimodal checker CLI, and the baked baseline/embedded EAGLE commands
all passed without network access, GPU or host code/recipe mounts. Its
private namespace, bridge and three original-container checks passed as well.

Both private daemon pairs were stopped after validation. The archive checks
use the same physical host and kernel; full GPU quantization and serving were
validated separately using the same frozen images.

The final read-only cleanup check found only the three preexisting containers
running, zero private validation daemons and zero new intermediate `.pt`
files under the rerun's outputs, logs and results directories. The validation
driver did not instrument or save intermediate model tensors.

## Inputs and evidence

The verified model is `Qwen3.8-27B`, with 18 source shards and 1,199 source
tensors: 851 language, 333 vision and 15 MTP. Calibration uses 256 rows,
including 128 images and 128 text rows, seed 42, batch size 1 and maximum
sequence length 1024. The preflight checked these inputs; both new export
audits independently checked calibration coverage and weight structure.

The expected calibration JSONL SHA256 is
`e2a48a43e732995c36a2096dd95f2edbad3f13eaa85a042ced7094a149af6eed`; the expected
source index SHA256 is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

New evidence is retained locally under `artifacts/spark-release-20261001/`.
The companion JSON names 170 individual evidence hashes, image manifests, export
coverage, response results and archive identities. The source archive hash
and all retained evidence hashes were checked. No new intermediate model
tensors were captured by this rerun.
