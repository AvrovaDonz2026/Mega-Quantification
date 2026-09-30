# DGX Spark clean container validation - 2026-09-30

**Status: incomplete.** Full W4A8/W4A4 quantization and export audits passed at
revision `15561d8`. The original official-based serving image passed W4A8
vision/MTP checks but reproduced a stable W4A4 baseline/MTP token mismatch.
The revised serving image at revision `44bf3d7` built cleanly twice with matching
container manifests, passed its GDN GPU regression and isolated archive
roundtrip, and passed strict W4A4 visual/MTP validation. Its W4A8 functional
checks passed but Python baseline/MTP tokens diverged. Later v4/v5 clean builds
also passed, and v5 passed its isolated image roundtrip. Projection repairs
have not established full-model W4A8 parity. Further serving repair and
clean-image validation remain pending.
Complete container reproduction is not yet established.

The companion
[JSON record](dgx-spark-clean-container-20260930.json) keeps pending results as
`null` and records `complete: false`. Commands are in the
[Spark container runbook](../DGX_SPARK.md).

| Stage | Runtime revision | Git source archive SHA256 |
|---|---|---|
| Quantization and original serving validation | `15561d8c445ab6e5637b78f3207e683fcc7c3869` | `5716b7ed292d7be27ce035158363121849753a2fd8dc797006e1da5bf60c8968` |
| Corrected serving validation | `44bf3d7e3a5679c722f27b27e04d3bfb26784d8b` | `cb5cbe73d05fa2259b3e9a330c0a3935aa9c52d5e30888fb778637cbda7f455d` |
| FP8 serving repair, v4 | `4897adb63227e4ce62ba68042ccd21e2dbb5de19` | `a6eae547194cbde33d5c51708acc0cc16bab1f60a31032298d6cbdc2e7ec6cb4` |
| BF16 GDN gate repair, v5 | `fc8925db4b59187bfacc920fefbc0192bf3593d1` | `4749b63443b374209b0bfc9eeb1730213fa101bb2c01e326a0420a5e178f9da9` |

The corrected serving runs reuse the audited exports produced by the quantization
runtime. The full PTQ runs were not repeated at the later serving revisions. Separately verified
dependency wheelhouses are not part of either Git archive.

## Scope and inputs

One ARM64 NVIDIA GB10 / SM121 was used, with a single GPU and TP1. The PTQ image
uses the pinned public NGC PyTorch 26.08 base; the serving build uses the pinned
official SGLang 0.5.20 CUDA 13 base. The builds install MegaQuant as a wheel,
record a container manifest, and retain baked code and recipes. External mounts
are restricted to model weights, calibration data, output, download cache and
offload storage. The model and calibration data do not enter either image.

Preflight checks decoded and checksum-verified 128 calibration images and found
256 image/text rows. Calibration uses seed 42, batch 1 and maximum sequence
length 1024. Source inspection found 1,199 BF16 tensors: 851 language, 333 vision
and 15 MTP. There are 18 source weight shards; their individual SHA256 values
were retained separately. The calibration JSONL SHA256 is
`e2a48a43e732995c36a2096dd95f2edbad3f13eaa85a042ced7094a149af6eed` and the source
index SHA256 is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

## Clean builds and repeatability

The final PTQ image was constructed from the official digest-pinned NGC base
with the repository Dockerfile, no cached build steps, an explicit wheelhouse,
and networking disabled for the Docker build. The wheelhouse contains the
fixed direct and transitive PTQ dependencies and build backend. No local
`quant-env` base, editable install or live code/recipe mount was used.

| PTQ build | OCI image index / Docker inspect ID | Container manifest SHA256 |
|---|---|---|
| First clean build | `sha256:39d6859eb7a7fe2a9195441f6df4cfa9236d17258c55f3abf0eae6a3c8dab180` | `418d877c13fce8b7e4a8b836f3acb87dfa4253401a1385bfcd80661a0def00b9` |
| Independent repeat | `sha256:32b6e68d4597923a7a289b32491da957865c568be7ba4f90626332e32e8c2ff2` | `418d877c13fce8b7e4a8b836f3acb87dfa4253401a1385bfcd80661a0def00b9` |

Both builds succeeded and produced the same recorded baked-file hashes,
installed MegaQuant hashes, package version map and architecture. Their image
indexes differ; this is software-input reproduction, not a byte-identical image
claim. Platform manifest and configuration digests are separately named in JSON.
The normal container entrypoint checks the baked manifest before invoking the
installed CLI.

The official SGLang ARM64 base was acquired as OCI content and all 71 blobs,
including its platform content and attestations, were SHA256-verified. Compressed
content totals 14,714,398,493 bytes; the canonical pinned upstream index was
preserved. Both serving builds then used the repository Dockerfile with no
cached build steps and no networking during the build.

| Original SGLang build (`15561d8`, patch v2) | OCI image index / Docker inspect ID | Container manifest SHA256 |
|---|---|---|
| First clean build | `sha256:42c9f3a1cda593163e0cbb4a05cbba570c6642c5853ee5ced5e64550b8108aa0` | `c62626c2a8d4a7d015b4b4d7ccce74d811c2b4891d8538320a1687eb9447b015` |
| Independent repeat | `sha256:796fcdc592a051adf711905770452d4798bdb74dd9444ec22f7fe220a6755336` | `c62626c2a8d4a7d015b4b4d7ccce74d811c2b4891d8538320a1687eb9447b015` |

Both CPU manifest checks and guarded loader-patch checks passed. The two complete
CPU check JSON records were also byte-identical. The serving image retains the
official base's PyTorch `2.13.0+cu130`, SGLang `0.5.20`, Transformers `5.12.1`,
sglang-kernel `0.4.7`, FlashInfer `0.6.18` and ModelOpt `0.46.1`. Serving ModelOpt
intentionally differs from PTQ's `0.47.0`; actual fresh-export inference is the
compatibility gate for the corrected serving runtime. The prior
[serving result](dgx-spark-serving-20260930.md) used a locally derived CUDA image
and is not a substitute for that new inference test.

### Revised Serving Builds

The `44bf3d7` serving revision bakes patch `megaquant-spark-v3`. It aligns the
packed GDN decode beta computation with MTP verification using an FP32
expression, eliminating the previous intermediate BF16 conversion. It also
includes a GPU regression checker covering variable sequence lengths and
unselected state-slot isolation. The normal GPU preflight command, which runs
the GDN checker and requires exact state/output equality, exited successfully.
This kernel check did not establish full-model W4A8 token parity.

Both new builds used the same official base with `--no-cache`, build networking
disabled, an explicit offline wheelhouse, and the normal repository Dockerfile.
Both baked manifest and installed patch verifications passed.

| Revised SGLang build (`44bf3d7`, patch v3) | OCI image index / Docker inspect ID | Container manifest SHA256 |
|---|---|---|
| First clean build | `sha256:05ae15d7873b6ea52fda4628b85a5a1fb0ccabb53f56232f8ac76f941f86cc72` | `5aaca3e443652d7057b8c261570ad7352fe56ccbcf364a73667b1d415fe2ed6a` |
| Independent repeat | `sha256:63eb53dd495d37f4b1f4e79024e3a32f8b7572fba619d3c198a42ac939b2a798` | `5aaca3e443652d7057b8c261570ad7352fe56ccbcf364a73667b1d415fe2ed6a` |

The new manifests record the same main upstream runtime versions as the original
serving image. The two complete manifest files are byte-identical; the OCI image
indexes differ. Its selected configuration was checked by the archive roundtrip
below. Other descriptor fields are separately named in JSON.

## Original Portable Image Roundtrip

Both finished `15561d8` images passed save/checksum/load checks in separate empty
Docker stores. The reruns used distinct mount and network namespaces, private
containerd and unique containerd namespaces, with no source/recipe mounts, GPU
attachment or network access for validation containers. Archive hashes before
and after load, image configurations, layer diff IDs and baked manifests matched.

| Image | Archive bytes | Archive SHA256 |
|---|---|---|
| PTQ | 12,038,956,032 | `458712cbca8906b0daee4c43b76239aa8f8151cb9a69873a1d30b076901ebc58` |
| SGLang | 14,752,670,208 | `d227794ebbc56bdc051fb9cc535f352c4412e6c8d2f52161217a1fd3e4871862` |

PTQ executed the baked W4A8/W4A4 plans and imported ModelOpt, Transformers,
Pillow and PyArrow. SGLang checked the loader patch, baked FP32 baseline and
embedded EAGLE commands, runtime imports and multimodal checker CLI. Docker's
isolated legacy store identified the loaded images by their configuration
digests rather than the source store's OCI indexes; both selected configurations
and layer lists were unchanged. This verifies CPU portability on the same ARM64
host and kernel, not GPU execution on a second physical machine or TP2.
These checks apply to the PTQ and original serving images at `15561d8`; they do
not establish portability of the corrected `44bf3d7` serving image.

An initial attempt shared the host containerd socket and network namespace.
Moby's bridge disabling removed the host `docker0` bridge outside a private
network namespace. The bridge was restored without restarting the three
pre-existing containers. The corrected reruns preserved the host bridge identity,
all three original container start records and gateway reachability before and
after. The initial attempt is superseded and is not counted as an isolated test.

## GPU and installed-wheel preflight

The PTQ preflight passed actual CUDA matrix multiplication and a small GDN
fallback forward on the GB10. Its manifest recorded quantization revision
`15561d8`.

| Component | Observed PTQ value |
|---|---|
| Architecture / GPU | `aarch64`, NVIDIA GB10, SM121 |
| PyTorch / compiled CUDA | `2.14.0a0+4fdf77b940.nv26.8.63802676` / `13.4` |
| Transformers / ModelOpt | `5.12.1` / `0.47.0` |
| Accelerate / Pillow | `1.14.0` / `12.3.0` |
| Manifest / CUDA matmul / GDN forward | Passed |
| Source / vision / MTP tensors | 1,199 / 333 / 15 |
| Calibration rows / verified images | 256 / 128 |

## Reproduced Provenance Failure

The first clean-image W4A8 run at revision `03949ba` completed calibration and
export with exit code 0 in 775.21 seconds. Its audit nevertheless failed: the
export directory `Qwen3.8-27B-NVFP4-W4A8-spark` contains a dot, so the old
suffix-based path heuristic wrote `provenance.json` beside the export rather
than inside it. The image had no Git checkout, so its Git provenance was also
null. The failed audit is retained as a failure, not counted as a passing export.

The fix gives an existing directory precedence over the filename suffix and
uses a valid full revision from the baked manifest before falling back to Git.
It preserves file-export behavior and the previous heuristic for paths that do
not exist. The 22 new regressions cover dotted directories, file paths,
independent export provenance, missing/invalid manifests and Git fallback.

Required push and PR CI for quantization revision `15561d8` both passed. The PR run
reported **312 passed, 1 skipped** in 22.67 seconds; the skipped test required
ModelOpt. A separate CPU-only run in the real clean PTQ image executed
`tests/test_mtp_model.py`: **5 passed, 0 skipped**, including ModelOpt collection
of positive finite activation ranges for all eight MTP projections. It used
the installed MegaQuant wheel, a read-only test mount at `/validation`, and a
temporary test runner. CUDA was unavailable and the final image was unchanged.
That real ModelOpt run used the preceding `03949ba` image with the same backend
versions; it is not a claim that every test ran on the final image.

## Full PTQ and Export Audit

Both full-27B runs use the baked Spark recipes and the normal image entrypoint.
The final W4A8 run completed with exit code 0 in 773.692 seconds; its strict
audit passed in 24.6 seconds with no errors or warnings. The corrected
`provenance.json` is inside the export and identifies runtime revision
`15561d8c445ab6e5637b78f3207e683fcc7c3869`. W4A4 completed with exit code 0 in
779.28 seconds; its strict audit passed in 34.154 seconds, also with no errors or
warnings and the same correct runtime revision.
The final audit must require correct per-export provenance and runtime revision,
complete source tensor coverage, expected projection formats and shapes,
activation calibration coverage and finite scale payloads.

| Format | Full calibration / export | Strict export audit | Vision / MTP coverage |
|---|---|---|---|
| Mixed W4A8 | Passed, 256 samples, 773.692 s | Passed, 2,544 tensors in 3 shards; 1,345 finite scales | 165 visual quantizers x 128 calls; 8 MTP quantizers x 256 calls |
| Uniform W4A4 | Passed, 256 samples, 779.28 s | Passed, 2,756 tensors in 2 shards; 1,557 finite scales | 165 visual quantizers x 128 calls; 8 MTP quantizers x 256 calls |

The passing final audit retained all source tensor groups. Language projections
are 193 NVFP4 plus 208 FP8, vision has 110 NVFP4 projections, and MTP has four
NVFP4 plus four FP8 projections. It scanned 1,197,496,120 bytes of scale payloads
and found no NaN or infinity. The audit checks source tensor mapping, logical
element counts, packed shapes and quantization metadata; it does not measure
numerical model quality.

W4A4 retained the same source tensor groups and quantized all 401 language,
110 visual and eight MTP projections to NVFP4. Its audit streamed 1,654,937,400
bytes of scale payloads and found no NaN or infinity.

CPU-only streaming SHA256 over every safetensors file found **all three W4A8
shards byte-identical** between the initial `03949ba` and final `15561d8` runs.
Each export contains 20,701,792,184 safetensors bytes; the checksum comparison
took 54.555 seconds using read-only mounts. The JSON record contains each shard's
size and digest. This is an observed match for this pair of runs, not a guarantee
for every recalibration or environment.

## Original Fresh-Export Vision and MTP

Each fresh export was loaded through the original official-based SGLang image
at `15561d8`, with
separate baseline and MTP starts through the normal Compose/CLI entrypoint.
Both recipes use FP32 SSM state, FlashInfer attention, Triton GDN, FP8 E4M3 KV,
32,768 context, memory fraction 0.70, and no host HiCache or CUDA graphs.

The public client checked arithmetic, nonempty Python output, red/blue
counterfactual images and digit OCR. Both formats returned all expected short
answers and finite output logprobs for all 205 tokens per mode. MTP executed in
all five cases. Strict MTP success additionally requires exact baseline
`output_ids` for each request; W4A4 did not satisfy that requirement.

| Export | Baseline functional checks | MTP functional checks | Exact IDs / accepted drafts |
|---|---|---|---|
| Mixed W4A8 | 5/5 passed | 5/5 passed; strict gate passed | 205/205 exact; 137/142 accepted (96.48%) |
| Uniform W4A4 | 5/5 passed | 5/5 functional passed; strict gate failed | Four short cases exact; Python diverged at index 16; 138/140 accepted (98.57%) |

W4A8 performed 71 MTP verification steps. Its 192-token Python request accepted
127/130 drafts (97.69%). W4A4 performed 70 verification steps and accepted
127/128 drafts in Python (99.22%), but its client exited with validation failure.
Acceptance does not override the token-equivalence gate or establish a speedup.

### Stable W4A4 Token Divergence

The first differing Python token is index 16, the 17th output token: baseline
chose token `449` (` with`), while MTP chose `264` (` a`). Both responses reached
the 192-token cap. Arithmetic, both color images and OCR exactly matched their
baselines. Two additional baseline requests each reproduced the original
baseline's 192 IDs; two additional MTP requests each reproduced the original
MTP's 192 IDs. Every repeat reported zero cached tokens. Thus the mismatch is
stable within each decoding mode in these tests.

The repeat requests also collected top-five output logprobs at that position:

| Mode | Token 449 logprob | Token 264 logprob | Greedy choice |
|---|---|---|---|
| Baseline, both repeats | -0.6295747161 | -1.7545747757 | 449 |
| MTP, both repeats | -1.1769373417 | -1.0519373417 | 264 |

The candidates changed rank rather than tying. This establishes different
verification/decode distributions for the same recorded prefix; it does not by
itself identify the responsible kernel operation. The revised GDN precision
patch is now baked into both successful `44bf3d7` clean serving builds. Full
model parity and portability must be rechecked on that corrected image.

## Revised Serving Archive Roundtrip

The `44bf3d7` serving image also passed a separate save/checksum/load roundtrip
using an empty Docker store, private containerd and distinct network/mount
namespaces. Its 14,752,680,448-byte archive has SHA256
`60a061d248899b2da0582dcd045d93724ca187ba281985275a6969a58ceee837`.
The loaded image identified configuration
`sha256:0bf34831245ef6cd700b9e1ffdb60f92225e2eebb8260868133e814e727609af`;
configuration, layers, the baked manifest and all five installed patch guards
matched. Baked baseline/MTP commands, runtime imports and checker CLI passed.

Validation containers had networking disabled, no GPU and no source/recipe
mounts. The host bridge identity and addresses, all three original container
starts and networks, and gateway reachability were preserved. The scope remains
CPU portability on the same ARM64 host/kernel; a subsequent serving revision
needs a fresh roundtrip.

## Revised Fresh-Export Vision and MTP

The revised serving runtime is `44bf3d7`; the quantized weights retain
provenance `15561d8`. The normal serving preflight command exited successfully.
All four fresh-export runs passed five functional cases, including the color
counterfactual and OCR; all 205 output logprobs per mode were finite. Embedded
MTP executed in every MTP case.

| Export | Baseline functional checks | MTP functional checks | Exact IDs / accepted drafts |
|---|---|---|---|
| Mixed W4A8 | 5/5 passed | 5/5 functional passed; strict gate failed | Four short cases exact (13 IDs); Python diverged at index 16; 137/142 accepted (96.48%) |
| Uniform W4A4 | 5/5 passed | 5/5 passed; strict gate passed | 205/205 exact; 138/140 accepted (98.57%) |

W4A4 performed 70 verification steps; its 192-token Python request matched the
baseline exactly and accepted 127/128 drafts. W4A8 performed 71 verification
steps and accepted 127/130 Python drafts, but its 17th Python token differed:
baseline chose `264` (` a`), while MTP chose `449` (` with`). Both Python
responses reached the 192-token cap. This is a new W4A8 failure at the same token
index as the historical W4A4 failure, with the opposite token direction. It is
recorded separately from the original `15561d8` results.

### Further Serving Repair History

Two subsequent revisions also built twice from the same official pinned base
with `--no-cache` and no build networking. Their baked manifest and installed
patch checks passed, with the same manifest in each pair. Image indexes differed.

| Revision / patch | First OCI image index | Repeat OCI image index | Container manifest SHA256 |
|---|---|---|---|
| `4897adb` / v4 | `sha256:bbac0d24b96728905cfb5c716a5561f6b5b66c4b6bfaef3aa6727b4055a84796` | `sha256:081abacb0d9b138d53bf828c88fbe7e90a93cdab161c99644575fb3ed827c97f` | `c7d5ea85551dd23e71c1db9942f3a3a8d120cfbd4cf6a6ba24e9d41a481c506b` |
| `fc8925d` / v5 | `sha256:231b0aecce3161dbbe0156f2e61cb09597843fefa225adbef96d8b841e31d82d` | `sha256:8de68795847eed70ef1f75e1cb33a95046afd745bb85decd8a4ed02861be1640` | `2a449292f25da864861e63bef08100a3d406fa7793324f5256f96fd14e99a058` |

V4 aligns the static FP8 decode/verify backends on SM121. Its GPU preflight
passed, but its full W4A8 client still failed at Python index 16. All five
functional cases and finite logprobs passed; the four short cases matched
exactly. Accepted drafts remained 137/142 over 71 verification steps. This is
a retained failing clean-image result.

The subsequent hash-only experiment used identical hidden row 0 inputs for
decode batch 1 and verify batch 3. At layer index 2, the BF16 `a` gate projection
output differed while `b` and `qkv` matched. Layers 0 and 1 matched. This isolates
a projection difference for that same recorded input; it does not establish the
entire cause of the later token mismatch. The record retained hashes rather
than the source hidden tensor.

Real gate-weight GPU comparisons covered 54 rows across three layers and six
seeds. The original computation differed for 18 rows, with maximum absolute
difference `0.0078125`; the fixed computation was bitwise equal across batch
sizes for all 54 comparisons. The initial harness attempt failed an assertion
and is not counted as passing evidence. V5 bakes the gate reduction repair and
its additional regression checker. The v5 clean image's GPU preflight passed
the baked manifest, CUDA matmul, GDN and FP8 batch-invariance checks, plus its
BF16 gate checker. A separate earlier run of that checker also passed 30
comparisons using bounded seeded BF16 weights and inputs, with batch sizes 3
and 12 against single-row execution. Every row was bitwise equal and finite.
Its independent CPU FP64 reference was within the declared budget
(`rtol=0.00400625`, `atol=0.001`), with zero out-of-budget elements. This
synthetic regression complements the real-weight comparison; the full-model
token-equivalence gate still needs to pass.

The experimental gate-repair image also passed all five functional W4A8 cases
but failed Python token parity at index 16. That experiment accepted 136/146
drafts over 73 verification steps. It is a candidate result, separate from the
reproducible v5 clean build.

After the gate repair, a bounded v5 raw-trace comparison recorded 1,176 matched
comparisons: 593 identical and 583 different. Its earliest recorded difference
was layer index 4, before convolution, in gate `a` at position 52 for input
token `71093`: 36 of 48 BF16 elements differed, maximum absolute difference
`0.015625`. The previous layer-2 same-input gate difference was repaired, but
the new trace does not establish attention-layer causality. Investigation now
includes attention layer 3. Common token/position matches alone do not prove an
identical earlier prefix or accepted draft; the comparison preserves that limit.

The 378 baseline and 378 MTP raw tensor files were deleted after comparison as
requested, totaling 2,853,597,144 bytes. Only comparison and cleanup summaries
were retained. The comparison cannot now be replayed from those deleted raw
tensors. Official deterministic-mode candidates have not passed the complete
model gate and are not enabled by default. In the recorded `fc8925d` test with
`--enable-deterministic-inference`, all five functional cases passed but Python
still differed at index 16; the four short cases matched. It accepted 136/146
drafts over 73 verification steps. The complete reproduction gate remains open.

### V5 Portable Roundtrip

The clean `fc8925d` image passed a separate roundtrip through an empty image
store with private containerd and distinct mount/network namespaces. Its
14,752,694,272-byte archive SHA256 is
`39467f466895ff7fc00bbb219f8817a8d796caede6bdf1e6a2726bed4907491a`.
The loaded configuration is
`sha256:6a08b0129170d95597fb1091d4388c06b7b9ad16e283071149e600646c1d8628`.
Configuration, layers, baked manifest and all six installed patch guards
matched. Baked baseline/MTP commands, imports and checker CLI passed.

The original three containers, their start records and networks, host bridge
identity/addresses and gateway reachability were preserved. Validation used
no GPU, no live source/recipe mounts and no container networking. This is a
passing CPU portability check for v5 on the same ARM64 host/kernel; it does
not change the failing full-model result.

### BF16 KV Candidate

A further `fc8925d` test changed only `--kv-cache-dtype` to `bfloat16` from the
usual baked serving options. All five functional cases passed in both modes,
but the Python request still diverged at index 16 and the strict MTP client
exited with code 1. The four short cases matched; MTP accepted 138/144 drafts
over 72 verification steps. Inspection of both current export indexes found
no layer-3 K/V/KV scale entries. The installed KV processing uses unit scales
when they are absent (or `None` for an unquantized cache), and the FlashInfer
replay used `k_scale=v_scale=1`. This excludes an erroneous non-unit checkpoint
scale for the inspected exports and layer; it is not a general guarantee for
other weights or a model-quality measurement.

### Attention Trace Observation

The retained comparison summary records equal layer-3 input, QKV,
rotary-position output and gate tensors at position 52/input token `71093`, but attention output differed
in 1,217 of 6,144 BF16 elements with maximum absolute difference `0.0078125`.
Across the bounded trace, 126 comparisons were made, 48 differed and none were
missing; the first recorded difference was that attention output. After wrapper
and repeated-query replay, all 140 attention tensor files (25,353,760 bytes)
were deleted and the trace root contained zero tensor files. Raw attention
tensors were never downloaded; comparison, replay and cleanup summaries remain.
Together with the earlier GDN run, the two separate cleanups removed 896 files
and 2,878,950,904 bytes.

Both instrumented decoding modes chose token `449` in this attention run.
It therefore records a mathematical attention-output difference without
reproducing the earlier `264` versus `449` token failure. The previous raw GDN
run did reproduce the 192-token Python failure before its tensors were deleted.
These experiments have different observation coverage and are recorded
separately. No complete attention repair or clean-image inference pass is claimed.
Observers add synchronization and allocations and may alter the result; the
summary also limits comparisons to common token/position rows, which do not
prove equal earlier draft ancestry.

### FlashInfer Wrapper Replay

An actual GPU replay on SM121 with FlashInfer `0.6.18` reconstructed both
observed FP8 decode and verification attention outputs exactly, with zero
different elements. Prefill K/V and the first token's Q/K/V/gate inputs were
byte-identical between modes. The replay used 24 query heads, four KV heads,
head dimension 256, 52 prefix tokens, page size 1 and unit K/V scales.

| KV representation | Single-query versus three-query attention | Maximum absolute difference |
|---|---|---|
| FP8 E4M3 | 1,217/6,144 BF16 output elements differ | `0.0078125` |
| BF16 | 749/6,144 BF16 output elements differ | `0.00390625` |

For each KV representation, causal versus custom-mask execution at the same
query count was exact for both one and three queries. The single-query output
was unchanged when using the longer KV buffer with masked future entries.
These controls isolate a difference between the one-query and three-query
execution paths for the observed input, rather than the tested mask choice or
future KV entries. A subsequent three-query replay repeated the first Q row
and exposed only the first 53 KV entries. Its first output matched the original
MTP first output exactly for both BF16 and FP8 KV. Extending the buffer to 55
entries while masking the future entries also left the output unchanged. This
adds an identical-query control to the observed query-count difference.

This replay covers one observed first query and a linear verification chain.
It does not cover branched speculative trees or establish complete serving
parity. BF16 wrapper arithmetic used unit scales independently of checkpoint
KV quantization.

### Prefill/Decode Candidate

The experimental prefill/decode image passed all five functional W4A8 cases
with finite output logprobs for all 205 tokens per mode, but strict Python
baseline/MTP equality still failed at index 16. The four short cases matched.
MTP accepted 136/146 drafts over 73 verification steps (93.15%). This candidate
is separate from the clean `fc8925d` image and is not a verified final repair.

## Limits

Complete reproduction remains unverified until the W4A8 token divergence is
resolved and a subsequent clean serving image passes the required gates. A local
mirror or offline base archive must preserve the pinned
official manifest/layer bytes; using a preconfigured derived image would change
the scope. Manifest checks detect recorded file and package-version drift; they
do not cryptographically authenticate every installed third-party package file.

The final functional checks are TP1 on one Spark. They do not establish TP2,
video, general OCR/vision accuracy, BF16 quality equivalence, long-context or
concurrent throughput, a GPQA score, or a controlled MTP speedup. No public
registry image publication is claimed.
