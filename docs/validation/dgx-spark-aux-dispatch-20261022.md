# Auxiliary Dispatch Probe

## Seeded public CUDA regression

The public checker passed on the distinct small-test image with source revision
`e11b673`, ModelOpt 0.47.0, seed `19031`, input fingerprint
`f11307676a14adb219097c8ff08bca10b928693e21e871bcff3afa7cf8ad163a`, and
14.939 seconds elapsed. Vendor and compatibility exports both detected
Accelerate offload and matched the input fingerprint. Vendor auxiliary weights
were BF16 `[32,32]`; the compatibility export produced U8 `[32,16]`, block FP8
`[32,2]` scales and scalar FP32 scales for both auxiliary modules. Cleanup
completed and raw tensor values were not retained. This validates the packing
contract only; it does not establish numeric parity or full-model behavior.

## Earlier small CUDA regression

After the original paired probe, the exact-source-gated compatibility path was
run on a small fresh Llama fixture with ModelOpt 0.47.0. Two auxiliary modules
remained resident on CUDA while the decoder used Accelerate CPU offload. The vendor export retained
both auxiliary weights as BF16 `[32,32]`; the patched export packed both as U8
`[32,16]`, with block FP8 `[32,2]` scales and scalar FP32 scales. The run used
source revision `acdc723`, completed in 14.178 seconds, and cleaned up its
container. The small-test image was
`sha256:7d9ba02b9968e61435174e541141e4f4dd6df03fcb9a5f99108bee67531294be`,
distinct from the full-27B PTQ image
`sha256:d3f9770ae061eeedd10100971ce77e2ac253cf5a45914b3d65c1bee86847dfe1`.
Raw tensor values were removed from the retained evidence.

This is a CUDA packing-contract regression only. The fresh fixture did not use
fixed seeded weights, so it does not establish numeric parity, full-27B export
success, vision/MTP functionality, serving behavior, GPQA quality, or a
repetition repair. The test image was distinct from the full-27B PTQ image.

The reproducible public checker is
[`scripts/check_modelopt_streaming_export.py`](../../scripts/check_modelopt_streaming_export.py).
Run `python scripts/check_modelopt_streaming_export.py` for its dependency-free
plan. For execution, use the pinned Spark CUDA image with ModelOpt 0.47.0,
Accelerate 1.14.0, the image's CUDA PyTorch and Transformers 5.12.1; verify the
container manifest, set the Hugging Face and pip offline flags, and run
`--execute`. The checker performs both vendor and compatibility exports on the
same fresh fixture, compares input fingerprints, audits headers and full file
hashes, removes temporary outputs, and returns `unsupported` nonzero when CUDA,
the pinned dependencies, or required free memory are unavailable. It never
downloads a model or retains raw tensor values.

The fresh paired probe used 256 calibration batches at max sequence length 1024,
batch size 1, with vision and MTP enabled under the same recipe and ModelOpt
0.47.0. It completed the `static-max` arm and failed the `static-MSE` arm before
a comparable pair was available.

- `static-max`: 1,084.633522 seconds, return code 0, export SHA
  `770aed36de2eb5c445b473da6f7449c7f34f9a8a428b924cd424363fa6a14d6d`.
  All 118 auxiliary `QuantLinear` modules (110 vision, 8 MTP) were prepared as
  NVFP4 CUDA BF16 with `requires_weight_materialization=false` and were CUDA U8
  after dispatch. This field does not claim that no materialization occurred.
  It recorded 3,815 dispatch calls and 889 auxiliary calls; the latter is not a
  quantized-layer count.
- `static-MSE`: 1,480.491465 seconds, return code 2. It reached the same 118
  auxiliary modules in the prepare phase, but recorded zero auxiliary dispatch
  calls; all 118 remained CUDA BF16 and unpacked. The backend failure is retained
  only as a redacted hash, not as a raw log or path.
- Pair comparison was skipped because one arm was incomplete, so this is not a
  new matched full-export fingerprint. The evidence supports a candidate
  ModelOpt streaming resident non-decoder dispatch path for the missing MSE
  auxiliary dispatch, but does not establish the complete root cause.
- The MSE backend export failed and the coordinator reported failure, but both
  original services were recovered; cleanup observed the owned container absent,
  both services were running without restart or OOM, the health check returned
  HTTP 200, and the fixed arithmetic smoke returned `is_two=true` with `stop`.
  No new matched fingerprint was produced.

The original probe above did not complete a patched full-27B export; later pair
and restoration evidence is recorded separately below. No repetition fix, GPQA
result, serving pass, or TP2 result is claimed by this report.

## Patched full-27B pair update

The subsequent patched ModelOpt 0.47 pair completed both export arms with return
code 0, and comparison completed with `ptq_comparable=true`. Each export had
`matched_protocol=true` and 118/118 eligible packed auxiliary tensors (U8): 110
vision and 8 MTP. Each export index validated 2,756 keys across two shards and
16 files. Arm durations were 1,102.736 seconds for `static-max` and 1,526.581
seconds for `static-MSE`. The matched protocol used ModelOpt 0.47.0, seed 42,
256 ordered calibration batches, and equal source/model/calibration fingerprints.
The complete quantizer scope was 401 core, 110 vision and 8 MTP weight modules;
the 118 auxiliary count excludes the 401 core modules.
The final coordinator process was stable. A later API restoration check passed
on both GLM services; the fixed arithmetic request returned `2` with native
`stop`. A combined serving attempt subsequently failed at `RuntimeGate` with
checker return code 1. Lifecycle restoration was
`process_stable`, but the harness lost the raw checker diagnostics, so the
specific failed checks are unknown. This is not a serving pass. Export reload,
vision/MTP functionality, replay and trajectory metrics remain unvalidated.
This is paired export evidence only: no serving or vision/MTP functionality,
replay or repetition result, GPQA score, TP2 result, or release/fix claim is
established here.

A subsequent reviewed recovery completed with `observation_passed` after
121.684 seconds of stable-process observation. Its fixed arithmetic API smoke
passed in 0.221 seconds with answer `2` and native `stop`. The pinned-image
dependency preflight then passed all five imports with writable, stage-matching
cache configuration. These checks establish service recovery and dependency
availability only; at that point model loading and MTP functionality were still
pending. The scoped serving result below supersedes that point-in-time status.

## Pre-cache-fix attempts

Before the writable-cache fix, the first combined check did not complete. The `static-max` baseline checker
returned 1: arithmetic, red-square and blue-square cases passed; Python code
produced nonempty output with finite logprobs but reached 192 tokens and ended
with `length`, which establishes neither code correctness nor natural stop.
OCR expected `3729` and returned `3720` with `stop`. The current response hash
is recorded for this incorrect `3720` output; no historical response-hash
comparison was performed. The OCR image hash and Pillow 12.3.0 match the
recorded historical inputs. This rules out image byte drift for this attempt but does not identify
the OCR cause. The `torchcodec` log notice did not prevent
server readiness or the image request; zero loader skip/unknown/unloaded
counters do not establish vision or MTP functionality.

At that point, the first MTP startup failure was a read-only-filesystem `OSError`; errno and
stack were not captured. The `static-MSE` arm and replay did not run. Both
workers used the same pinned image. Process restoration was stable; a later
0.217-second API recovery check passed with answer `2` and native `stop`. This attempt does
not establish current vision/MTP success or an MSE fix; earlier successful
checks remain historical evidence for their own cases.

### Repeat attempt

A second run with the same pinned image, weights and parameters reproduced the
same five `static-max` baseline outputs by token IDs and response hashes,
including the OCR response `3720` for expected `3729`. It again failed MTP
startup before requests, this time with errno 30 and a stack through
`eagle_worker_v2._configure_qsa_mtp_index_share`, `__new__`, `_create_dirs` and
`os.makedirs`. A pinned-image source audit then identified the cause: importing
`QSAIndexer` happens before the MTP flag check; its `qsa/mqa.py` import of
TileLang initializes `KernelCache`, whose directory creation uses the default
`~/.tilelang/cache` (`/root/.tilelang/cache` in the container). The diagnostic
failed-directory hash matched the bytes of `/root/.tilelang`, supporting this
path identification. A separate CPU-only import smoke in the pinned image
reproduced errno 30 with TileLang cache variables unset, then successfully
imported TileLang with `TILELANG_CACHE_DIR=/tmp/tilelang` and
`TILELANG_TMP_DIR=/tmp/tilelang/tmp` under a read-only container root. This
confirms the cache-directory cause for TileLang import. The proposed service
override was subsequently exercised in the scoped serving check below. This
identifies the MTP
startup read-only-filesystem cause only; OCR and MSE
behavior remain unresolved. MSE and replay did not run in these two pre-cache-fix
attempts. GLM process restoration was stable and the fixed API check passed; the
later writable-cache serving result follows.

A short max/MSE replay completed as a bounded diagnostic: three 512-token
requests per arm all finished by length, and outputs were identical within each
arm. The arms shared only a two-token prefix; no short-period candidate was
detected. Prompts had 288 tokens, zero cached tokens and zero retractions; flush
acknowledgements passed. `spec_verify_ct` was absent, so cold-cache verification
was not established. This does not prove repetition is fixed or exclude a later
long-context/periodic loop.

## Writable-cache MTP serving check

With TileLang cache and temporary paths redirected to writable storage, both
`static-max` and `static-MSE` completed baseline and MTP modes for all five
cases. MTP verification and acceptance ran for every case, and baseline/MTP
output IDs matched within each arm. `static-MSE` passed all five checks in both
modes, including OCR `3729`; max returned `3720` instead of `3729` in both modes.
Python remained a 192-token nonempty finite output ending at `length`, so code
correctness and natural stop are unproven. The `torchcodec` notice was nonfatal;
vision/MTP skip, unknown and unloaded counters were zero, which does not prove
numerical parity. The paired multimodal gate remains false; repetition remains
unresolved. The separate [cache configuration PR #73](https://github.com/AvrovaDonz2026/Mega-Quantification/pull/73)
proposes writable TileLang cache variables and temporary storage. Compose 5.6.1
parsing passed locally, but actual build and service start remain unverified.
Eighteen packaging tests pass. Recovery API arithmetic passed in 0.214 seconds
with answer `2` and native `stop`, with both services running and no restart/OOM
observed.
