# Full static-max export validation: 2026-10-23

## Status

The full static-max single-arm export completed its export and recovery gates,
but the overall protocol remains advisory: a post-close heartbeat watchdog
reported a late heartbeat failure after the worker had closed. This does not
invalidate the read-back export evidence. Strict-bound successor adjudication
passed the artifact-usability check for the matched experiment; the original
lifecycle protocol remains failed/advisory. The max journal's primary failure
is `heartbeat_failed` with no recovery failures. Recovery success is a separate
observation and does not make the overall protocol pass.

The static-MSE v1 arm was staged and launched separately with the same frozen
payload, but active execution terminated on an SSH transport EOF/heartbeat
expiry before producing an export. Its independent read-back confirmed
restoration, running services, no OOM/restarts, and a passing service API smoke.
This is an execution/lifecycle failure, not a model or quantization diagnosis.
No paired max/MSE quality or repetition
conclusion is made here.

A separate static-MSE retry v3 passed staging but failed before service mutation,
during worker installation: a deployment-unit TTL hard gate compared legacy
`30600` seconds with the planned `9000` seconds. The mismatch was independently
reproduced. No quantization or export was launched, so this is an installation
gate failure, not a reproduction failure of the quantizer. The original v1
active-execution failure above remains a separate historical attempt. Retry v4
job `0fc25e19d0c9bcdeae7c7afc` began execution at 20:51:25 CST and failed with
`transport_timeout` at 20:57:06 CST, before producing an export. Coordinator
recovery was stable. Additive V7 independent readback later passed recovery:
both services were running with zero restarts and no OOM, the experiment was
absent, journals were closed/unarmed with no primary failure, and the service API
smoke passed. This recovery readback does not change the v4 execution failure or
create an export; it remains an execution/lifecycle failure, not a quantizer
reproduction result. Earlier V4/V5/V6 readback failures were verifier defects.

## No-GPU transport diagnostic

A separate 420-second no-GPU echo probe completed successfully on both peers:
357/357 and 356/356 messages were received with no natural failure and the
processes remained alive before intentional teardown. The post-teardown exit
code 1 and 37-byte stderr digest matched the historical post-close diagnostic;
because both were captured after teardown, they do not establish the original
v4 SSH root cause. Separate idle-container inspection sampled 118 observations
per peer with maximum Docker inspection durations of 49.461 ms and 13.287 ms, with no
errors. This probe does not prove transport stability under PTQ load or provide
a quantization/export result.

## Static-MSE v6 staging checkpoint

Static-MSE v6 job `d4f368ca832d87e3ce0d4a78` passed dual staging with both
services unchanged and source, calibration, and pinned image identities
verified. The focused preflight/test gate passed 26 tests; planned arm,
experiment, and worker timeouts were 1,800, 2,400, and 3,000 seconds. No GPU
quantization or export started because the certificate expired before execution.
Against the frozen max plan, 39 of 43 payload files are byte-identical. The four
differing files are the execution descriptor, maintenance worker, deployment
worker, and runtime validator. The descriptor carries the intended max/MSE arm
selection and timeout; the other differences are operational. This is staging
evidence only.

A separate tiny public-codec hardware probe failed before codec coverage. It did
not complete candidate audit coverage: the probe used a stand-in quantizer that
the vendor classifier did not recognize, then read a nonexistent `.data`
attribute instead of `_quantized_data`. V2 uses real vendor quantizers and the
correct accessor, but has not executed on a GPU. Neither result establishes
full-export correctness or a repetition fix.

## Synthetic rounding diagnostic

An additive CPU diagnostic compares the frozen candidate's FP32 division path
with an independent FP64 nearest-distance reference, checked against exact
rational distances. Each input dtype uses six scales and 288 synthetic values
at quantization thresholds, their representable neighbours, signed zero, and
tiny values. FP16 and BF16 produced no code differences; FP32 produced 12 code
differences, corresponding to six differing packed bytes. The result repeated
with the same aggregate hash. Its focused tests passed 17 cases, with the CUDA
case skipped.

This shows sensitivity to rounding at constructed boundaries. It does not show
that actual model weights differ, that the Triton and PyTorch CUDA paths differ,
or that rounding causes the long-generation repetition. Those hardware and
full-model comparisons remain pending.

## CPU readiness follow-up: 2026-10-24

The pinned ModelOpt CPU classification follow-up validated the real vendor
`TensorQuantizer` path for `nvfp4`, static-quantizer classification, and the
quantized-data accessor. V2 retained its original byte identity
`f0cbcbac5958ce1b21b8b2f2677d3ca0dbb667aa041369c6af9c64845055fe46` and passed
5 tests with 1 CUDA skip. V3 fixed bounded exception handling and passed 6 tests
with 1 CUDA skip; its probe SHA-256 is
`12d5c2730df9ca183e3bd6775a723a4a7e5cafa64479d3a874554d636ebb352b`.
Ruff and driver plan/compile checks passed. The V1 plus V2 rounding suite
totals 35 passed and 2 CUDA skipped; its additive V2 source SHA-256 is
`709c8722cd84d8d9b058ff7ccd33c097f3194f435da3db220ceaf3565cb35dbf`.

These are CPU and source-readiness results only. The default Torch codec and
frozen Triton candidate have not been shown arithmetically identical on GPU;
the FP32 12-code/6-packed-byte difference remains a CPU boundary observation,
not a bug or root-cause proof. GPU, full-model MSE, visual/MTP, and matched-long
validation remain pending. The current certificate is expired and no new GPU
execution or GLM pause was performed.

## CPU checkpoint persistence follow-up

A static-state CPU probe (SHA-256
`1b779c159015c499fbedc8c2e5823bab29be5d710fffd223eb1f43d511b3b207`) used the
pinned ModelOpt runtime and repository static transform with block-16 E4M3
scale metadata, including weight per-block amax shape `[64,1]` and global
scalar amax state. It saved a 9,895-byte checkpoint, restored it in two fresh
subprocesses, and matched the full four-entry state fingerprint exactly
(`0654d60134263c9a88521c85bed9a68305c21e05430ef04098032df16d5fc874`,
`3 passed`). The temporary
checkpoint was removed. This establishes persistence of the same calibrated
static state only; it does not establish packed-export parity, GPU behavior,
offload behavior, full-27B memory/performance, or serving quality.

## CPU reference-grid follow-up

A pinned-vendor CPU comparison used the same FP32 quotient inputs across 720
values per dtype. Against the frozen candidate, FP16 and BF16 had zero code
differences; FP32 had 9 raw-code differences, all negative tiny values that
underflowed to negative zero, with zero magnitude differences and unchanged
decoded values. The exact-real Fraction oracle comparison produced 3 code/2
packed-byte differences for FP16, 3/2 for BF16, and 25/24 for FP32 (31 code
and 28 packed-byte differences in total). These are constructed CPU reference
boundaries, not evidence of a codec defect, GPU equivalence, real-weight
parity, or a repetition fix. The comparison remains pending GPU execution and
full-model validation.

## Export evidence

The static-max arm verified all 519 expected static-NVFP4 targets with 519
candidate calls and exact candidate/oracle results. The candidate and oracle
used separate implementations and call paths, but both use Triton's default
division and the same E2M1 thresholds, sign handling, and packing. This is not
a live ModelOpt fake-quant comparison or an independent arithmetic-equivalence
proof. Scales were unchanged.
Vision retention was 333/333 entries across 110 quantizers, and MTP retention
was 15/15 entries across 8 quantizers. The export index contained 2,756 keys
across two shards and 16 export files, totaling 17,499,733,224 bytes. Header
and key mapping checks passed, and the retained export tree is identified by
SHA-256 `63ca89e911cabe4e04106ce7173f21853fdda93cc73ddbfac3cef22ba225a914`.
Evidence summary SHA-256 is
`8b0cf011d818f8f270f7a058fc84990db47432eba0fd0a02a537e2f9fd8207ea`.

These are export-contract and artifact-integrity results. Reload, serving,
vision/MTP quality, full-model numeric parity, and natural-generation behavior
are separate gates.

## Recovery

Independent read-back found both original GLM services restored and stable:
both were running with restart count zero and no OOM, the experiment container
was absent, and the service arithmetic smoke returned `2` with `stop`. Recovery
passed despite the late watchdog advisory.
The MSE v1 read-back recorded primary failure `heartbeat_failed`, an empty
recovery-failures list, successful restoration, and no generated export. MSE v4
V7 recovery passed additively after the failed execution; it does not rewrite the
failed run as successful.
The max artifact adjudication does not change either MSE status.

## Scope

This is one completed static-max export arm plus a separately failed static-MSE
execution. It is not a matched max/MSE result, does not prove the MSE hypothesis
or a repetition fix, and does not establish a production serving pass or
release readiness.
