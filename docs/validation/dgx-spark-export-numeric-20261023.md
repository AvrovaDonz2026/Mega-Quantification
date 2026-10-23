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

## Export evidence

The static-max arm verified all 519 expected static-NVFP4 targets with 519
candidate calls and exact candidate/oracle results. Scales were unchanged.
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
