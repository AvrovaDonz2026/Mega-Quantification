# Full static-max export validation: 2026-10-23

## Status

The full static-max single-arm export completed its export and recovery gates,
but the overall protocol remains advisory: a post-close heartbeat watchdog
reported a late heartbeat failure after the worker had closed. This does not
invalidate the read-back export evidence, but it prevents calling the whole
protocol passed until that watchdog event is adjudicated. The max journal's
primary failure is `heartbeat_failed` with no recovery failures; recovery
success is a separate observation and is not adjudication success.

The static-MSE v1 arm was staged and launched separately with the same frozen
payload, but active execution terminated on an SSH transport EOF/heartbeat
expiry before producing an export. Its independent read-back confirmed
restoration, running services, no OOM/restarts, and a passing service API smoke.
This is an execution/lifecycle failure, not a model or quantization diagnosis.
No paired max/MSE quality or repetition
conclusion is made here.

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
The MSE read-back likewise recorded primary failure `heartbeat_failed`, an empty
recovery-failures list, successful restoration, and no generated export. A later
strict adjudication is still required; neither arm's recovery observation makes
the overall protocol passed.

## Scope

This is one completed static-max export arm plus a separately failed static-MSE
execution. It is not a matched max/MSE result, does not prove the MSE hypothesis
or a repetition fix, and does not establish a production serving pass or
release readiness.
