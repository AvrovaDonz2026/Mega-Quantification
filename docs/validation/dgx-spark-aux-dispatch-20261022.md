# Auxiliary Dispatch Probe

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

No patched GPU run, repetition fix, GPQA result, serving result, or TP2 result is
claimed by this report.
