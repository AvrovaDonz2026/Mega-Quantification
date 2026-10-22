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

No completed patched full-27B GPU export, repetition fix, GPQA result, serving
result, or TP2 result is claimed by this report.
