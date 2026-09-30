# DGX Spark vision and MTP serving validation - 2026-09-30

The full Qwen3.8-27B mixed W4A8 and uniform W4A4 exports passed text, image and
embedded MTP inference on one DGX Spark with FP32 SSM state. Each format's five
greedy MTP responses exactly matched its baseline `output_ids`, including three
image requests. The original BF16 SSM W4A4 run had a reproducible text mismatch.
The weights are the exports from the [2026-09-29 quantization validation](dgx-spark-pr46-20260929.md).

The [machine-readable record](dgx-spark-serving-20260930.json) contains curated
results and SHA256 hashes of the retained request/response records and logs.

## Environment and fix

| Component | Observed value |
|---|---|
| GPU / parallelism | NVIDIA GB10, ARM64, SM121; one GPU, TP1 |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| Transformers | 5.12.1 |
| SGLang / sglang-kernel | 0.5.20 / 0.4.7 |
| FlashInfer | 0.6.18 |
| Tested serving image | `sha256:c0c1992d72102ef74b7d3894a657b0a717c3e6576751a88a76e12b31651f52c9` |
| Reused CUDA base image | `sha256:4cd0f8c8e31730b24d7949166b0f2b6516e69b2d4435388cbe841222cd60970c` |
| Loader patch | `megaquant-spark-v2`, revision `5fae056242aaf9399cba2b8a83294ea975c0b311` |
| FP32 recipe / public client revision | `41969f150c7b2a91139a8b2c6096196218543b5a` |

Direct and mirrored pulls of the pinned official serving image failed with
network timeouts or interrupted layer downloads. The isolated test image reused
the existing CUDA base and installed the pinned SGLang wheel and matching kernel
packages. This validates that derived image; a fresh build from the pinned
official image was not completed.

That image embeds the v2 loader patch and the original BF16 SSM recipe. W4A4's
FP32 comparison used an external recipe override. W4A8's final FP32 run mounted
the exact updated repository recipe from revision `41969f1` into the same image
and used the normal recipe-to-serving-argument path.

Before the fix, image warmup reached a rank-three vision activation and crashed
in `ModelOptFp4LinearMethod.apply` with
`ValueError: too many values to unpack (expected 2)`. Patch v2 flattens leading
activation dimensions before the matrix-only NVFP4 path and restores them after
the projection. Two-dimensional text inputs and prequantized tuples keep their
original path. The rebuilt image passed image warmup and **36/36** ARM64 patch
regressions, covering shape restoration, noncontiguous inputs, bias, tuple
preservation and rejection of an older patch manifest.

## Protocol

Serving used `recipes/eval-gpqa-diamond.spark.yaml`: 32,768-token context,
FlashInfer attention, Triton GDN, FP8 E4M3 KV cache, memory fraction 0.70, no host
HiCache and no CUDA graphs. Requests ran sequentially through native `/generate`
with temperature 0, thinking disabled in the tokenizer chat template,
`return_logprob: true` and a 300-second timeout. Baseline and MTP used separate
server starts with the same weights and request payloads. The original runs used
BF16 SSM state; the FP32 comparison changed only `mamba_ssm_dtype`.

The image fixtures were generated with Pillow: red and blue squares on identical
320 x 320 white backgrounds, and black digits `3729` on a 640 x 320 white
background. Both square requests used the same question. Correctly changing the
answer from `red` to `blue` checks that the response depends on the image.

MTP used the checkpoint's quantized embedded head with `EAGLE`, two speculative
steps, top-k 1 and three draft tokens. No external draft checkpoint was supplied.

## Mixed W4A8 with BF16 SSM state

| Request | Baseline and MTP result | Output token IDs | Accepted / proposed drafts |
|---|---|---|---|
| Arithmetic `17 * 23` | `391` | Exact match, 4 tokens | 3 / 4 |
| Python generation | Nonempty; finite token logprobs | Exact match, 192 tokens | 127 / 132 |
| Red square | `red` | Exact match, 2 tokens | 2 / 2 |
| Blue square | `blue` | Exact match, 2 tokens | 2 / 2 |
| Digit OCR | `3729` | Exact match, 5 tokens | 4 / 4 |

All five cases passed with finite output logprobs. MTP executed for every case:
72 verification steps accepted 138 of 144 proposed draft tokens (**95.83%**).
The longer Python request accepted 127 of 132 (**96.21%**) with an average
acceptance length of 2.909, which includes the verifier's bonus token.

## Uniform W4A4 with BF16 SSM state

All five baseline and MTP requests returned nonempty output with finite token
logprobs. Arithmetic, both color requests and OCR produced their expected answers
and exactly matched their baseline token IDs. MTP accepted 137 of 144 proposed
draft tokens (**95.14%**), including 126 of 132 (**95.45%**) in the Python request.

The Python response differed at output token index 41 (the 42nd token): baseline
token 264 began `a, b = ...`, while MTP token 413 began `if n == 0: ...`. Both
responses reached the 192-token cap. Repeating the MTP request before and after an
explicit cache flush reproduced the same mismatch; both repeats reported zero
cached tokens. Therefore the BF16 SSM W4A4 run **failed
strict token-equivalence validation**, despite passing the image and arithmetic
checks and executing MTP. Its client record has `validation_passed: false` and
the client exited with status 1.

## Uniform W4A4 with FP32 SSM state

Changing only `serve.mamba_ssm_dtype` to `float32` restored exact baseline/MTP
equality for all five cases and all 205 output tokens. The FP32 baseline itself
matched the original BF16 baseline's Python token IDs. MTP repeated before and
after an explicit cache flush also matched both its first response and the
baseline. Acceptance remained 137/144 overall and 126/132 for Python.

This is consistent with a rounding difference in the Triton GDN paths: decode
writes BF16 SSM state after each step, while verification retains FP32 state
within its multi-token loop. It is not a guarantee for every prompt; packed
decode and verification also compute the beta sigmoid at different precision.

The Spark recipe now selects FP32 SSM state. Each cached SSM element occupies
four bytes rather than BF16's two. This changes state-cache precision; W4A4
weights and quantized projection activations remain NVFP4. The related recipe and
serve-argument regressions passed **53/53** on ARM64 Python 3.12.

## Final FP32 recipe check

| Export | Baseline / MTP checks | Exact output IDs | Accepted / proposed drafts | Python acceptance |
|---|---|---|---|---|
| Mixed W4A8 | 5 / 5 each | 5 cases, 205 tokens | 138 / 144 (95.83%) | 127 / 132 (96.21%) |
| Uniform W4A4 | 5 / 5 each | 5 cases, 205 tokens | 137 / 144 (95.14%) | 126 / 132 (95.45%) |

Both strict client validations passed. W4A8's final run verifies the new FP32
recipe default with the normal loader; the W4A4 comparison used the equivalent
FP32 setting through its external recipe override.

## Repeat the checks

Run the client inside the serving environment or another environment with the
same tokenizer and Pillow available. Set `MODEL_DIR` to the selected export,
start the baseline server with the Spark recipe, then run:

```bash
python3 scripts/check_sglang_multimodal.py \
  --model "$MODEL_DIR" --base-url http://127.0.0.1:30000 \
  --mode baseline --output baseline.json
```

Restart that server with these fields enabled under the recipe's `serve` section:

```yaml
speculative_algorithm: EAGLE
speculative_num_steps: 2
speculative_eagle_topk: 1
speculative_num_draft_tokens: 3
```

```bash
python3 scripts/check_sglang_multimodal.py \
  --model "$MODEL_DIR" --base-url http://127.0.0.1:30000 \
  --mode mtp --baseline baseline.json --output mtp.json
```

The public client exits successfully only when `validation_passed` is true.
For MTP, that requires functional checks, exact baseline token IDs and at least
one accepted draft token. Inspect `color_counterfactual_passed`,
`greedy_outputs_identical` and the per-request `spec_metrics` for the details.
The earlier W4A8 records predate the combined `validation_passed` field; their
individual fields satisfy the same gate.

## Limits

This is a small functional test. Python generation hit the 192-token limit;
generated code was not executed or graded. No GPQA score, general vision/OCR
accuracy, long-context behavior, concurrent throughput, video or cross-node TP2
result is established. Recorded single-request timings are not a controlled
performance benchmark. Token equality is between baseline and MTP for the same
quantized export, not a comparison with BF16 or between quantization formats.
