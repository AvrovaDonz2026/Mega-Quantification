# GDN beta regression

`probe.py` is a scalar-only GPU regression for the packed GDN decode and the
fused GDN gating prefill path. It uses one copied fixture for every variant so
the negative controls cannot pass from fixture drift.

The decode cases cover a 32-token chain at GQA `(4, 8)` and `(16, 48)`, with
one-hot `isolated-beta` and `random` Q/K inputs. The state pool is FP32 and
envelope-strided; guard rows and verify source are checked after the run. The
prefill case feeds BF16 beta logits whose sigmoid values are not BF16-exact and
compares `fused_gdn_gating` with the explicit FP32 expression.

Expected behavior with `FLA_USE_FAST_OPS=0` is derived from the requested
variant and checked against source expressions as well as observed counts:

* `original`: decode mismatch, prefill mismatch.
* `decode-beta`: decode match, prefill mismatch.
* `all-beta`: decode match, prefill match.

Fast ops mode is run as an observational matrix (`validation_gate: false`) so
its different exponential implementation cannot be mistaken for the exact
FP32 acceptance gate.

The prefill reference is a local Triton kernel using the same `tl.sigmoid` and
softplus expression as the implementation. Beta must match exactly in FP32;
the gate `g` must be finite and within `1e-5`, avoiding false failures from
Torch/Triton math differences.

Run from this artifact directory or point at a source worktree:

```bash
SGLANG_ROOT=/path/to/sglang-gdn-20261019 ./run_regression.sh all-beta 0 report.json BASE_REVISION
```

`BASE_REVISION` is optional metadata. A source archive without `.git` is
accepted; when omitted, a local git revision is recorded if available and is
never used as a validation gate.

Reports contain only scalar counts, maxima, runtime/source hashes, and guards;
they do not include tensors, prompts, token IDs, or raw token data.
