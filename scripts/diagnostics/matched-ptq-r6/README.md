# Matched PTQ R6 offline harness

This directory contains an offline planning, execution-supervision, evidence,
and comparison harness for a matched `static-max` / `static-MSE` PTQ run. The
bootstrap planner reads local inputs and emits a deterministic package and
execution-disabled container plan. The lifecycle code validates that plan and
uses a bounded, owned container workflow when explicitly driven by a reviewed
caller. The harness does not provide a remote deployment command.

The fixed container mount destinations are `/workspace`, `/model`,
`/calibration`, `/output`, and `/evidence`. The model and calibration mounts are
read-only. The image reference is pinned by digest. Container defaults disable
network access and enable offline Hugging Face settings. Planning and tests do
not launch a container or contact a deployment host.

PTQ evidence and replay evidence are separate gates. PTQ comparison checks
whether both exports completed with matching source, model, calibration,
configuration, and quantization scope. Replay adds same-prompt native response
and trajectory checks. Neither gate establishes production serving readiness;
the comparison reports `serving_ready: false`.

The prior MSE discrepancy has no established algorithmic root cause. Existing
exports had mismatched auxiliary vision/MTP quantization coverage, which is a
known comparison confounder. A controlled matched export is needed to attribute
quality differences to the quantization algorithm. This directory contains no
GPU completion evidence and makes no claim that the matched run completed.

Run the offline suite from the repository root with:

```sh
pytest -q tests/test_matched_ptq_r6.py
```

That wrapper invokes pytest in a subprocess scoped to this directory, so its
test modules cannot collide with similarly named modules elsewhere in the
repository.
