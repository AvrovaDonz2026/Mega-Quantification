# SGLang upstream patch validation — 2026-10-13

Two contributions extract runtime fixes from the MegaQuant integration into
upstream SGLang. Their validation uses public SGLang source at the exact commits
below. These runs are separate from the frozen MegaQuant containers used for
the [Spark GPQA results](README.md#完整-spark-gpqa-diamond).

| Contribution | Tested source | Current scope |
|---|---|---|
| [PR #43990](https://github.com/sgl-project/sglang/pull/43990): host-list logprob compatibility | [`60bd318b3be0ba1cb18192bdae92c64ba2893c32`](https://github.com/sgl-project/sglang/commit/60bd318b3be0ba1cb18192bdae92c64ba2893c32) | CPU regression, direct registered-test CLI, and full contribution formatting checks passed; upstream review and CI remain pending. |
| [Draft PR #44032](https://github.com/sgl-project/sglang/pull/44032): explicit Qwen vision/MTP ModelOpt NVFP4 metadata | [`f22670dcca213a0718f3950943e0442971f3bb33`](https://github.com/sgl-project/sglang/commit/f22670dcca213a0718f3950943e0442971f3bb33) | CPU regressions, direct registered-test CLI, full contribution formatting checks, and a synthetic NVFP4 linear regression on a real GB10 GPU passed. Full-model validation remains pending. |

## Contribution checks

The [contribution guide](https://github.com/sgl-project/sglang/blob/7ff3a112846a3fcc12a27bba09d644885a226e4d/docs/docs/developer_guide/contribution_guide.mdx),
[quantization guide](https://github.com/sgl-project/sglang/blob/7ff3a112846a3fcc12a27bba09d644885a226e4d/docs/docs/developer_guide/quantization_contribution_guide.mdx),
registered-test instructions, maintainer guide, and PR template were reviewed.
Their validation requirements
include full `pre-commit run --all-files`, executable registered tests, changed
line coverage, and representative model accuracy and performance checks for
quantization changes.

`pre-commit run --all-files` passed on disposable source snapshots of both exact
commits: exit 0, 30 hooks passed, one path-specific hook skipped, and no tracked
file changes. This includes the registered-test checks and the Rust formatting
and clippy hooks. Changed-file checks also passed. Homebrew's `rustup` and
isolated Rust toolchains resolved the earlier missing-toolchain limitation. Optional manual-stage
documentation link hooks were outside this command's scope.

## CPU regressions and registered-test CLI

At `60bd318b3be0ba1cb18192bdae92c64ba2893c32`,
`test/registered/unit/layers/test_logprob_chunk_stitching.py` passed with
**5 tests and 17 warnings** in 14.78 seconds. The ARM64 validation container
exposed no GPU. Direct `python testfile.py -f` execution also passed all five
tests. Running the same regression against pinned upstream baseline
`7ff3a112846a3fcc12a27bba09d644885a226e4d` failed as expected at
`batch_result_processor.py:1125`: the host list has no `.tolist()` method.

The vision/MTP CPU suites ran on
[`0042e3d8d29ed3399123cc79e60e2349b589f783`](https://github.com/sgl-project/sglang/commit/0042e3d8d29ed3399123cc79e60e2349b589f783).
SHA-256 comparison established that all four changed production files and all
three CPU test files are byte-identical to final `f22670d`; the final commit adds
the GPU regression. These CPU results are attributed through that source
identity comparison, rather than described as a new final-commit pytest run.

| CPU test file | Pytest result | Direct `python testfile.py -f` |
|---|---|---|
| `test_qwen3_5_mtp_quant_config.py` | 13 passed, 18 warnings | 13 passed |
| `test_modelopt_nvfp4.py` | 18 passed, 26 subtests, 17 warnings | 18 passed |
| `test_model_config.py` | 12 passed, 8 subtests, 17 warnings | 12 passed |

The three suites total **43 tests and 34 subtests**, with no failures or skips.
The new tests also detect the corresponding faults in pinned upstream baseline
`7ff3a112846a3fcc12a27bba09d644885a226e4d`: visual alias resolution returns no
NVFP4 configuration, the quantized MTP `fc` selects a plain `Linear` instead of
`ReplicatedLinear`, and draft depth remains 8 instead of being normalized to 1.

Coverage tools were installed only in disposable validation containers. The
vision/MTP coverage run passed **43 tests and 34 subtests**, with 18 warnings;
the logprob coverage run passed five tests, with 17 warnings. `diff-cover`
measured changed production lines against the same pinned upstream baseline:

| Contribution | Changed production-line coverage | Changed lines | Uncovered lines |
|---|---:|---:|---:|
| Logprob, `60bd318b` | 100% | 2 | 0 |
| Vision/MTP, tested `0042e3d8` with final-source file identity | 87% | 81 | 10 |

Both checks exited 0 with `--fail-under=60`, satisfying the registered unit
test guide's changed-line coverage threshold. These percentages describe the
changed production lines in the CPU test scope; they do not measure model
accuracy or full GPU execution coverage.

## Real GPU NVFP4 regression

The final public source `f22670dcca213a0718f3950943e0442971f3bb33` ran
`test/registered/kernels/ops/quantization/test_modelopt_nvfp4_vision.py -f`
in a fresh GPU container. The test passed: **1 test in 1.770 seconds**, exit 0,
with no skips or accelerator mocks.

| Runtime | Recorded value |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1 |
| Driver | 580.173.02 |
| Python / PyTorch | 3.12.3 / 2.13.0+cu130 |
| CUDA / FlashInfer | 13.0 / 0.6.18 |
| Container digest | `sha256:20f41d6ea8897ac17a3a2c43edbcc1ff051da71607297a0d107d24f9715bacd0` |
| Quantization / GEMM | Serialized ModelOpt NVFP4 W4A4 / `flashinfer_cutlass` |

The test loads packed synthetic checkpoint weights through the production
weight loader and postprocessing path, checks a two-dimensional numerical
reference, and checks contiguous and noncontiguous `[3, 2, 256]` activations
against the same matrix input. It also checks prequantized tuple output. The
dequantized reference tolerance is 0.05; reshaped results match the matrix
results exactly.

The same final regression test was then run against upstream baseline
[`7ff3a112846a3fcc12a27bba09d644885a226e4d`](https://github.com/sgl-project/sglang/commit/7ff3a112846a3fcc12a27bba09d644885a226e4d)
in a second fresh GPU container. Its matrix checks passed, but the first
multidimensional input failed at `x_m, _ = x.shape` with
`ValueError: too many values to unpack (expected 2)`. This establishes that the
regression detects the original shape fault and that the final patch handles
the tested inputs.

## Remaining scope

The GPU result establishes synthetic linear/kernel integration. It does not
establish full-model `/generate` behavior, visual or MTP accuracy, warmed-up
inference speed, TP2 operation, or recovery from repetitive generation. The
vision/MTP PR remains a draft while those representative model checks are
pending. Prior MegaQuant fixed visual/MTP checks and GPQA scores retain their
original runtime identities and are not transferred to these new upstream
commits.
