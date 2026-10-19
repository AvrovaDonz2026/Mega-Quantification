# NVFP4 row shape probe

`probe.py` is a synthetic, read-only probe for the Qwen3.8 ModelOpt NVFP4
dense-linear path on DGX Spark/GB10 (SM121). It uses one fixed BF16 input row
and one fixed BF16 weight matrix per `KxN` shape, repeats the input row at
`M=1,2,3,4,8`, quantizes through `flashinfer.fp4_quantize`, and calls
`flashinfer.mm_fp4` with the same packed/transposed weight and scale layout as
`ModelOptFp4LinearMethod.apply`.

The default shapes are the first non-GDN projections to check:

```text
5120x12288  (GDN in_proj_qkvz, logical TP1 shape)
5120x64     (GDN in_proj_ba, logical TP1 shape)
4096x5120   (GDN out_proj, logical TP1 shape)
```

Run on the GPU environment that contains the repository's FlashInfer install:

```bash
python probe.py
python probe.py --shapes 5120x12288 --rows 1,3 --backend cutlass
python probe.py --backend cudnn
```

Output is one compact JSON object. It includes tensor shapes/strides, finite
and scalar summaries, SHA256 hashes, first-row equality against M=1, and exact
row-repeat checks. It intentionally emits no raw tensor values. On SM121 the
reported `qwen3x_custom_runtime_gate` must be false: the SGLang Qwen3.x custom
gate requires `torch.cuda.get_device_capability() == (12, 0)`. Therefore this
probe is explicitly the FlashInfer path, even for M=1.

Interpretation: `input_first_scale_exact_m1` describes the physically swizzled
first scale row, so it is not a value that can be compared across M. A changed
input scale hash or that boolean at M>1 is therefore not model evidence. The
round's evidence is the output comparison: a nonzero
`first_row_max_abs_vs_m1`, changed output first-row hash, or failed
`first_row_exact_m1` indicates an M-dependent FlashInfer quantization/GEMM
schedule for these synthetic inputs. It is useful for prioritizing a
full-model probe, but cannot establish the first divergent Qwen3.8 activation
by itself.
