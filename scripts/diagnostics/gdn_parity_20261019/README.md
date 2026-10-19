# GDN parity reproduction bundle

This directory is a small, reproducible copy of the diagnostic runners used
for the 2026-10-19 AITOP validation. It targets NVIDIA GB10 / SM121 and keeps
the source identity separate from the large source archives and model files.
The subdirectories contain the parity-v7 runner, beta regression,
experiment driver, candidate text/vision/MTP matrix, same-prefix prefill logits
probe, and the synthetic NVFP4 shape probe. The runners retain the completed experiments' protocol and add
an enforced source lock so future runs reject unlisted commits or archives.

Run commands from this bundle directory. The paths beginning with
`/path/to/` in `example-config.json` are adjustable examples for the model
directories, provenance file, and fixture repository; replace them with paths
on the machine where the experiment runs. The image and image ID are pinned to
the validated digest unless the target environment has a separately validated
image.

```bash
cd scripts/diagnostics/gdn_parity_20261019
python3 build_sources.py --repo /Users/zxy/sglang --output source-package
python3 experiment-driver/driver.py --action launch --phase all \
  --config example-config.json --root /fresh/evidence
```

`/Users/zxy/sglang` and `/path/to/...` are local-path examples only; the
builder does not fetch a remote. The evidence `--root` must be a new path
outside this bundle, and in particular must not be inside `parity-v7`, because
the parity runner rejects roots that could overlap its own files. Use
`python3 experiment-driver/driver.py --action status --root /fresh/evidence`
to inspect a running or completed root.

## Pinned source

The four source variants are the real commits recorded in `source-lock.json`:

| variant | commit |
| --- | --- |
| original | `4a667614d810f7b5570be23eb75956c603a85c3a` |
| decode-beta | `a3b5d8376f4206ce78d11d31f4a7f4a8d1ffb568` |
| all-beta | `49d3acfd4d54b0cf1afffcfeb369c63af4f56499` |
| shared-decode | `c7c3f48f4a8aa9e3cf28a87da26c5eb8c1017116` |

The source lock points at a local `source-package/summary.json`. Build the
archives and compact, compatible manifests from a worktree that contains
those commits:

```bash
python3 build_sources.py --repo /Users/zxy/sglang --output source-package
```

`build_sources.py` uses only the Python standard library and `git archive`.
Each archive uses the exact `sglang-<variant>/` prefix. It refuses to pass if
the revision is not a commit or if the generated archive SHA differs from
`source-lock.json`; `summary.json` then records the same SHA and commit for
the experiment driver. Use `--repo` to point at another local clone containing
the pinned objects. No network or remote fetch is performed.

Copy or adapt `example-config.json` after building `source-package`; supply
the model paths, provenance files, fixture repository, and serving image for
the target machine. The existing runners require the source archive and its
matching manifest for every parity job.

## Scope and limits

This round completed the W4A4 `python_code` whole-model parity for the
all-beta source: baseline and MTP both produced 277 tokens and passed the
strict parity gates. The full W4A8 visual/OCR/MTP matrix is still pending; see
the [validation record](../../../docs/validation/dgx-spark-gdn-parity-20261019.md)
for the exact scope and evidence. Synthetic NVFP4 shape results are diagnostic
signals and do not replace GPQA or a full-model visual evaluation. Reports
should retain only the sanitized scalar outputs produced by the runners; do
not add raw prompts, token data, tensors, or model files to this bundle.
