# Candidate matrix

`matrix_candidate.py` is an independent detached driver for the all-beta
candidate matrix. It leaves the existing experiment-driver and active
evidence roots untouched. Its default child, `matrix_runner.py`, imports the
adjacent `parity-v7` gate/server/cleanup helpers. Set `GDN_PARITY_RUNNER_DIR`
only when using a separately pinned copy of those helpers.

The matrix runs two scheme jobs and ten scheme/case comparisons in sequence:

- W4A4 and W4A8
- `python_code`, `arithmetic`, `red_square`, `blue_square`, `ocr_3729`
- two baseline and two MTP repetitions per case (40 requests total)

Each scheme starts one baseline and one MTP server, then runs all five cases
twice per side. The candidate client flushes cache before every request and
checks expected text in memory. The runner performs the pinned image, GPU
idle, exact source archive/manifest, model identity, runtime imported source
hash, container ownership, and cleanup gates. Reports contain only hashes,
lengths, stop reasons, AST structure, answer booleans, and whitelisted scalar
metrics; native output, token IDs, and tensors are never written.

## Config

```json
{
  "runner": "matrix_runner.py",
  "image": "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf",
  "fixture_repo": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/fixtures",
  "sources": {
    "all-beta": {
      "source": "/home/zhiman/megaquant-aitop-validation-20261019/source-package/sglang-all-beta",
      "archive": "/home/zhiman/megaquant-aitop-validation-20261019/source-package/sglang-all-beta.tar.gz",
      "manifest": "/home/zhiman/megaquant-aitop-validation-20261019/source-package/sglang-all-beta.manifest.json",
      "revision": "49d3acfd4d54b0cf1afffcfeb369c63af4f56499"
    }
  },
  "models": {
    "w4a4": {"path": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/models/w4a4", "provenance": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/model-identity.json"},
    "w4a8": {"path": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/models/w4a8", "provenance": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/model-identity.json"}
  }
}
```

Run detached:

```text
python matrix_candidate.py --action launch --config config.json --root /fresh/candidate-matrix
python matrix_candidate.py --action status --root /fresh/candidate-matrix
```

The wrapper supplies `expected_client.py` to the candidate runner. The client
requires the pinned fixture SHA and uses 768 generated tokens for code and 32
for other cases. The host checks `391`, `red`, `blue`, and `3729` while the
native response is still in memory. Answer mismatches remain recorded failures
and allow later cases to run. Code is checked for AST structure; generated code
is not executed. Every case requires two repetitions per side, natural stop,
prompt/image identity, and stable output. Every MTP request must report a finite,
positive verify count. The report does not establish visual/MTP quantized apply;
that requires a separate execution probe.

## Offline mock verification

The wrapper can be syntax-checked without Docker or a GPU:

```text
PYTHONPYCACHEPREFIX=/tmp/candidate-matrix-pyc python3 -m py_compile matrix_candidate.py
python matrix_candidate.py --action status --root /tmp/no-candidate-matrix
```
