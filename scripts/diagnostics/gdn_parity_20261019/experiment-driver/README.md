# Experiment driver

`driver.py` runs the local beta regression and parity matrix in strict
sequence, with one evidence directory per job. It uses only the Python
standard library and starts the existing `beta-regression/probe.py` and
`parity-v7/run_v7.py` as subprocesses. The parity runner owns Docker/GPU
gates and cleanup; a `parity_failed` report is recorded and the next job is
still started. Invalid input, a crash, timeout, or cleanup failure stops the
sequence.

## Actions

```text
python driver.py --action launch --phase local --config config.json --root /fresh/evidence
python driver.py --action launch --phase parity --config config.json --root /fresh/evidence
python driver.py --action launch --phase all --config config.json --root /fresh/evidence
python driver.py --action status --root /fresh/evidence
```

`launch` requires a nonexistent root and detaches a supervisor. `status` is
read-only. The default root is
`/home/zhiman/megaquant-aitop-validation-20261019`.

## Config

Paths are relative to the config file. The source package layout is accepted
directly:

```json
{
  "source_package": "/home/zhiman/megaquant-aitop-validation-20261019/source-package",
  "source_summary": "summary.json",
  "source_variants": {
    "original": {"package": "source-package", "revision": "<40-char-revision>"},
    "decode-beta": {"package": "source-package", "revision": "<40-char-revision>"},
    "all-beta": {"package": "source-package", "revision": "<40-char-revision>"},
    "shared-decode": {"package": "source-package", "revision": "<40-char-revision>"}
  },
  "models": {
    "w4a4": {"path": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/models/w4a4", "provenance": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/model-identity.json"},
    "w4a8": {"path": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/models/w4a8", "provenance": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/model-identity.json"}
  },
  "fixture_repo": "/home/zhiman/megaquant-aitop-validation-20261014/full-model/fixtures",
  "image": "sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf"
}
```

Each `source_variants` entry may instead provide explicit `archive`, `path`,
`manifest`, and `revision` values. The driver does not infer a git revision;
it reads the configured revision or the source package summary. Source
archives are extracted only after rejecting path traversal and links.

The four local jobs are `original`, `decode-beta`, `all-beta`, and
`shared-decode` (the last is reported as `all-beta`). The four parity jobs are
W4A4 `python_code`: decode-beta, all-beta, shared-decode force 0, and
shared-decode force 1. Each runner uses two baseline and two MTP repetitions.

Job records are in `jobs/<job-id>.json`; child reports and logs are under the
same directory. No raw model responses or tensors are written by this driver.
