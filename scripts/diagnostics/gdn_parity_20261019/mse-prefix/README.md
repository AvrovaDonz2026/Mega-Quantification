# MSE native-prefix reproduction

This directory contains byte-for-byte copies of the two scripts used by the
AITOP validation run:

| file | source SHA-256 |
| --- | --- |
| `controller.py` | `f0c8a2a1b6a44363e7a77e8ddcab380926127b1b45a53218ae98eb0518fe30be` |
| `create_model_view.py` | `ae5ad0712ccb5e53556707b37fb8e8eb28ab1f4d7ec892dfb4bc0bd73f1bca9c` |

The controller is the fresh-prefill native-prefix probe. It does not restore
incremental SSM/KV state and it does not expose prompts, token IDs, responses,
or raw evidence in the output report.

## Prepare the remote root

Run these commands from this directory. Use a new remote root on the machine
where the model and serving image are available:

```bash
REMOTE_ROOT=/fresh/mse-prefix-remote
mkdir -p "$REMOTE_ROOT"
```

Build the pinned source package in the remote root. The builder uses the
parent bundle's `source-lock.json`; it requires the local SGLang clone to
contain the locked commits and does not fetch from a network:

```bash
python3 ../build_sources.py \
  --repo /path/to/sglang \
  --output "$REMOTE_ROOT/source-package" \
  --lock ../source-lock.json
```

The controller consumes the `original` and `all-beta` packages from this
directory, including their matching `.tar.gz` archives and `.manifest.json`
files. The candidate runner must be available at the fixed remote-root path;
copy the current parity-v7 bundle there:

```bash
mkdir -p "$REMOTE_ROOT/candidate-bundle-r1"
cp -a ../parity-v7 "$REMOTE_ROOT/candidate-bundle-r1/parity-v7"
cp ../source_lock.py ../source-lock.json "$REMOTE_ROOT/candidate-bundle-r1/"
```

## Establish the MSE model view

Start from the scored original MSE ModelOpt export. The output must be a new
directory named `mse-model-view-r1`; the helper and helper revision are pinned
below:

```text
helper revision: a0c6163d15601529a424384124471f44ea579310
helper SHA-256:  98ba655dc897f078787f888fa741b199bac5fbc480b3df7e4cbca7320b26cf6f
```

The helper is `src/megaquant/sglang_export.py`. Use the input gates shipped in
the candidate bundle and pass the exact helper SHA and revision:

```bash
python3 create_model_view.py \
  --original-model /path/to/scored-mse-model \
  --output "$REMOTE_ROOT/mse-model-view-r1" \
  --helper /path/to/Mega-Quantification/src/megaquant/sglang_export.py \
  --helper-sha256 98ba655dc897f078787f888fa741b199bac5fbc480b3df7e4cbca7320b26cf6f \
  --helper-revision a0c6163d15601529a424384124471f44ea579310 \
  --input-gates "$REMOTE_ROOT/candidate-bundle-r1/parity-v7/input_gates.py"
```

The generated view must contain `models/w4a4` and its generated identity and
metadata reports. Do not replace the hard-linked weights or edit the generated
identity files after creation.

## Run the probe

Keep the native-private file on the evaluation machine only. It is an input gate,
not part of this bundle or the evidence root. Its fixed SHA-256 begins with
`8e151d` and ends with `7218`; the controller rejects any other file. Choose a
new evidence root outside this directory:

```bash
python3 controller.py \
  --root /fresh/mse-prefix-evidence \
  --remote-root "$REMOTE_ROOT" \
  --native-private /private/path/native-private.json \
  --budget 512
```

The controller enforces the validated protocol: context length `65536`,
native prefix `41472` plus prompt `288`, two trials per arm, cold cache with
radix cache disabled, no MTP/speculative decoding, and the pinned image,
source, model, and runtime identities. It writes sanitized status,
checkpoint, report, runtime-source, and cleanup records under the evidence
root. The controller finishes by generating its stats and cleanup records.
New reports require and retain the canonical integer `num_retractions=0`.
The frozen 2026-10-19 controller checked the noncanonical `retraction_count`
field and defaulted an absent value to zero; its reports do not prove that the
canonical counter was zero. This correction does not alter the recorded
continuation hashes or turn the fixed-prefix result into a natural replay.

## Scope

This is a fresh-prefill sensitivity probe. It is not an incremental-state
reconstruction, and a `512` token continuation is shorter than one historical
`2032`-token cycle. The completed 2026-10-19 run reproduced the historical
512-token continuation exactly in both trials of both source variants;
all four stopped at the token limit without EOS or `</think>`. This is a
negative result for breaking that fixed-prefix continuation with the beta
patch. It does not support claims about throughput, GPQA improvement, or
natural-generation repair. See the [validation record](../../../../docs/validation/dgx-spark-gdn-parity-20261019.md).
