# Parity v7

`run_v7.py` is a detached first-divergence probe for one quantized model
scheme. It validates the exact source archive/manifest, selected model files,
serving image ID, container labels, server configuration, and imported source
file hashes before sending a fixture request.

The default fixture is `python_code`; `arithmetic`, `red_square`,
`blue_square`, and `ocr_3729` are also accepted. Baseline and MTP each run the
same fixture twice. A repeated output or prompt mismatch fails the run before
the parity comparison. The report contains only hashes, lengths, stop reasons,
AST structure, selected-token margins/probabilities, whitelisted scalar
metrics, source hashes, and environment identity. Native responses are kept in
memory only.

When outputs diverge, `comparison.first_divergence` additionally records the
first divergence index, a bounded window of up to four token IDs on each side,
and numeric top-k token-ID/logprob summaries (up to five entries per side).
Token text, prompts, and complete token sequences are never added to this
diagnostic field. A sequence-length mismatch points at the first missing token
and uses `null` for the missing side's token and top-k summary.

The client verifies the fixture file SHA before importing its cases. For
`python_code`, it requests up to 768 generated tokens so the fixture can reach
its natural completion. The parity gate requires all four responses (two
baseline and two MTP repetitions) to finish with the server's natural `stop`
reason; `length` is not accepted. It also requires complete prompt/image/file
identity and repeated-output stability. For `python_code`, both returned
programs must pass the complete AST gate, including `fibonacci`, `factorial`,
and `is_prime`. The report records the token budget and these gate results.

Example actions:

```text
python run_v7.py --action launch --root /fresh/evidence ...
python run_v7.py --action status --root /fresh/evidence
```

`--fla-use-fast-ops` is explicit (`0` or `1`) and defaults to `0`; it is passed
as the imported FLA implementation's real `FLA_USE_FAST_OPS` variable.
`--force-shared-decode` is an independent diagnostic switch for
`SGLANG_GDN_FORCE_SHARED_DECODE` and also defaults to `0`. The runtime gate
checks both values and the imported FLA/GDN module origins and hashes. The
launch action requires a new evidence root and the run action refuses to
proceed when `nvidia-smi` reports active compute processes.

For shared-decode comparisons, both ReplaySSM switches must remain closed:
do not enable `--enable-linear-replayssm` or
`--enable-linear-replayssm-spec`. The current measured runtime defaults have
both flags off; a run with either flag enabled is a different configuration
and is outside this parity evidence.
