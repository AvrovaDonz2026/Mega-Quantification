# Same-prefix prefill logits probe

`client.py` scores two candidate token IDs at one fixed token prefix through a fresh `/generate` prefill. It does not start or stop servers; a v7 runner can invoke it in an already-owned client container and pass the control object on stdin. The native response, prefix IDs, and candidate IDs are held in memory. Stdout contains only scalar logprobs, finite/probability checks, and SHA256 fingerprints.

## Control protocol

The caller writes one JSON object to stdin:

```json
{"prefix_ids": [1, 2, 3], "candidate_ids": [101, 102], "prefix_role": "baseline", "target_index": 16}
```

`prefix_ids` must be a nonempty list of nonnegative token IDs. `candidate_ids` must contain two distinct nonnegative IDs. `prefix_role` is metadata only and must be `baseline`, `mtp`, or `unspecified`; `target_index` is optional.

Example inside a v7-owned client container:

```bash
python /prefill-logits-probe/client.py --base-url http://127.0.0.1:31200 --timeout 300 <<'JSON'
{"prefix_ids":[1,2,3],"candidate_ids":[101,102],"prefix_role":"baseline","target_index":16}
JSON
```

The probe flushes the prefix cache before every request. It appends the first candidate to `input_ids`, sets `logprob_start_len` to the supplied prefix length, and requests both candidates with `token_ids_logprob`. It accepts the response only when `meta_info.input_token_ids_logprobs` has exactly one row and that row contains both requested IDs with finite values. Any missing or ambiguous row alignment fails the request.

## Interpretation

The output is a **same-prefix fresh-prefill** comparison. For each `k`, invoke this client against baseline and MTP servers with identical `prefix_ids` and compare the two scalar reports by `prefix_sha256` and candidate hashes. The first exact or thresholded logprob difference is a prefill/LM-head divergence candidate. This probe does **not** align speculative decode state, KV state, or GDN recurrent state; `decode_state_aligned` is always `false`. Do not compare a baseline-generated prefix with an MTP-generated prefix and call that a numeric difference.

The v7 runner remains responsible for server lifecycle, source/model/image identity gates, and persistence policy. A runner should keep raw control IDs and native responses in memory and persist only the sanitized client output.
