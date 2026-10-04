# Spark GPQA responses without final answers

Both formal Spark runs completed all 198 GPQA Diamond questions at temperature 0 with a 262144 context. Each has four empty final answers and zero length-truncated responses. This investigation started with the first three W4A8 failures at a 194-record snapshot on 2026-10-03. The [complete benchmark report](dgx-spark-gpqa-256k-20261002.md) preserves the formal scores; this report adds stopping evidence and a separate three-case retry.

| Output tokens | Observed behavior | Same question in the earlier 32k W4A8/16-request run |
|---:|---|---|
| 21,468 | Repeated `U` characters; no final answer | Stopped without an answer after 14,270 tokens |
| 22,565 | Repeated reasoning; no final answer | Returned an answer after 23,934 tokens |
| 158,333 | Extended repeated reasoning; no final answer | Reached the length limit after 32,365 tokens |

The installed SGLang Qwen3 parser was checked on CPU. It preserves a final answer after `</think>` and places an unclosed thinking trace in `reasoning_content`. The tokenizer preserves `</think>` even when skipping special tokens. Reconstructed prompt token counts match all three recorded usages. Re-parsing the saved reasoning reproduces the empty final answer. There is no evidence that the answer extractor or reasoning parser discarded an emitted final answer.

The client synthesizes the journal's thinking delimiters when combining `reasoning_content` and `content`. Those delimiters cannot establish that the model closed thinking. The original three W4A8 records do not retain the raw response, output token IDs, or SGLang's `matched_stop`. Their actual stopping tokens therefore cannot be recovered from the journals. A fourth W4A8 failure completed with 70,466 output tokens and an empty final answer; it also has no matching passive stopping record and was outside the frozen three-case selection. Premature EOS remains a hypothesis for these four W4A8 failures.

Passive diagnostics captured all four W4A4 responses without final answers. Each reports `finish_reason=stop` and `matched_stop=248046`, the tokenizer's EOS token `<|im_end|>`, with zero answer characters. The captures were matched to their journal records using reasoning/answer hashes and prompt/completion counts. This establishes EOS stopping for these W4A4 responses. The captures retain no raw output IDs and cannot establish whether a thinking-close token was emitted.

| W4A4 output tokens | Reasoning characters | Final answer characters | Matched stop |
|---:|---:|---:|---|
| 21,996 | 63,084 | 0 | EOS 248046 |
| 30,467 | 80,665 | 0 | EOS 248046 |
| 17,406 | 24,550 | 0 | EOS 248046 |
| 49,181 | 113,508 | 0 | EOS 248046 |

The separate retry of the original three W4A8 cases finished at `2026-10-03T14:31:46Z`, after both formal runs passed verification. It used the frozen client and serving image at revision `79b5dfef54618d0e385c5974d0ea508cd2fbdd40`, the same weights, prompts, shuffled choices, seed 0, temperature 0, preserved xhigh thinking and 262144 context. Returned prompt token IDs exactly match the reconstructed prompts. Initial generation budgets retained the original `len(prompt)//4` estimate plus the 256-token template reserve; only `return_token_ids` and `return_meta_info` were added to the HTTP request.

| Case | Original output tokens | Retry output tokens | Retry final answer | Retry correctness |
|---|---:|---:|---|---|
| 1 | 21,468 | 18,095 | Present | Correct |
| 2 | 22,565 | 20,271 | Present | Wrong |
| 3 | 158,333 | 48,186 | Present | Correct |

All three retry responses contain the model-emitted `</think>` token 248069, followed by a final answer, and terminate with token 248046 matching SGLang's `matched_stop`. The retained response IDs match each completion-token count and contain valid integer token IDs. There were zero HTTP errors, truncations, continuations or request retractions. Three requests started concurrently under a server limit of 16; the actual decoding batch differs from the formal run and changes as requests finish. These results do not establish bitwise reproducibility or isolate quantization, a kernel, or batch-dependent numerical changes as the cause of the original failures. The new EOS evidence does not recover the original three W4A8 stopping tokens.

Scoring remains unchanged: an empty final answer counts as wrong, and tentative choices inside reasoning do not become final answers. The retry's two correct answers belong to its independent three-case diagnostic summary. Both formal journals and benchmark scores remain unchanged; the original W4A8 journal SHA-256 is `0b6b1085707bbcbad6a76013c9b21bc9d45f77f7e5ec0497b7459157a20ce5dc` before and after the retry. Passive diagnostics added no GPU requests and saved only metadata; the separately authorized retry retained raw responses and token IDs in private diagnostic files.

[Sanitized investigation evidence](dgx-spark-gpqa-unparsed-20261003.json) includes CPU parser checks, source/template hashes, passive W4A4 stopping evidence and audited retry statistics, without questions, record IDs, raw traces or private deployment paths.
