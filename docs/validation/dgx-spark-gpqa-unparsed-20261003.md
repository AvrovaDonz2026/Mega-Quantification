# Spark GPQA responses without final answers

On 2026-10-03, the 262144-context W4A8 run had 194 completed records. Three responses ended with `finish_reason=stop`, `truncated=false`, and an empty final answer. This is an investigation of those responses; the complete benchmark result remains pending.

| Output tokens | Observed behavior | Same question in the earlier 32k W4A8/16-request run |
|---:|---|---|
| 21,468 | Repeated `U` characters; no final answer | Stopped without an answer after 14,270 tokens |
| 22,565 | Repeated reasoning; no final answer | Returned an answer after 23,934 tokens |
| 158,333 | Extended repeated reasoning; no final answer | Reached the length limit after 32,365 tokens |

The installed SGLang Qwen3 parser was checked on CPU. It preserves a final answer after `</think>` and places an unclosed thinking trace in `reasoning_content`. The tokenizer preserves `</think>` even when skipping special tokens. Reconstructed prompt token counts match all three recorded usages. Re-parsing the saved reasoning reproduces the empty final answer. There is no evidence that the answer extractor or reasoning parser discarded an emitted final answer.

The client synthesizes the journal's thinking delimiters when combining `reasoning_content` and `content`. Those delimiters cannot establish that the model closed thinking. The three records do not retain the raw response, output token IDs, or SGLang's `matched_stop`. Their actual stopping tokens therefore cannot be recovered from the journals. Premature EOS is a hypothesis; the evidence does not isolate quantization, a kernel, or batch-dependent numerical changes as its cause.

Scoring remains unchanged: an empty final answer counts as wrong. Tentative choices inside reasoning do not become final answers. Passive diagnostics have been attached to the existing evaluation client's responses to record `matched_stop`, reasoning/answer lengths, token counts and hashes. The diagnostics add no GPU requests, change no sampling parameters, and do not store raw responses. The W4A4 client will receive the same diagnostics when it starts.

A separate rerun of these three original W4A8 cases is queued to start after both formal 198-case runs finish and pass verification. It uses the same frozen image, weights, prompts, shuffled choices, seed 0, temperature 0, preserved xhigh thinking and 262144 context. The retry has three concurrent requests, so its actual batch differs from the formal run. It saves the raw responses, token IDs and `matched_stop` in private diagnostic files, with an independent three-case summary. The original formal results remain the benchmark record.

[Sanitized CPU and historical comparison evidence](dgx-spark-gpqa-unparsed-20261003.json) includes source/template hashes and aggregate case details, without questions, record IDs, raw traces or private deployment paths.
