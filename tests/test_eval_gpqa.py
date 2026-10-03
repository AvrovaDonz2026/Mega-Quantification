"""GPQA Diamond protocol — no GPU, no Hub, no 27B."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from dataclasses import asdict
from pathlib import Path

import pytest

from megaquant.eval_gpqa import (
    CHAT_TEMPLATE_TOKEN_RESERVE,
    DEFAULT_EVAL_BASE_URL,
    DEFAULT_EVAL_RECIPE,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MAX_MODEL_LEN,
    NVIDIA_SGLANG_SERVE,
    QWEN_THINKING_SAMPLING,
    GenerationResult,
    GPQAItem,
    ItemResult,
    auto_kv_offload_gib,
    build_gpqa_item,
    default_eval_base_url,
    describe_eval,
    eval_base_url_from_env,
    extract_choice,
    format_gpqa_prompt,
    gpqa_trace_dir,
    load_eval_recipe,
    load_gpqa_journal,
    openai_chat_complete,
    remaining_new_tokens,
    resolve_eval_recipe_path,
    run_gpqa,
    score_items,
    serve_argv_from_recipe,
    sglang_serve_argv,
    sglang_serve_argv_from_recipe,
    sglang_serve_environ,
    visible_answer_span,
    vllm_serve_argv,
    vllm_serve_argv_from_recipe,
)
from megaquant.exceptions import EvalError

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / "recipes" / "eval-gpqa-diamond.yaml"


def test_recipe_matches_qwen_and_nvidia_cards() -> None:
    recipe = load_eval_recipe(RECIPE)
    assert recipe.benchmark == "gpqa_diamond"
    assert recipe.thinking.enable is True
    assert recipe.thinking.preserve is True
    assert recipe.thinking.reasoning_effort == "xhigh"
    assert recipe.generation.max_new_tokens == 0
    assert recipe.generation.continue_on_length is True
    assert recipe.generation.max_model_len == DEFAULT_MAX_MODEL_LEN
    assert recipe.generation.seed == 0
    assert recipe.generation.http_timeout_seconds == 86400.0
    samp = recipe.sampling.model_dump()
    assert samp["temperature"] == 0.0
    for key, value in QWEN_THINKING_SAMPLING.items():
        if key == "temperature":
            continue
        assert samp[key] == value, key
    serve = recipe.serve.model_dump()
    assert serve["engine"] == "sglang"
    assert serve["kv_cache_dtype"] == NVIDIA_SGLANG_SERVE["kv_cache_dtype"]
    assert serve["reasoning_parser"] == NVIDIA_SGLANG_SERVE["reasoning_parser"]
    assert serve["chunked_prefill_size"] == NVIDIA_SGLANG_SERVE["chunked_prefill_size"]
    assert serve["mamba_full_memory_ratio"] == NVIDIA_SGLANG_SERVE["mamba_full_memory_ratio"]
    assert serve["mamba_radix_cache_strategy"] == "extra_buffer_lazy"
    assert serve["enable_hierarchical_cache"] is True
    assert serve["kv_offloading_backend"] == "native"
    assert serve["kv_offloading_size_gb"] == 12
    assert serve["swap_space_gb"] == 0
    assert serve["cpu_reserve_gib"] == 6
    assert serve["disable_cuda_graph"] is True
    assert serve["attention_backend"] == "flashinfer"
    assert serve["sampling_backend"] is None
    assert serve["enable_deterministic_inference"] is False
    default_argv = sglang_serve_argv_from_recipe(recipe)
    assert "--attention-backend" in default_argv
    assert default_argv[default_argv.index("--attention-backend") + 1] == "flashinfer"
    assert "--sampling-backend" not in default_argv
    assert "--enable-deterministic-inference" not in default_argv
    plan = describe_eval(recipe)
    assert plan["engine"] == "sglang"
    assert plan["generation"]["http_timeout_seconds"] == 86400.0
    assert plan["base_url"] is None
    assert "remaining context" in plan["max_new_tokens_policy"]
    assert plan["kv_offloading_backend"] == "hicache"
    assert plan["kv_cpu_offload_gib"] == 12
    assert default_eval_base_url(recipe) == DEFAULT_EVAL_BASE_URL
    assert default_eval_base_url(recipe) == "http://127.0.0.1:30000/v1"


def test_default_traces_sit_inside_the_model_directory() -> None:
    recipe = load_eval_recipe(RECIPE)
    assert recipe.output_dir == ""
    traced = gpqa_trace_dir(recipe.model, recipe.output_dir)
    assert traced == Path(recipe.model) / "gpqa_diamond"
    assert describe_eval(recipe)["journal_dir"] == str(traced)
    custom = gpqa_trace_dir(recipe.model, "/tmp/keep-this")
    assert custom == Path("/tmp/keep-this")


def test_5090_recipe_uses_triton_when_flashinfer_cannot_see_sm120() -> None:
    recipe = load_eval_recipe(REPO / "recipes" / "eval-gpqa-diamond.5090.yaml")
    assert recipe.serve.attention_backend == "triton"
    assert recipe.serve.sampling_backend == "pytorch"
    assert recipe.serve.linear_attn_backend == "triton"
    assert recipe.serve.fp8_gemm_backend == "triton"
    assert recipe.serve.fp4_gemm_backend == "marlin"
    assert recipe.serve.force_fp8_marlin is True
    assert recipe.serve.disable_silu_fp4_quant_fusion is True
    assert recipe.serve.disable_cuda_graph is True
    assert recipe.serve.kv_offloading_size_gb == 64
    assert recipe.concurrency == 24
    assert recipe.serve.max_mamba_cache_size == 96
    assert recipe.serve.max_running_requests == 24
    assert recipe.serve.mamba_ssm_dtype == "bfloat16"
    assert recipe.serve.mem_fraction_static == 0.95
    argv = sglang_serve_argv_from_recipe(recipe)
    joined = " ".join(argv)
    assert "--attention-backend triton" in joined
    assert "--attention-backend flashinfer" not in joined
    assert "--sampling-backend pytorch" in joined
    assert argv[argv.index("--sampling-backend") + 1] == "pytorch"
    assert "--linear-attn-backend triton" in joined
    assert "--fp8-gemm-backend triton" in joined
    assert "--fp4-gemm-backend marlin" in joined
    assert "--hicache-size 64" in joined
    assert "--max-mamba-cache-size 96" in joined
    assert "--mamba-ssm-dtype bfloat16" in joined
    assert "--max-running-requests 24" in joined
    assert "--mem-fraction-static 0.95" in joined
    assert sglang_serve_environ(recipe) == {
        "SGLANG_FORCE_FP8_MARLIN": "1",
        "SGLANG_DISABLE_SILU_FP4_QUANT_FUSION": "1",
    }


def test_6000d_recipe_keeps_cuda_graph_and_skips_flashinfer_fusion() -> None:
    recipe = load_eval_recipe(REPO / "recipes" / "eval-gpqa-diamond.6000d.yaml")
    assert recipe.concurrency == 64
    assert recipe.serve.disable_cuda_graph is False
    assert recipe.serve.enable_hierarchical_cache is False
    assert recipe.serve.max_mamba_cache_size == 256
    assert recipe.serve.fp8_gemm_backend == "cutlass"
    argv = sglang_serve_argv_from_recipe(recipe)
    joined = " ".join(argv)
    assert "--disable-cuda-graph" not in joined
    assert "--enable-hierarchical-cache" not in joined
    assert "--hicache-size" not in joined
    assert "--attention-backend triton" in joined
    assert "--mamba-backend triton" in joined
    assert "--fp8-gemm-backend cutlass" in joined
    assert "--fp4-gemm-backend marlin" in joined
    assert "--disable-radix-cache" in joined
    assert "--disable-flashinfer-autotune" in joined
    assert "--max-running-requests 64" in joined
    assert "--max-mamba-cache-size 256" in joined
    assert "--chunked-prefill-size 4096" in joined
    assert "--mem-fraction-static 0.9" in joined
    assert sglang_serve_environ(recipe) == {
        "SGLANG_FORCE_FP8_MARLIN": "1",
        "SGLANG_DISABLE_SILU_FP4_QUANT_FUSION": "1",
        "SGLANG_IS_FLASHINFER_AVAILABLE": "0",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    }


def test_spark_recipe_serves_images_without_double_budgeting_unified_memory(monkeypatch) -> None:
    recipe = load_eval_recipe(REPO / "recipes" / "eval-gpqa-diamond.spark.yaml")

    def reject_host_memory_budget(*_args, **_kwargs):
        pytest.fail("HiCache must not budget Spark CPU/GPU shared memory twice")

    monkeypatch.setattr("megaquant.eval_gpqa.auto_kv_offload_gib", reject_host_memory_budget)
    argv = serve_argv_from_recipe(recipe)
    assert argv[:2] == ["sglang", "serve"]
    assert argv[argv.index("--model-path") + 1] == recipe.model
    assert argv[argv.index("--tp-size") + 1] == "1"
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.7"
    assert argv[argv.index("--context-length") + 1] == "32768"
    assert argv[argv.index("--max-running-requests") + 1] == "4"
    assert argv[argv.index("--attention-backend") + 1] == "flashinfer"
    assert argv[argv.index("--linear-attn-backend") + 1] == "triton"
    assert argv[argv.index("--quantization") + 1] == "modelopt"
    assert argv[argv.index("--mamba-ssm-dtype") + 1] == "float32"
    assert "--enable-multimodal" in argv
    assert "--disable-cuda-graph" in argv
    assert "--enable-hierarchical-cache" not in argv
    assert "--hicache-size" not in argv
    assert not any(arg.startswith("--speculative-") for arg in argv)
    plan = describe_eval(recipe)
    assert plan["kv_cpu_offload_gib"] == 0
    assert plan["kv_offloading_backend"] == "none"
    assert sglang_serve_environ(recipe) == {"SGLANG_ENABLE_JIT_DEEPGEMM": "0"}


@pytest.mark.parametrize("quantization", [None, "modelopt", "modelopt_mixed"])
def test_sglang_recipe_honors_quantization_and_multimodal_options(quantization) -> None:
    recipe = load_eval_recipe(RECIPE)
    recipe.serve.quantization = quantization
    recipe.serve.enable_multimodal = False
    argv = sglang_serve_argv_from_recipe(recipe)
    assert "--enable-multimodal" not in argv
    if quantization is None:
        assert "--quantization" not in argv
    else:
        assert argv[argv.index("--quantization") + 1] == quantization
    recipe.serve.enable_multimodal = True
    assert "--enable-multimodal" in sglang_serve_argv_from_recipe(recipe)


@pytest.mark.parametrize("enabled", [False, True])
def test_sglang_deterministic_setting_maps_once_and_does_not_leak_to_vllm(enabled) -> None:
    flag = "--enable-deterministic-inference"
    argv = sglang_serve_argv("/ckpt", enable_deterministic_inference=enabled)
    assert argv.count(flag) == int(enabled)
    recipe = load_eval_recipe(RECIPE)
    recipe.serve.enable_deterministic_inference = enabled
    assert sglang_serve_argv_from_recipe(recipe).count(flag) == int(enabled)
    recipe.serve.engine = "vllm"
    assert flag not in serve_argv_from_recipe(recipe)
    assert flag not in vllm_serve_argv_from_recipe(recipe)


@pytest.mark.parametrize("name", ["eval-gpqa-diamond.spark", "eval-gpqa-diamond.spark-mtp"])
def test_spark_recipes_leave_deterministic_inference_disabled(name) -> None:
    recipe = load_eval_recipe(REPO / "recipes" / f"{name}.yaml")
    assert recipe.serve.enable_deterministic_inference is False
    assert "--enable-deterministic-inference" not in serve_argv_from_recipe(recipe)


def test_sglang_recipe_can_opt_into_embedded_mtp() -> None:
    recipe = load_eval_recipe(REPO / "recipes" / "eval-gpqa-diamond.spark.yaml")
    recipe.serve.speculative_algorithm = "EAGLE"
    recipe.serve.speculative_num_steps = 2
    recipe.serve.speculative_eagle_topk = 1
    recipe.serve.speculative_num_draft_tokens = 3
    argv = sglang_serve_argv_from_recipe(recipe)
    assert argv[argv.index("--speculative-algorithm") + 1] == "EAGLE"
    assert argv[argv.index("--speculative-num-steps") + 1] == "2"
    assert argv[argv.index("--speculative-eagle-topk") + 1] == "1"
    assert argv[argv.index("--speculative-num-draft-tokens") + 1] == "3"
    assert argv[argv.index("--mamba-ssm-dtype") + 1] == "float32"
    # SGLang selects the embedded Qwen MTP from the target export.
    assert "--speculative-draft-model-path" not in argv


def test_remaining_tokens_never_uses_small_default_cap() -> None:
    assert remaining_new_tokens(2000, 262144, 0) == 260144
    assert remaining_new_tokens(2000, 262144, None) == 260144
    assert remaining_new_tokens(2000, 32768, 0) == 30768
    assert remaining_new_tokens(100, 262144, 4096) == 4096
    assert remaining_new_tokens(262000, 262144, 0) == 144


def test_extract_choice_ignores_letters_inside_thinking() -> None:
    text = (
        "<think>\nThe answer might be A but wait, also B.\n</think>\n"
        "The correct option is C.\nAnswer: C\n"
    )
    assert visible_answer_span(text).strip().startswith("The correct")
    assert extract_choice(text) == "C"
    boxed = "reasoning done.\n\\boxed{B}\n"
    assert extract_choice(boxed) == "B"


def test_prompt_asks_for_untruncated_reasoning() -> None:
    item = GPQAItem(
        item_id="t1",
        question="Which particle?",
        choices={"A": "up", "B": "down", "C": "strange", "D": "charm"},
        gold="C",
    )
    prompt = format_gpqa_prompt(item)
    assert "Do not stop until the reasoning is complete" in prompt
    assert "Answer: $LETTER" in prompt
    assert "A. up" in prompt


def test_choice_shuffle_is_stable() -> None:
    row = {
        "Question": "Q?",
        "Correct Answer": "gold",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
        "Record ID": "r1",
    }
    a = build_gpqa_item(row, index=0, seed=0, shuffle=True)
    b = build_gpqa_item(row, index=0, seed=0, shuffle=True)
    assert a.choices == b.choices
    assert a.gold == b.gold
    assert a.choices[a.gold] == "gold"


def test_continue_on_length_appends_full_text(tmp_path: Path) -> None:
    recipe = load_eval_recipe(RECIPE, overrides={"output_dir": str(tmp_path), "limit": 1})
    item = GPQAItem(
        item_id="t1",
        question="Which particle?",
        choices={"A": "up", "B": "down", "C": "strange", "D": "charm"},
        gold="C",
    )
    calls: list[int] = []

    def complete(**kwargs):
        max_tokens = kwargs["max_tokens"]
        calls.append(max_tokens)
        if len(calls) == 1:
            return GenerationResult(
                text="<think>long trace",
                finish_reason="length",
                prompt_tokens=100,
                completion_tokens=50,
                truncated=True,
            )
        return GenerationResult(
            text="</think>\nAnswer: C",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=10,
            truncated=False,
        )

    summary = run_gpqa(
        recipe,
        items=[item],
        complete=complete,
        count_prompt_tokens=lambda _t: 100,
    )
    assert summary["scores"]["correct"] == 1
    assert summary["scores"]["n"] == 1
    assert len(calls) == 2
    journal = (tmp_path / "gpqa_diamond.jsonl").read_text()
    assert "long trace" in journal
    assert "Answer: C" in journal
    assert calls[0] == remaining_new_tokens(
        100 + CHAT_TEMPLATE_TOKEN_RESERVE, recipe.generation.max_model_len, 0
    )


def test_run_gpqa_resumes_existing_journal_without_rewriting(tmp_path: Path) -> None:
    recipe = load_eval_recipe(RECIPE, overrides={"output_dir": str(tmp_path)})
    done = GPQAItem(
        item_id="done",
        question="Q1?",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
        gold="A",
    )
    pending = GPQAItem(
        item_id="pending",
        question="Q2?",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
        gold="B",
    )
    first = run_gpqa(
        recipe,
        items=[done],
        complete=lambda **_k: GenerationResult(
            text="Answer: A",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
        ),
        count_prompt_tokens=lambda _t: 8,
    )
    assert first["scores"]["headline"] == "1/1"
    prior = load_gpqa_journal(tmp_path / "gpqa_diamond.jsonl")
    assert "done" in prior

    calls: list[str] = []

    def complete(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return GenerationResult(
            text="Answer: B",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
        )

    summary = run_gpqa(
        recipe,
        items=[done, pending],
        complete=complete,
        count_prompt_tokens=lambda _t: 8,
    )
    assert calls  # pending only
    assert all("Q1?" not in text for text in calls)
    assert summary["scores"]["headline"] == "2/2"
    assert summary["scores"]["n"] == 2
    lines = [ln for ln in (tmp_path / "gpqa_diamond.jsonl").read_text().splitlines() if ln]
    assert len(lines) == 2
    ids = [json.loads(ln)["item_id"] for ln in lines]
    assert ids == ["done", "pending"]


def test_run_gpqa_concurrency_fans_out(tmp_path: Path) -> None:
    recipe = load_eval_recipe(
        RECIPE, overrides={"output_dir": str(tmp_path), "concurrency": 4}
    )
    items = [
        GPQAItem(
            item_id=f"p{i}",
            question=f"Q{i}?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            gold="A",
        )
        for i in range(4)
    ]
    barrier = threading.Barrier(4, timeout=5)

    def complete(**_kwargs):
        barrier.wait()
        return GenerationResult(
            text="Answer: A",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
        )

    summary = run_gpqa(
        recipe,
        items=items,
        complete=complete,
        count_prompt_tokens=lambda _t: 8,
    )
    assert summary["scores"]["headline"] == "4/4"
    lines = [ln for ln in (tmp_path / "gpqa_diamond.jsonl").read_text().splitlines() if ln]
    assert len(lines) == 4


def test_run_gpqa_request_failure_preserves_other_inflight_results(tmp_path: Path) -> None:
    recipe = load_eval_recipe(
        RECIPE, overrides={"output_dir": str(tmp_path), "concurrency": 3}
    )
    items = [
        GPQAItem(key, key, {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
        for key in ("failing", "success-1", "success-2")
    ]
    # All three requests begin together. The successes finish after the failed
    # request so they exercise preserving work while reporting a real failure.
    barrier = threading.Barrier(3, timeout=5)
    delay = threading.Event()

    def complete(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        barrier.wait()
        if "\nfailing\n" in prompt:
            raise EvalError("transient request failure")
        delay.wait(0.1)
        return GenerationResult(
            text="Answer: A", finish_reason="stop", prompt_tokens=8, completion_tokens=4
        )

    # Stale output must not advertise a finished score after an interrupted run.
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"scores": {"headline": "3/3"}}\n')
    with pytest.raises(EvalError, match="completed rows saved"):
        run_gpqa(recipe, items=items, complete=complete)
    rows = load_gpqa_journal(tmp_path / "gpqa_diamond.jsonl")
    assert set(rows) == {"success-1", "success-2"}
    assert all(row.correct for row in rows.values())
    assert not summary_path.exists()

    retried: list[str] = []

    def recover(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        retried.append(prompt)
        return GenerationResult(
            text="Answer: A", finish_reason="stop", prompt_tokens=8, completion_tokens=4
        )

    summary = run_gpqa(recipe, items=items, complete=recover)
    assert len(retried) == 1
    assert "\nfailing\n" in retried[0]
    assert summary["scores"]["headline"] == "3/3"
    assert len(load_gpqa_journal(tmp_path / "gpqa_diamond.jsonl")) == 3


def test_score_counts_truncated_and_unparsed_against_full_denominator() -> None:
    items = [
        GPQAItem("a", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A"),
        GPQAItem("b", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "B"),
    ]
    rows = [
        ItemResult("a", "A", "A", True, False, "stop", 1, 1, 0, "Answer: A", items[0].choices),
        ItemResult("b", "B", None, False, True, "length", 1, 1, 0, "incomplete", items[1].choices),
    ]
    scores = score_items(items, rows)
    assert scores["headline"] == "1/2"
    assert scores["truncated"] == 1
    assert scores["unparsed"] == 1


def test_score_recomputes_cached_correct_flags() -> None:
    items = [
        GPQAItem(key, "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A")
        for key in ("valid", "truncated", "unparsed", "wrong")
    ]
    rows = [
        ItemResult("valid", "A", "A", False, False, "stop", 1, 1, 0, "Answer: A", {}),
        ItemResult("truncated", "A", "A", True, True, "length", 1, 1, 0, "Answer: A", {}),
        ItemResult("unparsed", "A", None, True, False, "stop", 1, 1, 0, "", {}),
        ItemResult("wrong", "A", "B", True, False, "stop", 1, 1, 0, "Answer: B", {}),
    ]
    scores = score_items(items, rows)
    assert scores["headline"] == "1/4"
    assert scores["truncated"] == 1
    assert scores["unparsed"] == 1


@pytest.mark.parametrize("truncated,text", [(True, "Answer: A"), (False, "")])
def test_run_gpqa_never_scores_truncated_or_unparsed_answers_correct(
    tmp_path: Path, truncated: bool, text: str
) -> None:
    recipe = load_eval_recipe(RECIPE, overrides={"output_dir": str(tmp_path)})
    recipe.generation.continue_on_length = False
    item = GPQAItem("a", "q", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
    summary = run_gpqa(
        recipe,
        items=[item],
        complete=lambda **_k: GenerationResult(
            text=text,
            finish_reason="length" if truncated else "stop",
            prompt_tokens=8,
            completion_tokens=4,
            truncated=truncated,
        ),
    )
    assert summary["scores"]["headline"] == "0/1"
    row = json.loads((tmp_path / "gpqa_diamond.jsonl").read_text())
    assert row["correct"] is False


@pytest.mark.parametrize("retry_context_overflow", [False, True])
def test_openai_http_serializes_sglang_extensions_at_top_level(
    monkeypatch: pytest.MonkeyPatch, retry_context_overflow: bool
) -> None:
    recipe = load_eval_recipe(RECIPE)
    recipe.sampling.min_p = 0.1
    recipe.sampling.repetition_penalty = 1.2
    recipe.thinking.preserve = False
    requests: list[dict] = []

    def urlopen(request, timeout):
        assert request.full_url == "http://sglang.local/v1/chat/completions"
        assert timeout == DEFAULT_HTTP_TIMEOUT_SECONDS
        body = json.loads(request.data)
        requests.append(body)
        if retry_context_overflow and len(requests) == 1:
            detail = (
                "maximum context length of 256 tokens: "
                "100 tokens from the input messages and 512 tokens for the completion"
            )
            raise urllib.error.HTTPError(
                request.full_url, 400, "context overflow", None, io.BytesIO(detail.encode())
            )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"reasoning_content": "trace", "content": "Answer: B"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 4},
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    messages = [{"role": "user", "content": "Question"}]
    result = openai_chat_complete(
        base_url="http://sglang.local/v1",
        model="test-model",
        messages=messages,
        sampling=recipe.sampling,
        thinking=recipe.thinking,
        max_tokens=512,
        seed=0,
    )
    assert len(requests) == (2 if retry_context_overflow else 1)
    for body in requests:
        assert "extra_body" not in body
        assert body["model"] == "test-model"
        assert body["messages"] == messages
        assert body["temperature"] == 0.0
        assert body["top_p"] == 0.95
        assert body["top_k"] == 20
        assert body["min_p"] == 0.1
        assert body["repetition_penalty"] == 1.2
        assert body["chat_template_kwargs"] == {
            "enable_thinking": True,
            "preserve_thinking": False,
        }
        assert body["reasoning_effort"] == "xhigh"
    if retry_context_overflow:
        assert requests[1] == {**requests[0], "max_tokens": 148}
    assert result.text == "<think>\ntrace\n</think>\nAnswer: B"
    assert result.completion_tokens == 4
    assert result.truncated is False


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_recipe_rejects_invalid_http_timeout(timeout: float) -> None:
    with pytest.raises(EvalError):
        load_eval_recipe(RECIPE, overrides={"generation": {"http_timeout_seconds": timeout}})


def test_run_gpqa_uses_configured_http_timeout_for_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_eval_recipe(
        RECIPE,
        overrides={
            "output_dir": str(tmp_path),
            "base_url": "http://sglang.local/v1",
            "generation": {"http_timeout_seconds": 43210.5, "max_new_tokens": 32},
        },
    )
    timeouts: list[float] = []

    def complete(**kwargs):
        assert kwargs["base_url"] == recipe.base_url
        timeouts.append(kwargs["timeout"])
        if len(timeouts) == 1:
            return GenerationResult(
                text="<think>trace",
                finish_reason="length",
                prompt_tokens=100,
                completion_tokens=32,
                truncated=True,
            )
        return GenerationResult(
            text="</think>\nAnswer: A",
            finish_reason="stop",
            prompt_tokens=150,
            completion_tokens=4,
        )

    monkeypatch.setattr("megaquant.eval_gpqa.openai_chat_complete", complete)
    item = GPQAItem("a", "q", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
    summary = run_gpqa(recipe, items=[item])
    assert timeouts == [43210.5, 43210.5]
    assert summary["plan"]["generation"]["http_timeout_seconds"] == 43210.5
    assert summary["scores"]["headline"] == "1/1"


def test_openai_http_timeout_reports_configured_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = load_eval_recipe(RECIPE)

    def urlopen(_request, timeout):
        assert timeout == 43210.5
        raise TimeoutError("socket timed out")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(EvalError, match="timed out after 43210.5 seconds"):
        openai_chat_complete(
            base_url="http://sglang.local/v1",
            model="test-model",
            messages=[{"role": "user", "content": "Question"}],
            sampling=recipe.sampling,
            thinking=recipe.thinking,
            max_tokens=512,
            seed=0,
            timeout=43210.5,
        )


def test_sglang_argv_matches_cookbook_and_hicache() -> None:
    argv = sglang_serve_argv("/ckpt", tp=1, context_length=262144, kv_offloading_size_gb=58)
    joined = " ".join(argv)
    assert argv[:2] == ["sglang", "serve"]
    assert "--model-path /ckpt" in joined
    assert "--kv-cache-dtype fp8_e4m3" in joined
    assert "--context-length 262144" in joined
    assert "--chunked-prefill-size 2048" in joined
    assert "--reasoning-parser qwen3" in joined
    assert "--tool-call-parser qwen3_coder" in joined
    assert "--mamba-full-memory-ratio 4.59" in joined
    assert "--mamba-radix-cache-strategy extra_buffer_lazy" in joined
    assert "--attention-backend flashinfer" in joined
    assert "--enable-hierarchical-cache" in joined
    assert "--hicache-size 58" in joined
    assert "--port 30000" in joined
    assert "vllm" not in joined
    assert "--seed" not in argv
    assert "--disable-cuda-graph" not in argv
    assert "--fp8-gemm-backend" not in argv
    assert "--fp4-gemm-backend" not in argv
    assert "--linear-attn-backend" not in argv
    assert "--sampling-backend" not in argv
    assert "--enable-deterministic-inference" not in argv
    default_recipe = load_eval_recipe(RECIPE)
    assert sglang_serve_environ(default_recipe) == {}
    with_graphs_off = sglang_serve_argv("/ckpt", disable_cuda_graph=True)
    assert "--disable-cuda-graph" in with_graphs_off


def test_vllm_argv_matches_nvidia_card_flags() -> None:
    argv = vllm_serve_argv("/ckpt", tp=1, max_model_len=262144, kv_offloading_size_gb=58)
    joined = " ".join(argv)
    assert "--kv-cache-dtype fp8_e4m3" in joined
    assert "--max-model-len 262144" in joined
    assert "--reasoning-parser qwen3" in joined
    assert "--seed 0" in joined
    assert "--max-num-batched-tokens 32768" in joined
    assert "--enable-chunked-prefill" in joined
    assert "--quantization modelopt" in joined
    assert "--kv-offloading-backend native" in joined
    assert "--kv-offloading-size 58" in joined
    assert "--swap-space" not in joined


def test_auto_kv_offload_uses_memtotal_minus_reserve() -> None:
    meminfo = "MemTotal:       67108864 kB\nMemAvailable:      1234 kB\n"
    # 24 GiB floor then /2: GDN pins a matching Mamba host pool (~2x).
    assert auto_kv_offload_gib(6, meminfo=meminfo) == 20
    assert auto_kv_offload_gib(6, meminfo=meminfo, explicit_gb=24) == 24
    assert auto_kv_offload_gib(8, meminfo=meminfo) == 20
    assert auto_kv_offload_gib(32, meminfo=meminfo) == 16
    assert auto_kv_offload_gib(6, meminfo="", explicit_gb=0) == 20


def test_vllm_argv_omits_swap_by_default_and_honors_explicit_swap() -> None:
    native = vllm_serve_argv("/ckpt", kv_offloading_size_gb=58, swap_space_gb=0)
    assert "--swap-space" not in native
    both = vllm_serve_argv("/ckpt", kv_offloading_size_gb=12, swap_space_gb=8)
    assert both[both.index("--swap-space") + 1] == "8"
    assert both[both.index("--kv-offloading-size") + 1] == "12"


def test_sglang_argv_from_recipe_follows_yaml_kv_offload() -> None:
    recipe = load_eval_recipe(RECIPE)
    recipe.serve.kv_offloading_size_gb = 40
    argv = sglang_serve_argv_from_recipe(recipe, "/export")
    joined = " ".join(argv)
    assert "--model-path" in argv
    assert argv[argv.index("--model-path") + 1] == "/export"
    assert "--enable-hierarchical-cache" in joined
    assert "--hicache-size 40" in joined
    assert "--context-length 262144" in joined
    assert "--disable-cuda-graph" in joined
    assert describe_eval(recipe)["kv_cpu_offload_gib"] == 40
    default = serve_argv_from_recipe(recipe, "/export")
    assert default[:2] == ["sglang", "serve"]


def test_vllm_argv_from_recipe_follows_yaml_kv_offload() -> None:
    recipe = load_eval_recipe(RECIPE)
    recipe.serve.engine = "vllm"
    recipe.serve.kv_offloading_size_gb = 40
    argv = vllm_serve_argv_from_recipe(recipe, "/export")
    joined = " ".join(argv)
    assert argv[2] == "/export"
    assert "--kv-offloading-backend native" in joined
    assert "--kv-offloading-size 40" in joined
    assert "--max-model-len 262144" in joined


def test_eval_base_url_prefers_sglang_then_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "MEGAQUANT_SGLANG_BASE_URL",
        "MEGAQUANT_BASE_URL",
        "MEGAQUANT_VLLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    assert eval_base_url_from_env() is None
    monkeypatch.setenv("MEGAQUANT_VLLM_BASE_URL", "http://vllm.local:8000/v1")
    assert eval_base_url_from_env() == "http://vllm.local:8000/v1"
    monkeypatch.setenv("MEGAQUANT_SGLANG_BASE_URL", "http://sglang.local:30000/v1")
    assert eval_base_url_from_env() == "http://sglang.local:30000/v1"


def test_cli_eval_and_serve_dry_run(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    from megaquant.cli import main

    for key in (
        "MEGAQUANT_SGLANG_BASE_URL",
        "MEGAQUANT_BASE_URL",
        "MEGAQUANT_VLLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    import json

    code = main(["eval", "-c", str(RECIPE), "--dry-run"])
    assert code == 0
    eval_out = capsys.readouterr().out
    assert "remaining context" in eval_out
    assert "native" in eval_out
    assert "http://127.0.0.1:30000/v1" in eval_out
    assert '"engine": "sglang"' in eval_out

    code = main(["serve", "-c", str(RECIPE), "--model", "/ckpt", "--dry-run"])
    assert code == 0
    serve_out = capsys.readouterr().out
    payload = serve_out[serve_out.find("{") :]
    data = json.loads(payload)
    joined = " ".join(data["argv"])
    assert data["argv"][:2] == ["sglang", "serve"]
    assert data["engine"] == "sglang"
    assert "--enable-hierarchical-cache" in joined
    assert "--kv-cache-dtype fp8_e4m3" in joined
    assert data["argv"][data["argv"].index("--model-path") + 1] == "/ckpt"
    assert data["kv_cpu_offload_gib"] >= 4
    assert "--port" in data["argv"]
    assert data["argv"][data["argv"].index("--port") + 1] == "30000"
    assert data["plan"]["base_url"] == "http://127.0.0.1:30000/v1"
    assert data["plan"]["sglang_quant"] is None
    assert data["sglang_quant"] is None


def test_dry_run_does_not_need_a_model() -> None:
    recipe = load_eval_recipe(RECIPE)
    out = run_gpqa(recipe, dry_run=True)
    assert out["dry_run"] is True
    assert out["plan"]["sampling"]["temperature"] == 0.0
    assert out["plan"]["sglang_quant"] is None


def _write_hf_quant(tmp_path: Path, quantization: dict) -> None:
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": quantization}),
        encoding="utf-8",
    )


def test_describe_eval_inspects_local_export(tmp_path: Path) -> None:
    from megaquant.sglang_export import inspect_sglang_quant_config

    _write_hf_quant(
        tmp_path,
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"lm_head": {"quant_algo": "NVFP4"}},
        },
    )
    recipe = load_eval_recipe(RECIPE, overrides={"model": str(tmp_path)})
    plan = describe_eval(recipe)
    assert plan["sglang_quant"] == inspect_sglang_quant_config(tmp_path)
    assert plan["sglang_quant"]["sglang_ok"] is True
    assert plan["sglang_quant"]["has_quantized_layers"] is True

    missing = load_eval_recipe(RECIPE, overrides={"model": "/ckpt"})
    assert describe_eval(missing)["sglang_quant"] is None


def test_serve_rejects_trtllm_w4a8_but_dry_run_does_not(
    tmp_path: Path, capsys
) -> None:
    from megaquant.cli import main

    _write_hf_quant(tmp_path, {"quant_algo": "W4A8_NVFP4_FP8", "group_size": 32})
    code = main(
        ["serve", "-c", str(RECIPE), "--model", str(tmp_path), "--dry-run"]
    )
    assert code == 0
    serve_out = capsys.readouterr().out
    data = json.loads(serve_out[serve_out.find("{") :])
    assert data["sglang_quant"]["sglang_ok"] is False
    assert data["plan"]["sglang_quant"]["quant_algo"] == "W4A8_NVFP4_FP8"
    assert data["argv"][:2] == ["sglang", "serve"]

    code = main(["serve", "-c", str(RECIPE), "--model", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "W4A8_NVFP4_FP8" in err
    assert "not the fix" in err
    assert "block 32" in err
    assert "w4a8_nvfp4_fp8" not in err
    assert "rewrite-sglang" in err
    assert "nvfp4_w4a8" in err


def _journal_item(item_id: str, gold: str = "A") -> ItemResult:
    return ItemResult(
        item_id=item_id,
        gold=gold,
        predicted=gold,
        correct=True,
        truncated=False,
        finish_reason="stop",
        prompt_tokens=8,
        completion_tokens=4,
        continued=0,
        text=f"Answer: {gold}",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
    )


def test_load_eval_recipe_missing_file_raises_eval_error(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-eval.yaml"
    with pytest.raises(EvalError, match="Eval recipe not found") as excinfo:
        load_eval_recipe(missing)
    assert str(missing) in str(excinfo.value)


def test_load_eval_recipe_invalid_yaml_raises_eval_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text(":\n  {", encoding="utf-8")
    with pytest.raises(EvalError, match="Invalid eval recipe YAML"):
        load_eval_recipe(bad)


def test_load_eval_recipe_non_mapping_raises_eval_error(tmp_path: Path) -> None:
    listed = tmp_path / "list.yaml"
    listed.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(EvalError, match="must be a mapping"):
        load_eval_recipe(listed)


def test_default_eval_recipe_resolves_from_checkout_when_cwd_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_eval_recipe_path(DEFAULT_EVAL_RECIPE)
    assert resolved.is_file()
    assert resolved == RECIPE
    recipe = load_eval_recipe(DEFAULT_EVAL_RECIPE)
    assert recipe.benchmark == "gpqa_diamond"


def test_cli_eval_and_serve_missing_recipe_prints_error_not_traceback(
    tmp_path: Path, capsys
) -> None:
    from megaquant.cli import main

    missing = str(tmp_path / "no-such-eval.yaml")
    for command in ("eval", "serve"):
        code = main([command, "-c", missing, "--dry-run"])
        captured = capsys.readouterr()
        assert code == 1
        assert captured.err.startswith("error:")
        assert "Eval recipe not found" in captured.err
        assert missing in captured.err
        assert "Traceback" not in captured.err
        assert "Traceback" not in captured.out


def test_cli_eval_default_recipe_works_outside_repo_root(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megaquant.cli import main

    monkeypatch.chdir(tmp_path)
    code = main(["eval", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "remaining context" in out
    assert '"engine": "sglang"' in out


def test_load_gpqa_journal_skips_truncated_last_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "gpqa_diamond.jsonl"
    good = json.dumps(asdict(_journal_item("done")), ensure_ascii=False)
    path.write_text(good + "\n{\"item_id\": \"pending\", \"gold\":", encoding="utf-8")
    rows = load_gpqa_journal(path)
    err = capsys.readouterr().err
    assert "done" in rows
    assert "pending" not in rows
    assert "warning" in err
    assert "dropping truncated journal line" in err
    assert path.read_text(encoding="utf-8") == good + "\n"


@pytest.mark.parametrize("truncated,predicted", [(True, "A"), (False, None)])
def test_run_gpqa_rescores_invalid_cached_answers_without_rewriting_journal(
    tmp_path: Path, truncated: bool, predicted: str | None
) -> None:
    recipe = load_eval_recipe(RECIPE, overrides={"output_dir": str(tmp_path)})
    row = _journal_item("done")
    row.truncated = truncated
    row.predicted = predicted
    row.finish_reason = "length" if truncated else "stop"
    original = (json.dumps(asdict(row)) + "\n").encode()
    journal = tmp_path / "gpqa_diamond.jsonl"
    journal.write_bytes(original)
    assert load_gpqa_journal(journal)["done"].correct is False

    def unexpected_generation(**_kwargs):
        pytest.fail("A complete cached row must not be regenerated")

    item = GPQAItem("done", "q", row.choices, row.gold)
    summary = run_gpqa(recipe, items=[item], complete=unexpected_generation)
    assert summary["scores"]["headline"] == "0/1"
    assert journal.read_bytes() == original


def test_load_gpqa_journal_truncate_keeps_earlier_bytes(tmp_path: Path) -> None:
    path = tmp_path / "gpqa_diamond.jsonl"
    first = json.dumps(asdict(_journal_item("a")), ensure_ascii=False)
    retry = json.dumps(asdict(_journal_item("a")))
    legacy = json.dumps({"item_id": "old", "note": "missing fields"})
    kept = f"{first}\n\n{retry}\n{legacy}\n"
    path.write_bytes(kept.encode("utf-8") + b'{"item_id": "b", "text": "\xe4\xb8')
    rows = load_gpqa_journal(path)
    assert set(rows) == {"a"}
    assert path.read_bytes() == kept.encode("utf-8")


def test_load_gpqa_journal_terminates_unterminated_valid_row(tmp_path: Path) -> None:
    path = tmp_path / "gpqa_diamond.jsonl"
    good = json.dumps(asdict(_journal_item("done")), ensure_ascii=False)
    path.write_text(good, encoding="utf-8")
    assert "done" in load_gpqa_journal(path)
    assert path.read_text(encoding="utf-8") == good + "\n"


def test_load_gpqa_journal_raises_on_corrupt_middle_line(tmp_path: Path) -> None:
    path = tmp_path / "gpqa_diamond.jsonl"
    first = json.dumps(asdict(_journal_item("a")), ensure_ascii=False)
    last = json.dumps(asdict(_journal_item("c")), ensure_ascii=False)
    path.write_text(f"{first}\n{{not json}}\n{last}\n", encoding="utf-8")
    with pytest.raises(EvalError, match="Corrupt GPQA journal") as excinfo:
        load_gpqa_journal(path)
    assert "line 2" in str(excinfo.value)
    # Middle corruption must not be rewritten away.
    assert "{not json}" in path.read_text(encoding="utf-8")


def test_run_gpqa_resumes_after_truncated_last_line(tmp_path: Path) -> None:
    recipe = load_eval_recipe(RECIPE, overrides={"output_dir": str(tmp_path)})
    journal = tmp_path / "gpqa_diamond.jsonl"
    done = json.dumps(asdict(_journal_item("done")), ensure_ascii=False)
    journal.write_text(done + "\n{\"item_id\": \"pending\"", encoding="utf-8")
    pending = GPQAItem(
        item_id="pending",
        question="Q2?",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
        gold="B",
    )
    already = GPQAItem(
        item_id="done",
        question="Q1?",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
        gold="A",
    )
    calls: list[str] = []

    def complete(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        return GenerationResult(
            text="Answer: B",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
        )

    summary = run_gpqa(
        recipe,
        items=[already, pending],
        complete=complete,
        count_prompt_tokens=lambda _t: 8,
    )
    assert calls
    assert all("Q1?" not in text for text in calls)
    assert summary["scores"]["headline"] == "2/2"
    lines = [ln for ln in journal.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
    ids = [json.loads(line)["item_id"] for line in lines]
    assert ids == ["done", "pending"]
