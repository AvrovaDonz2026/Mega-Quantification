"""GPQA Diamond protocol — no GPU, no Hub, no 27B."""

from __future__ import annotations

from pathlib import Path

from megaquant.eval_gpqa import (
    DEFAULT_MAX_MODEL_LEN,
    NVIDIA_VLLM_SERVE,
    QWEN_THINKING_SAMPLING,
    GenerationResult,
    GPQAItem,
    auto_kv_offload_gib,
    build_gpqa_item,
    describe_eval,
    extract_choice,
    format_gpqa_prompt,
    load_eval_recipe,
    remaining_new_tokens,
    run_gpqa,
    score_items,
    visible_answer_span,
    vllm_serve_argv,
    vllm_serve_argv_from_recipe,
)

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
    samp = recipe.sampling.model_dump()
    for key, value in QWEN_THINKING_SAMPLING.items():
        assert samp[key] == value, key
    serve = recipe.serve.model_dump()
    assert serve["kv_cache_dtype"] == NVIDIA_VLLM_SERVE["kv_cache_dtype"]
    assert serve["reasoning_parser"] == NVIDIA_VLLM_SERVE["reasoning_parser"]
    assert serve["max_num_batched_tokens"] == NVIDIA_VLLM_SERVE["max_num_batched_tokens"]
    assert serve["gpu_memory_utilization"] == NVIDIA_VLLM_SERVE["gpu_memory_utilization"]
    assert serve["kv_offloading_backend"] == "native"
    assert serve["kv_offloading_size_gb"] == 0
    assert serve["swap_space_gb"] == 0
    assert serve["cpu_reserve_gib"] == 6
    plan = describe_eval(recipe)
    assert "remaining context" in plan["max_new_tokens_policy"]
    assert plan["kv_offloading_backend"] == "native"
    assert plan["kv_cpu_offload_gib"] >= 4


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
    assert calls[0] == remaining_new_tokens(100, recipe.generation.max_model_len, 0)


def test_score_counts_truncated_and_unparsed_against_full_denominator() -> None:
    items = [
        GPQAItem("a", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A"),
        GPQAItem("b", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "B"),
    ]
    from megaquant.eval_gpqa import ItemResult

    rows = [
        ItemResult("a", "A", "A", True, False, "stop", 1, 1, 0, "Answer: A", items[0].choices),
        ItemResult("b", "B", None, False, True, "length", 1, 1, 0, "incomplete", items[1].choices),
    ]
    scores = score_items(items, rows)
    assert scores["headline"] == "1/2"
    assert scores["truncated"] == 1
    assert scores["unparsed"] == 1


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
    assert auto_kv_offload_gib(6, meminfo=meminfo) == 58
    assert auto_kv_offload_gib(6, meminfo=meminfo, explicit_gb=24) == 24
    assert auto_kv_offload_gib(8, meminfo=meminfo) == 56
    assert auto_kv_offload_gib(6, meminfo="", explicit_gb=0) == 58


def test_vllm_argv_omits_swap_by_default_and_honors_explicit_swap() -> None:
    native = vllm_serve_argv("/ckpt", kv_offloading_size_gb=58, swap_space_gb=0)
    assert "--swap-space" not in native
    both = vllm_serve_argv("/ckpt", kv_offloading_size_gb=12, swap_space_gb=8)
    assert both[both.index("--swap-space") + 1] == "8"
    assert both[both.index("--kv-offloading-size") + 1] == "12"


def test_vllm_argv_from_recipe_follows_yaml_kv_offload() -> None:
    recipe = load_eval_recipe(RECIPE)
    recipe.serve.kv_offloading_size_gb = 40
    argv = vllm_serve_argv_from_recipe(recipe, "/export")
    joined = " ".join(argv)
    assert argv[2] == "/export"
    assert "--kv-offloading-backend native" in joined
    assert "--kv-offloading-size 40" in joined
    assert "--max-model-len 262144" in joined
    assert describe_eval(recipe)["kv_cpu_offload_gib"] == 40


def test_cli_eval_and_serve_dry_run(capsys) -> None:
    from megaquant.cli import main

    code = main(["eval", "-c", str(RECIPE), "--dry-run"])
    assert code == 0
    eval_out = capsys.readouterr().out
    assert "remaining context" in eval_out
    assert "native" in eval_out

    code = main(["serve", "-c", str(RECIPE), "--model", "/ckpt", "--dry-run"])
    assert code == 0
    serve_out = capsys.readouterr().out
    payload = serve_out[serve_out.find("{") :]
    import json

    data = json.loads(payload)
    joined = " ".join(data["argv"])
    assert "--kv-offloading-backend native" in joined
    assert "--kv-cache-dtype fp8_e4m3" in joined
    assert data["argv"][2] == "/ckpt"
    assert data["kv_cpu_offload_gib"] >= 4


def test_dry_run_does_not_need_a_model() -> None:
    recipe = load_eval_recipe(RECIPE)
    out = run_gpqa(recipe, dry_run=True)
    assert out["dry_run"] is True
    assert out["plan"]["sampling"]["temperature"] == 1.0
