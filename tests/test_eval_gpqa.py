"""GPQA Diamond protocol — no GPU, no Hub, no 27B."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megaquant.eval_gpqa import (
    DEFAULT_EVAL_BASE_URL,
    DEFAULT_MAX_MODEL_LEN,
    NVIDIA_SGLANG_SERVE,
    QWEN_THINKING_SAMPLING,
    GenerationResult,
    GPQAItem,
    auto_kv_offload_gib,
    build_gpqa_item,
    default_eval_base_url,
    describe_eval,
    eval_base_url_from_env,
    extract_choice,
    format_gpqa_prompt,
    load_eval_recipe,
    remaining_new_tokens,
    run_gpqa,
    score_items,
    serve_argv_from_recipe,
    sglang_serve_argv,
    sglang_serve_argv_from_recipe,
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
    assert serve["engine"] == "sglang"
    assert serve["kv_cache_dtype"] == NVIDIA_SGLANG_SERVE["kv_cache_dtype"]
    assert serve["reasoning_parser"] == NVIDIA_SGLANG_SERVE["reasoning_parser"]
    assert serve["chunked_prefill_size"] == NVIDIA_SGLANG_SERVE["chunked_prefill_size"]
    assert serve["mamba_full_memory_ratio"] == NVIDIA_SGLANG_SERVE["mamba_full_memory_ratio"]
    assert serve["mamba_radix_cache_strategy"] == "extra_buffer_lazy"
    assert serve["enable_hierarchical_cache"] is True
    assert serve["kv_offloading_backend"] == "native"
    assert serve["kv_offloading_size_gb"] == 0
    assert serve["swap_space_gb"] == 0
    assert serve["cpu_reserve_gib"] == 6
    assert serve["disable_cuda_graph"] is True
    plan = describe_eval(recipe)
    assert plan["engine"] == "sglang"
    assert plan["base_url"] is None
    assert "remaining context" in plan["max_new_tokens_policy"]
    assert plan["kv_offloading_backend"] == "hicache"
    assert plan["kv_cpu_offload_gib"] >= 4
    assert default_eval_base_url(recipe) == DEFAULT_EVAL_BASE_URL
    assert default_eval_base_url(recipe) == "http://127.0.0.1:30000/v1"


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
    assert out["plan"]["sampling"]["temperature"] == 1.0
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
    assert "w4a8_nvfp4_fp8" in err
    assert "rewrite-sglang" in err
    assert "nvfp4_w4a8" in err
