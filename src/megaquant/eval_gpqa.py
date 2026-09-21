"""GPQA Diamond eval matching the Qwen3.8-27B / NVIDIA NVFP4 model cards.

Official protocol this module encodes
-------------------------------------
Qwen/Qwen3.8-27B thinking-mode sampling (model card + generation_config.json)::

    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=0.0, repetition_penalty=1.0, do_sample=True
    enable_thinking=True, preserve_thinking=True, reasoning_effort=xhigh

NVIDIA ``nvidia/Qwen3.8-27B-NVFP4`` vLLM serve flags (GB300 card)::

    --kv-cache-dtype fp8_e4m3 --max-model-len 262144 --reasoning-parser qwen3
    --seed 0 --gpu-memory-utilization 0.85 --max-num-seqs 32
    --max-num-batched-tokens 32768 --enable-chunked-prefill

On a 32 GB card the 262144 window does not fit in HBM. KV that does not fit
is offloaded to host RAM (``--kv-offloading-backend native`` plus
``--kv-offloading-size`` = MemTotal − reserve). Do not shrink ``max_new_tokens``
to dodge VRAM.

Generation is **not** capped at a small ``max_new_tokens``. The budget is the
remaining context (``max_model_len - prompt_tokens``). If the server still
stops on length, the client continues until EOS. Completions are stored in
full in the JSONL journal (stdout only prints a short status line).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from megaquant.exceptions import EvalError
from megaquant.runtime import parse_meminfo_kib

LETTERS = ("A", "B", "C", "D")
DEFAULT_MAX_MODEL_LEN = 262144
# Qwen thinking traces on GPQA can run tens of thousands of tokens; never
# default to a 512/2048-style cap.
UNLIMITED = 0

QWEN_THINKING_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "do_sample": True,
}

NVIDIA_VLLM_SERVE: dict[str, Any] = {
    "kv_cache_dtype": "fp8_e4m3",
    "max_model_len": DEFAULT_MAX_MODEL_LEN,
    "reasoning_parser": "qwen3",
    "seed": 0,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 32768,
    "enable_chunked_prefill": True,
    "enable_auto_tool_choice": True,
    "tool_call_parser": "qwen3_coder",
    "mm_encoder_tp_mode": "data",
    "kv_offloading_backend": "native",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalSampling(StrictModel):
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    do_sample: bool = True


class EvalThinking(StrictModel):
    enable: bool = True
    preserve: bool = True
    reasoning_effort: str = "xhigh"


class EvalGeneration(StrictModel):
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    # 0 = use the entire remaining context window (no extra cap).
    max_new_tokens: int = UNLIMITED
    continue_on_length: bool = True
    seed: int = 0


class EvalServe(StrictModel):
    kv_cache_dtype: str = "fp8_e4m3"
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 32768
    reasoning_parser: str = "qwen3"
    enable_chunked_prefill: bool = True
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "qwen3_coder"
    mm_encoder_tp_mode: str = "data"
    quantization: str | None = "modelopt"
    tensor_parallel_size: int | None = None
    kv_offloading_backend: str = "native"
    # 0 = auto: MemTotal − cpu_reserve_gib. KV that does not fit HBM goes to RAM.
    kv_offloading_size_gb: float = 0
    # Legacy vLLM paging. Leave 0 when native KV offload is on (avoid double RAM).
    swap_space_gb: float = 0
    cpu_reserve_gib: int = 6


class EvalRecipe(StrictModel):
    name: str
    benchmark: str = "gpqa_diamond"
    dataset: str = "Idavidrein/gpqa"
    dataset_config: str = "gpqa_diamond"
    model: str = "outputs/Qwen3.8-27B-NVFP4-W4A8"
    backend: str = "auto"
    base_url: str | None = None
    thinking: EvalThinking = Field(default_factory=EvalThinking)
    sampling: EvalSampling = Field(default_factory=EvalSampling)
    generation: EvalGeneration = Field(default_factory=EvalGeneration)
    serve: EvalServe = Field(default_factory=EvalServe)
    output_dir: str = "outputs/eval/gpqa_diamond"
    limit: int | None = None
    shuffle_choices: bool = True


@dataclass
class GPQAItem:
    item_id: str
    question: str
    choices: dict[str, str]
    gold: str


@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    truncated: bool = False
    continued: int = 0


@dataclass
class ItemResult:
    item_id: str
    gold: str
    predicted: str | None
    correct: bool
    truncated: bool
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    continued: int
    text: str
    choices: dict[str, str]


def remaining_new_tokens(
    prompt_tokens: int,
    max_model_len: int,
    cap: int | None = UNLIMITED,
) -> int:
    """Tokens still available before the context fills. ``cap<=0`` means no extra cap."""
    room = max(1, int(max_model_len) - max(0, int(prompt_tokens)))
    if cap is None or int(cap) <= 0:
        return room
    return max(1, min(room, int(cap)))


def auto_kv_offload_gib(
    reserve_gib: int = 6,
    meminfo: str | None = None,
    explicit_gb: float | int | None = 0,
) -> int:
    """Host RAM reserved for KV that does not fit in GPU HBM.

    ``explicit_gb > 0`` wins. Otherwise ``MemTotal − reserve`` (min 4 GiB).
    A 64 GiB box with reserve 6 → 58 GiB CPU KV buffer.
    """
    if explicit_gb is not None and float(explicit_gb) > 0:
        return max(1, int(float(explicit_gb)))
    text = meminfo
    if text is None:
        try:
            text = Path("/proc/meminfo").read_text()
        except OSError:
            text = ""
    total_kib, _avail = parse_meminfo_kib(text or "")
    reserve = max(1, int(reserve_gib))
    if total_kib is None:
        return max(4, 64 - reserve)
    total_gib = int(total_kib) // 1024 // 1024
    return max(4, total_gib - reserve)


def load_eval_recipe(path: str | Path, overrides: dict[str, Any] | None = None) -> EvalRecipe:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise EvalError(f"{path} must be a mapping")
    if overrides:
        payload = {**payload, **{k: v for k, v in overrides.items() if v is not None}}
    try:
        return EvalRecipe.model_validate(payload)
    except Exception as exc:
        raise EvalError(str(exc)) from exc


def describe_eval(recipe: EvalRecipe) -> dict[str, Any]:
    return {
        "name": recipe.name,
        "benchmark": recipe.benchmark,
        "dataset": recipe.dataset,
        "model": recipe.model,
        "backend": recipe.backend,
        "thinking": recipe.thinking.model_dump(),
        "sampling": recipe.sampling.model_dump(),
        "generation": recipe.generation.model_dump(),
        "serve": recipe.serve.model_dump(),
        "truncation": "remaining_context_then_continue",
        "max_new_tokens_policy": (
            "remaining context (max_model_len - prompt_tokens); no small cap"
            if recipe.generation.max_new_tokens <= 0
            else f"min(remaining, {recipe.generation.max_new_tokens})"
        ),
        "official_thinking_sampling": QWEN_THINKING_SAMPLING,
        "official_vllm_serve": NVIDIA_VLLM_SERVE,
        "kv_cpu_offload_gib": auto_kv_offload_gib(
            recipe.serve.cpu_reserve_gib,
            explicit_gb=recipe.serve.kv_offloading_size_gb,
        ),
        "kv_offloading_backend": recipe.serve.kv_offloading_backend,
    }


def vllm_serve_argv(
    model: str,
    *,
    tp: int = 1,
    max_model_len: int | None = None,
    quantization: str | None = "modelopt",
    port: int = 8000,
    kv_cache_dtype: str | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    reasoning_parser: str | None = None,
    enable_chunked_prefill: bool = True,
    enable_auto_tool_choice: bool = True,
    tool_call_parser: str | None = None,
    mm_encoder_tp_mode: str | None = None,
    seed: int | None = None,
    kv_offloading_backend: str = "native",
    kv_offloading_size_gb: float | int = 0,
    swap_space_gb: float | int = 0,
    cpu_reserve_gib: int = 6,
) -> list[str]:
    """NVIDIA-card vLLM flags plus CPU KV offload so 262k gen fits a 32 GB card.

    Native KV CPU offload and ``--swap-space`` both consume host RAM. Leave
    ``swap_space_gb`` at 0 when native offload is on so a 64 GiB box is not
    double-booked.
    """
    length = int(max_model_len or NVIDIA_VLLM_SERVE["max_model_len"])
    kv_ram = auto_kv_offload_gib(cpu_reserve_gib, explicit_gb=kv_offloading_size_gb)
    argv = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--kv-cache-dtype",
        str(kv_cache_dtype or NVIDIA_VLLM_SERVE["kv_cache_dtype"]),
        "--tensor-parallel-size",
        str(max(1, tp)),
        "--max-model-len",
        str(length),
        "--reasoning-parser",
        str(reasoning_parser or NVIDIA_VLLM_SERVE["reasoning_parser"]),
        "--mm-encoder-tp-mode",
        str(mm_encoder_tp_mode or NVIDIA_VLLM_SERVE["mm_encoder_tp_mode"]),
        "--seed",
        str(NVIDIA_VLLM_SERVE["seed"] if seed is None else seed),
        "--gpu-memory-utilization",
        str(
            NVIDIA_VLLM_SERVE["gpu_memory_utilization"]
            if gpu_memory_utilization is None
            else gpu_memory_utilization
        ),
        "--max-num-seqs",
        str(NVIDIA_VLLM_SERVE["max_num_seqs"] if max_num_seqs is None else max_num_seqs),
        "--max-num-batched-tokens",
        str(
            NVIDIA_VLLM_SERVE["max_num_batched_tokens"]
            if max_num_batched_tokens is None
            else max_num_batched_tokens
        ),
        "--trust-remote-code",
        "--kv-offloading-backend",
        str(kv_offloading_backend or "native"),
        "--kv-offloading-size",
        str(kv_ram),
    ]
    if enable_auto_tool_choice:
        argv.extend(
            [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                str(tool_call_parser or NVIDIA_VLLM_SERVE["tool_call_parser"]),
            ]
        )
    if enable_chunked_prefill:
        argv.append("--enable-chunked-prefill")
    if quantization:
        argv.extend(["--quantization", quantization])
    if swap_space_gb is not None and float(swap_space_gb) > 0:
        argv.extend(["--swap-space", str(int(float(swap_space_gb)))])
    return argv


def vllm_serve_argv_from_recipe(
    recipe: EvalRecipe,
    model: str | None = None,
    *,
    port: int = 8000,
) -> list[str]:
    serve = recipe.serve
    tp = serve.tensor_parallel_size if serve.tensor_parallel_size else 1
    return vllm_serve_argv(
        model or recipe.model,
        tp=int(tp),
        max_model_len=recipe.generation.max_model_len,
        quantization=serve.quantization,
        port=port,
        kv_cache_dtype=serve.kv_cache_dtype,
        gpu_memory_utilization=serve.gpu_memory_utilization,
        max_num_seqs=serve.max_num_seqs,
        max_num_batched_tokens=serve.max_num_batched_tokens,
        reasoning_parser=serve.reasoning_parser,
        enable_chunked_prefill=serve.enable_chunked_prefill,
        enable_auto_tool_choice=serve.enable_auto_tool_choice,
        tool_call_parser=serve.tool_call_parser,
        mm_encoder_tp_mode=serve.mm_encoder_tp_mode,
        seed=recipe.generation.seed,
        kv_offloading_backend=serve.kv_offloading_backend,
        kv_offloading_size_gb=serve.kv_offloading_size_gb,
        swap_space_gb=serve.swap_space_gb,
        cpu_reserve_gib=serve.cpu_reserve_gib,
    )


def _stable_rng(seed: int, text: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\n{text}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def format_gpqa_prompt(item: GPQAItem) -> str:
    lines = [
        "Answer the following multiple-choice question.",
        "Think step by step. Do not stop until the reasoning is complete.",
        "The last line of your response must be exactly: Answer: $LETTER",
        "where LETTER is one of A, B, C, or D.",
        "",
        item.question.strip(),
        "",
    ]
    for letter in LETTERS:
        lines.append(f"{letter}. {item.choices[letter]}")
    return "\n".join(lines)


def build_gpqa_item(
    row: dict[str, Any],
    *,
    index: int,
    seed: int,
    shuffle: bool,
) -> GPQAItem:
    question = str(row.get("Question") or row.get("question") or "").strip()
    gold_text = str(row.get("Correct Answer") or row.get("correct_answer") or "").strip()
    wrong = [
        str(row.get("Incorrect Answer 1") or row.get("incorrect_answer_1") or "").strip(),
        str(row.get("Incorrect Answer 2") or row.get("incorrect_answer_2") or "").strip(),
        str(row.get("Incorrect Answer 3") or row.get("incorrect_answer_3") or "").strip(),
    ]
    options = [gold_text, *wrong]
    if shuffle:
        _stable_rng(seed, question).shuffle(options)
    choices = {letter: options[i] for i, letter in enumerate(LETTERS)}
    gold_letter = next(letter for letter, text in choices.items() if text == gold_text)
    item_id = str(row.get("Record ID") or row.get("id") or f"gpqa-{index:03d}")
    return GPQAItem(item_id=item_id, question=question, choices=choices, gold=gold_letter)


def load_gpqa_diamond(
    dataset: str = "Idavidrein/gpqa",
    config: str = "gpqa_diamond",
    *,
    seed: int = 0,
    shuffle: bool = True,
    limit: int | None = None,
    csv_path: str | Path | None = None,
) -> list[GPQAItem]:
    rows: list[dict[str, Any]]
    if csv_path:
        import csv

        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise EvalError(
                "datasets is required to load GPQA. pip install datasets "
                "or pass a local CSV via GPQA_CSV."
            ) from exc
        ds = load_dataset(dataset, config, split="train")
        rows = [dict(row) for row in ds]
    items = [
        build_gpqa_item(row, index=i, seed=seed, shuffle=shuffle) for i, row in enumerate(rows)
    ]
    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


_THINK_SPLIT = re.compile(
    r"</think>|</thinking>|</reasoning>",
    flags=re.IGNORECASE,
)
_ANSWER_LINE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*[:：]\s*\(?([A-D])\)?\s*[.\s]*$",
)
_BOXED = re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", flags=re.IGNORECASE)
_SOLE_LETTER = re.compile(r"(?im)^\s*\(?([A-D])\)?\s*[.\s]*$")


def visible_answer_span(text: str) -> str:
    """Drop thinking blocks so we do not score letters inside the trace."""
    parts = _THINK_SPLIT.split(text)
    if len(parts) > 1:
        return parts[-1]
    return text


def extract_choice(text: str) -> str | None:
    span = visible_answer_span(text)
    for pattern in (_ANSWER_LINE, _BOXED, _SOLE_LETTER):
        matches = list(pattern.finditer(span))
        if matches:
            return matches[-1].group(1).upper()
    loose = list(re.finditer(r"\b([A-D])\b", span))
    if loose:
        return loose[-1].group(1).upper()
    return None


def chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def openai_chat_complete(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    sampling: EvalSampling,
    thinking: EvalThinking,
    max_tokens: int,
    seed: int,
    timeout: float = 3600.0,
) -> GenerationResult:
    url = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": 0.0,
        "max_tokens": int(max_tokens),
        "seed": seed,
        "stream": False,
        "extra_body": {
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": thinking.enable,
                "preserve_thinking": thinking.preserve,
            },
            "reasoning_effort": thinking.reasoning_effort,
        },
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EvalError(f"chat completion HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise EvalError(f"chat completion failed: {exc}") from exc

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if reasoning and "<think>" not in content:
        text = f"<think>\n{reasoning}\n</think>\n{content}"
    else:
        text = content
    usage = data.get("usage") or {}
    finish = str(choice.get("finish_reason") or "stop")
    return GenerationResult(
        text=text,
        finish_reason=finish,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        truncated=finish in {"length", "max_tokens"},
    )


def generate_untruncated(
    prompt: str,
    *,
    recipe: EvalRecipe,
    prompt_tokens: int,
    complete: Callable[..., GenerationResult],
    model: str,
) -> GenerationResult:
    """Fill the remaining context; continue if the server still hits length."""
    max_len = recipe.generation.max_model_len
    cap = recipe.generation.max_new_tokens
    messages = chat_messages(prompt)
    budget = remaining_new_tokens(prompt_tokens, max_len, cap)
    first = complete(
        model=model,
        messages=messages,
        sampling=recipe.sampling,
        thinking=recipe.thinking,
        max_tokens=budget,
        seed=recipe.generation.seed,
    )
    if first.prompt_tokens:
        prompt_tokens = first.prompt_tokens
    truncated = first.truncated
    continued = 0
    text = first.text
    finish = first.finish_reason
    completion_tokens = first.completion_tokens
    if recipe.generation.continue_on_length:
        while truncated:
            spent = prompt_tokens + completion_tokens
            leftover = remaining_new_tokens(spent, max_len, cap)
            if leftover <= 1:
                break
            continued += 1
            follow = complete(
                model=model,
                messages=[
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "Continue from where you left off. Do not restart. "
                        "Finish the reasoning and end with `Answer: $LETTER`.",
                    },
                ],
                sampling=recipe.sampling,
                thinking=recipe.thinking,
                max_tokens=leftover,
                seed=recipe.generation.seed + continued,
            )
            text = text + follow.text
            completion_tokens += follow.completion_tokens
            finish = follow.finish_reason
            truncated = follow.truncated
            if continued >= 8:
                break
    return GenerationResult(
        text=text,
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        truncated=truncated,
        continued=continued,
    )


def score_items(
    items: Iterable[GPQAItem],
    results: Iterable[ItemResult],
) -> dict[str, Any]:
    items_list = list(items)
    rows = list(results)
    n = len(items_list)
    correct = sum(1 for row in rows if row.correct)
    truncated = sum(1 for row in rows if row.truncated)
    unparsed = sum(1 for row in rows if row.predicted is None)
    return {
        "n": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "denominator": n,
        "truncated": truncated,
        "unparsed": unparsed,
        "headline": f"{correct}/{n}",
    }


def run_gpqa(
    recipe: EvalRecipe,
    *,
    items: list[GPQAItem] | None = None,
    complete: Callable[..., GenerationResult] | None = None,
    count_prompt_tokens: Callable[[str], int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = describe_eval(recipe)
    if dry_run:
        return {"dry_run": True, "plan": plan}

    if items is None:
        csv_path = None
        import os

        csv_path = os.environ.get("GPQA_CSV") or None
        items = load_gpqa_diamond(
            recipe.dataset,
            recipe.dataset_config,
            seed=recipe.generation.seed,
            shuffle=recipe.shuffle_choices,
            limit=recipe.limit,
            csv_path=csv_path,
        )
    if complete is None:
        base = recipe.base_url or ""
        if recipe.backend in {"auto", "openai"} and base:
            def complete(**kwargs: Any) -> GenerationResult:
                return openai_chat_complete(base_url=base, **kwargs)
        else:
            raise EvalError(
                "Set --base-url to an OpenAI-compatible vLLM server "
                "(NVIDIA-card flags) or inject a generator in tests."
            )

    def n_tokens(text: str) -> int:
        if count_prompt_tokens is not None:
            return int(count_prompt_tokens(text))
        return max(1, len(text) // 4)

    out_dir = Path(recipe.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "gpqa_diamond.jsonl"
    rows: list[ItemResult] = []
    with journal_path.open("w", encoding="utf-8") as journal:
        for item in items:
            prompt = format_gpqa_prompt(item)
            gen = generate_untruncated(
                prompt,
                recipe=recipe,
                prompt_tokens=n_tokens(prompt),
                complete=complete,
                model=recipe.model,
            )
            predicted = extract_choice(gen.text)
            row = ItemResult(
                item_id=item.item_id,
                gold=item.gold,
                predicted=predicted,
                correct=predicted == item.gold,
                truncated=gen.truncated,
                finish_reason=gen.finish_reason,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                continued=gen.continued,
                text=gen.text,
                choices=item.choices,
            )
            rows.append(row)
            journal.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            journal.flush()
            print(
                f"[gpqa] {item.item_id} gold={item.gold} pred={predicted} "
                f"ok={int(row.correct)} trunc={int(row.truncated)} "
                f"tok={gen.completion_tokens} cont={gen.continued}",
                flush=True,
            )

    summary = {
        "plan": plan,
        "scores": score_items(items, rows),
        "journal": str(journal_path),
        "model": recipe.model,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[gpqa] {summary['scores']['headline']} "
        f"truncated={summary['scores']['truncated']} "
        f"unparsed={summary['scores']['unparsed']}",
        flush=True,
    )
    return summary
