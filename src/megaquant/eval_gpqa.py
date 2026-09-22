"""GPQA Diamond eval for the Qwen3.8-27B / NVIDIA NVFP4 checkpoints.

Sampling this module sends
--------------------------
Eval recipes use temperature 0. ``QWEN_THINKING_SAMPLING`` keeps the published
Qwen / NVIDIA card (temperature 1.0) so the two protocols stay distinct.
Everything else matches Qwen/Qwen3.8-27B thinking mode::

    temperature=0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=0.0, repetition_penalty=1.0, do_sample=True
    enable_thinking=True, preserve_thinking=True, reasoning_effort=xhigh

Default serve engine is **SGLang** (NVIDIA Qwen3.8 cookbook + mixed NVFP4)::

    sglang serve --model-path … --kv-cache-dtype fp8_e4m3
    --mem-fraction-static 0.85 --chunked-prefill-size 2048
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder
    --mamba-full-memory-ratio 4.59 --mamba-radix-cache-strategy extra_buffer_lazy
    --mamba-ssm-dtype float32 --attention-backend flashinfer
    --context-length 262144 --port 30000

On a 32 GB card the 262144 window does not fit in HBM. KV that does not fit
is offloaded to host RAM (SGLang ``--enable-hierarchical-cache`` plus
``--hicache-size`` = MemTotal − reserve). Optional ``engine: vllm`` keeps the
NVIDIA GB300 vLLM flags (``--kv-offloading-backend native``). Do not shrink
``max_new_tokens`` to dodge VRAM.

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
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from megaquant.exceptions import EvalError
from megaquant.runtime import parse_meminfo_kib
from megaquant.sglang_export import sglang_quant_snapshot

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

DEFAULT_SGLANG_PORT = 30000
DEFAULT_VLLM_PORT = 8000
DEFAULT_EVAL_BASE_URL = f"http://127.0.0.1:{DEFAULT_SGLANG_PORT}/v1"

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

# SGLang cookbook (Qwen3.8-27B NVFP4) + NVIDIA mixed card extras.
NVIDIA_SGLANG_SERVE: dict[str, Any] = {
    "kv_cache_dtype": "fp8_e4m3",
    "mem_fraction_static": 0.85,
    "chunked_prefill_size": 2048,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "mamba_full_memory_ratio": 4.59,
    "mamba_radix_cache_strategy": "extra_buffer_lazy",
    "mamba_ssm_dtype": "float32",
    "attention_backend": "flashinfer",
    "context_length": DEFAULT_MAX_MODEL_LEN,
    "host": "0.0.0.0",
    "port": DEFAULT_SGLANG_PORT,
    "enable_hierarchical_cache": True,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalSampling(StrictModel):
    temperature: float = 0.0
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
    engine: Literal["sglang", "vllm"] = "sglang"
    host: str = "0.0.0.0"
    kv_cache_dtype: str = "fp8_e4m3"
    gpu_memory_utilization: float = 0.85
    mem_fraction_static: float | None = None
    max_num_seqs: int = 32
    max_running_requests: int | None = None
    max_num_batched_tokens: int = 32768
    chunked_prefill_size: int = 2048
    reasoning_parser: str = "qwen3"
    enable_chunked_prefill: bool = True
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = "qwen3_coder"
    mm_encoder_tp_mode: str = "data"
    quantization: str | None = "modelopt"
    tensor_parallel_size: int | None = None
    attention_backend: str | None = "flashinfer"
    sampling_backend: str | None = None
    linear_attn_backend: str | None = None
    fp8_gemm_backend: str | None = None
    fp4_gemm_backend: str | None = None
    # ModelOpt FP8 linear on SM120 tries KDA, then FlashInfer BMM, and only
    # then --fp8-gemm-backend. CUDA 12.8 FlashInfer JIT cannot see SM 12.0, so
    # 5090 recipes set this and cmd_serve exports SGLANG_FORCE_FP8_MARLIN=1.
    force_fp8_marlin: bool = False
    # Fused SiLU+FP4 quant imports FlashInfer even when the FP4 GEMM backend
    # is Marlin. SM120 + CUDA 12.8 then dies in the SM120f JIT. 80 GB recipes
    # set this; cmd_serve exports SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1.
    disable_silu_fp4_quant_fusion: bool = False
    # None leaves the process env alone. False exports the SM120 workaround.
    flashinfer_available: bool | None = None
    enable_jit_deepgemm: bool | None = None
    mamba_backend: str | None = None
    disable_radix_cache: bool = False
    disable_flashinfer_autotune: bool = False
    mamba_full_memory_ratio: float | None = 4.59
    mamba_radix_cache_strategy: str = "extra_buffer_lazy"
    mamba_ssm_dtype: str = "float32"
    max_mamba_cache_size: int | None = None
    enable_hierarchical_cache: bool = True
    kv_offloading_backend: str = "native"
    # 0 = auto: MemTotal − cpu_reserve_gib. KV that does not fit HBM goes to RAM.
    kv_offloading_size_gb: float = 0
    # Legacy vLLM paging. Leave 0 when native KV offload is on (avoid double RAM).
    swap_space_gb: float = 0
    cpu_reserve_gib: int = 6
    # Mixed NVFP4 + Mamba + HiCache on a 32 GB card OOMs during CUDA-graph capture.
    # Cookbook GB300 boxes can set this false.
    disable_cuda_graph: bool = False


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
    # Empty means `<model>/gpqa_diamond` (full traces beside the weights).
    output_dir: str = ""
    limit: int | None = None
    shuffle_choices: bool = True
    # HTTP client fan-out. Serve max_running_requests is the hard cap (GDN
    # mamba slots). Sequential (1) wastes a 5090 that can hold ~8 in-flight
    # Diamond traces once HiCache is sized for them.
    concurrency: int = 1


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


# Chat-template / special-token headroom. The eval client estimates prompt
# tokens before the first HTTP round trip; Qwen3.8 templates add tens of
# tokens, and SGLang 400s if prompt + max_tokens exceeds context_length.
CHAT_TEMPLATE_TOKEN_RESERVE = 256

_CONTEXT_OVERFLOW = re.compile(
    r"maximum context length of (\d+) tokens.*?"
    r"(\d+) tokens from the input messages and "
    r"(\d+) tokens for the completion",
    flags=re.IGNORECASE | re.DOTALL,
)


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


HICACHE_MIN_HOST_RESERVE_GIB = 24


def auto_kv_offload_gib(
    reserve_gib: int = 6,
    meminfo: str | None = None,
    explicit_gb: float | int | None = 0,
) -> int:
    """Host RAM reserved for KV that does not fit in GPU HBM.

    ``explicit_gb > 0`` wins. Otherwise half of
    ``MemTotal − max(reserve, 24)`` (min 4 GiB). Qwen3.5 GDN HiCache
    also pins a matching Mamba host pool, so ``--hicache-size N`` costs
    ~2N GiB of pinned RAM. Cookbook 58 GiB is for GB300-class hosts.
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
    reserve = max(int(reserve_gib), HICACHE_MIN_HOST_RESERVE_GIB)
    if total_kib is None:
        usable = max(4, 64 - reserve)
    else:
        total_gib = int(total_kib) // 1024 // 1024
        usable = max(4, total_gib - reserve)
    return max(4, usable // 2)


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


def eval_base_url_from_env() -> str | None:
    """Prefer SGLang URL; keep MEGAQUANT_VLLM_BASE_URL as a fallback alias."""
    import os

    for key in (
        "MEGAQUANT_SGLANG_BASE_URL",
        "MEGAQUANT_BASE_URL",
        "MEGAQUANT_VLLM_BASE_URL",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def gpqa_trace_dir(model: str, output_dir: str | None = None) -> Path:
    """Folder for GPQA traces. The default sits inside the model directory."""
    raw = (output_dir or "").strip()
    if not raw:
        return Path(model) / "gpqa_diamond"
    return Path(raw)


def default_eval_base_url(recipe: EvalRecipe | None = None) -> str:
    """Local OpenAI-compatible URL for the selected serve engine."""
    if recipe is not None and recipe.serve.engine == "vllm":
        return f"http://127.0.0.1:{DEFAULT_VLLM_PORT}/v1"
    return DEFAULT_EVAL_BASE_URL


def describe_eval(recipe: EvalRecipe) -> dict[str, Any]:
    kv_backend = recipe.serve.kv_offloading_backend
    if recipe.serve.engine == "sglang" and kv_backend in {"native", "hicache", ""}:
        kv_backend = "hicache"
    return {
        "name": recipe.name,
        "benchmark": recipe.benchmark,
        "dataset": recipe.dataset,
        "model": recipe.model,
        "backend": recipe.backend,
        "base_url": recipe.base_url,
        "engine": recipe.serve.engine,
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
        "official_sglang_serve": NVIDIA_SGLANG_SERVE,
        "official_vllm_serve": NVIDIA_VLLM_SERVE,
        "kv_cpu_offload_gib": auto_kv_offload_gib(
            recipe.serve.cpu_reserve_gib,
            explicit_gb=recipe.serve.kv_offloading_size_gb,
        ),
        "concurrency": recipe.concurrency,
        "journal_dir": str(gpqa_trace_dir(recipe.model, recipe.output_dir)),
        "kv_offloading_backend": kv_backend,
        # Local export only: missing dir -> null (never Hub-download 27B).
        "sglang_quant": sglang_quant_snapshot(recipe.model),
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


def sglang_serve_argv(
    model: str,
    *,
    tp: int = 1,
    port: int = DEFAULT_SGLANG_PORT,
    host: str = "0.0.0.0",
    context_length: int | None = None,
    kv_cache_dtype: str | None = None,
    mem_fraction_static: float | None = None,
    max_running_requests: int | None = None,
    chunked_prefill_size: int | None = None,
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
    mamba_full_memory_ratio: float | None = None,
    mamba_radix_cache_strategy: str | None = None,
    mamba_ssm_dtype: str | None = None,
    attention_backend: str | None = None,
    sampling_backend: str | None = None,
    linear_attn_backend: str | None = None,
    fp8_gemm_backend: str | None = None,
    fp4_gemm_backend: str | None = None,
    mamba_backend: str | None = None,
    disable_radix_cache: bool = False,
    disable_flashinfer_autotune: bool = False,
    max_mamba_cache_size: int | None = None,
    seed: int | None = None,
    enable_hierarchical_cache: bool = True,
    kv_offloading_backend: str = "native",
    kv_offloading_size_gb: float | int = 0,
    cpu_reserve_gib: int = 6,
    disable_cuda_graph: bool = False,
) -> list[str]:
    """NVIDIA/SGLang cookbook flags plus HiCache so 262k gen fits a 32 GB card.

    ``kv_offloading_backend`` in ``{native, hicache}`` turns on
    ``--enable-hierarchical-cache`` with ``--hicache-size`` = MemTotal − reserve.
    """
    length = int(context_length or NVIDIA_SGLANG_SERVE["context_length"])
    kv_ram = auto_kv_offload_gib(cpu_reserve_gib, explicit_gb=kv_offloading_size_gb)
    argv = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--kv-cache-dtype",
        str(kv_cache_dtype or NVIDIA_SGLANG_SERVE["kv_cache_dtype"]),
        "--mem-fraction-static",
        str(
            NVIDIA_SGLANG_SERVE["mem_fraction_static"]
            if mem_fraction_static is None
            else mem_fraction_static
        ),
        "--context-length",
        str(length),
        "--chunked-prefill-size",
        str(
            NVIDIA_SGLANG_SERVE["chunked_prefill_size"]
            if chunked_prefill_size is None
            else chunked_prefill_size
        ),
        "--reasoning-parser",
        str(reasoning_parser or NVIDIA_SGLANG_SERVE["reasoning_parser"]),
        "--tool-call-parser",
        str(tool_call_parser or NVIDIA_SGLANG_SERVE["tool_call_parser"]),
        "--tp-size",
        str(max(1, tp)),
        "--max-running-requests",
        str(
            NVIDIA_VLLM_SERVE["max_num_seqs"]
            if max_running_requests is None
            else max_running_requests
        ),
        "--mamba-radix-cache-strategy",
        str(
            mamba_radix_cache_strategy
            or NVIDIA_SGLANG_SERVE["mamba_radix_cache_strategy"]
        ),
        "--mamba-ssm-dtype",
        str(mamba_ssm_dtype or NVIDIA_SGLANG_SERVE["mamba_ssm_dtype"]),
    ]
    ratio = (
        NVIDIA_SGLANG_SERVE["mamba_full_memory_ratio"]
        if mamba_full_memory_ratio is None
        else mamba_full_memory_ratio
    )
    if ratio is not None:
        argv.extend(["--mamba-full-memory-ratio", str(ratio)])
    backend = attention_backend
    if backend is None:
        backend = NVIDIA_SGLANG_SERVE["attention_backend"]
    if backend:
        argv.extend(["--attention-backend", str(backend)])
    if sampling_backend:
        argv.extend(["--sampling-backend", str(sampling_backend)])
    if linear_attn_backend:
        argv.extend(["--linear-attn-backend", str(linear_attn_backend)])
    if fp8_gemm_backend:
        argv.extend(["--fp8-gemm-backend", str(fp8_gemm_backend)])
    if fp4_gemm_backend:
        argv.extend(["--fp4-gemm-backend", str(fp4_gemm_backend)])
    if mamba_backend:
        argv.extend(["--mamba-backend", str(mamba_backend)])
    if disable_radix_cache:
        argv.append("--disable-radix-cache")
    if disable_flashinfer_autotune:
        argv.append("--disable-flashinfer-autotune")
    if max_mamba_cache_size is not None and int(max_mamba_cache_size) > 0:
        argv.extend(["--max-mamba-cache-size", str(int(max_mamba_cache_size))])
    offload = (kv_offloading_backend or "native").strip().lower()
    if enable_hierarchical_cache and offload not in {"none", "off", "false"}:
        argv.extend(["--enable-hierarchical-cache", "--hicache-size", str(kv_ram)])
    if disable_cuda_graph:
        argv.append("--disable-cuda-graph")
    # SGLang 0.5.x `serve` has no --seed; sampling seed stays on the eval client.
    _ = seed
    return argv


def sglang_serve_argv_from_recipe(
    recipe: EvalRecipe,
    model: str | None = None,
    *,
    port: int | None = None,
) -> list[str]:
    serve = recipe.serve
    tp = serve.tensor_parallel_size if serve.tensor_parallel_size else 1
    mem = serve.mem_fraction_static
    if mem is None:
        mem = serve.gpu_memory_utilization
    running = serve.max_running_requests
    if running is None:
        running = serve.max_num_seqs
    return sglang_serve_argv(
        model or recipe.model,
        tp=int(tp),
        port=int(port or DEFAULT_SGLANG_PORT),
        host=serve.host,
        context_length=recipe.generation.max_model_len,
        kv_cache_dtype=serve.kv_cache_dtype,
        mem_fraction_static=mem,
        max_running_requests=running,
        chunked_prefill_size=serve.chunked_prefill_size,
        reasoning_parser=serve.reasoning_parser,
        tool_call_parser=serve.tool_call_parser,
        mamba_full_memory_ratio=serve.mamba_full_memory_ratio,
        mamba_radix_cache_strategy=serve.mamba_radix_cache_strategy,
        mamba_ssm_dtype=serve.mamba_ssm_dtype,
        attention_backend=serve.attention_backend,
        sampling_backend=serve.sampling_backend,
        linear_attn_backend=serve.linear_attn_backend,
        fp8_gemm_backend=serve.fp8_gemm_backend,
        fp4_gemm_backend=serve.fp4_gemm_backend,
        mamba_backend=serve.mamba_backend,
        disable_radix_cache=serve.disable_radix_cache,
        disable_flashinfer_autotune=serve.disable_flashinfer_autotune,
        max_mamba_cache_size=serve.max_mamba_cache_size,
        seed=recipe.generation.seed,
        enable_hierarchical_cache=serve.enable_hierarchical_cache,
        kv_offloading_backend=serve.kv_offloading_backend,
        kv_offloading_size_gb=serve.kv_offloading_size_gb,
        cpu_reserve_gib=serve.cpu_reserve_gib,
        disable_cuda_graph=serve.disable_cuda_graph,
    )


def sglang_serve_environ(recipe: EvalRecipe) -> dict[str, str]:
    """Extra process env for ``sglang serve`` (not CLI flags)."""
    env: dict[str, str] = {}
    serve = recipe.serve
    if serve.engine != "sglang":
        return env
    if serve.force_fp8_marlin:
        env["SGLANG_FORCE_FP8_MARLIN"] = "1"
    if serve.disable_silu_fp4_quant_fusion:
        env["SGLANG_DISABLE_SILU_FP4_QUANT_FUSION"] = "1"
    if serve.flashinfer_available is False:
        env["SGLANG_IS_FLASHINFER_AVAILABLE"] = "0"
    if serve.enable_jit_deepgemm is False:
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"
    return env


def serve_argv_from_recipe(
    recipe: EvalRecipe,
    model: str | None = None,
    *,
    port: int | None = None,
) -> list[str]:
    if recipe.serve.engine == "vllm":
        return vllm_serve_argv_from_recipe(
            recipe, model, port=int(port or DEFAULT_VLLM_PORT)
        )
    return sglang_serve_argv_from_recipe(recipe, model, port=port)


def default_serve_port(recipe: EvalRecipe) -> int:
    if recipe.serve.engine == "vllm":
        return DEFAULT_VLLM_PORT
    return DEFAULT_SGLANG_PORT


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
    # xhigh traces at ~20 tok/s run well past one hour. 21600 matches the
    # 80 GB GPQA client that already scored this checkpoint.
    timeout: float = 21600.0,
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
    current_max = int(max_tokens)
    data: dict[str, Any] | None = None
    last_detail = ""
    for attempt in range(2):
        body["max_tokens"] = current_max
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
            break
        except urllib.error.HTTPError as exc:
            last_detail = exc.read().decode("utf-8", errors="replace")
            match = _CONTEXT_OVERFLOW.search(last_detail) if exc.code == 400 else None
            if match is None or attempt == 1:
                raise EvalError(
                    f"chat completion HTTP {exc.code}: {last_detail[:500]}"
                ) from exc
            ctx, prompt_n, _comp = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            retried = max(1, ctx - prompt_n - 8)
            if retried >= current_max:
                raise EvalError(
                    f"chat completion HTTP {exc.code}: {last_detail[:500]}"
                ) from exc
            current_max = retried
        except urllib.error.URLError as exc:
            raise EvalError(f"chat completion failed: {exc}") from exc
    if data is None:
        raise EvalError(f"chat completion HTTP 400: {last_detail[:500]}")

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
    budget = remaining_new_tokens(
        prompt_tokens + CHAT_TEMPLATE_TOKEN_RESERVE, max_len, cap
    )
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
            leftover = remaining_new_tokens(
                spent + CHAT_TEMPLATE_TOKEN_RESERVE, max_len, cap
            )
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


def load_gpqa_journal(path: Path) -> dict[str, ItemResult]:
    """Replay a JSONL journal so a restarted eval does not wipe finished items."""
    rows: dict[str, ItemResult] = {}
    if not path.is_file():
        return rows
    fields = ItemResult.__dataclass_fields__
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        kwargs = {key: payload[key] for key in fields if key in payload}
        try:
            row = ItemResult(**kwargs)
        except TypeError:
            continue
        rows[row.item_id] = row
    return rows


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
                "Set --base-url to an OpenAI-compatible SGLang server "
                "(cookbook flags, default http://127.0.0.1:30000/v1) "
                "or inject a generator in tests."
            )

    def n_tokens(text: str) -> int:
        if count_prompt_tokens is not None:
            return int(count_prompt_tokens(text))
        return max(1, len(text) // 4)

    out_dir = gpqa_trace_dir(recipe.model, recipe.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "gpqa_diamond.jsonl"
    prior = load_gpqa_journal(journal_path)
    rows: list[ItemResult] = []
    lock = threading.Lock()
    mode = "a" if prior else "w"
    workers = max(1, int(recipe.concurrency or 1))

    def eval_one(item: GPQAItem) -> ItemResult:
        prompt = format_gpqa_prompt(item)
        gen = generate_untruncated(
            prompt,
            recipe=recipe,
            prompt_tokens=n_tokens(prompt),
            complete=complete,
            model=recipe.model,
        )
        predicted = extract_choice(gen.text)
        return ItemResult(
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

    def write_row(row: ItemResult) -> None:
        with lock:
            rows.append(row)
            journal.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            journal.flush()
            print(
                f"[gpqa] {row.item_id} gold={row.gold} pred={row.predicted} "
                f"ok={int(row.correct)} trunc={int(row.truncated)} "
                f"tok={row.completion_tokens} cont={row.continued}",
                flush=True,
            )

    with journal_path.open(mode, encoding="utf-8") as journal:
        if prior:
            print(
                f"[gpqa] resume {len(prior)} rows from {journal_path}",
                flush=True,
            )
        pending: list[GPQAItem] = []
        for item in items:
            if item.item_id in prior:
                rows.append(prior[item.item_id])
                continue
            pending.append(item)
        if workers > 1 and len(pending) > 1:
            print(
                f"[gpqa] concurrency={workers} pending={len(pending)}/{len(items)}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(eval_one, item) for item in pending]
                for fut in as_completed(futs):
                    write_row(fut.result())
        else:
            for item in pending:
                write_row(eval_one(item))

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
