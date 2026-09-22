"""Command-line interface for Mega-Quantification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from megaquant.config import CLI_OVERRIDE_MAP, load_recipe
from megaquant.eval_gpqa import (
    default_eval_base_url,
    default_serve_port,
    describe_eval,
    eval_base_url_from_env,
    load_eval_recipe,
    run_gpqa,
    serve_argv_from_recipe,
    sglang_serve_environ,
)
from megaquant.exceptions import MegaQuantError
from megaquant.pipeline import QuantPipeline
from megaquant.registry import list_families, list_schemes
from megaquant.sglang_export import sglang_quant_snapshot


def _add_recipe_flags(parser: argparse.ArgumentParser, *, with_dry_run: bool) -> None:
    parser.add_argument("-c", "--config", required=True, help="Recipe YAML path")
    if with_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the resolved plan without loading weights",
        )
    parser.add_argument("--model", help="Override model.source")
    parser.add_argument("--output", help="Override export.output_dir")
    parser.add_argument(
        "--backend",
        choices=["auto", "modelopt", "llmcompressor"],
        help="Override backend",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override calibration.num_samples",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        dest="max_seq_length",
        help="Override calibration.max_seq_length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        help="Override calibration.batch_size (larger batches amortize CPU↔GPU offload)",
    )
    parser.add_argument("--algorithm", help="Override algorithm")
    parser.add_argument("--scheme", help="Override scheme catalog key")


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for attr, dotted in CLI_OVERRIDE_MAP.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[dotted] = value
    return overrides


def cmd_quantize(args: argparse.Namespace) -> int:
    overrides = _overrides_from_args(args)
    recipe = load_recipe(args.config, overrides=overrides or None)
    pipeline = QuantPipeline(recipe)
    result = pipeline.run(dry_run=bool(getattr(args, "dry_run", False)))
    if isinstance(result, Path):
        print(result)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    args.dry_run = True
    return cmd_quantize(args)


def cmd_schemes(_args: argparse.Namespace) -> int:
    names = list_schemes()
    if not names:
        print("No schemes registered (catalog not loaded).")
        return 0
    for name in names:
        print(name)
    return 0


def cmd_families(_args: argparse.Namespace) -> int:
    names = list_families()
    if not names:
        print("No families registered.")
        return 0
    for name in names:
        print(name)
    return 0


def cmd_rewrite_sglang(args: argparse.Namespace) -> int:
    import json

    from megaquant.sglang_export import rewrite_sglang_mixed_export

    layers = rewrite_sglang_mixed_export(args.export_dir)
    print(
        json.dumps(
            {
                "export_dir": str(args.export_dir),
                "quant_algo": "MIXED_PRECISION",
                "n_quantized_layers": len(layers),
            },
            indent=2,
        )
    )
    return 0


def _eval_recipe_from_args(args: argparse.Namespace):
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "output", None):
        overrides["output_dir"] = args.output
    if getattr(args, "limit", None) is not None:
        overrides["limit"] = args.limit
    if getattr(args, "concurrency", None) is not None:
        overrides["concurrency"] = args.concurrency
    recipe = load_eval_recipe(args.config, overrides or None)
    engine = getattr(args, "engine", None)
    if engine:
        recipe.serve.engine = engine
    recipe.base_url = (
        getattr(args, "base_url", None)
        or eval_base_url_from_env()
        or recipe.base_url
        or default_eval_base_url(recipe)
    )
    return recipe


def cmd_eval(args: argparse.Namespace) -> int:
    import json

    recipe = _eval_recipe_from_args(args)
    result = run_gpqa(recipe, dry_run=bool(getattr(args, "dry_run", False)))
    payload = result.get("plan") if result.get("dry_run") else result.get("scores")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import json
    import os
    import sys

    recipe = _eval_recipe_from_args(args)
    port = getattr(args, "port", None)
    if port is None:
        port = default_serve_port(recipe)
    argv = serve_argv_from_recipe(
        recipe,
        getattr(args, "model", None),
        port=int(port),
    )
    extra_env = sglang_serve_environ(recipe)
    if getattr(args, "dry_run", False):
        plan = describe_eval(recipe)
        print(
            json.dumps(
                {
                    "argv": argv,
                    "plan": plan,
                    "engine": recipe.serve.engine,
                    "environ": extra_env,
                    "kv_cpu_offload_gib": plan["kv_cpu_offload_gib"],
                    "sglang_quant": plan.get("sglang_quant"),
                },
                indent=2,
            )
        )
        return 0
    snapshot = sglang_quant_snapshot(recipe.model)
    if recipe.serve.engine == "sglang" and snapshot is not None and not snapshot["sglang_ok"]:
        algo = snapshot.get("quant_algo") or "W4A8_NVFP4_FP8"
        raise MegaQuantError(
            f"This checkpoint is TRT-LLM {algo}; "
            "SGLang rejects that quant_algo. --engine vllm is not the fix. "
            "Use mixed W4A8 (nvfp4_w4a8 / MIXED_PRECISION), not uniform NVFP4 "
            "block 32. `megaquant rewrite-sglang` is only for mixed exports "
            "with a bare NVFP4 tag; do not rewrite these weights in place."
        )
    print("[serve]", " ".join(argv), flush=True)
    os.environ.update(extra_env)
    try:
        os.execvp(argv[0], argv)
    except FileNotFoundError as exc:
        if argv[:2] == ["sglang", "serve"]:
            alt = [sys.executable, "-m", "sglang.launch_server", *argv[2:]]
            print("[serve]", " ".join(alt), flush=True)
            try:
                os.execvp(alt[0], alt)
            except FileNotFoundError as inner:
                raise MegaQuantError(
                    "sglang is not installed. Install a recent SGLang "
                    "(lmsysorg/sglang:dev / `uv pip install --prerelease=allow sglang`) "
                    "then retry."
                ) from inner
        raise MegaQuantError(
            "vllm is not on PATH. Install a Blackwell vLLM build, then retry."
        ) from exc
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="megaquant",
        description="Generic PTQ pipeline (NVFP4 / FP8 / mixed-precision).",
    )
    sub = parser.add_subparsers(dest="command")

    quantize = sub.add_parser("quantize", help="Quantize a model from a recipe YAML")
    _add_recipe_flags(quantize, with_dry_run=True)
    quantize.set_defaults(func=cmd_quantize)

    plan = sub.add_parser("plan", help="Alias of quantize --dry-run")
    _add_recipe_flags(plan, with_dry_run=False)
    plan.set_defaults(func=cmd_plan, dry_run=True)

    schemes = sub.add_parser("schemes", help="List registered scheme catalog keys")
    schemes.set_defaults(func=cmd_schemes)

    families = sub.add_parser("families", help="List registered model families")
    families.set_defaults(func=cmd_families)

    rewrite = sub.add_parser(
        "rewrite-sglang",
        help=(
            "Rewrite an exported HF dir to quant_algo=MIXED_PRECISION + "
            "quantized_layers (SGLang modelopt_mixed)"
        ),
    )
    rewrite.add_argument(
        "export_dir",
        type=Path,
        help="Directory with model.safetensors.index.json / hf_quant_config.json",
    )
    rewrite.set_defaults(func=cmd_rewrite_sglang)

    evaluate = sub.add_parser(
        "eval",
        help="GPQA Diamond via OpenAI-compatible SGLang (Qwen thinking; no short truncation)",
    )
    evaluate.add_argument(
        "-c",
        "--config",
        default="recipes/eval-gpqa-diamond.yaml",
        help=(
            "Eval recipe. Default FlashInfer; CUDA 12.8 / 32 GB SM120 uses "
            "recipes/eval-gpqa-diamond.5090.yaml; 80 GB SM120 uses "
            "recipes/eval-gpqa-diamond.6000d.yaml."
        ),
    )
    evaluate.add_argument(
        "--model",
        help="Local NVFP4 export (outputs/Qwen3.8-27B-NVFP4-W4A8), not Qwen/Qwen3.8-27B",
    )
    evaluate.add_argument("--output", help="Eval journal directory")
    evaluate.add_argument(
        "--base-url",
        dest="base_url",
        help="OpenAI-compatible SGLang URL (default http://127.0.0.1:30000/v1)",
    )
    evaluate.add_argument(
        "--engine",
        choices=["sglang", "vllm"],
        help="Override serve.engine in the eval plan (default sglang)",
    )
    evaluate.add_argument("--limit", type=int, help="Optional item cap (full run omits this)")
    evaluate.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallel GPQA HTTP requests (recipe concurrency; 5090 default 24)",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the eval plan; no GPU, no 27B Hub download",
    )
    evaluate.set_defaults(func=cmd_eval)

    serve = sub.add_parser(
        "serve",
        help="SGLang serve (NVIDIA Qwen3.8 cookbook); KV that misses HBM goes to host RAM",
    )
    serve.add_argument(
        "-c",
        "--config",
        default="recipes/eval-gpqa-diamond.yaml",
        help=(
            "Serve recipe. Default FlashInfer; CUDA 12.8 / 32 GB SM120 uses "
            "recipes/eval-gpqa-diamond.5090.yaml; 80 GB SM120 uses "
            "recipes/eval-gpqa-diamond.6000d.yaml."
        ),
    )
    serve.add_argument(
        "--model",
        help="Local NVFP4 export dir (not the BF16 Hub id Qwen/Qwen3.8-27B)",
    )
    serve.add_argument(
        "--engine",
        choices=["sglang", "vllm"],
        help="Override serve.engine (default sglang)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port (default 30000 for SGLang, 8000 for vLLM)",
    )
    serve.add_argument(
        "--dry-run",
        action="store_true",
        help="Print argv; do not exec, load weights, or download 27B",
    )
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    from megaquant.runtime import configure_host_parallelism

    configure_host_parallelism()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except MegaQuantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
