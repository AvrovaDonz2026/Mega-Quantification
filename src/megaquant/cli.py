"""Command-line interface for Mega-Quantification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from megaquant.config import CLI_OVERRIDE_MAP, load_recipe
from megaquant.eval_gpqa import (
    describe_eval,
    load_eval_recipe,
    run_gpqa,
    vllm_serve_argv_from_recipe,
)
from megaquant.exceptions import MegaQuantError
from megaquant.pipeline import QuantPipeline
from megaquant.registry import list_families, list_schemes


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


def _eval_recipe_from_args(args: argparse.Namespace):
    import os

    overrides: dict[str, Any] = {}
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "output", None):
        overrides["output_dir"] = args.output
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    elif os.environ.get("MEGAQUANT_VLLM_BASE_URL"):
        overrides["base_url"] = os.environ["MEGAQUANT_VLLM_BASE_URL"]
    if getattr(args, "limit", None) is not None:
        overrides["limit"] = args.limit
    return load_eval_recipe(args.config, overrides or None)


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

    recipe = _eval_recipe_from_args(args)
    argv = vllm_serve_argv_from_recipe(
        recipe,
        getattr(args, "model", None),
        port=int(getattr(args, "port", 8000) or 8000),
    )
    if getattr(args, "dry_run", False):
        plan = describe_eval(recipe)
        print(
            json.dumps(
                {
                    "argv": argv,
                    "plan": plan,
                    "kv_cpu_offload_gib": plan["kv_cpu_offload_gib"],
                },
                indent=2,
            )
        )
        return 0
    print("[serve]", " ".join(argv), flush=True)
    try:
        os.execvp(argv[0], argv)
    except FileNotFoundError as exc:
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

    evaluate = sub.add_parser(
        "eval",
        help="GPQA Diamond (Qwen thinking + NVIDIA vLLM card flags; no short truncation)",
    )
    evaluate.add_argument("-c", "--config", default="recipes/eval-gpqa-diamond.yaml")
    evaluate.add_argument("--model", help="Export dir or served model name")
    evaluate.add_argument("--output", help="Eval journal directory")
    evaluate.add_argument("--base-url", dest="base_url", help="OpenAI-compatible vLLM URL")
    evaluate.add_argument("--limit", type=int, help="Optional item cap (full run omits this)")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    serve = sub.add_parser(
        "serve",
        help="vLLM serve with NVIDIA-card flags; KV that misses HBM goes to host RAM",
    )
    serve.add_argument("-c", "--config", default="recipes/eval-gpqa-diamond.yaml")
    serve.add_argument("--model", help="Export dir (NVFP4 checkpoint)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dry-run", action="store_true", help="Print argv; do not exec vLLM")
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
