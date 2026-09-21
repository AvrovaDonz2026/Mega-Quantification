"""Command-line interface for Mega-Quantification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from megaquant.config import CLI_OVERRIDE_MAP, load_recipe
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

    return parser


def main(argv: list[str] | None = None) -> int:
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
