#!/usr/bin/env python3
"""Verify a complete temperature-zero GPQA Diamond run without changing its files.

The evidence contains hashes and aggregate checks, never questions, choices,
model responses, item identifiers, or local paths. Install the repository first
(``pip install -e .``); no datasets, Torch, model weights, or network are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from megaquant.eval_gpqa import LETTERS, ItemResult, build_gpqa_item, extract_choice

DIAMOND_ITEMS = 198
TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def verify_results(
    csv_path: Path,
    run_dir: Path,
    *,
    seed: int = 0,
    shuffle_choices: bool = True,
) -> dict[str, Any]:
    """Independently check saved traces and their summary; never repair journals."""
    report: dict[str, Any] = {
        "passed": False,
        "complete": False,
        "benchmark": "gpqa_diamond",
        "expected_items": DIAMOND_ITEMS,
        "protocol": {"temperature": 0.0, "seed": seed, "shuffle_choices": shuffle_choices},
        "files": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]
    blobs: dict[str, bytes] = {}
    for label, path in (
        ("csv", csv_path),
        ("journal", run_dir / "gpqa_diamond.jsonl"),
        ("summary", run_dir / "summary.json"),
    ):
        try:
            blob = path.read_bytes()
        except OSError:
            errors.append(f"{label}: missing or unreadable file")
            continue
        blobs[label] = blob
        report["files"][label] = {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
        }
    if len(blobs) != 3:
        return report

    try:
        raw_rows = list(csv.DictReader(io.StringIO(blobs["csv"].decode("utf-8"))))
        items = [
            build_gpqa_item(row, index=i, seed=seed, shuffle=shuffle_choices)
            for i, row in enumerate(raw_rows)
        ]
    except (UnicodeError, csv.Error, TypeError, ValueError):
        errors.append("csv: invalid dataset")
        return report
    if len(items) != DIAMOND_ITEMS:
        errors.append(f"csv: expected {DIAMOND_ITEMS} rows, found {len(items)}")
    if len({item.item_id for item in items}) != len(items):
        errors.append("csv: duplicate item IDs")
    if any(
        not item.question
        or not item.item_id
        or any(not choice for choice in item.choices.values())
        or len(set(item.choices.values())) != len(LETTERS)
        for item in items
    ):
        errors.append("csv: empty question, ID, or option, or duplicate options")
    if errors:
        return report
    expected = {item.item_id: item for item in items}

    try:
        summary = _json(blobs["summary"])
        if not isinstance(summary, dict):
            raise ValueError("not an object")
    except (UnicodeError, ValueError):
        errors.append("summary: invalid JSON object")
        return report
    plan = summary.get("plan")
    if not isinstance(plan, dict):
        errors.append("summary: missing evaluation plan")
        plan = {}
    if plan.get("benchmark") != "gpqa_diamond":
        errors.append("summary: benchmark is not gpqa_diamond")
    sampling = plan.get("sampling")
    if not isinstance(sampling, dict):
        sampling = {}
    temperature = sampling.get("temperature")
    if not _number(temperature) or temperature != 0:
        errors.append("summary: temperature must be exactly zero")
    generation = plan.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    summary_seed = generation.get("seed")
    if type(summary_seed) is not int or summary_seed != seed:
        errors.append("summary: generation seed does not match verification seed")
    # Older summaries omit shuffle_choices; the CSV comparison still checks
    # the explicitly supplied shuffle policy against every saved choice.
    if "shuffle_choices" in plan and (
        type(plan["shuffle_choices"]) is not bool
        or plan["shuffle_choices"] != shuffle_choices
    ):
        errors.append("summary: choice shuffle policy does not match verification policy")

    rows: list[dict[str, Any]] = []
    row_ids: list[str] = []
    recomputed_correct = 0
    truncated = 0
    unparsed = 0
    required = set(ItemResult.__dataclass_fields__)
    lines = [raw for raw in blobs["journal"].splitlines() if raw.strip()]
    for line_no, raw in enumerate(lines, start=1):
        prefix = f"journal row {line_no}"
        try:
            row = _json(raw)
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError("missing fields")
        except (UnicodeError, ValueError):
            errors.append(f"{prefix}: malformed or incomplete JSON record")
            continue
        rows.append(row)
        item_id = row["item_id"]
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{prefix}: invalid item ID")
            continue
        row_ids.append(item_id)
        item = expected.get(item_id)
        if item is None:
            errors.append(f"{prefix}: unexpected item ID")
            continue
        if row["gold"] != item.gold or row["choices"] != item.choices:
            errors.append(f"{prefix}: gold or choices do not match the CSV and shuffle policy")
        if not isinstance(row["text"], str):
            errors.append(f"{prefix}: response text is not a string")
            continue
        if type(row["truncated"]) is not bool or type(row["correct"]) is not bool:
            errors.append(f"{prefix}: truncated and correct must be booleans")
            continue
        if not isinstance(row["finish_reason"], str) or not row["finish_reason"]:
            errors.append(f"{prefix}: invalid finish reason")
        elif row["truncated"] != (row["finish_reason"] in TRUNCATED_FINISH_REASONS):
            errors.append(f"{prefix}: truncation flag disagrees with finish reason")
        if any(
            not _integer(row[field])
            for field in ("prompt_tokens", "completion_tokens", "continued")
        ):
            errors.append(f"{prefix}: invalid token or continuation count")
        predicted = extract_choice(row["text"])
        if row["predicted"] != predicted:
            errors.append(f"{prefix}: saved prediction disagrees with response extraction")
        correct = predicted == item.gold and not row["truncated"]
        if row["correct"] != correct:
            errors.append(f"{prefix}: saved correctness violates strict scoring")
        recomputed_correct += int(correct)
        truncated += int(row["truncated"])
        unparsed += int(predicted is None)

    counts = Counter(row_ids)
    missing = len(set(expected) - set(counts))
    extra = len(set(counts) - set(expected))
    duplicate_rows = sum(count - 1 for count in counts.values())
    report["coverage"] = {
        "dataset_rows": len(items),
        "journal_rows": len(lines),
        "valid_record_objects": len(rows),
        "unique_item_ids": len(counts),
        "missing_items": missing,
        "unexpected_items": extra,
        "duplicate_rows": duplicate_rows,
    }
    complete = (
        len(lines) == len(rows) == len(counts) == DIAMOND_ITEMS
        and missing == extra == duplicate_rows == 0
    )
    report["complete"] = complete
    if not complete:
        errors.append("journal: expected exactly 198 unique records matching the dataset")

    recomputed = {
        "n": DIAMOND_ITEMS,
        "correct": recomputed_correct,
        "accuracy": recomputed_correct / DIAMOND_ITEMS,
        "denominator": DIAMOND_ITEMS,
        "truncated": truncated,
        "unparsed": unparsed,
        "headline": f"{recomputed_correct}/{DIAMOND_ITEMS}",
    }
    scores = summary.get("scores")
    if not isinstance(scores, dict):
        errors.append("summary: missing scores")
    elif complete:
        for key, expected_value in recomputed.items():
            value = scores.get(key)
            if key == "accuracy":
                matches = _number(value) and math.isclose(
                    value, expected_value, rel_tol=0, abs_tol=1e-12
                )
            else:
                matches = type(value) is type(expected_value) and value == expected_value
            if not matches:
                errors.append(f"summary: {key} disagrees with independent strict scoring")
    report["passed"] = not errors
    # Publishing a partial or inconsistent score would undermine this check.
    if report["passed"]:
        report["scores"] = recomputed
        report["verified"] = [
            "198 unique item IDs exactly match the supplied Diamond CSV",
            "every gold letter and shuffled choice matches the CSV and supplied seed",
            "saved predictions match the repository response extractor",
            "truncated and unparsed responses count as wrong",
            "summary scores match independently recomputed scores",
            "summary records temperature zero and the supplied generation seed",
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, dest="csv_path")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle-choices", action="store_true")
    args = parser.parse_args(argv)
    input_paths = (
        args.csv_path,
        args.run_dir / "gpqa_diamond.jsonl",
        args.run_dir / "summary.json",
    )
    aliases_input = args.output_json.resolve() in {path.resolve() for path in input_paths}
    if args.output_json.exists():
        aliases_input = aliases_input or any(
            path.exists() and args.output_json.samefile(path) for path in input_paths
        )
    if aliases_input:
        parser.error("--output-json must not overwrite an input file")
    report = verify_results(
        args.csv_path, args.run_dir, seed=args.seed, shuffle_choices=not args.no_shuffle_choices
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
