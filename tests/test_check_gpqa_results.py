"""Reject incomplete or misleading GPQA evidence using synthetic Diamond rows."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from megaquant.eval_gpqa import load_gpqa_diamond

_SPEC = importlib.util.spec_from_file_location(
    "check_gpqa_results", Path(__file__).parents[1] / "scripts" / "check_gpqa_results.py"
)
check_gpqa_results = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_gpqa_results)


@pytest.fixture
def run_fixture(tmp_path):
    dataset = tmp_path / "private-dataset.csv"
    run_dir = tmp_path / "private-run"
    run_dir.mkdir()
    fieldnames = [
        "Record ID",
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(198):
            writer.writerow(
                {
                    "Record ID": f"private-item-{index}",
                    "Question": f"private question {index}",
                    "Correct Answer": f"private correct option {index}",
                    **{
                        f"Incorrect Answer {i}": f"private incorrect option {index}-{i}"
                        for i in range(1, 4)
                    },
                }
            )
    items = load_gpqa_diamond(csv_path=dataset, seed=0, shuffle=True)
    rows = [
        {
            "item_id": item.item_id,
            "gold": item.gold,
            "predicted": item.gold,
            "correct": True,
            "truncated": False,
            "finish_reason": "stop",
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "continued": 0,
            "text": f"<think>private reasoning says A</think>\nAnswer: {item.gold}",
            "choices": item.choices,
        }
        for item in items
    ]
    summary = {
        "plan": {
            "benchmark": "gpqa_diamond",
            "sampling": {"temperature": 0.0},
            "generation": {"seed": 0},
        },
        "scores": {
            "n": 198,
            "correct": 198,
            "accuracy": 1.0,
            "denominator": 198,
            "truncated": 0,
            "unparsed": 0,
            "headline": "198/198",
        },
    }

    def save():
        (run_dir / "gpqa_diamond.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    save()
    return dataset, run_dir, rows, summary, save


def test_complete_run_is_verified_and_evidence_has_no_private_content(run_fixture):
    dataset, run_dir, _, _, _ = run_fixture
    before = {path: path.read_bytes() for path in (dataset, *run_dir.iterdir())}
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert report["passed"] and report["complete"]
    assert report["scores"]["headline"] == "198/198"
    assert report["coverage"]["unique_item_ids"] == 198
    assert report["files"]["csv"]["sha256"] == hashlib.sha256(before[dataset]).hexdigest()
    assert "private" not in json.dumps(report)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_correct_looking_truncated_and_unparsed_responses_count_wrong(run_fixture):
    dataset, run_dir, rows, summary, save = run_fixture
    rows[0].update(truncated=True, finish_reason="length", correct=False)
    rows[1].update(text="no response", predicted=None, correct=False)
    summary["scores"].update(
        correct=196, accuracy=196 / 198, truncated=1, unparsed=1, headline="196/198"
    )
    save()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert report["passed"], report["errors"]
    assert report["scores"]["correct"] == 196


def test_repeated_wrong_options_in_the_dataset_are_accepted(run_fixture):
    dataset, run_dir, rows, _, save = run_fixture
    with dataset.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        dataset_rows = list(reader)
    # The official Diamond CSV contains a row with repeated option strings.
    # Verification must preserve the source choices rather than reject them.
    dataset_rows[0]["Incorrect Answer 2"] = dataset_rows[0]["Incorrect Answer 1"]
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
    item = load_gpqa_diamond(csv_path=dataset, seed=0, shuffle=True)[0]
    rows[0].update(
        gold=item.gold,
        choices=item.choices,
        predicted=item.gold,
        text=f"Answer: {item.gold}",
    )
    save()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert report["passed"], report["errors"]
    assert report["scores"]["headline"] == "198/198"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "unexpected"])
def test_missing_duplicate_and_unexpected_items_never_publish_score(run_fixture, defect):
    dataset, run_dir, rows, _, save = run_fixture
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[0].copy()
    elif defect == "extra":
        rows.append(rows[0].copy())
    else:
        rows[-1]["item_id"] = "private-unexpected-item"
    save()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert not report["passed"] and not report["complete"]
    assert "scores" not in report
    assert "private" not in json.dumps(report)


@pytest.mark.parametrize(
    "defect",
    ["gold", "choices", "prediction", "correct", "truncation", "boolean", "tokens"],
)
def test_record_integrity_failures_are_detected(run_fixture, defect):
    dataset, run_dir, rows, _, save = run_fixture
    row = rows[0]
    if defect == "gold":
        row["gold"] = "wrong"
    elif defect == "choices":
        row["choices"]["A"] = "private modified choice"
    elif defect == "prediction":
        row["predicted"] = None
    elif defect == "correct":
        row["truncated"] = True
        row["finish_reason"] = "length"
    elif defect == "truncation":
        row["finish_reason"] = "length"
    elif defect == "boolean":
        row["correct"] = 1
    else:
        row["completion_tokens"] = -1
    save()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert not report["passed"]
    assert "scores" not in report
    assert any("journal row 1:" in error for error in report["errors"])


@pytest.mark.parametrize(
    "defect", ["temperature", "temperature_bool", "seed", "shuffle", "score", "accuracy"]
)
def test_summary_protocol_and_scores_must_match(run_fixture, defect):
    dataset, run_dir, _, summary, save = run_fixture
    if defect == "temperature":
        summary["plan"]["sampling"]["temperature"] = 1.0
    elif defect == "temperature_bool":
        summary["plan"]["sampling"]["temperature"] = False
    elif defect == "seed":
        summary["plan"]["generation"]["seed"] = 42
    elif defect == "shuffle":
        summary["plan"]["shuffle_choices"] = False
    elif defect == "score":
        summary["scores"]["correct"] = 197
    else:
        summary["scores"]["accuracy"] = float("nan")
    save()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert not report["passed"] and "scores" not in report


def test_wrong_verification_shuffle_policy_fails_against_choices(run_fixture):
    dataset, run_dir, _, _, _ = run_fixture
    report = check_gpqa_results.verify_results(dataset, run_dir, shuffle_choices=False)
    assert not report["passed"]
    assert any("shuffle policy" in error for error in report["errors"])


@pytest.mark.parametrize("defect", ["partial_tail", "duplicate_key", "missing_field"])
def test_corrupt_journal_is_rejected_without_repair(run_fixture, defect):
    dataset, run_dir, rows, _, save = run_fixture
    journal = run_dir / "gpqa_diamond.jsonl"
    if defect == "partial_tail":
        with journal.open("ab") as handle:
            handle.write(b'{"item_id":')
    elif defect == "duplicate_key":
        data = journal.read_text()
        journal.write_text(data.replace('"correct": true', '"correct": false, "correct": true', 1))
    else:
        del rows[0]["text"]
        save()
    before = journal.read_bytes()
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert not report["passed"] and "scores" not in report
    assert journal.read_bytes() == before


def test_missing_summary_is_a_nonzero_cli_result_with_sanitized_evidence(run_fixture, capsys):
    dataset, run_dir, _, _, _ = run_fixture
    (run_dir / "summary.json").unlink()
    evidence = run_dir / "evidence.json"
    assert check_gpqa_results.main(
        ["--csv", str(dataset), "--run-dir", str(run_dir), "--output-json", str(evidence)]
    ) == 1
    report = json.loads(evidence.read_text())
    assert report["errors"] == ["summary: missing or unreadable file"]
    assert "scores" not in report
    assert "private" not in capsys.readouterr().out


def test_cli_refuses_to_overwrite_input_evidence(run_fixture):
    dataset, run_dir, _, _, _ = run_fixture
    before = dataset.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        check_gpqa_results.main(
            ["--csv", str(dataset), "--run-dir", str(run_dir), "--output-json", str(dataset)]
        )
    assert dataset.read_bytes() == before


def test_cli_refuses_to_overwrite_an_input_through_a_hard_link(run_fixture):
    dataset, run_dir, _, _, _ = run_fixture
    output = run_dir / "linked-evidence.json"
    output.hardlink_to(dataset)
    before = dataset.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        check_gpqa_results.main(
            ["--csv", str(dataset), "--run-dir", str(run_dir), "--output-json", str(output)]
        )
    assert dataset.read_bytes() == before


def test_dataset_must_itself_contain_198_unique_rows(run_fixture):
    dataset, run_dir, _, _, _ = run_fixture
    data = dataset.read_text().splitlines(keepends=True)
    dataset.write_text("".join(data[:-1]))
    report = check_gpqa_results.verify_results(dataset, run_dir)
    assert not report["passed"] and report["errors"] == ["csv: expected 198 rows, found 197"]
