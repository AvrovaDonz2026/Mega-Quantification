from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pair_supervisor as supervisor


def _fake_runner(path: Path, *, stdout_probe: str | None = None) -> Path:
    probe = repr(stdout_probe)
    path.write_text(
        f"""import argparse, pathlib, subprocess, sys, time
p=argparse.ArgumentParser()
p.add_argument('--arm')
p.add_argument('--evidence-dir')
p.add_argument('--recipe')
p.add_argument('--execute', action='store_true')
a=p.parse_args()
pathlib.Path(a.evidence_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(a.evidence_dir, 'evidence.json').write_text('ok')
if {probe} is not None: print({probe})
if a.arm == 'static-MSE' and pathlib.Path(a.recipe).read_text().strip() == 'FAIL_MSE':
    raise SystemExit(7)
if a.arm == 'static-max' and pathlib.Path(a.recipe).read_text().strip() == 'FAIL_MAX':
    raise SystemExit(9)
if pathlib.Path(a.recipe).read_text().strip() == 'TIMEOUT':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    pathlib.Path(a.evidence_dir, 'child.pid').write_text(str(child.pid))
    time.sleep(30)
"""
    )
    return path


def _recipe(tmp_path: Path, text: str = "OK") -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(text)
    return recipe


def test_runs_in_order_and_writes_private_logs(tmp_path, monkeypatch):
    runner = _fake_runner(tmp_path / "runner.py")
    compare = tmp_path / "compare.py"
    compare.write_text("import json; print(json.dumps({'ptq_comparable': True}))\n")
    evidence = tmp_path / "evidence"
    summary = supervisor.supervise_pair(
        recipe=_recipe(tmp_path),
        runner=runner,
        evidence_root=evidence,
        timeout_per_arm=2,
        pair_timeout=5,
        compare_script=compare,
    )
    assert [x["arm"] for x in summary["arms"]] == ["static-max", "static-MSE"]
    assert all(x["status"] == "completed" for x in summary["arms"])
    assert summary["status"] == "observation_passed"
    assert (evidence / "pair-summary.json").is_file()
    assert (evidence / "logs" / "static-max.log").is_file()
    assert "verified" not in json.dumps(summary).lower()
    assert summary["compare"]["result"]["ptq_comparable"] is True


def test_omitted_comparator_cannot_pass(tmp_path):
    runner = _fake_runner(tmp_path / "runner.py")
    summary = supervisor.supervise_pair(
        recipe=_recipe(tmp_path),
        runner=runner,
        evidence_root=tmp_path / "evidence",
        timeout_per_arm=2,
        pair_timeout=5,
        compare_script=tmp_path / "missing-compare.py",
    )
    assert summary["status"] == "failed"
    assert summary["compare"]["status"] == "failed"
    assert summary["compare"]["reason"] == "comparator_missing"


def test_first_failure_skips_second(tmp_path):
    runner = _fake_runner(tmp_path / "runner.py")
    summary = supervisor.supervise_pair(
        recipe=_recipe(tmp_path, "FAIL_MAX"),
        runner=runner,
        evidence_root=tmp_path / "evidence",
        timeout_per_arm=2,
        pair_timeout=5,
    )
    assert summary["arms"][0]["status"] == "failed"
    assert summary["arms"][1]["status"] == "skipped"


def test_timeout_kills_process_group(tmp_path):
    runner = _fake_runner(tmp_path / "runner.py")
    evidence = tmp_path / "evidence"
    summary = supervisor.supervise_pair(
        recipe=_recipe(tmp_path, "TIMEOUT"),
        runner=runner,
        evidence_root=evidence,
        timeout_per_arm=0.2,
        pair_timeout=1,
    )
    record = summary["arms"][0]
    assert record["status"] == "timeout"
    assert record["process_group_exited"]
    child_pid = int((evidence / "static-max" / "child.pid").read_text())
    time.sleep(0.05)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("child process survived timeout cleanup")


def test_compare_failure_is_recorded_without_verified(tmp_path):
    runner = _fake_runner(tmp_path / "runner.py")
    compare = tmp_path / "compare.py"
    compare.write_text("raise SystemExit(4)\n")
    summary = supervisor.supervise_pair(
        recipe=_recipe(tmp_path),
        runner=runner,
        evidence_root=tmp_path / "evidence",
        timeout_per_arm=2,
        pair_timeout=5,
        compare_script=compare,
    )
    assert summary["compare"]["status"] == "failed"
    assert "verified" not in json.dumps(summary).lower()


def test_summary_and_arm_logs_do_not_expose_prompt_paths_or_argv(tmp_path):
    secret_prompt = "PRIVATE_PROMPT_SHOULD_NOT_BE_PERSISTED"
    runner = _fake_runner(tmp_path / "runner.py", stdout_probe=secret_prompt)
    recipe = _recipe(tmp_path)
    evidence = tmp_path / "private-secret-evidence"
    summary = supervisor.supervise_pair(
        recipe=recipe,
        runner=runner,
        evidence_root=evidence,
        timeout_per_arm=2,
        pair_timeout=5,
        compare_script=tmp_path / "missing-compare.py",
    )
    serialized = json.dumps(summary)
    assert secret_prompt not in serialized
    assert str(tmp_path) not in serialized
    assert str(recipe) not in serialized
    assert all("argv" not in record for record in summary["arms"])
    assert summary["arms"][0]["input_identity"]["recipe"] == "recipe.yaml"
    assert summary["arms"][0]["log_file"] == "logs/static-max.log"
    assert secret_prompt not in (evidence / "logs" / "static-max.log").read_text()
