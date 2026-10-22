from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_matched_ptq_r6_offline_suite():
    suite = Path(__file__).resolve().parents[1] / "scripts/diagnostics/matched-ptq-r6"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(suite)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        raise AssertionError(f"R6 offline suite exceeded 60 seconds\n{output}") from exc

    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
    assert result.returncode == 0
