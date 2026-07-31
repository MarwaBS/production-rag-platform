"""What the default embedder does once the corpus outgrows the reference size.

Its 128 buckets alias distinct vocabulary, so gold documents fall out of the
top 3 as competition grows. That is a bounded-reference limitation rather than a
bug — but an unmeasured limitation reads as a capability, so the curve is
measured, committed, and re-measured here.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _committed() -> dict:
    return json.loads(
        (ROOT / "scale_cliff_derivation.json").read_text(encoding="utf-8")
    )


def test_the_scale_curve_reproduces_through_the_serving_path() -> None:
    """Re-running the producer measures the live service again, so an embedder
    or retrieval change that moves the curve fails here rather than quietly
    dating the committed file."""
    producer = ROOT / "scripts" / "derive_scale_cliff.py"
    assert producer.exists(), (
        "the scale curve has no committed producer: scripts/derive_scale_cliff.py"
    )
    rerun = subprocess.run(
        [sys.executable, str(producer), "--print"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert rerun.returncode == 0, f"the producer failed: {rerun.stderr[-400:]}"
    assert json.loads(rerun.stdout) == _committed(), (
        "the producer does not reproduce the committed curve"
    )


def test_the_committed_curve_records_the_degradation_it_documents() -> None:
    """A flat curve would reproduce perfectly and document nothing."""
    curve = _committed()["recall_at_3"]
    sizes = sorted(int(size) for size in curve)
    assert curve[str(sizes[-1])] < curve[str(sizes[0])], curve
