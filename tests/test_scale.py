"""What the default embedder does once the corpus outgrows the reference size.

Its 128 buckets alias distinct vocabulary, so gold documents fall out of the
top 3 as competition grows. An unmeasured limitation reads as a capability, so
the curve is measured, committed, and re-measured here.
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


def test_the_curve_separates_competition_volume_from_vocabulary_growth() -> None:
    """More documents costs recall on its own. Charging the whole fall to the
    distractors' new vocabulary needs a control at the same sizes that adds no
    token the corpus did not already have; the gap between the two is the cost
    of the vocabulary, and it is the only part the buckets explain."""
    committed = _committed()
    curve, control = committed["recall_at_3"], committed["recall_at_3_control"]
    assert control.keys() == curve.keys(), "the control was measured elsewhere"
    sizes = sorted(int(size) for size in curve)
    largest, smallest = str(sizes[-1]), str(sizes[0])
    assert control[largest] < control[smallest], (
        "the control does not fall, so volume alone would explain nothing"
    )
    assert curve[largest] < control[largest], (
        "the new vocabulary costs nothing beyond the volume that carries it"
    )


def test_the_producer_refuses_a_corpus_that_dedup_would_shrink() -> None:
    """Distractors that repeat measure a few hundred documents while reporting
    the size they meant to: the collapse is silent unless the producer checks."""
    import pytest

    from scripts import derive_scale_cliff

    original = derive_scale_cliff._distractor
    derive_scale_cliff._distractor = lambda index, vocabulary: "identical"
    try:
        with pytest.raises(RuntimeError, match="dedup shrank the corpus"):
            derive_scale_cliff.derive()
    finally:
        derive_scale_cliff._distractor = original


def test_the_recorded_embedder_width_follows_the_embedder() -> None:
    """Recorded as a matching literal it would keep saying 128 after the
    embedder changed, and the field would be decoration."""
    import app.embedder as embedder
    from scripts import derive_scale_cliff

    original_dim, original_sizes = embedder._DIM, derive_scale_cliff.CORPUS_SIZES
    embedder._DIM, derive_scale_cliff.CORPUS_SIZES = 7, (24,)
    try:
        assert derive_scale_cliff.derive()["embedder_dimensions"] == 7
    finally:
        embedder._DIM, derive_scale_cliff.CORPUS_SIZES = original_dim, original_sizes
