"""Retrieval-quality gate — a gate that CAN go red.

The reference service ships a Mock LLM, so answer text is a fixed template;
what can actually regress is RETRIEVAL. `evals.harness` runs a fixed Q/gold set
through the real embed + NumPy store + top-k path. This module gates CI on a
measured recall/MRR floor that sits with margin BELOW the observed baseline
(recall@1 = recall@3 = 1.000, MRR = 1.000), so a genuine regression — a broken
embedder, a store bug, a bad `k` default — trips it, while normal variation does
not. `test_each_floor_is_load_bearing` proves the floors are not inert:
degrade the embedder and every one of them must reject it.

Scope, stated because the floors do not yet mean what they should: they are
derived from a baseline built out of queries that share words with their gold
document, so the shipped set measures word matching. The paraphrase set measures
meaning and is marked `semantic`, deselected by default. Re-deriving the floors
from an honest baseline is ingestion-phase work, not gate work.
"""

from __future__ import annotations

import numpy as np
import pytest

import evals.harness as harness
from evals.harness import evaluate

# Floors below the measured baseline (all 1.000) — a real retrieval regression
# drops beneath these; run-to-run noise (there is none; the embedder is
# deterministic) does not. Not a can't-fail assertion: the true values clear
# these with headroom AND can fall under them.
RECALL_AT_1_FLOOR = 0.75
RECALL_AT_3_FLOOR = 0.90
MRR_FLOOR = 0.85


def test_retrieval_meets_recall_floor() -> None:
    top1 = evaluate(k=1)
    top3 = evaluate(k=3)
    assert top1.recall_at_k >= RECALL_AT_1_FLOOR, top1.summary()
    assert top3.recall_at_k >= RECALL_AT_3_FLOOR, top3.summary()
    assert top3.mrr >= MRR_FLOOR, top3.summary()


def test_paraphrase_queries_share_no_word_with_their_gold_document() -> None:
    """Guards the paraphrase set itself: a query that reuses the document's words
    measures token matching, and would quietly turn the gate below into the one
    above."""
    import re

    for query, gold in harness.PARAPHRASE_QUERIES:
        shared = set(re.findall(r"[a-z0-9]+", query.lower())) & set(
            re.findall(r"[a-z0-9]+", harness.CORPUS[gold].lower())
        )
        assert not shared, f"{query!r} reuses {sorted(shared)} from its gold document"


def _paraphrased() -> tuple[tuple[str, str], ...]:
    return tuple(
        (query, harness.CORPUS[gold]) for query, gold in harness.PARAPHRASE_QUERIES
    )


def test_the_default_embedder_cannot_match_a_paraphrase() -> None:
    """Records what the shipped bag-of-tokens embedder actually does.

    It matches words, not meaning, so a query that shares no word with its
    document is beyond it. That is a limitation, not a bug — but an unrecorded
    limitation becomes an implied capability, and this is the number that keeps
    the published floor from being read as a claim about semantic retrieval.
    """
    result = evaluate(k=3, queries=_paraphrased())
    assert result.recall_at_k < RECALL_AT_3_FLOOR, (
        "the default embedder now clears the paraphrase floor — if the semantic "
        "backend became the default, this limitation note is stale"
    )


@pytest.mark.semantic
def test_retrieval_meets_the_recall_floor_on_paraphrased_queries() -> None:
    """The published floor, asked of meaning rather than of vocabulary.

    Marked `semantic` and deselected by default: the shipped embedder matches
    words, so it cannot satisfy this, and selecting it deliberately is how the
    gap is measured. The floor is the one the repository already publishes,
    asked of queries that cannot be solved by copying words.
    """
    result = evaluate(k=3, queries=_paraphrased())
    assert result.recall_at_k >= RECALL_AT_3_FLOOR, result.summary()


def test_each_floor_is_load_bearing_through_the_serving_path(monkeypatch) -> None:
    """All three floors must be able to fail, not only recall@3 — and they must
    fail because the SERVICE degraded, not a private reimplementation of it.

    Patching the app's own embedder proves both at once: if the eval did not
    flow through app.main, collapsing app.main's embedder would not move these
    numbers and every assertion here would be false.
    """
    import app.main as main

    monkeypatch.setattr(
        main, "embed", lambda texts: np.ones((len(texts), 128), dtype="float32")
    )
    top1, top3 = evaluate(k=1), evaluate(k=3)
    assert top1.recall_at_k < RECALL_AT_1_FLOOR, top1.summary()
    assert top3.recall_at_k < RECALL_AT_3_FLOOR, top3.summary()
    assert top3.mrr < MRR_FLOOR, top3.summary()


def test_the_floors_are_derived_from_a_committed_measured_baseline() -> None:
    """Every floor traces to scripts/derive_eval_floors.py's committed output,
    and that baseline is a real measurement — not the saturated 1.000 a
    token-overlap eval set produces by construction."""
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "scripts" / "derive_eval_floors.py").exists(), (
        "the eval floors have no committed producer: scripts/derive_eval_floors.py"
    )
    derivation_file = root / "eval_floors_derivation.json"
    assert derivation_file.exists(), (
        "no committed derivation for the eval floors: eval_floors_derivation.json"
    )
    derivation = json.loads(derivation_file.read_text(encoding="utf-8"))
    baseline = derivation["measured"]["semantic_paraphrase"]
    assert all(value < 1.0 for value in baseline.values()), (
        f"the baseline is saturated ({baseline}); floors derived from it "
        "cannot discriminate"
    )
    floors = derivation["derived_floors"]
    assert RECALL_AT_1_FLOOR == floors["recall_at_1"]
    assert RECALL_AT_3_FLOOR == floors["recall_at_3"]
    assert MRR_FLOOR == floors["mrr"]


@pytest.mark.semantic
def test_the_floor_derivation_reproduces_under_the_semantic_backend() -> None:
    """Running the producer must regenerate the committed derivation exactly —
    an artefact nobody can regenerate is a number someone typed. Needs the
    semantic extra, so it runs where that extra is installed."""
    import json
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    rerun = subprocess.run(
        [sys.executable, str(root / "scripts" / "derive_eval_floors.py"), "--print"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert rerun.returncode == 0, f"the producer failed: {rerun.stderr[-400:]}"
    assert json.loads(rerun.stdout) == json.loads(
        (root / "eval_floors_derivation.json").read_text(encoding="utf-8")
    ), "the producer does not reproduce the committed derivation"
