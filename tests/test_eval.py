"""Retrieval-quality gate — a gate that CAN go red.

The reference service ships a Mock LLM, so answer text is a fixed template;
what can actually regress is RETRIEVAL. `evals.harness` runs a fixed Q/gold set
through the real embed + NumPy store + top-k path. This module gates CI on a
measured recall/MRR floor that sits with margin BELOW the observed baseline
(recall@1 = recall@3 = 1.000, MRR = 1.000), so a genuine regression — a broken
embedder, a store bug, a bad `k` default — trips it, while normal variation does
not. `test_eval_gate_detects_a_retrieval_regression` proves the gate is not
inert: degrade the embedder and recall collapses below the floor.
"""

from __future__ import annotations

import numpy as np

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


def test_eval_gate_detects_a_retrieval_regression(monkeypatch) -> None:
    """Proof the gate is falsifiable: collapse the embedder to a constant vector
    (all docs identical → retrieval can't distinguish them) and recall must fall
    below the floor the healthy system clears. A gate that cannot go red
    certifies nothing."""
    monkeypatch.setattr(
        harness, "embed", lambda texts: np.ones((len(texts), 128), dtype="float32")
    )
    degraded = evaluate(k=3)
    assert degraded.recall_at_k < RECALL_AT_3_FLOOR, (
        "a constant-vector embedder should fail the retrieval gate, but recall "
        f"stayed at {degraded.recall_at_k:.3f}"
    )
