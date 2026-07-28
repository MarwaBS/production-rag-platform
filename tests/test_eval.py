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


def test_each_floor_is_load_bearing(monkeypatch) -> None:
    """All three floors must be able to fail, not only recall@3.

    A floor no measurement can fall beneath is a number in a file. The constant
    embedder collapses every metric, so each floor must reject it.
    """
    monkeypatch.setattr(
        harness, "embed", lambda texts: np.ones((len(texts), 128), dtype="float32")
    )
    top1, top3 = evaluate(k=1), evaluate(k=3)
    assert top1.recall_at_k < RECALL_AT_1_FLOOR, top1.summary()
    assert top3.recall_at_k < RECALL_AT_3_FLOOR, top3.summary()
    assert top3.mrr < MRR_FLOOR, top3.summary()
