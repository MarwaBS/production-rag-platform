"""Retrieval-quality gates over the service's own /index and /query routes.

The floors come from scripts/derive_eval_floors.py (semantic backend on the
paraphrase set, the one non-saturated baseline). The default hash path is gated
against floors derived from its own literal-set measurement, at the same half a
miss of margin; its own bar, not this one.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import evals.harness as harness
from evals.harness import evaluate

from scripts.gate_report import SUBPROCESS_TIMEOUT_S

_DERIVATION = json.loads(
    (
        pathlib.Path(__file__).resolve().parent.parent / "eval_floors_derivation.json"
    ).read_text(encoding="utf-8")
)
_FLOORS = _DERIVATION["derived_floors"]
RECALL_AT_1_FLOOR = _FLOORS["recall_at_1"]
RECALL_AT_3_FLOOR = _FLOORS["recall_at_3"]
MRR_FLOOR = _FLOORS["mrr"]
# The default path is held to its own measurement: the semantic floors sit far
# below what it does on the literal set, so they would not notice a regression.
_DEFAULT_FLOORS = _DERIVATION["derived_floors_default_path"]


def test_retrieval_meets_recall_floor() -> None:
    top1 = evaluate(k=1)
    top3 = evaluate(k=3)
    assert top1.recall_at_k >= _DEFAULT_FLOORS["recall_at_1"], top1.summary()
    assert top3.recall_at_k >= _DEFAULT_FLOORS["recall_at_3"], top3.summary()
    assert top3.mrr >= _DEFAULT_FLOORS["mrr"], top3.summary()


def test_the_eval_leaves_the_service_index_as_it_found_it() -> None:
    """The eval indexes its own corpus through /index. A run that does not put
    the previous snapshot back changes what every later caller retrieves."""
    import app.main as main

    main._index = None
    evaluate(k=1)
    assert main._index is None, "the eval left its own corpus indexed in the service"


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
    """Records what the shipped bag-of-tokens embedder cannot do: match a query
    sharing no word with its document. An unrecorded limitation reads as a
    capability, and this keeps the floor from implying semantic retrieval."""
    result = evaluate(k=3, queries=_paraphrased())
    assert result.recall_at_k < RECALL_AT_3_FLOOR, (
        "the default embedder now clears the paraphrase floor; if the semantic "
        "backend became the default, this limitation note is stale"
    )


@pytest.mark.semantic
def test_retrieval_meets_every_floor_on_paraphrased_queries(monkeypatch) -> None:
    """The floors, asked of meaning rather than of vocabulary; the measurement
    they were derived from. Selects the semantic backend itself, so the gate
    cannot silently run against the hash path."""
    import app.embedder as embedder

    monkeypatch.setattr(embedder._settings, "embedding_backend", "semantic")
    top1 = evaluate(k=1, queries=_paraphrased())
    top3 = evaluate(k=3, queries=_paraphrased())
    assert top1.recall_at_k >= RECALL_AT_1_FLOOR, top1.summary()
    assert top3.recall_at_k >= RECALL_AT_3_FLOOR, top3.summary()
    assert top3.mrr >= MRR_FLOOR, top3.summary()


def test_each_floor_is_load_bearing_through_the_serving_path(monkeypatch) -> None:
    """Each floor must reject a degraded SERVICE: if the eval did not flow
    through app.main, collapsing app.main's embedder would move nothing here."""
    import app.main as main

    monkeypatch.setattr(
        main, "embed", lambda texts: np.ones((len(texts), 128), dtype="float32")
    )
    top1, top3 = evaluate(k=1), evaluate(k=3)
    assert top1.recall_at_k < _DEFAULT_FLOORS["recall_at_1"], top1.summary()
    assert top3.recall_at_k < _DEFAULT_FLOORS["recall_at_3"], top3.summary()
    assert top3.mrr < _DEFAULT_FLOORS["mrr"], top3.summary()


def test_the_floors_are_derived_from_a_committed_measured_baseline() -> None:
    """Every floor traces to the committed producer output, whose baseline is
    a real, non-saturated measurement."""
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

    # Recompute the floors from the measurement: reading the same key back out
    # of the same file compares it against itself and cannot fail.
    def floors_from(measured: dict, n: int) -> dict:
        return {metric: round(value - 0.5 / n, 4) for metric, value in measured.items()}

    recomputed = floors_from(baseline, derivation["n_paraphrase_queries"])
    assert derivation["derived_floors"] == recomputed, (
        f"the published floors do not follow {derivation['floor_rule']!r}"
    )
    assert (RECALL_AT_1_FLOOR, RECALL_AT_3_FLOOR, MRR_FLOOR) == (
        recomputed["recall_at_1"],
        recomputed["recall_at_3"],
        recomputed["mrr"],
    )
    # The default path's floors follow the same rule from its own measurement.
    assert _DEFAULT_FLOORS == floors_from(
        derivation["measured"]["hash_literal"], derivation["n_literal_queries"]
    )


@pytest.mark.semantic
def test_the_floor_derivation_reproduces_under_the_semantic_backend() -> None:
    """The producer must regenerate the committed derivation exactly; an
    artefact nobody can regenerate is a number someone typed."""
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
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    assert rerun.returncode == 0, f"the producer failed: {rerun.stderr[-400:]}"
    assert json.loads(rerun.stdout) == json.loads(
        (root / "eval_floors_derivation.json").read_text(encoding="utf-8")
    ), "the producer does not reproduce the committed derivation"
