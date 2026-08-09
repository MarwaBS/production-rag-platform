"""What the default embedder does once the corpus outgrows the reference size.

Gold documents fall out of the top 3 as the corpus grows, and the reason is not
the document count: distractors that reuse the corpus's own words compete on the
very terms the queries use. A control that changes nothing else is what tells
those two apart. An unmeasured limitation reads as a capability, so the curves
are measured, committed, and re-measured here.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

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


def test_the_control_is_the_shipped_construction_over_another_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the control at the corpus's own pool and it must come out as the
    shipped distractor, character for character. Naming the variables to hold
    still leaves every unnamed one free; word count, prefix, index arithmetic,
    separators; and any of those moves the curve it is supposed to isolate."""
    from scripts import derive_scale_cliff as producer

    monkeypatch.setattr(producer, "FOREIGN", producer.VOCABULARY)
    for index in (0, 1, 7, 196, 197, 1_000, 9_975):
        assert producer._control(index, producer.VOCABULARY) == producer._distractor(
            index, producer.VOCABULARY
        ), index
    # And it must be the pool doing that: a control reading the corpus's words
    # directly matches under the swap above while ignoring the pool entirely.
    monkeypatch.setattr(producer, "FOREIGN", ["elsewhere"] * len(producer.VOCABULARY))
    assert producer._control(0, producer.VOCABULARY) != producer._distractor(
        0, producer.VOCABULARY
    )


def test_the_two_pools_are_the_same_size_and_share_no_word() -> None:
    """Swapping pools only isolates the wording if the pools are otherwise
    alike: a shorter one would repeat sooner, a longer one would compete more."""
    from scripts import derive_scale_cliff as producer

    assert len(producer.FOREIGN) == len(producer.VOCABULARY)
    assert not set(producer.FOREIGN) & set(producer.VOCABULARY)
    # Built from the index and nothing else. A pool allowed to consult the query
    # set could be picked to miss it, and would then hold flat by being absent.
    assert producer.FOREIGN == [
        f"unrelated{index}" for index in range(len(producer.VOCABULARY))
    ]


def _buckets(words: list[str]) -> set[int]:
    """Through the embedder, not a second copy of its tokenising: a copy agrees
    with it right up until one of them changes."""
    from app.embedder import _hash_embed

    counts = _hash_embed([" ".join(words)])[0]
    return {index for index, count in enumerate(counts) if count}


def test_the_control_still_lands_in_the_buckets_the_queries_use() -> None:
    """A pool the retriever cannot see is not a control, it is an absence: it
    would hold flat whatever the corpus did, and 'holds flat' is the whole
    finding. A pool chosen to dodge the queries reaches none of their buckets;
    197 words over 128 buckets reach any given bucket about four times in five,
    so half of them is a floor no pool picked without looking will fall under.

    A coarse floor: what forbids the selection itself is the pinned expression
    above, which has no way to read a query. This catches a pool that got past
    that."""
    from evals.harness import QUERIES
    from scripts import derive_scale_cliff as producer

    queries = _buckets([query for query, _ in QUERIES])
    assert queries, "fixture: the queries occupy no buckets"
    reached = _buckets(producer.FOREIGN) & queries
    assert len(reached) >= len(queries) // 2, (
        f"the control reaches {len(reached)} of the queries' {len(queries)} "
        "buckets: it competes too little to be evidence of anything"
    )


def test_the_curve_separates_the_document_count_from_the_words_they_use() -> None:
    """The control holds the corpus's own words out of the distractors and
    changes nothing else, so what it does at the largest size is what the
    document count alone costs. The shipped curve's shortfall against it is what
    reusing the queries' own words costs."""
    committed = _committed()
    curve, control = committed["recall_at_3"], committed["recall_at_3_control"]
    assert control.keys() == curve.keys(), "the control was measured elsewhere"
    sizes = sorted(int(size) for size in curve)
    largest = str(sizes[-1])
    # Every size, not the two ends: the smallest is the corpus itself, where
    # neither construction has produced a single distractor yet.
    assert len(set(control.values())) == 1, (
        f"the control no longer holds flat, so the attribution needs re-deriving: {control}"
    )
    assert curve[largest] < control[largest], (
        "reusing the corpus's words costs nothing, so there is nothing to explain"
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


def test_a_gold_tied_at_the_cut_scores_the_same_whichever_way_the_tie_broke() -> None:
    """The store orders equal scores arbitrarily, so the curve was measuring
    argpartition rather than retrieval: CI produced 0.8333 where the committed
    file said 0.75, on identical code."""
    from scripts.derive_scale_cliff import reliably_top3

    def hit(text: str, score: float) -> dict:
        return {"text": text, "score": score}

    tied_in = [hit("a", 0.9), hit("b", 0.9), hit("gold", 0.5), hit("d", 0.5)]
    tied_out = [hit("a", 0.9), hit("b", 0.9), hit("d", 0.5), hit("gold", 0.5)]
    assert reliably_top3(tied_in, "gold") == reliably_top3(tied_out, "gold")
    assert not reliably_top3(tied_in, "gold")

    clear = [hit("a", 0.9), hit("b", 0.8), hit("gold", 0.7), hit("d", 0.5)]
    assert reliably_top3(clear, "gold")

    # Nothing else was in contention, so there is no tie to lose.
    assert reliably_top3([hit("gold", 0.4)], "gold")
