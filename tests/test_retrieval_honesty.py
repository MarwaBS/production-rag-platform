"""An answer may only be called grounded when retrieval actually carried signal.

Three situations, two contracts:

* the query holds nothing the embedder can represent -> 200, grounded=false, and
  the vector store is never consulted,
* the query is representable but matches nothing     -> 200, grounded=false,
* the query matches                                  -> 200, grounded=true, with a
  score and a stable id per hit so a caller can audit the answer.

A query in a script this embedder cannot tokenise is a valid request the system
cannot answer, not a malformed one. Rejecting it as a client error would blame
the caller for a limit of the service.

The failure these prevent: a zero-norm query is renormalised by the vector store
rather than rejected, so every similarity is 0.0 and the top-k is an arbitrary
slice of the corpus — which the service then presents as evidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)

CORPUS = [
    "FAISS performs in-process vector similarity search over dense embeddings",
    "Prometheus scrapes metrics endpoints and stores time series for alerting",
]

# The demo embedder keeps only [a-z0-9]+ tokens, so each of these embeds to the
# zero vector: no similarity it produces means anything. Whitespace-only text
# belongs here with the rest — it is a valid string carrying no signal, which is
# the system's limit to report, not the caller's error to be blamed for.
UNREPRESENTABLE = ["日本語のクエリ", "   ", "\t\n", "!!!", "…—–", "Кириллица", "→←↑↓"]


class _SpyStore:
    """Wraps the real store and counts searches, to prove where a query stopped."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.searches = 0

    def add(self, embeddings: Any) -> Any:
        return self._inner.add(embeddings)

    def search(self, queries: Any, k: int) -> Any:
        self.searches += 1
        return self._inner.search(queries, k)


@pytest.fixture
def spy(monkeypatch) -> Iterator[_SpyStore]:
    """Index the corpus through a store this test can observe."""
    main._index = None
    real = main.get_vector_store
    created: list[_SpyStore] = []

    def _spied(backend: str) -> _SpyStore:
        store = _SpyStore(real(backend))
        created.append(store)
        return store

    monkeypatch.setattr(main, "get_vector_store", _spied)
    client.post("/index", json={"documents": CORPUS})
    store = created[0]
    store.searches = 0  # indexing is not a search; start the count at the query
    yield store
    main._index = None


@pytest.mark.parametrize("query", UNREPRESENTABLE)
def test_query_the_embedder_cannot_represent_is_not_grounded(spy, query: str) -> None:
    response = client.post("/query", json={"query": query, "k": 2})
    assert response.status_code == 200, response.text[:160]
    body = response.json()
    assert body["grounded"] is False, (
        f"{query!r} embeds to the zero vector, so any documents returned for it are "
        f"arbitrary; got {body.get('retrieved')}"
    )
    assert body["retrieved"] == []
    assert body["answer"] is None


@pytest.mark.parametrize("query", UNREPRESENTABLE)
def test_unrepresentable_query_never_reaches_the_vector_store(spy, query: str) -> None:
    """The refusal must be this service's, not the dependency's.

    An empty result cannot tell the two apart; a store that is never asked can.
    """
    client.post("/query", json={"query": query, "k": 2})
    assert spy.searches == 0, (
        "an unrepresentable query must be refused before the vector store is asked"
    )


def test_query_matching_nothing_is_answered_as_not_grounded(spy) -> None:
    body = client.post("/query", json={"query": "braised lamb shoulder", "k": 2}).json()
    assert body["grounded"] is False
    assert body["retrieved"] == []
    assert body["answer"] is None


def test_a_match_is_grounded_and_every_hit_is_auditable(spy) -> None:
    body = client.post(
        "/query", json={"query": "vector similarity search", "k": 1}
    ).json()
    assert body["grounded"] is True
    assert len(body["retrieved"]) == 1
    hit = body["retrieved"][0]
    assert hit["text"] == CORPUS[0]
    assert 0.0 < hit["score"] <= 1.0
    assert hit["id"] and hit["doc_id"]


def test_hits_are_ordered_by_descending_score(spy) -> None:
    body = client.post("/query", json={"query": "metrics vector search", "k": 2}).json()
    scores = [hit["score"] for hit in body["retrieved"]]
    # Ordering is trivially true for one hit and for none, so an empty result
    # would satisfy the very assertion that is meant to catch it.
    assert len(scores) == 2, f"expected both documents to match, got {scores}"
    assert scores == sorted(scores, reverse=True), scores
    # The query shares two words with the FAISS document and one with the
    # Prometheus document, so their similarities cannot be equal — a constant
    # "score" satisfies range and ordering checks while auditing nothing.
    assert scores[0] > scores[1], (
        "two documents of different relevance carry the same score, so the "
        "score is a constant, not a similarity"
    )
    assert body["retrieved"][0]["text"] == CORPUS[0], (
        "the less relevant document outranked the more relevant one — the "
        "surfaced number is not the similarity that ordered the hits"
    )


def test_ids_are_stable_across_requests_and_re_indexing(spy) -> None:
    """An id a caller quotes must still mean the same passage tomorrow.

    Otherwise it identifies a response, not a document.
    """
    ask = lambda: client.post(  # noqa: E731 — two identical calls, read as one
        "/query", json={"query": "vector similarity search", "k": 1}
    ).json()["retrieved"][0]

    first, again = ask(), ask()
    assert first["id"] == again["id"], "the id changed between two identical queries"
    assert first["doc_id"] == again["doc_id"]

    # Re-index REVERSED. An id derived from position is stable across an
    # identical re-index and means nothing the moment the corpus is reordered,
    # so identity has to come from the document, not from where it sat.
    main._index = None
    client.post("/index", json={"documents": list(reversed(CORPUS))})
    after_reindex = ask()
    assert after_reindex["doc_id"] == first["doc_id"], (
        "the same document got a different id after the corpus was reordered, so "
        "the id identifies a position rather than a document"
    )

    # Re-index a SUPERSET whose new document sorts before every existing one
    # (a leading digit precedes letters in any collation) and sits first in the
    # submitted list. Reversal alone is survived by a canonicalised position —
    # the index in the SORTED corpus. Growth shifts every position, raw or
    # canonical, so only an id derived from the content itself holds still.
    main._index = None
    client.post(
        "/index",
        json={
            "documents": ["0 aardvark burrows appear across open grassland", *CORPUS]
        },
    )
    after_growth = ask()
    assert after_growth["doc_id"] == first["doc_id"], (
        "the same document got a different id after the corpus grew, so the id "
        "derives from a position (raw or canonicalised), not from the content"
    )


def test_the_refusal_tracks_the_embedding_not_the_characters(spy, monkeypatch) -> None:
    """The guard consults what the embedder PRODUCED, not what the query
    looks like. A character heuristic agrees with the norm check on every query
    in UNREPRESENTABLE — and silently diverges the day a backend that can
    represent them lands. Steering the embedder directly separates the two.
    """
    import numpy as np

    real = main.embed

    def zero_for_vectors(texts: list[str]) -> Any:
        out = real(texts)
        return np.zeros_like(out) if texts == ["vectors"] else out

    monkeypatch.setattr(main, "embed", zero_for_vectors)
    body = client.post("/query", json={"query": "vectors", "k": 1}).json()
    assert body["grounded"] is False and body["retrieved"] == [], (
        "the embedding was zero yet the query was served — the guard reads the "
        "characters, not the embedding"
    )
    assert spy.searches == 0


def test_a_query_the_embedder_represents_is_served_whatever_its_script(
    spy, monkeypatch
) -> None:
    """The converse pin: representable means SERVED. Under a semantic backend
    these scripts embed fine; a guard keyed to the characters would keep
    refusing them and turn into a language filter.
    """
    real = main.embed

    def represent_cjk(texts: list[str]) -> Any:
        return real(["vectors"]) if texts == ["日本語のクエリ"] else real(texts)

    monkeypatch.setattr(main, "embed", represent_cjk)
    client.post("/query", json={"query": "日本語のクエリ", "k": 1})
    assert spy.searches == 1, (
        "the embedding carried signal yet the store was never consulted — the "
        "guard is a character filter, not an embedding check"
    )


def test_the_same_document_indexed_twice_is_one_document() -> None:
    """A duplicated document must not come back as two corroborating sources."""
    main._index = None
    response = client.post(
        "/index", json={"documents": [CORPUS[0], CORPUS[0], CORPUS[1]]}
    )
    assert response.json()["indexed"] == 2
    hits = client.post(
        "/query", json={"query": "vector similarity search", "k": 3}
    ).json()["retrieved"]
    # Both fixture documents fit inside one window, so hits are 1:1 with
    # documents here; what follows is a claim about dedup, not about windowing.
    assert all(
        len(document) <= main.settings.max_chunk_chars
        for document in (CORPUS[0], CORPUS[1])
    )
    assert len({hit["doc_id"] for hit in hits}) == len(hits), (
        "a document was cited twice"
    )
