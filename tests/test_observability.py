"""Observability answers the third operator question: is retrieval healthy?

/health says the process is up and rag_requests_total says it is serving, but
nothing today says whether retrieval is returning evidence or quietly answering
"grounded: false" to everything. These gates pin the retrieval-health series,
a status-labelled response series an error rate can be computed from, and a
correlation id that ties a response to its log lines.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import app.main as main
from app.main import app

client = TestClient(app)


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.fixture(autouse=True)
def _reset():
    main._index = None
    yield
    main._index = None


def test_the_corpus_gauge_tracks_the_indexed_documents() -> None:
    """/index REPLACES the corpus, so the gauge must follow it down as well as
    up — a counter here would silently misreport every re-index."""
    client.post("/index", json={"documents": ["gauge doc one", "gauge doc two"]})
    assert sample("rag_corpus_documents") == 2.0
    client.post("/index", json={"documents": ["solo gauge doc"]})
    assert sample("rag_corpus_documents") == 1.0


def test_hit_scores_are_observed() -> None:
    """The score distribution is the earliest signal of retrieval decay: a
    corpus drifting away from its queries shows up here before users complain."""
    before = sample("rag_hit_score_count")
    client.post(
        "/index", json={"documents": ["vectors are dense", "vectors index fast"]}
    )
    body = client.post("/query", json={"query": "vectors", "k": 2}).json()
    assert body["retrieved"], "fixture: the query must retrieve something"
    assert sample("rag_hit_score_count") == before + len(body["retrieved"]), (
        "every returned hit's score must be observed"
    )


def test_unanswered_queries_are_counted() -> None:
    """A service can be 100% up while answering nothing; the unanswered rate is
    the metric that says so."""
    before = sample("rag_unanswered_total")
    client.post("/index", json={"documents": ["alpha beta gamma"]})
    assert client.post("/query", json={"query": "zzz"}).json()["grounded"] is False
    assert sample("rag_unanswered_total") == before + 1, "no-evidence refusal"
    assert client.post("/query", json={"query": "   "}).json()["grounded"] is False
    assert sample("rag_unanswered_total") == before + 2, "zero-signal refusal"
    assert client.post("/query", json={"query": "alpha"}).json()["grounded"] is True
    assert sample("rag_unanswered_total") == before + 2, (
        "a grounded answer must not count as unanswered"
    )


def test_every_response_carries_a_status_labelled_series() -> None:
    """rag_requests_total has no status label, so a 100% failure rate and a
    healthy service are the same number; an error rate needs status."""
    client.post("/index", json={"documents": ["a doc about vectors"]})
    ok_before = sample("rag_responses_total", {"endpoint": "/query", "status": "200"})
    client.post("/query", json={"query": "vectors"})
    assert (
        sample("rag_responses_total", {"endpoint": "/query", "status": "200"})
        == ok_before + 1
    )
    main._index = None
    conflict_before = sample(
        "rag_responses_total", {"endpoint": "/query", "status": "409"}
    )
    client.post("/query", json={"query": "vectors"})
    assert (
        sample("rag_responses_total", {"endpoint": "/query", "status": "409"})
        == conflict_before + 1
    ), "error responses must be countable, or an error rate cannot exist"


def test_responses_echo_or_mint_a_request_id() -> None:
    """A caller-supplied id must survive the round trip so the caller's trace
    and the server's logs name the same request; absent one, the server mints."""
    client.post("/index", json={"documents": ["a doc about vectors"]})
    echoed = client.post(
        "/query",
        json={"query": "vectors"},
        headers={"X-Request-ID": "corr-abc-123"},
    )
    assert echoed.headers.get("x-request-id") == "corr-abc-123"
    minted = client.post("/query", json={"query": "vectors"})
    assert minted.headers.get("x-request-id"), (
        "a response with no request id cannot be correlated with its logs"
    )


def test_the_request_id_reaches_the_logs(caplog) -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    with caplog.at_level("INFO", logger="app.main"):
        client.post(
            "/query",
            json={"query": "vectors"},
            headers={"X-Request-ID": "corr-log-456"},
        )
    answered = [r for r in caplog.records if r.getMessage() == "query answered"]
    assert answered, "fixture: expected the 'query answered' log line"
    assert getattr(answered[-1], "request_id", None) == "corr-log-456", (
        "the log line for a request must carry that request's id"
    )
