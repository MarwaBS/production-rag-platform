"""Integration tests for the reference service (no network, no API key).

Covers the happy path plus the error surface and edge cases the service must
handle: readiness before indexing, query-before-index, input validation
(empty corpus, non-positive k), corpus-replace semantics, the re-index
torn-read regression, and optional API-key auth on the destructive write.
"""
import threading

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_index():
    # The corpus lives in a module-level reference; reset it around every test
    # so cases don't leak state into each other regardless of run order.
    main._index = None
    yield
    main._index = None


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_exposes_prometheus() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text


def test_ready_503_before_index() -> None:
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_ready_true_after_index() -> None:
    client.post("/index", json={"documents": ["faiss vector search", "qdrant database"]})
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_index_then_query_grounds_answer() -> None:
    docs = ["FAISS in-process vector similarity search", "Qdrant vector database with gRPC"]
    r = client.post("/index", json={"documents": docs})
    assert r.status_code == 201
    assert r.json() == {"indexed": 2}
    body = client.post("/query", json={"query": "vector similarity search", "k": 1}).json()
    assert body["retrieved"] == ["FAISS in-process vector similarity search"]
    assert "grounded" in body["answer"]


def test_query_before_index_returns_409() -> None:
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 409
    assert r.json()["error"] == "index documents first"


def test_index_rejects_empty_documents_422() -> None:
    r = client.post("/index", json={"documents": []})
    assert r.status_code == 422
    # An empty corpus must not slip through and flip readiness to 200.
    assert client.get("/ready").status_code == 503


def test_query_rejects_nonpositive_k_422() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    for bad_k in (0, -2):
        r = client.post("/query", json={"query": "vectors", "k": bad_k})
        assert r.status_code == 422, f"k={bad_k} should be rejected, got {r.status_code}"


def test_index_replaces_corpus_not_additive() -> None:
    client.post("/index", json={"documents": ["first corpus alpha"]})
    client.post("/index", json={"documents": ["second corpus beta"]})
    body = client.post("/query", json={"query": "corpus", "k": 5}).json()
    assert body["retrieved"] == ["second corpus beta"]  # the old corpus is gone


def test_reindex_with_smaller_corpus_never_500s() -> None:
    # Replace a large corpus with a smaller one and immediately query. The
    # atomic snapshot guarantees docs and store always match, so retrieved
    # indices never exceed the current doc list.
    client.post("/index", json={"documents": [f"doc number {i} about vectors" for i in range(20)]})
    client.post("/index", json={"documents": ["only one doc about vectors"]})
    r = client.post("/query", json={"query": "vectors", "k": 5})
    assert r.status_code == 200
    assert r.json()["retrieved"] == ["only one doc about vectors"]


def test_concurrent_reindex_and_query_never_5xx() -> None:
    # Concurrency regression for the torn read: with the old two-key state a
    # query interleaved with a re-index could pair a new store with stale docs
    # and 500. The single atomic snapshot makes that impossible — no request
    # should ever see a 5xx, no matter the interleaving.
    client.post("/index", json={"documents": [f"d{i} vectors" for i in range(10)]})
    errors: list[int] = []

    def reindexer() -> None:
        for i in range(40):
            n = 1 if i % 2 else 10
            client.post("/index", json={"documents": [f"d{j} vectors" for j in range(n)]})

    def querier() -> None:
        for _ in range(40):
            r = client.post("/query", json={"query": "vectors", "k": 5})
            if r.status_code >= 500:
                errors.append(r.status_code)

    threads = [threading.Thread(target=reindexer) for _ in range(2)]
    threads += [threading.Thread(target=querier) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"saw 5xx responses under concurrency: {errors}"


def test_index_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    assert client.post("/index", json={"documents": ["x doc"]}).status_code == 401
    assert (
        client.post("/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    assert (
        client.post("/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "s3cret"}).status_code
        == 201
    )
