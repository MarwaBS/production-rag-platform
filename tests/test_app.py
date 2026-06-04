"""Integration tests for the reference service (no network, no API key)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_exposes_prometheus() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text


def test_ready_true_after_index() -> None:
    client.post("/index", json={"documents": ["faiss vector search", "qdrant database"]})
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_index_then_query_grounds_answer() -> None:
    docs = ["FAISS in-process vector similarity search", "Qdrant vector database with gRPC"]
    assert client.post("/index", json={"documents": docs}).json() == {"indexed": 2}
    body = client.post("/query", json={"query": "vector similarity search", "k": 1}).json()
    assert body["retrieved"] == ["FAISS in-process vector similarity search"]
    assert "grounded" in body["answer"]
