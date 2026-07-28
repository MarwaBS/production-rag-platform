"""Every axis of the request is bounded, and the bound is enforced by the schema.

The data-plane is unauthenticated by default and the reference pod runs with a
512Mi limit, so an unbounded /index is a single-request kill: the corpus lives in
process, so the pod that dies takes the whole index with it.

Bounds are read from Settings rather than written here, so this file pins that a
bound EXISTS and BITES without also inventing its value.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


def bound(name: str) -> int:
    value = getattr(main.settings, name, None)
    assert value is not None, (
        f"Settings.{name} does not exist — the input contract is unbounded on this axis"
    )
    return int(value)


@pytest.fixture(autouse=True)
def _reset():
    main._index = None
    yield
    main._index = None


@pytest.mark.parametrize("documents", [[""], ["   "], ["\t\n"], ["ok doc", ""]])
def test_blank_documents_are_rejected(documents: list[str]) -> None:
    """An item-level bound, not a list-level one: a corpus of empty strings
    otherwise indexes cleanly and flips /ready to 200 over nothing."""
    assert client.post("/index", json={"documents": documents}).status_code == 422


def test_an_empty_query_is_rejected() -> None:
    """An empty string is a missing field, so the schema rejects it.

    Whitespace-only text is a valid string carrying no signal, and belongs to the
    retrieval contract instead (tests/test_retrieval_honesty.py).
    """
    client.post("/index", json={"documents": ["a doc about vectors"]})
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_document_count_is_capped() -> None:
    over = bound("max_documents") + 1
    response = client.post(
        "/index", json={"documents": [f"doc {i}" for i in range(over)]}
    )
    assert response.status_code == 422, f"{over} documents were accepted"


def test_document_length_is_capped() -> None:
    over = "x" * (bound("max_document_chars") + 1)
    assert client.post("/index", json={"documents": [over]}).status_code == 422


def test_query_length_is_capped() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    over = "x" * (bound("max_query_chars") + 1)
    assert client.post("/query", json={"query": over}).status_code == 422


def test_k_is_capped() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    over = bound("max_top_k") + 1
    assert (
        client.post("/query", json={"query": "vectors", "k": over}).status_code == 422
    )


def test_total_request_size_is_capped() -> None:
    """The per-field bounds above still permit a request that is enormous in
    aggregate: many documents, each individually legal. The budget that protects
    the process is the total one."""
    budget = bound("max_request_bytes")
    per_document = bound("max_document_chars")
    # Build the oversize body from the FEWEST, largest legal documents. Filling it
    # with many small ones would trip the count cap first and prove nothing about
    # the aggregate.
    needed = budget // per_document + 1
    assert needed <= bound("max_documents"), (
        f"{bound('max_documents')} documents of {per_document} chars cannot reach the "
        f"{budget}-byte budget, so the budget can never be the binding limit"
    )
    documents = ["x" * per_document] * needed
    response = client.post("/index", json={"documents": documents})
    assert response.status_code in (413, 422), (
        f"a body over the {budget}-byte budget was accepted with {response.status_code}"
    )
