"""Every axis of the request is bounded, and the bound is enforced by the schema.

The data-plane is unauthenticated by default and the reference pod runs with a
512Mi limit, so an unbounded /index is a single-request kill: the corpus lives in
process, so the pod that dies takes the whole index with it.

Bounds are read from Settings rather than written here, so this file pins that a
bound EXISTS and BITES without also inventing its value. The schema is also
strict and canonical: unknown fields are rejected rather than ignored, and text
is normalised to NFC so canonically equivalent inputs are the same input.
"""

from __future__ import annotations

import unicodedata

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


def test_unknown_fields_are_rejected() -> None:
    """A silently ignored extra field turns a typo into a silent misconfiguration:
    `top_k` instead of `k` runs the query at the default and reports nothing."""
    client.post("/index", json={"documents": ["a doc about vectors"]})
    typo = client.post("/query", json={"query": "vectors", "top_k": 5})
    assert typo.status_code == 422, "a typo'd `top_k` field was silently ignored"
    stray = client.post("/index", json={"documents": ["a doc"], "replace": True})
    assert stray.status_code == 422, "an unknown /index field was silently ignored"


def test_canonically_equivalent_documents_are_one_document() -> None:
    """NFC and NFD are two byte encodings of the same text; without
    normalisation they embed differently and dedup keeps both, so one document
    comes back as two independent sources corroborating each other."""
    nfc = unicodedata.normalize("NFC", "café résumé naïveté détaillé")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # control: the two encodings really are different bytes
    body = client.post("/index", json={"documents": [nfc, nfd]}).json()
    assert body == {"indexed": 1}, "NFC and NFD forms of one text indexed as two"


def test_a_query_matches_its_document_whatever_the_unicode_form() -> None:
    """A query that IS the document, in the other canonical form, must score as
    the document itself — otherwise retrieval depends on which keyboard/OS
    produced the bytes."""
    nfc = unicodedata.normalize("NFC", "café résumé naïveté détaillé")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    client.post("/index", json={"documents": [nfc]})
    body = client.post("/query", json={"query": nfd, "k": 1}).json()
    assert body["grounded"] is True and body["retrieved"], (
        "an NFD query failed to match its own NFC document"
    )
    assert body["retrieved"][0]["score"] == pytest.approx(1.0), (
        "the NFD form of the document scored below the document itself"
    )


def test_a_body_with_no_declared_length_is_refused() -> None:
    """A chunked body declares no Content-Length, so the size budget cannot be
    checked without buffering the whole stream first — the exact cost the budget
    exists to prevent. Refusing with 411 keeps the cap unbypassable."""
    body = (chunk for chunk in [b'{"documents": ["a doc"]}'])
    response = client.post(
        "/index", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 411, (
        f"a length-less chunked body was accepted with {response.status_code}"
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
