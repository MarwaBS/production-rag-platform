"""Documents are split into windows the embedder can actually represent.

A whole document is the wrong unit twice over: it is one point in the vector
space however long it is, so a passage that answers the question competes with
everything else the document says; and the text that reaches the model is bounded
by a context window the document may not fit.

The property that matters is the second test here. Fixed-width splitting fractures
the sentence carrying the answer across two windows, and then neither window
retrieves it. Overlap is what prevents that, which is also what fixes the size of
the overlap: it must be at least as wide as the longest sentence to be guaranteed.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main
from app.config import Settings
from app.main import app

client = TestClient(app)

MAX_CHARS = 100
OVERLAP = 40
SENTENCE = "the margin widens the served interval"  # 37 chars, inside the overlap


def chunk(*args, **kwargs):
    """Imported per call so a missing splitter fails each test, not collection."""
    from app.chunking import chunk as _chunk

    return _chunk(*args, **kwargs)


def test_a_document_longer_than_the_window_becomes_several_windows() -> None:
    pieces = chunk("word " * 200, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    assert len(pieces) > 1


def test_no_window_exceeds_the_limit() -> None:
    pieces = chunk("word " * 200, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    assert all(len(piece) <= MAX_CHARS for piece in pieces), [
        len(p) for p in pieces if len(p) > MAX_CHARS
    ]


@pytest.mark.parametrize("offset", range(0, 240, 11))
def test_a_sentence_shorter_than_the_overlap_is_never_fractured(offset: int) -> None:
    """At every position in the document, the sentence survives whole in one window."""
    text = ("pad " * offset) + SENTENCE + (" pad" * 60)
    pieces = chunk(text, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    assert any(SENTENCE in piece for piece in pieces), (
        f"the sentence was split across windows at offset {offset}"
    )


def test_a_word_wider_than_the_window_is_split_rather_than_dropped() -> None:
    pieces = chunk("z" * 250, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    assert pieces and all(len(piece) <= MAX_CHARS for piece in pieces)
    assert "".join(pieces).count("z") >= 250


@pytest.mark.parametrize("length", [160, 220, 280])
def test_a_length_landing_on_a_boundary_emits_no_redundant_tail(length: int) -> None:
    """Off by one, the stop test appends a final window wholly inside the one
    before it — an extra vector and a duplicate hit for the same text."""
    text = "".join(chr(0x100 + position) for position in range(length))
    pieces = chunk(text, max_chars=MAX_CHARS, overlap_chars=OVERLAP)
    assert not any(
        later in earlier
        for index, earlier in enumerate(pieces)
        for later in pieces[index + 1 :]
    ), [len(piece) for piece in pieces]


def test_the_splitter_refuses_arguments_it_cannot_advance_through() -> None:
    """A non-positive stride appends windows until memory runs out instead of
    raising, so the refusal has to happen before the loop. The wide-overlap case
    comes first because it is the one an unguarded splitter still returns from."""
    with pytest.raises(ValueError):
        chunk("text", max_chars=10, overlap_chars=10)
    with pytest.raises(ValueError):
        chunk("text", max_chars=0, overlap_chars=0)


def test_an_overlap_as_wide_as_the_window_is_refused_at_startup() -> None:
    """The two bounds are independent env-settable fields, so the pairing is
    reachable; it must die at boot, not on every /index."""
    with pytest.raises(ValidationError):
        Settings(max_chunk_chars=100, chunk_overlap_chars=100)


def setting(name: str) -> int:
    value = getattr(main.settings, name, None)
    assert value is not None, f"Settings.{name} does not exist"
    return int(value)


def test_the_shipped_defaults_can_produce_an_overlap() -> None:
    """Zero is smaller than the window and still overlaps nothing."""
    assert 0 < setting("chunk_overlap_chars") < setting("max_chunk_chars")


def test_indexing_a_long_document_reports_more_chunks_than_documents() -> None:
    main._index = None
    long_document = "vectors " * (setting("max_chunk_chars") // 4)
    body = client.post("/index", json={"documents": [long_document]}).json()
    main._index = None
    assert body["indexed"] == 1
    assert body["chunks"] > 1


_MARKER = "quokka narwhal zebra"  # shares no token with the padding around it


def _multi_window_document() -> str:
    return ("padding " * (setting("max_chunk_chars") // 4)) + _MARKER


def test_retrieval_returns_the_window_that_carries_the_answer() -> None:
    """The retrieval unit must be the window. Indexing whole documents instead
    leaves every count in the response unchanged and surfaces only as evidence
    that does not contain what was asked for."""
    main._index = None
    body = client.post("/index", json={"documents": [_multi_window_document()]}).json()
    assert body["chunks"] > 1, "the fixture is not multi-window"
    hits = client.post("/query", json={"query": _MARKER, "k": 1}).json()["retrieved"]
    main._index = None
    assert hits and _MARKER in hits[0]["text"], hits


def test_the_answer_never_claims_more_documents_than_it_retrieved() -> None:
    """Hits are windows now, so counting them as documents turns one document
    into several apparently corroborating sources."""
    main._index = None
    client.post("/index", json={"documents": [_multi_window_document()]})
    body = client.post("/query", json={"query": f"padding {_MARKER}", "k": 3}).json()
    main._index = None
    hits = body["retrieved"]
    documents = {hit["doc_id"] for hit in hits}
    assert len(hits) > len(documents), "the fixture returned one window per document"
    assert len({hit["id"] for hit in hits}) == len(hits), "windows share a chunk id"
    claimed = re.search(r"(\d+) document\(s\)", body["answer"])
    assert claimed is None or int(claimed.group(1)) <= len(documents), body["answer"]


def test_the_shipped_overlap_is_wide_enough_for_the_sentence_it_guarantees() -> None:
    """The overlap must be at least the sentence length it claims to carry.

    What this checks is that the number is REPRODUCIBLE: a committed producer
    regenerates the artefact exactly. It cannot check that the producer measured
    anything rather than printing a literal — that is what reading the producer
    is for.
    """
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    producer = root / "scripts" / "derive_chunking.py"
    assert producer.exists(), (
        "the chunking constants have no committed producer: scripts/derive_chunking.py"
    )
    derivation = root / "chunking_derivation.json"
    assert derivation.exists(), (
        "no committed derivation for the chunking constants: chunking_derivation.json"
    )
    # An artefact nobody can regenerate is a number someone typed. Running the
    # producer must reproduce the committed file exactly.
    import subprocess
    import sys

    rerun = subprocess.run(
        [sys.executable, str(producer), "--print"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert rerun.returncode == 0, f"the producer failed: {rerun.stderr[-400:]}"
    assert json.loads(rerun.stdout) == json.loads(
        derivation.read_text(encoding="utf-8")
    ), "the producer does not reproduce the committed derivation"
    measured = json.loads(derivation.read_text(encoding="utf-8"))
    longest = measured["longest_sentence_chars"]
    assert setting("chunk_overlap_chars") >= longest, (
        f"overlap {setting('chunk_overlap_chars')} is narrower than the longest "
        f"sentence it must carry ({longest})"
    )
    assert setting("max_chunk_chars") <= measured["embedder_window_chars"], (
        "the window must not exceed what the embedder can actually represent"
    )
    # Bounds alone let a shipped default drift away from the measurement it
    # claims to come from, so pin each default to the producer's own output.
    defaults = measured["derived_defaults"]
    assert setting("chunk_overlap_chars") == defaults["chunk_overlap_chars"]
    assert setting("max_chunk_chars") == defaults["max_chunk_chars"]


def test_the_type_checker_scaffold_is_removed_once_the_splitter_lands() -> None:
    """The override that lets the type checker ignore a module that does not yet
    exist must not outlive it, or it hides real errors in the shipped one."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    if not (root / "app" / "chunking.py").exists():
        pytest.skip("the splitter has not landed yet")
    assert 'module = ["app.chunking"]' not in (root / "pyproject.toml").read_text(
        encoding="utf-8"
    ), "app/chunking.py exists, so its missing-import override is now dead config"
