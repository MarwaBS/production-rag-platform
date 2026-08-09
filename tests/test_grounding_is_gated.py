"""The answer must be produced from the retrieved evidence, not from its count.

This is the repository's headline claim, and it is the one with no gate: the
default answer is assembled from `len(retrieved)`, so removing the context from
the prompt — or the instruction that constrains the model to it — changes
nothing observable. An answer that would read identically for a different corpus
is not grounded in that corpus, whatever the sentence says.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def invoke(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return "an answer"


@pytest.fixture(autouse=True)
def _reset():
    main._index = None
    yield
    main._index = None


def _prompt(monkeypatch: Any, documents: list[str], query: str) -> str:
    recorder = _Recorder()
    monkeypatch.setattr(main, "get_llm", lambda *a, **kw: recorder)
    client.post("/index", json={"documents": documents})
    client.post("/query", json={"query": query, "k": len(documents)})
    return "\n".join(message["content"] for message in recorder.messages)


def test_every_retrieved_document_reaches_the_prompt(monkeypatch) -> None:
    documents = ["vectors are dense arrays", "vectors index quickly"]
    prompt = _prompt(monkeypatch, documents, "vectors")
    for document in documents:
        assert document in prompt, (
            f"retrieved but never sent to the model: {document!r}"
        )


def test_the_prompt_constrains_the_model_to_the_context(monkeypatch) -> None:
    """The constraint must live in the SYSTEM turn.

    The same words appear in the question and in the documents, so searching the
    whole prompt would find them with no instruction present.
    """
    recorder = _Recorder()
    monkeypatch.setattr(main, "get_llm", lambda *a, **kw: recorder)
    client.post("/index", json={"documents": ["vectors are dense arrays"]})
    client.post("/query", json={"query": "vectors", "k": 1})
    system = " ".join(
        message["content"]
        for message in recorder.messages
        if message.get("role") == "system"
    ).lower()
    assert system, "the model must receive a system instruction"
    assert "context" in system
    assert "only" in system


def test_the_answer_differs_when_the_evidence_differs() -> None:
    """Same number of documents, different content, on the shipped default path.

    A response built from the count alone is identical for both, which is the
    proof that nothing about the evidence reached it.
    """
    # One query for both, so an answer echoing the question cannot tell them
    # apart; the answer must carry a word from ITS OWN evidence, because "the two
    # answers differ" is satisfied by a timestamp; and each corpus holds a second
    # document the query does not retrieve, so repeating the whole corpus is not
    # grounding either.
    corpora = {
        "arrays": ["vectors are stored as dense arrays", "unrelated zzz filler"],
        "prometheus": ["vectors are scraped by prometheus", "unrelated zzz filler"],
    }
    answers = {}
    for marker, corpus in corpora.items():
        main._index = None
        client.post("/index", json={"documents": corpus})
        answers[marker] = client.post(
            "/query", json={"query": "vectors", "k": 1}
        ).json()["answer"]

    for marker, answer in answers.items():
        assert marker in answer.lower(), (
            f"the answer does not carry its own evidence: {answer!r} was grounded "
            f"in a document about {marker!r}"
        )
        assert "zzz" not in answer.lower(), (
            f"the answer repeats a document retrieval did not return: {answer!r}"
        )
    assert len(set(answers.values())) == len(answers), (
        f"both corpora produced the same answer: {answers}"
    )


def test_the_route_description_says_what_grounded_does_not_mean() -> None:
    """`grounded` is true on every answered query, so the name alone reads as a
    check on the answer. The README points a reader at this description."""
    description = " ".join(
        app.openapi()["paths"]["/query"]["post"]["description"].split()
    )
    assert "not whether the answer was checked against it" in description, description
