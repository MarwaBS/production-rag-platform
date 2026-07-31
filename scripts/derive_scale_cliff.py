"""Producer for the corpus-scale curve (scale_cliff_derivation.json).

Measures recall@3 of the literal query set through the service's own /index and
/query as the corpus grows, on the default hash backend. Recall falls for two
reasons at once, so the same sizes are measured twice. Any distractor aliases
into the 128 buckets simply by existing; a distractor drawn from the corpus's
own vocabulary also competes on the words the queries use. The control curve
uses distractors sharing no token with the corpus, so it is the cost of the
buckets alone, and the shipped curve is that plus the same-domain competition.

Run: python scripts/derive_scale_cliff.py            # rewrite the committed file
     python scripts/derive_scale_cliff.py --print    # print, no write (CI gate)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from typing import Any, Callable, Dict, List

os.environ["APP_EMBEDDING_BACKEND"] = "hash"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.harness import CORPUS, QUERIES  # noqa: E402

CORPUS_SIZES: tuple[int, ...] = (24, 100, 1_000, 10_000)
DISTRACTOR_WORDS = 8


def _words(index: int, vocabulary: List[str], count: int) -> str:
    return " ".join(
        vocabulary[(index * 7 + offset * 13) % len(vocabulary)]
        for offset in range(count)
    )


def _distractor(index: int, vocabulary: List[str]) -> str:
    """Same-domain competition plus one token of its own, so distinct vocabulary
    grows with the corpus — and the token also keeps content dedup off."""
    return f"note{index} {_words(index, vocabulary, DISTRACTOR_WORDS)}"


def _control(index: int, vocabulary: List[str]) -> str:
    """The same document length sharing no token with the corpus, so nothing is
    left to compete but bucket collision. Substituting corpus words instead was
    tried and measured worse than the shipped curve at 1000: they carry the
    queries' own terms, which is the effect the control has to hold still."""
    return " ".join(f"n{index}w{offset}" for offset in range(DISTRACTOR_WORDS + 1))


def _curve(
    client: Any,
    make_document: Callable[[int, List[str]], str],
    vocabulary: List[str],
) -> Dict[str, float]:
    curve: Dict[str, float] = {}
    for size in CORPUS_SIZES:
        documents = list(CORPUS) + [
            make_document(index, vocabulary) for index in range(size - len(CORPUS))
        ]
        indexed = client.post("/index", json={"documents": documents}).json()
        if indexed["indexed"] != size:
            # A construction that repeats measures a few hundred documents while
            # reporting the size it meant to measure.
            raise RuntimeError(f"dedup shrank the corpus: {indexed}")
        found = 0
        for query, gold in QUERIES:
            body = client.post("/query", json={"query": query, "k": 3}).json()
            found += gold in [hit["text"] for hit in body["retrieved"]]
        curve[str(size)] = round(found / len(QUERIES), 4)
    return curve


def derive() -> dict:
    from fastapi.testclient import TestClient

    import app.embedder as embedder
    import app.main as main

    vocabulary = sorted(
        {
            token
            for document in CORPUS
            for token in re.findall(r"[a-z0-9]+", document.lower())
        }
    )
    client = TestClient(main.app)
    previous = main._index
    try:
        shipped = _curve(client, _distractor, vocabulary)
        control = _curve(client, _control, vocabulary)
    finally:
        main._index = previous
    return {
        "embedder": "hash",
        "embedder_dimensions": embedder._DIM,
        "queries": "evals.harness.QUERIES",
        "queries_evaluated": len(QUERIES),
        "distractor_construction": (
            f"note<i> plus {DISTRACTOR_WORDS} words cycling through the corpus's "
            f"own vocabulary, which repeats every {len(vocabulary)} documents: "
            "same-domain competition on the words the queries themselves use"
        ),
        "control_construction": (
            "the same document length in tokens the corpus does not contain, so "
            "the only competition left is aliasing into the buckets. The gap "
            "between the two curves is what the same-domain wording costs"
        ),
        "recall_at_3": shipped,
        "recall_at_3_control": control,
    }


def main() -> None:
    payload = json.dumps(derive(), indent=2) + "\n"
    if "--print" in sys.argv:
        sys.stdout.write(payload)
        return
    out = pathlib.Path(__file__).resolve().parent.parent / "scale_cliff_derivation.json"
    out.write_text(payload, encoding="utf-8")
    sys.stdout.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
