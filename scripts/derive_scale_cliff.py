"""Producer for the corpus-scale curve (scale_cliff_derivation.json).

Measures recall@3 of the literal query set through the service's own /index and
/query as the corpus grows, on the default hash backend. The same sizes are
measured twice by one construction over two word pools of equal size, so the
word count and the index arithmetic are identical and only the words being
reused change. One thing does not match, and it cuts against the control: its
pool is itself vocabulary the corpus lacks, so it carries that many more
distinct tokens into the buckets rather than fewer.

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

# Both chosen, neither derived. The sizes step by decades from the shipped
# corpus to where the measured curve falls; eight words per distractor is a
# judgement — enough to compete for buckets, cheap enough at ten thousand.
CORPUS_SIZES: tuple[int, ...] = (24, 100, 1_000, 10_000)
DISTRACTOR_WORDS = 8
VOCABULARY: List[str] = sorted(
    {
        token
        for document in CORPUS
        for token in re.findall(r"[a-z0-9]+", document.lower())
    }
)
# A pool of the same size that the corpus does not contain a word of.
FOREIGN: List[str] = [f"unrelated{index}" for index in range(len(VOCABULARY))]


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
    """The construction above word for word, over a pool of the same size that
    the corpus shares nothing with. Same length, same index arithmetic, same one
    new token per document: only the words being reused change."""
    return f"note{index} {_words(index, FOREIGN, DISTRACTOR_WORDS)}"


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

    client = TestClient(main.app)
    previous = main._index
    try:
        shipped = _curve(client, _distractor, VOCABULARY)
        control = _curve(client, _control, VOCABULARY)
    finally:
        main._index = previous
    return {
        "embedder": "hash",
        "embedder_dimensions": embedder._DIM,
        "queries": "evals.harness.QUERIES",
        "queries_evaluated": len(QUERIES),
        "distractor_construction": (
            f"note<i> plus {DISTRACTOR_WORDS} words cycling through the corpus's "
            f"own vocabulary, which repeats every {len(VOCABULARY)} documents: "
            "competition on the words the queries themselves use"
        ),
        "control_construction": (
            "the identical construction over a pool of the same size that the "
            f"corpus shares no word with. Beyond one new token per document, "
            f"the pool's own {len(FOREIGN)} words are unseen too, so the control "
            "carries more distinct vocabulary than the curve above, not less"
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
