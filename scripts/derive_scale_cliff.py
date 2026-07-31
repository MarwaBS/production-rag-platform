"""Producer for the corpus-scale curve (scale_cliff_derivation.json).

Measures recall@3 of the literal query set through the service's own /index and
/query as the corpus grows, on the default hash backend. How far it falls
depends on how much vocabulary the competing documents share, so the distractor
construction is recorded beside the numbers rather than left implied.

Run: python scripts/derive_scale_cliff.py            # rewrite the committed file
     python scripts/derive_scale_cliff.py --print    # print, no write (CI gate)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

os.environ["APP_EMBEDDING_BACKEND"] = "hash"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.harness import CORPUS, QUERIES  # noqa: E402

CORPUS_SIZES = (24, 100, 1_000, 10_000)
DISTRACTOR_WORDS = 8


def _distractor(index: int, vocabulary: list[str]) -> str:
    # Drawn from the corpus's own vocabulary so the competition is same-domain,
    # with a unique token so content dedup cannot shrink the corpus measured.
    words = " ".join(
        vocabulary[(index * 7 + offset * 13) % len(vocabulary)]
        for offset in range(DISTRACTOR_WORDS)
    )
    return f"note{index} {words}"


def derive() -> dict:
    from fastapi.testclient import TestClient

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
    curve: dict[str, float] = {}
    try:
        for size in CORPUS_SIZES:
            documents = list(CORPUS) + [
                _distractor(index, vocabulary) for index in range(size - len(CORPUS))
            ]
            indexed = client.post("/index", json={"documents": documents}).json()
            assert indexed["indexed"] == size, f"dedup shrank the corpus: {indexed}"
            found = 0
            for query, gold in QUERIES:
                body = client.post("/query", json={"query": query, "k": 3}).json()
                found += gold in [hit["text"] for hit in body["retrieved"]]
            curve[str(size)] = round(found / len(QUERIES), 4)
    finally:
        main._index = previous
    return {
        "embedder": "hash",
        "embedder_dimensions": 128,
        "queries": "evals.harness.QUERIES",
        "gold_documents": len(QUERIES),
        "distractor_construction": (
            f"note<i> plus {DISTRACTOR_WORDS} words drawn from the corpus's own "
            "vocabulary; the size of the fall depends on this overlap"
        ),
        "recall_at_3": curve,
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
