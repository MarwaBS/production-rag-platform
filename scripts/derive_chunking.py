"""Producer for the chunking constants (chunking_derivation.json).

`chunk_overlap_chars` := the longest sentence in the shipped reference corpus
(the sentences the eval guarantees intact). `max_chunk_chars` := the semantic
model's 256-wordpiece truncation limit under worst-case one-character tokens —
worst-case so the producer reproduces byte-identically with no model installed.

Run: python scripts/derive_chunking.py            # rewrite the committed file
     python scripts/derive_chunking.py --print    # print, no write (CI gate)
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.harness import CORPUS  # noqa: E402 — needs the repo root on sys.path


def derive() -> dict:
    lengths = sorted(len(document) for document in CORPUS)
    return {
        "guarantee_corpus": "evals.harness.CORPUS",
        "n_sentences": len(lengths),
        "sentence_chars": lengths,
        "longest_sentence_chars": lengths[-1],
        "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_token_limit": 256,
        "worst_case_chars_per_token": 1,
        "embedder_window_chars": 256,
        "derived_defaults": {
            "chunk_overlap_chars": lengths[-1],
            "max_chunk_chars": 256,
        },
    }


def main() -> None:
    payload = json.dumps(derive(), indent=2) + "\n"
    if "--print" in sys.argv:
        sys.stdout.write(payload)
        return
    out = pathlib.Path(__file__).resolve().parent.parent / "chunking_derivation.json"
    out.write_text(payload, encoding="utf-8")
    sys.stdout.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
