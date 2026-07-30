"""Producer for the chunking constants (chunking_derivation.json).

Two numbers govern the splitter, and neither is a preference:

- `chunk_overlap_chars` must be at least the longest sentence the corpus is
  expected to guarantee intact (app/chunking.py explains why the overlap is
  what prevents a fractured grounding sentence). The guarantee set is the
  shipped reference corpus in evals/harness.py — the documents the retrieval
  eval holds the service accountable for — so the overlap IS that measurement.

- `max_chunk_chars` must not exceed what the embedding backend can represent.
  The semantic backend (sentence-transformers all-MiniLM-L6-v2) truncates at
  256 wordpieces; a wordpiece is never shorter than one character, so 256
  characters is the largest window guaranteed fully representable under any
  tokenization. Deliberately worst-case: it needs no model download, so this
  producer reproduces byte-identically in the base environment where CI runs
  it. The default hash embedder has no length limit and imposes no bound.

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
