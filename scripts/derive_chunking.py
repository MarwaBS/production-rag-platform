"""Producer for the chunking constants (chunking_derivation.json).

`chunk_overlap_chars` := the longest sentence in the shipped reference corpus
(the sentences the eval guarantees intact). `max_chunk_chars` := the semantic
model's 256-token limit less the two special tokens it adds, at one wordpiece
per character. Those three constants are literals here so this reproduces with
no model installed; a semantic-marked gate holds them against the real
tokenizer. The assumption bounds text whose characters cost a wordpiece each —
scripts that cost more per character can still overflow the limit.

Run: python scripts/derive_chunking.py            # rewrite the committed file
     python scripts/derive_chunking.py --print    # print, no write (CI gate)
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.harness import CORPUS  # noqa: E402 — needs the repo root on sys.path


def _sentences(document: str) -> list[str]:
    # The guarantee is stated per sentence, so measure sentences: every line of
    # this corpus happens to be one, and a corpus of prose would not be.
    return [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", document) if part.strip()
    ]


def derive() -> dict:
    lengths = sorted(
        len(sentence) for document in CORPUS for sentence in _sentences(document)
    )
    token_limit, special_tokens, chars_per_token = 256, 2, 1
    window = (token_limit - special_tokens) * chars_per_token
    return {
        "guarantee_corpus": "evals.harness.CORPUS",
        "n_sentences": len(lengths),
        "sentence_chars": lengths,
        "longest_sentence_chars": lengths[-1],
        "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_token_limit": token_limit,
        "special_tokens_reserved": special_tokens,
        "assumed_chars_per_wordpiece": chars_per_token,
        "assumption_limit": (
            "text whose characters cost one wordpiece each; scripts that cost "
            "more per character can still overflow the sequence limit"
        ),
        "embedder_window_chars": window,
        "derived_defaults": {
            "chunk_overlap_chars": lengths[-1],
            "max_chunk_chars": window,
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
