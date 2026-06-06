"""Deterministic demo embedder (no model download).

Reproducible bag-of-tokens hashing. Production swaps this for
`rag_llm_infra.EmbeddingEngine` (real sentence embeddings).
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_DIM = 128


def embed(texts: List[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), _DIM), dtype="float32")
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            # md5 is a fast token->bucket hash here, not a security primitive;
            # usedforsecurity=False makes the intent explicit and keeps this
            # importable under FIPS-mode interpreters.
            digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
            vecs[row, int(digest, 16) % _DIM] += 1.0
    return vecs
