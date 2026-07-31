"""Text embedding behind one function, backend selected by APP_EMBEDDING_BACKEND.

The default hash embedder matches words, not meaning: it retrieves few of the
paraphrases that share no token with their document, and its 128 buckets alias
as vocabulary grows. Both are measured, in eval_floors_derivation.json and
scale_cliff_derivation.json; the paraphrase floors assume the semantic extra.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, List

import numpy as np

from .config import get_settings

_DIM = 128
_settings = get_settings()
_semantic_model: Any = None


def _hash_embed(texts: List[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), _DIM), dtype="float32")
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            # md5 is a fast token->bucket hash here, not a security primitive;
            # usedforsecurity=False makes the intent explicit and keeps this
            # importable under FIPS-mode interpreters.
            digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
            vecs[row, int(digest, 16) % _DIM] += 1.0
    return vecs


def _semantic_embed(texts: List[str]) -> np.ndarray:
    # Lazy and cached: the model costs seconds and ~90MB; boot already
    # verified the package exists when this backend is selected.
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer

        _semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(_semantic_model.encode(texts), dtype="float32")


def embed(texts: List[str]) -> np.ndarray:
    # Read off the settings object per call rather than captured into a
    # constant, so the backend stays switchable; the env is read once, at import.
    if _settings.embedding_backend == "semantic":
        return _semantic_embed(texts)
    return _hash_embed(texts)
