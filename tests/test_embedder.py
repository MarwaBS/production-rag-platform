"""The embedding backend is typed configuration: defaulted, validated, and —
like every other optional backend — refused at boot with the exact fix when
its package is missing."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import app.embedder as embedder
import app.main as main
from app.config import Settings


def test_the_default_backend_is_the_hash_embedder() -> None:
    assert Settings().embedding_backend == "hash"


def test_an_unknown_backend_is_rejected_at_startup() -> None:
    with pytest.raises(ValidationError):
        # The runtime rejection IS the assertion, so the bad literal is deliberate.
        Settings(embedding_backend="word2vec")  # type: ignore[arg-type]


def test_embed_dispatches_on_the_selected_backend(monkeypatch) -> None:
    """`embed` must consult the setting on every call — a binding frozen at
    import would make the CI job's env-var selection silently serve hashes."""
    sentinel = np.ones((1, 4), dtype="float32")
    monkeypatch.setattr(embedder, "_semantic_embed", lambda texts: sentinel)
    monkeypatch.setattr(embedder._settings, "embedding_backend", "semantic")
    assert embedder.embed(["anything"]) is sentinel


def test_a_missing_semantic_package_refuses_to_boot_with_the_fix(
    monkeypatch,
) -> None:
    """Mirrors the openai/faiss/qdrant boot guard. Simulated via find_spec
    because the dev environment may genuinely have the extra installed."""
    real_find_spec = main.importlib.util.find_spec
    monkeypatch.setattr(
        main.importlib.util,
        "find_spec",
        lambda name: None if name == "sentence_transformers" else real_find_spec(name),
    )
    with pytest.raises(RuntimeError, match=r"production-rag-platform\[semantic\]"):
        main._require_backend_packages(Settings(embedding_backend="semantic"))
    main._require_backend_packages(Settings())  # default stack: no raise
