"""Retrieval evaluation harness for the reference RAG service.

A small fixed question/gold-document set run through the SAME retrieval path the
service serves (`app.embedder.embed` + the NumPy vector store), reporting
recall@k and MRR. `tests/test_eval.py` gates CI on a measured recall floor — a
real retrieval regression (a broken embedder, a store bug, a bad `k` default)
drops recall below the floor and fails the build. See `harness.py`.
"""

from .harness import EvalResult, evaluate

__all__ = ["EvalResult", "evaluate"]
