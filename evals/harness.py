"""Fixed-dataset retrieval evaluation, exercised through the real serving path.

This is deliberately a *retrieval* eval, not an LLM-quality eval: the reference
service ships a Mock LLM by default, so generation is a fixed template — what can
actually regress here is RETRIEVAL (the embedder, the vector store, the top-k
decode). recall@k and MRR over a fixed Q/gold set catch exactly that, and the
gate in tests/test_eval.py can go red (drop the embedder to noise, break the
store, mis-default k → recall falls below the floor).

Run standalone:  python -m evals
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_llm_infra import get_vector_store

from app.embedder import embed

# Fixed corpus: one distinct-topic document per line. Retrieval must rank the
# gold document for each query above these distractors.
CORPUS: tuple[str, ...] = (
    "FAISS performs in-process vector similarity search using inner-product indexes",
    "Qdrant is a vector database served over gRPC with payload filtering",
    "Prometheus scrapes metrics endpoints and stores time series for alerting",
    "Kubernetes schedules containers across nodes and manages pod lifecycles",
    "Helm packages Kubernetes manifests into versioned reusable charts",
    "Docker builds container images from a layered Dockerfile specification",
    "FastAPI serves typed REST endpoints with Pydantic request validation",
    "Retrieval-augmented generation grounds model answers in retrieved documents",
    "Trivy scans container images for known operating-system and library vulnerabilities",
    "OpenAI provides hosted large language model completions through an HTTP API",
    "A CycloneDX software bill of materials lists every dependency in a build",
    "Structured JSON logs let operators ingest and query application events at scale",
)

# (query, gold document) — the gold is the corpus entry the query is about.
QUERIES: tuple[tuple[str, str], ...] = (
    ("inner product similarity search index", CORPUS[0]),
    ("vector database over grpc with payload filtering", CORPUS[1]),
    ("scrape metrics time series for alerting", CORPUS[2]),
    ("schedule containers across nodes and pods", CORPUS[3]),
    ("package kubernetes manifests into versioned charts", CORPUS[4]),
    ("build container images from a dockerfile", CORPUS[5]),
    ("typed rest endpoints with pydantic validation", CORPUS[6]),
    ("ground model answers in retrieved documents", CORPUS[7]),
    ("scan container images for library vulnerabilities", CORPUS[8]),
    ("hosted language model completions over an http api", CORPUS[9]),
    ("software bill of materials listing dependencies", CORPUS[10]),
    ("ingest and query json application event logs", CORPUS[11]),
)


@dataclass(frozen=True)
class EvalResult:
    k: int
    n: int
    recall_at_k: float  # fraction of queries whose gold doc is in the top-k
    mrr: float  # mean reciprocal rank of the gold doc (0 if outside top-k)
    misses: tuple[str, ...]  # queries whose gold fell outside the top-k

    def summary(self) -> str:
        lines = [
            f"retrieval eval - n={self.n}, k={self.k}",
            f"  recall@{self.k}: {self.recall_at_k:.3f}",
            f"  MRR:        {self.mrr:.3f}",
        ]
        if self.misses:
            lines.append("  misses:")
            lines.extend(f"    - {q}" for q in self.misses)
        return "\n".join(lines)


def evaluate(k: int = 3) -> EvalResult:
    """Index the fixed corpus and score each query's retrieval, via the real path."""
    store = get_vector_store("numpy")
    store.add(embed(list(CORPUS)))
    ranks: list[int] = []  # 1-based rank of the gold doc, or 0 if outside top-k
    misses: list[str] = []
    for query, gold in QUERIES:
        _, idx = store.search(embed([query]), k=k)
        retrieved = [CORPUS[int(i)] for i in idx[0] if i >= 0]
        if gold in retrieved:
            ranks.append(retrieved.index(gold) + 1)
        else:
            ranks.append(0)
            misses.append(query)
    n = len(QUERIES)
    recall = sum(1 for r in ranks if r > 0) / n
    mrr = sum((1.0 / r if r > 0 else 0.0) for r in ranks) / n
    return EvalResult(k=k, n=n, recall_at_k=recall, mrr=mrr, misses=tuple(misses))
