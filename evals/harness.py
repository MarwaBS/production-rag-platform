"""Fixed-dataset retrieval eval through the service's own /index and /query
routes — a regression anywhere on the serving path moves these numbers. A
retrieval eval, not an LLM eval: the default backend is a Mock LLM, so what
can regress is retrieval.

Run standalone:  python -m evals
"""

from __future__ import annotations

from dataclasses import dataclass

# One distinct-topic document per line: the first twelve are gold, the rest are
# adjacent-topic distractors. Without them a top-3 over so small a corpus is
# saturated and no floor derived from it can discriminate.
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
    "Grafana renders dashboards over stored measurement feeds",
    "etcd keeps cluster configuration in a replicated key-value log",
    "Redis caches hot data in memory with optional persistence to disk",
    "Elasticsearch tokenises text into inverted posting lists for keyword lookup",
    "Kafka partitions ordered event streams across broker replicas",
    "nginx terminates TLS and proxies requests to upstream workers",
    "Terraform plans and applies declarative cloud resource graphs",
    "Ansible pushes idempotent configuration tasks over SSH to hosts",
    "Git tracks source history as an immutable directed acyclic graph of commits",
    "Object storage buckets hold immutable blobs addressed by key",
    "OAuth grants scoped delegated access tokens without sharing passwords",
    "A service mesh injects sidecars that encrypt and route pod traffic",
)

# Paraphrases sharing NO word with their gold document (asserted in tests). The
# literal set below measures token matching; this one measures meaning, which a
# query set drawn from its gold documents' vocabulary cannot tell apart.
PARAPHRASE_QUERIES: tuple[tuple[str, int], ...] = (
    ("cosine nearest neighbour lookup across embedding matrices", 0),
    ("hosted engine holding embeddings queried by remote procedure call", 1),
    ("gather numeric telemetry on a schedule then raise alarms", 2),
    ("orchestrator placing workloads onto machines in a cluster", 3),
    ("templating tool bundling cluster resource definitions for reuse", 4),
    ("tooling that assembles runnable filesystem bundles layer by layer", 5),
    ("python web framework validating payloads against declared schemas", 6),
    ("answering questions using evidence fetched from a corpus first", 7),
    ("security tooling reporting weaknesses inside shipped artefacts", 8),
    ("commercial provider exposing text generation behind web endpoints", 9),
    ("machine readable inventory enumerating everything shipped with releases", 10),
    ("parseable event records engineers can search when volume grows", 11),
    # Oblique asks, worded so a near neighbour in the corpus competes: both
    # recall@3 misses fall here. recall@1 also misses one of the direct asks
    # above, so it is lower for another reason.
    ("sampled counters graphed over the day with pages raised on anomalies", 2),
    ("declarative reconciliation keeping desired workload state on a fleet", 3),
    ("immutable userland snapshots assembled stepwise for shipping", 5),
    ("inventory naming third-party code accompanying an artefact", 10),
    ("weaknesses lurking in base layers reported before deployment", 8),
    ("metered SaaS endpoint returning generated prose per request", 9),
    ("grep-able newline-delimited records emitted by busy services", 11),
    ("release bundles rendered from parameterised resource templates", 4),
    ("durable text emitted per event that people later filter, aggregate, search", 11),
    ("periodic polling of numeric health readouts feeding an on-call pager", 2),
    ("distributing an application's cluster setup as one installable unit", 4),
    ("ranking stored numeric representations of text by closeness to a probe", 0),
    ("python toolkit turning annotated functions into checked web routes", 6),
    ("self-healing placement of replicated processes over a machine pool", 3),
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


def evaluate(
    k: int = 3, queries: tuple[tuple[str, str], ...] | None = None
) -> EvalResult:
    """Index the corpus via POST /index, then score each query via POST /query."""
    # Lazy import: the app builds itself from settings at import time, and the
    # floor producer selects the backend via env before triggering it.
    from fastapi.testclient import TestClient

    import app.main as main

    client = TestClient(main.app)
    previous = main._index  # the eval must not leave its corpus behind
    try:
        response = client.post("/index", json={"documents": list(CORPUS)})
        if response.status_code != 201:
            # Not an assert: this is library code, and -O would delete the check.
            raise RuntimeError(f"indexing the eval corpus failed: {response.text}")
        ranks: list[int] = []  # 1-based rank of the gold doc, 0 if outside top-k
        misses: list[str] = []
        for query, gold in queries if queries is not None else QUERIES:
            body = client.post("/query", json={"query": query, "k": k}).json()
            retrieved = [hit["text"] for hit in body["retrieved"]]
            if gold in retrieved:
                ranks.append(retrieved.index(gold) + 1)
            else:
                ranks.append(0)
                misses.append(query)
    finally:
        main._index = previous
    n = len(ranks)
    recall = sum(1 for r in ranks if r > 0) / n
    mrr = sum((1.0 / r if r > 0 else 0.0) for r in ranks) / n
    return EvalResult(k=k, n=n, recall_at_k=recall, mrr=mrr, misses=tuple(misses))
