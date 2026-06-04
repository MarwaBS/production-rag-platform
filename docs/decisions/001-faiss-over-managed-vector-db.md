# ADR 001 — FAISS over a managed vector database

**Status:** Accepted

## Context
The retrieval layer needs semantic search over a small, per-request corpus.

## Decision
Use an in-process FAISS index (with a pure-NumPy fallback) rather than a managed vector database.

## Why
At this corpus size, brute-force in-process search returns in sub-millisecond time with zero
network hops and zero standing infrastructure cost. Per-request isolation is trivial — each
request gets its own index. A managed vector DB would add latency, monthly cost, and operational
surface with no quality gain at this scale.

## Trade-off
No cross-restart persistence (indexes are rebuilt from input and cached). Memory scales with
concurrency. Both are acceptable at current scale, and the retrieval layer sits behind a
`VectorStoreProtocol` interface, so a managed backend can be swapped in if scale demands it — the
reference implementation (FAISS / NumPy / Qdrant) is open-sourced in
[rag-llm-infra](https://github.com/MarwaBS/rag-llm-infra).

## Alternatives considered
- **Managed vector DB (Pinecone / Weaviate / Qdrant Cloud):** richer ops tooling, but adds a
  network hop, a monthly bill, and a standing service to monitor — no quality gain at per-request scale.
- **pgvector (reuse the primary database):** avoids a new dependency, but couples retrieval latency
  to database load and gives weaker ANN performance than FAISS for this access pattern.

## When to reconsider
Switch to a managed/shared index if (a) corpora become shared across requests (cross-request reuse),
(b) per-instance memory becomes the binding constraint at high concurrency, or (c) real-time index
updates without a rebuild are required. The `VectorStoreProtocol` keeps that a configuration change.
