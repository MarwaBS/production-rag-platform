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
