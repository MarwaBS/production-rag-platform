# ADR 001 — An in-process index over a managed vector database

**Status:** Accepted

**Scope:** the decision holds for both the full system and this repo. What differs is the
backend each one runs: this repo defaults to the pure-NumPy store, with `faiss` and `qdrant`
available as extras. No caching layer described here exists in `app/`.

## Context
The retrieval layer needs search over a corpus supplied by the caller, small enough to hold
in memory.

## Decision
Hold the index in the service process behind a `VectorStoreProtocol`, rather than calling a
managed vector database. NumPy is the default implementation; FAISS and Qdrant are selected
with `APP_VECTOR_BACKEND` and their extras.

## Why
At this corpus size, brute-force in-process search costs zero network hops and zero standing
infrastructure. A managed vector DB would add a round trip, a monthly bill, and a service to
monitor, for no quality gain at this scale. Keeping the choice behind a protocol means the
backend is a configuration change rather than a rewrite.

No latency figure is claimed here, because nothing in this repo measures one. The numbers this
repo does state — the chunk window, the eval floors, the scale curve — each carry a producer
under `scripts/` and a committed artefact that a gate re-derives.

## Trade-off
Three things follow from holding the index in the process, and all three are load-bearing:

- **No persistence.** The corpus is built by `POST /index`, which replaces it wholly. A
  restart leaves the service with no corpus, and `/query` answers 409 until it is indexed
  again. Nothing is written to disk and nothing is cached between restarts.
- **One corpus per process, not one per request.** `/index` swaps a single process-global
  snapshot; concurrent requests all read whichever snapshot is current. There is no
  per-caller isolation, so the reference service is single-tenant.
- **No horizontal scale-out.** A second replica starts empty and answers 409 until it
  receives its own `/index`, which is why the Helm chart pins one replica and ships no
  autoscaler.

The reference implementation of the protocol (FAISS / NumPy / Qdrant) is open-sourced in
[rag-llm-infra](https://github.com/MarwaBS/rag-llm-infra).

## Alternatives considered
- **Managed vector DB (Pinecone / Weaviate / Qdrant Cloud):** richer ops tooling, but adds a
  network hop, a monthly bill, and a standing service to monitor — no quality gain at this scale.
- **pgvector (reuse the primary database):** avoids a new dependency, but couples retrieval
  latency to database load and gives weaker ANN performance for this access pattern.

## When to reconsider
Move to a shared external store if (a) the corpus must survive a restart, (b) more than one
replica has to serve it, (c) callers need isolation from each other, or (d) per-instance
memory becomes the binding constraint. Any one of those makes the in-process index the wrong
choice; the protocol keeps the switch a configuration change.
