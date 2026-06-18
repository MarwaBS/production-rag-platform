"""Production RAG reference service — built on the published `rag-llm-infra` package.

A single-process reference service that demonstrates the production envelope around
the infra library: typed config, structured logging, Prometheus metrics,
liveness/readiness probes, and an index -> retrieve -> generate API. The corpus is
held **in process** (one vector store per pod), so the reference deployment runs a
single replica; see deploy/helm/values.yaml. Runs on the NumPy vector store + Mock
LLM with no API key; set `APP_LLM_BACKEND=openai` + `OPENAI_API_KEY` for real
generation.

    uvicorn app.main:app
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag_llm_infra import configure_logging, get_llm, get_vector_store

from .config import get_settings
from .embedder import embed

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="production-rag-platform", version="1.0.0")


@dataclass(frozen=True)
class _Index:
    """Immutable snapshot of the indexed corpus and its vector store.

    Held behind a single module-level reference so /index swaps the whole
    snapshot in one atomic assignment and /query reads one consistent
    (docs, store) pair. The previous design stored docs and store under two
    separate dict keys and read them in two steps, so a /query interleaved with
    a re-index could pair a new store with stale docs (or vice versa) and raise
    IndexError. A single reference makes that torn read impossible by
    construction.

    Atomicity caveat: the lock-free swap relies on a single name rebind being
    atomic, which holds under CPython's GIL (the supported runtime here, and the
    reference deployment is single-replica/single-process anyway). On a
    free-threaded build (PEP 703, Python 3.13+ `--disable-gil`) a rebind is no
    longer guaranteed atomic against a concurrent read; a multi-process or
    free-threaded deployment should guard the swap with a lock (or move the
    corpus to a shared external store).
    """

    docs: tuple[str, ...]
    store: Any


_index: _Index | None = None

_REQUESTS = Counter("rag_requests_total", "Total API requests", ["endpoint"])
_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency in seconds")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard the destructive /index write when APP_API_KEY is configured.

    Unset (the default) leaves /index open for the no-auth local/demo run. When
    set, /index requires a matching X-API-Key header — /index REPLACES the
    entire corpus, so it must not be world-writable in a shared deployment.
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class IndexRequest(BaseModel):
    # min_length=1: an empty index is meaningless and would otherwise flip
    # /ready to 200 with zero documents.
    documents: List[str] = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    # ge=1: a non-positive k reaches the store's argpartition and 500s.
    k: int = Field(default=settings.default_top_k, ge=1)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness: the process is up. This is the k8s readiness/liveness target."""
    _REQUESTS.labels("health").inc()
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """App-level readiness: has a corpus been indexed?

    Deliberately NOT the k8s pod-readiness probe (that targets /health). The
    index is populated at runtime via POST /index, so gating *pod* readiness on
    this would deadlock: the Service routes no traffic to an un-indexed pod, so
    the pod could never receive the /index call that would make it ready.
    Clients poll /ready to know when /query will return grounded answers.
    """
    _REQUESTS.labels("ready").inc()
    is_ready = _index is not None
    return JSONResponse({"ready": is_ready}, status_code=200 if is_ready else 503)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/index", status_code=201)
def index(req: IndexRequest, _: None = Depends(require_api_key)) -> Dict[str, int]:
    """Build a fresh vector store from `documents` and swap it in atomically.

    NOTE: this REPLACES the entire corpus — it is not additive. Single-tenant
    reference semantics; a multi-tenant system would namespace per caller and
    persist to a shared store.
    """
    _REQUESTS.labels("index").inc()
    store = get_vector_store(settings.vector_backend)
    store.add(embed(list(req.documents)))
    global _index
    _index = _Index(docs=tuple(req.documents), store=store)
    return {"indexed": len(req.documents)}


@app.post("/query")
def query(req: QueryRequest) -> Any:
    _REQUESTS.labels("query").inc()
    start = time.perf_counter()
    # Observe latency on EVERY exit path (success, 409, or a raised error) in a
    # finally — not only on the success tail. Otherwise the histogram silently
    # excludes the 409 "not indexed" and error paths, understating real latency
    # and hiding a slow failure mode.
    try:
        snapshot = _index  # single atomic read — docs and store are always consistent
        if snapshot is None:
            # Query before any corpus exists is a client error, not a 200 with an
            # error key buried in the body.
            return JSONResponse(
                {"error": "index documents first", "retrieved": [], "answer": ""},
                status_code=409,
            )
        docs, store = snapshot.docs, snapshot.store
        _, idx = store.search(embed([req.query]), k=min(req.k, len(docs)))
        retrieved = [docs[int(i)] for i in idx[0] if i >= 0]
        context = "\n".join(f"- {d}" for d in retrieved)
        if settings.llm_backend == "mock":
            llm = get_llm("mock", response=lambda _m: f"(answer grounded in {len(retrieved)} retrieved docs)")
        else:
            llm = get_llm(settings.llm_backend)
        answer = llm.invoke(
            [
                {"role": "system", "content": "Answer using ONLY the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
            ]
        )
        return {"retrieved": retrieved, "answer": answer}
    finally:
        _QUERY_LATENCY.observe(time.perf_counter() - start)
