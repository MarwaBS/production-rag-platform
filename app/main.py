"""Production RAG reference service — built on the published `rag-llm-infra` package.

Demonstrates the production envelope around the infra library: typed config,
structured logging, Prometheus metrics, liveness/readiness probes, and an
index -> retrieve -> generate API. Runs on the NumPy vector store + Mock LLM with
no API key; set `APP_LLM_BACKEND=openai` + `OPENAI_API_KEY` for real generation.

    uvicorn app.main:app
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from rag_llm_infra import configure_logging, get_llm, get_vector_store

from .config import get_settings
from .embedder import embed

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="production-rag-platform", version="1.0.0")
_state: Dict[str, Any] = {"docs": [], "store": None}

_REQUESTS = Counter("rag_requests_total", "Total API requests", ["endpoint"])
_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency in seconds")


class IndexRequest(BaseModel):
    documents: List[str]


class QueryRequest(BaseModel):
    query: str
    k: int = settings.default_top_k


@app.get("/health")
def health() -> Dict[str, str]:
    _REQUESTS.labels("health").inc()
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    _REQUESTS.labels("ready").inc()
    is_ready = _state["store"] is not None
    return JSONResponse({"ready": is_ready}, status_code=200 if is_ready else 503)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/index")
def index(req: IndexRequest) -> Dict[str, int]:
    _REQUESTS.labels("index").inc()
    store = get_vector_store(settings.vector_backend)
    store.add(embed(req.documents))
    _state["docs"], _state["store"] = req.documents, store
    return {"indexed": len(req.documents)}


@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    _REQUESTS.labels("query").inc()
    start = time.perf_counter()
    store, docs = _state["store"], _state["docs"]
    if store is None:
        return {"error": "index documents first", "retrieved": [], "answer": ""}
    _, idx = store.search(embed([req.query]), k=min(req.k, len(docs)))
    retrieved = [docs[int(i)] for i in idx[0] if i >= 0]
    context = "\n".join(f"- {d}" for d in retrieved)
    if settings.llm_backend == "mock":
        llm = get_llm("mock", response=lambda _m: f"(answer grounded in {len(retrieved)} retrieved docs)")
    else:
        llm = get_llm(settings.llm_backend)
    answer = llm.invoke([
        {"role": "system", "content": "Answer using ONLY the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
    ])
    _QUERY_LATENCY.observe(time.perf_counter() - start)
    return {"retrieved": retrieved, "answer": answer}
