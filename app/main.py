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

import hashlib
import importlib.util
import logging
import os
import secrets
import time
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)
from starlette.types import Receive, Scope, Send

from rag_llm_infra import configure_logging, get_llm, get_vector_store

from .config import Settings, get_settings
from .embedder import embed

settings = get_settings()


def _require_backend_packages(s: Settings) -> None:
    """Fail fast at startup when a selected non-default backend isn't installed.

    The base install ships only the NumPy vector store + Mock LLM; openai / faiss
    / qdrant live in optional extras (see pyproject). Without this check a
    misconfiguration (e.g. ``APP_LLM_BACKEND=openai`` on a base install) would
    surface as a 500 on the FIRST /query — get_llm / get_vector_store import the
    SDK lazily inside the request. Checking importability at boot (find_spec, no
    live credential needed) turns that into an immediate, actionable startup
    failure carrying the exact `pip install …[extra]` fix.
    """
    checks = (
        ("APP_LLM_BACKEND", s.llm_backend, "openai", "openai", "openai"),
        ("APP_VECTOR_BACKEND", s.vector_backend, "faiss", "faiss", "faiss"),
        ("APP_VECTOR_BACKEND", s.vector_backend, "qdrant", "qdrant_client", "qdrant"),
    )
    for env_var, selected, backend, module, extra in checks:
        if selected == backend and importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"{env_var}={backend} requires the optional '{extra}' extra, which is "
                f"not installed. Run `pip install 'production-rag-platform[{extra}]'` "
                f"or use the default backend (mock LLM / numpy store)."
            )


# Validate the configured backends are importable before the app serves a single
# request — a misconfigured deploy dies at boot with the fix, not mid-query.
_require_backend_packages(settings)

configure_logging(settings.log_level)

# Emitted through rag-llm-infra's logging config: human-readable in dev,
# single-line JSON when the process runs with ENV=prod (the Helm deploy does).
logger = logging.getLogger("app.main")


def _route_uvicorn_logs_through_json() -> None:
    """Under ENV=prod, make uvicorn's OWN loggers emit the same single-line JSON
    as the app logger, so prod stdout is one uniform format.

    rag-llm-infra installs its JSON formatter on the ROOT logger (keyed on the
    ENV=prod env var — the Helm deploy sets it), but uvicorn installs its own
    plain-text handlers on the `uvicorn` / `uvicorn.access` loggers with
    ``propagate=False``. A record on `uvicorn.error` even bubbles up to
    `uvicorn`'s plain handler and STOPS there (uvicorn.propagate is False), so
    ALL THREE uvicorn loggers emit plain text while the app logger emits JSON —
    prod stdout ends up a MIX of the two formats, which breaks log ingestion and
    contradicts the README's structured-logging claim.

    Clearing uvicorn's handlers and re-enabling propagation routes every
    uvicorn.* record up to the root JSON handler, so prod logs are uniform JSON.
    Gated on the exact condition rag-llm-infra keys its JSON formatter on
    (ENV=prod); in dev the root handler is human-readable and uvicorn's default
    formatting is left untouched.
    """
    if os.getenv("ENV", "dev").lower() != "prod":
        return
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


# Route uvicorn's loggers through the root JSON handler AT IMPORT TIME (under
# ENV=prod). uvicorn's launch order is: configure_logging() (installs uvicorn's
# plain handlers) -> import the app module (this line runs here) -> log "Started
# server process" / "Waiting for application startup." Rerouting at import
# therefore lands BEFORE those two banner lines, so they emit as JSON too —
# doing it only in the lifespan (below) left those first two prod-boot lines
# plain text, breaking strict-JSON ingestion on every restart.
_route_uvicorn_logs_through_json()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Belt-and-suspenders: also reroute at lifespan startup, covering launch
    # paths where the app module is imported before uvicorn installs its logging
    # config (so the import-time call above was a no-op). Idempotent.
    _route_uvicorn_logs_through_json()
    # Startup config summary — the one line an operator needs to confirm WHAT
    # is actually running (backends, auth posture) from the logs alone.
    logger.info(
        "service started",
        extra={
            "app_env": settings.env,
            "llm_backend": settings.llm_backend,
            "vector_backend": settings.vector_backend,
            "default_top_k": settings.default_top_k,
            "auth_enabled": bool(settings.api_key),
        },
    )
    yield
    logger.info("service stopping")


class _BodySizeLimit:
    """Refuse an oversized body from its declared Content-Length, before FastAPI
    buffers it whole to parse: the schema bounds cap every field, but only after
    the full body is already in memory. No parseable length on a body-bearing
    method is 411 — the alternative is buffering an unbounded stream to find out.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            length = dict(scope["headers"]).get(b"content-length")
            if length is None or not length.isdigit():
                await JSONResponse(
                    {"detail": "Content-Length required"}, status_code=411
                )(scope, receive, send)
                return
            if int(length) > self.max_bytes:
                await JSONResponse(
                    {"detail": "request body too large"}, status_code=413
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = FastAPI(title="production-rag-platform", version="1.0.0", lifespan=_lifespan)
app.add_middleware(_BodySizeLimit, max_bytes=settings.max_request_bytes)


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


def _doc_id(text: str) -> str:
    """Identity from the content, so an id survives reordering and growth.

    16 hex chars (64 bits) is a judgement call, not a derivation: accidental
    collision odds are ~n^2/2^65 — negligible at any corpus this service holds.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _grounded_answer(hits: List[Dict[str, Any]]) -> str:
    """Default-path answer that carries its own evidence, not just a count."""
    top = hits[0]
    return (
        f"Grounded in {len(hits)} document(s); "
        f"best evidence [{top['id']}]: {top['text']}"
    )


_REQUESTS = Counter("rag_requests_total", "Total API requests", ["endpoint"])
# Rejected (401) requests never reach a route body, so they are invisible to
# rag_requests_total. Count them separately so bad/missing-credential traffic is
# observable (alert on auth-failure spikes) instead of silently dropped.
_AUTH_FAILURES = Counter(
    "rag_auth_failures_total",
    "Requests rejected for a missing/invalid API key (HTTP 401)",
)
_QUERY_LATENCY = Histogram("rag_query_latency_seconds", "Query latency in seconds")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard the API's data-plane (/index and /query) when APP_API_KEY is configured.

    Unset (the default) leaves both open for the no-auth local/demo run. When
    set, /index AND /query require a matching X-API-Key header: /index REPLACES
    the entire corpus (must not be world-writable in a shared deployment), and
    /query reads that corpus back and spends LLM budget on every call — leaving
    it open while guarding /index would let anyone exfiltrate the indexed
    documents and burn the model allowance. The probes (/health, /ready,
    /metrics) stay open so liveness checks and Prometheus scraping need no key.

    The comparison is constant-time (secrets.compare_digest over the encoded
    bytes): a plain equality check short-circuits at the first differing byte,
    leaking a timing signal about how much of a guessed key prefix matched.
    Encoding to bytes also keeps a non-ASCII header value from raising inside
    compare_digest.
    """
    if not settings.api_key:
        return
    supplied = (x_api_key or "").encode()
    if not secrets.compare_digest(supplied, settings.api_key.encode()):
        # Count the rejection BEFORE raising: the dependency short-circuits the
        # route body, so rag_requests_total never sees this request — without a
        # dedicated counter, 401s would be invisible to metrics.
        _AUTH_FAILURES.inc()
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _nfc(text: str) -> str:
    # NFC and NFD are two byte encodings of one text; dedup and the hash
    # embedder both key on bytes, so unnormalised input splits one document
    # into two and stops a query from matching its own document.
    return unicodedata.normalize("NFC", text)


class IndexRequest(BaseModel):
    # extra="forbid": a typo'd field is a client error, not something to ignore.
    model_config = ConfigDict(extra="forbid")
    # Item min_length + strip: a blank document indexes cleanly and would flip
    # /ready to 200 over nothing. List min_length=1: an empty index is
    # meaningless.
    documents: List[
        Annotated[
            str,
            StringConstraints(
                min_length=1,
                max_length=settings.max_document_chars,
                strip_whitespace=True,
            ),
            AfterValidator(_nfc),
        ]
    ] = Field(min_length=1, max_length=settings.max_documents)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # No strip here: a whitespace-only query is a valid string carrying no
    # signal — the zero-norm guard answers it honestly (grounded: false).
    query: Annotated[
        str,
        StringConstraints(min_length=1, max_length=settings.max_query_chars),
        AfterValidator(_nfc),
    ]
    # ge=1: a non-positive k reaches the store's argpartition and 500s.
    k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)


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
    # Dedup by content: the same document indexed twice would come back as two
    # independent "sources" corroborating each other.
    seen: set[str] = set()
    docs: List[str] = []
    for document in req.documents:
        key = _doc_id(document)
        if key not in seen:
            seen.add(key)
            docs.append(document)
    store = get_vector_store(settings.vector_backend)
    store.add(embed(docs))
    global _index
    _index = _Index(docs=tuple(docs), store=store)
    # Counts only — document CONTENT never goes to the logs.
    logger.info(
        "corpus indexed",
        extra={
            "documents": len(docs),
            "vector_backend": settings.vector_backend,
        },
    )
    return {"indexed": len(docs)}


@app.post("/query")
def query(req: QueryRequest, _: None = Depends(require_api_key)) -> Any:
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
        query_vec = embed([req.query])
        # A zero-norm query embeds to nothing this store can rank: the library
        # renormalises it by /1.0 and argpartition then returns an ARBITRARY
        # slice of the corpus at score 0.0 — so the refusal happens here,
        # before the store is ever consulted.
        if not float((query_vec * query_vec).sum()):
            return {"grounded": False, "retrieved": [], "answer": None}
        # The store protocol pins shape and truncation to min(k, size);
        # descending order is implementation behaviour in all three backends —
        # not a protocol clause — and is pinned HERE by the ordering gate, so
        # neither a clamp nor a re-sort is earned.
        scores, idx = store.search(query_vec, k=req.k)
        hits: List[Dict[str, Any]] = []
        for score, i in zip(scores[0], idx[0]):
            # score <= 0: shares nothing with the query — arbitrary, not evidence.
            if float(score) <= 0.0:
                continue
            text = docs[int(i)]
            doc_id = _doc_id(text)
            hits.append(
                # One window per document until the splitter lands, hence ":0".
                {
                    "id": f"{doc_id}:0",
                    "doc_id": doc_id,
                    "score": float(score),
                    "text": text,
                }
            )
        if not hits:
            return {"grounded": False, "retrieved": [], "answer": None}
        context = "\n".join(f"- {hit['text']}" for hit in hits)
        if settings.llm_backend == "mock":
            llm = get_llm("mock", response=lambda _m: _grounded_answer(hits))
        else:
            llm = get_llm(settings.llm_backend)
        answer = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "Answer using ONLY the provided context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {req.query}",
                },
            ]
        )
        # Counts only — the query text itself (potential PII) never goes to
        # the logs.
        logger.info(
            "query answered",
            extra={
                "retrieved": len(hits),
                "k": req.k,
                "llm_backend": settings.llm_backend,
            },
        )
        return {"grounded": True, "retrieved": hits, "answer": answer}
    finally:
        _QUERY_LATENCY.observe(time.perf_counter() - start)
