"""Integration tests for the reference service (no network, no API key).

Covers the happy path plus the error surface and edge cases the service must
handle: readiness before indexing, query-before-index, input validation
(empty corpus, non-positive k), corpus-replace semantics, the re-index
torn-read regression, and optional API-key auth on the destructive write.
"""
import threading

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_index():
    # The corpus lives in a module-level reference; reset it around every test
    # so cases don't leak state into each other regardless of run order.
    main._index = None
    yield
    main._index = None


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_exposes_prometheus() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text


def test_ready_503_before_index() -> None:
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_ready_true_after_index() -> None:
    client.post("/index", json={"documents": ["faiss vector search", "qdrant database"]})
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_index_then_query_grounds_answer() -> None:
    docs = ["FAISS in-process vector similarity search", "Qdrant vector database with gRPC"]
    r = client.post("/index", json={"documents": docs})
    assert r.status_code == 201
    assert r.json() == {"indexed": 2}
    body = client.post("/query", json={"query": "vector similarity search", "k": 1}).json()
    assert body["retrieved"] == ["FAISS in-process vector similarity search"]
    assert "grounded" in body["answer"]


def test_query_before_index_returns_409() -> None:
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 409
    assert r.json()["error"] == "index documents first"


def test_query_409_path_is_observed_in_latency_histogram() -> None:
    """Regression: latency was observed only on the success tail, so the 409
    "not indexed" path (and errors) never entered the histogram, understating
    real latency. The observe() now runs in a finally, so a 409 must bump the
    histogram's count."""
    from prometheus_client import REGISTRY

    def _count() -> float:
        return REGISTRY.get_sample_value("rag_query_latency_seconds_count") or 0.0

    before = _count()
    r = client.post("/query", json={"query": "anything"})  # 409 — no index
    assert r.status_code == 409
    assert _count() == before + 1.0, "409 path must be recorded in the latency histogram"


def test_index_rejects_empty_documents_422() -> None:
    r = client.post("/index", json={"documents": []})
    assert r.status_code == 422
    # An empty corpus must not slip through and flip readiness to 200.
    assert client.get("/ready").status_code == 503


def test_query_rejects_nonpositive_k_422() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    for bad_k in (0, -2):
        r = client.post("/query", json={"query": "vectors", "k": bad_k})
        assert r.status_code == 422, f"k={bad_k} should be rejected, got {r.status_code}"


def test_index_replaces_corpus_not_additive() -> None:
    client.post("/index", json={"documents": ["first corpus alpha"]})
    client.post("/index", json={"documents": ["second corpus beta"]})
    body = client.post("/query", json={"query": "corpus", "k": 5}).json()
    assert body["retrieved"] == ["second corpus beta"]  # the old corpus is gone


def test_reindex_with_smaller_corpus_never_500s() -> None:
    # Replace a large corpus with a smaller one and immediately query. The
    # atomic snapshot guarantees docs and store always match, so retrieved
    # indices never exceed the current doc list.
    client.post("/index", json={"documents": [f"doc number {i} about vectors" for i in range(20)]})
    client.post("/index", json={"documents": ["only one doc about vectors"]})
    r = client.post("/query", json={"query": "vectors", "k": 5})
    assert r.status_code == 200
    assert r.json()["retrieved"] == ["only one doc about vectors"]


def test_concurrent_reindex_and_query_never_5xx() -> None:
    # Concurrency regression for the torn read: with the old two-key state a
    # query interleaved with a re-index could pair a new store with stale docs
    # and 500. The single atomic snapshot makes that impossible — no request
    # should ever see a 5xx, no matter the interleaving.
    client.post("/index", json={"documents": [f"d{i} vectors" for i in range(10)]})
    errors: list[int] = []

    def reindexer() -> None:
        for i in range(40):
            n = 1 if i % 2 else 10
            client.post("/index", json={"documents": [f"d{j} vectors" for j in range(n)]})

    def querier() -> None:
        for _ in range(40):
            r = client.post("/query", json={"query": "vectors", "k": 5})
            if r.status_code >= 500:
                errors.append(r.status_code)

    threads = [threading.Thread(target=reindexer) for _ in range(2)]
    threads += [threading.Thread(target=querier) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"saw 5xx responses under concurrency: {errors}"


def test_deployment_app_env_values_are_valid() -> None:
    """Guard config/deployment drift: every APP_ENV the deploy files set must be
    a valid Settings.env value. A 'dev' typo in docker-compose.yml used to crash
    the service at import once env became a Literal."""
    import pathlib
    import re

    from app.config import Settings

    root = pathlib.Path(__file__).resolve().parent.parent
    files = [root / "deploy" / "docker-compose.yml", root / "deploy" / "helm" / "values.yaml"]
    values = [m for f in files for m in re.findall(r"APP_ENV[:=]\s*([A-Za-z_]+)", f.read_text())]
    assert values, "expected APP_ENV declarations in the deploy files"
    for val in values:
        Settings(env=val)  # raises pydantic ValidationError if not a valid Literal


def test_index_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    assert client.post("/index", json={"documents": ["x doc"]}).status_code == 401
    assert (
        client.post("/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    # A non-ASCII guess must be a clean 401, not a 500 — the constant-time
    # comparison encodes to bytes precisely so compare_digest can't raise on it.
    # (Sent as raw bytes: httpx itself only allows ASCII in str header values.)
    assert (
        client.post(
            "/index",
            json={"documents": ["x doc"]},
            headers={"X-API-Key": "wröng".encode()},
        ).status_code
        == 401
    )
    assert (
        client.post("/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "s3cret"}).status_code
        == 201
    )


def test_api_key_comparison_is_constant_time() -> None:
    """Guard: the API-key check must go through secrets.compare_digest. A plain
    `!=` short-circuits at the first differing byte, leaking a timing signal
    about how much of a guessed key prefix matched."""
    import inspect

    src = inspect.getsource(main.require_api_key)
    assert "compare_digest" in src, "API-key check must use secrets.compare_digest"
    assert "!=" not in src, "no short-circuiting comparison in the API-key check"


def test_startup_emits_structured_config_summary(caplog) -> None:
    """Regression: the service used to ship ZERO log statements, so the
    'structured logs' claim was inert. Startup must emit a config summary
    through the app logger (JSON-formatted when the process runs with
    ENV=prod, as the Helm deploy does)."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(app):
            pass
    started = [
        r for r in caplog.records if r.name == "app.main" and r.getMessage() == "service started"
    ]
    assert started, "expected a 'service started' INFO record on startup"
    record = started[0]
    # The summary must carry the config an operator needs, as structured fields.
    assert getattr(record, "llm_backend") == main.settings.llm_backend
    assert getattr(record, "vector_backend") == main.settings.vector_backend
    assert getattr(record, "index_auth_enabled") == bool(main.settings.api_key)


def test_index_and_query_emit_count_logs(caplog) -> None:
    """Both write and read paths must leave an INFO trail (counts only — never
    document/query content)."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.main"):
        client.post("/index", json={"documents": ["a doc about vectors"]})
        client.post("/query", json={"query": "vectors", "k": 1})
    by_msg = {r.getMessage(): r for r in caplog.records if r.name == "app.main"}
    assert "corpus indexed" in by_msg and getattr(by_msg["corpus indexed"], "documents") == 1
    assert "query answered" in by_msg and getattr(by_msg["query answered"], "retrieved") == 1


def test_readme_hook_claims_only_tech_that_runs_here() -> None:
    """Guard doc drift: the hook (everything above the first ---) must not
    claim tech that the README's own boundary table declares private-only
    (Redis, OpenTelemetry, arq, slowapi) or that appears nowhere in this
    codebase at all (LangChain)."""
    import pathlib

    readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8"
    )
    hook = readme.split("\n---", 1)[0]
    for private_only in ("LangChain", "Redis", "OpenTelemetry", "arq", "slowapi"):
        assert private_only not in hook, (
            f"README hook claims '{private_only}', which does not run in this repo"
        )


def test_default_helm_image_tag_is_published_by_ci() -> None:
    """Regression: the chart's Deployment defaults the image tag to
    .Chart.AppVersion, but CI used to push only :latest + :<sha> — so a bare
    `helm install` referenced a tag that did not exist on GHCR and the pod
    ImagePullBackOff'd. CI must derive a pushed tag from the same Chart.yaml
    the template falls back to."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    deployment = (root / "deploy" / "helm" / "templates" / "deployment.yaml").read_text()
    assert ".Chart.AppVersion" in deployment  # the fallback the chart resolves
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert "deploy/helm/Chart.yaml" in ci, "CI must read the appVersion from the chart itself"
    assert "steps.chart.outputs.app_version" in ci, "CI must push the appVersion tag"


def test_helm_deploy_activates_json_logging() -> None:
    """Regression: rag-llm-infra keys its JSON log formatter on ENV=prod, but
    the chart used to set only APP_ENV=production (and the template hardcoded
    that single key) — so 'structured JSON logs' never activated in the shipped
    deploy. The values must carry both knobs and the template must render every
    key under .Values.env."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    values = (root / "deploy" / "helm" / "values.yaml").read_text()
    env_pairs = dict(re.findall(r"^\s{2}(APP_ENV|ENV):\s*(\S+)", values, flags=re.M))
    assert env_pairs.get("APP_ENV") == "production"  # app settings knob
    assert env_pairs.get("ENV") == "prod"  # rag-llm-infra JSON-log knob
    deployment = (root / "deploy" / "helm" / "templates" / "deployment.yaml").read_text()
    assert "range $name, $value := .Values.env" in deployment, (
        "Deployment must render ALL of .Values.env, not a hardcoded key"
    )
