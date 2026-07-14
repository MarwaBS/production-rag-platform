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
    client.post(
        "/index", json={"documents": ["faiss vector search", "qdrant database"]}
    )
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_index_then_query_grounds_answer() -> None:
    docs = [
        "FAISS in-process vector similarity search",
        "Qdrant vector database with gRPC",
    ]
    r = client.post("/index", json={"documents": docs})
    assert r.status_code == 201
    assert r.json() == {"indexed": 2}
    body = client.post(
        "/query", json={"query": "vector similarity search", "k": 1}
    ).json()
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
    assert _count() == before + 1.0, (
        "409 path must be recorded in the latency histogram"
    )


def test_index_rejects_empty_documents_422() -> None:
    r = client.post("/index", json={"documents": []})
    assert r.status_code == 422
    # An empty corpus must not slip through and flip readiness to 200.
    assert client.get("/ready").status_code == 503


def test_query_rejects_nonpositive_k_422() -> None:
    client.post("/index", json={"documents": ["a doc about vectors"]})
    for bad_k in (0, -2):
        r = client.post("/query", json={"query": "vectors", "k": bad_k})
        assert r.status_code == 422, (
            f"k={bad_k} should be rejected, got {r.status_code}"
        )


def test_index_replaces_corpus_not_additive() -> None:
    client.post("/index", json={"documents": ["first corpus alpha"]})
    client.post("/index", json={"documents": ["second corpus beta"]})
    body = client.post("/query", json={"query": "corpus", "k": 5}).json()
    assert body["retrieved"] == ["second corpus beta"]  # the old corpus is gone


def test_reindex_with_smaller_corpus_never_500s() -> None:
    # Replace a large corpus with a smaller one and immediately query. The
    # atomic snapshot guarantees docs and store always match, so retrieved
    # indices never exceed the current doc list.
    client.post(
        "/index",
        json={"documents": [f"doc number {i} about vectors" for i in range(20)]},
    )
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
            client.post(
                "/index", json={"documents": [f"d{j} vectors" for j in range(n)]}
            )

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
    files = [
        root / "deploy" / "docker-compose.yml",
        root / "deploy" / "helm" / "values.yaml",
    ]
    values = [
        m
        for f in files
        for m in re.findall(r"APP_ENV[:=]\s*([A-Za-z_]+)", f.read_text())
    ]
    assert values, "expected APP_ENV declarations in the deploy files"
    for val in values:
        Settings(env=val)  # raises pydantic ValidationError if not a valid Literal


def test_index_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    assert client.post("/index", json={"documents": ["x doc"]}).status_code == 401
    assert (
        client.post(
            "/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "wrong"}
        ).status_code
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
        client.post(
            "/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 201
    )


def test_query_requires_api_key_when_configured(monkeypatch) -> None:
    """When APP_API_KEY is set, /query (a read that touches the corpus and
    spends LLM budget) must require the key too — not only the destructive
    /index write. A shared deployment that guards /index but leaves /query open
    lets anyone read the indexed corpus and burn the LLM allowance."""
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    # Seed a corpus (with the key) so a served /query would be a 200 — proving
    # the 401 below is auth, not the empty-index 409.
    assert (
        client.post(
            "/index",
            json={"documents": ["a doc about vectors"]},
            headers={"X-API-Key": "s3cret"},
        ).status_code
        == 201
    )
    assert client.post("/query", json={"query": "vectors", "k": 1}).status_code == 401
    assert (
        client.post(
            "/query", json={"query": "vectors", "k": 1}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/query", json={"query": "vectors", "k": 1}, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )


def test_query_open_when_no_api_key_configured() -> None:
    """The no-auth local/demo default is preserved: with no APP_API_KEY set,
    /query needs no header."""
    client.post("/index", json={"documents": ["a doc about vectors"]})
    assert client.post("/query", json={"query": "vectors", "k": 1}).status_code == 200


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
        r
        for r in caplog.records
        if r.name == "app.main" and r.getMessage() == "service started"
    ]
    assert started, "expected a 'service started' INFO record on startup"
    record = started[0]
    # The summary must carry the config an operator needs, as structured fields.
    assert getattr(record, "llm_backend") == main.settings.llm_backend
    assert getattr(record, "vector_backend") == main.settings.vector_backend
    assert getattr(record, "auth_enabled") == bool(main.settings.api_key)


def test_index_and_query_emit_count_logs(caplog) -> None:
    """Both write and read paths must leave an INFO trail (counts only — never
    document/query content)."""
    import logging

    with caplog.at_level(logging.INFO, logger="app.main"):
        client.post("/index", json={"documents": ["a doc about vectors"]})
        client.post("/query", json={"query": "vectors", "k": 1})
    by_msg = {r.getMessage(): r for r in caplog.records if r.name == "app.main"}
    assert (
        "corpus indexed" in by_msg
        and getattr(by_msg["corpus indexed"], "documents") == 1
    )
    assert (
        "query answered" in by_msg
        and getattr(by_msg["query answered"], "retrieved") == 1
    )


def test_nondefault_backend_without_extra_fails_fast() -> None:
    """A selected non-default backend whose package isn't installed must fail at
    STARTUP with the exact install hint — not defer to a 500 on the first /query.
    openai/faiss/qdrant are optional extras; the base/dev env installs none of
    them, so selecting one here must raise a clear RuntimeError."""
    from app.config import Settings

    with pytest.raises(RuntimeError, match=r"production-rag-platform\[openai\]"):
        main._require_backend_packages(Settings(llm_backend="openai"))
    with pytest.raises(RuntimeError, match=r"production-rag-platform\[qdrant\]"):
        main._require_backend_packages(Settings(vector_backend="qdrant"))
    # The default stack (mock LLM + numpy store) is always available — no raise.
    main._require_backend_packages(Settings())


def test_pyproject_declares_every_nondefault_backend_extra() -> None:
    """Every selectable non-default backend must be pip-installable via an extra,
    so the boot guard's `pip install …[extra]` hint actually resolves. Guards
    against config.py offering a backend with no way to install its package."""
    import pathlib
    import re

    pyproject = (
        pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text()
    extras_block = re.search(
        r"\[project\.optional-dependencies\]\n((?:.*\n)+?)\n?\[", pyproject
    )
    assert extras_block, "expected an optional-dependencies table"
    for extra in ("openai", "faiss", "qdrant"):
        assert re.search(rf"(?m)^{extra}\s*=", extras_block.group(1)), (
            f"backend '{extra}' is selectable in config.py but has no install extra"
        )


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
    deployment = (
        root / "deploy" / "helm" / "templates" / "deployment.yaml"
    ).read_text()
    assert ".Chart.AppVersion" in deployment  # the fallback the chart resolves
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert "deploy/helm/Chart.yaml" in ci, (
        "CI must read the appVersion from the chart itself"
    )
    assert "steps.chart.outputs.app_version" in ci, "CI must push the appVersion tag"


def test_default_helm_ingress_is_disabled() -> None:
    """Secure default: a bare `helm install` must NOT publish the service to the
    internet. The data-plane is only authenticated when APP_API_KEY is set, so a
    default-on Ingress would expose an unauthenticated /index + /query. The
    ingress template is guarded on this flag, so default-false renders no
    Ingress; enabling it is a deliberate opt-in documented alongside the auth +
    TLS prerequisites."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    values = (root / "deploy" / "helm" / "values.yaml").read_text()
    ingress_block = re.search(r"^ingress:\n((?:\s+.*\n)+)", values, flags=re.M)
    assert ingress_block, "expected an ingress: block in values.yaml"
    enabled = re.search(r"^\s+enabled:\s*(\S+)", ingress_block.group(1), flags=re.M)
    assert enabled and enabled.group(1) == "false", (
        "ingress must default to disabled — a default-on Ingress publishes the "
        "unauthenticated data-plane to the internet"
    )
    template = (root / "deploy" / "helm" / "templates" / "ingress.yaml").read_text()
    assert "if .Values.ingress.enabled" in template, (
        "the Ingress must be guarded by ingress.enabled so default-false renders nothing"
    )


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
    deployment = (
        root / "deploy" / "helm" / "templates" / "deployment.yaml"
    ).read_text()
    assert "range $name, $value := .Values.env" in deployment, (
        "Deployment must render ALL of .Values.env, not a hardcoded key"
    )


def test_uvicorn_loggers_emit_json_under_prod(monkeypatch) -> None:
    """Regression: under ENV=prod the app logger emits single-line JSON, but
    uvicorn keeps its OWN plain-text handlers on the uvicorn / uvicorn.access
    loggers with propagate=False (and uvicorn.error's records bubble to
    uvicorn's plain handler and stop there) — so ALL uvicorn.* lines stay plain
    text while app lines are JSON. Prod stdout was therefore a MIX of formats,
    which breaks log ingestion and contradicts the README's structured-logging
    claim. Every uvicorn.* logger must emit through the root JSON handler."""
    import io
    import json
    import logging
    import logging.config

    import uvicorn.config as uv_config
    from rag_llm_infra.log_config import _JsonFormatter

    monkeypatch.setenv("ENV", "prod")

    root = logging.getLogger()
    saved_root_handlers = root.handlers[:]
    buf = io.StringIO()
    capture = logging.StreamHandler(buf)
    capture.setFormatter(_JsonFormatter())  # stand in for the prod JSON root handler
    root.handlers = [capture]

    uv_names = ("uvicorn", "uvicorn.access", "uvicorn.error")
    saved_uv = {
        n: (logging.getLogger(n).handlers[:], logging.getLogger(n).propagate)
        for n in uv_names
    }
    try:
        # Install uvicorn's REAL default logging (plain handlers, propagate=False).
        logging.config.dictConfig(uv_config.LOGGING_CONFIG)

        # Precondition (the bug): a uvicorn line never reaches the JSON root handler.
        logging.getLogger("uvicorn").info("startup line")
        assert buf.getvalue() == "", (
            "uvicorn logs must currently bypass the JSON root handler"
        )

        main._route_uvicorn_logs_through_json()

        # After the fix every uvicorn.* logger emits a single JSON line via root.
        for name in uv_names:
            buf.seek(0)
            buf.truncate(0)
            logging.getLogger(name).info("line from %s", name)
            out = buf.getvalue().strip()
            assert out, f"{name} produced no output through the root JSON handler"
            record = json.loads(out)  # must be a single valid JSON object
            assert record["logger"] == name
            assert record["msg"] == f"line from {name}"
    finally:
        root.handlers = saved_root_handlers
        for n, (handlers, propagate) in saved_uv.items():
            lg = logging.getLogger(n)
            lg.handlers = handlers
            lg.propagate = propagate


def test_uvicorn_reroute_happens_at_import_not_only_lifespan() -> None:
    """F1 regression: the uvicorn->JSON reroute must run at MODULE IMPORT under
    ENV=prod, not only in the lifespan. uvicorn's order is configure_logging()
    -> import the app -> log 'Started server process' / 'Waiting for application
    startup.', so a lifespan-only reroute leaves those first two prod-boot lines
    plain text.

    Verified in a FRESH subprocess (reloading app.main in-process re-registers
    its module-level Prometheus collectors and raises Duplicated timeseries). The
    child installs uvicorn's real logging config exactly as uvicorn does before
    importing an app, then imports app.main under ENV=prod — the import ALONE
    must clear the `uvicorn` logger's plain handler and set it to propagate to
    the root JSON handler. Remove the import-time reroute (leave only the
    lifespan one) and this fails: importing app.main no longer reroutes."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    child = (
        "import logging, logging.config, json\n"
        "import uvicorn.config as uc\n"
        "logging.config.dictConfig(uc.LOGGING_CONFIG)  # uvicorn's plain handlers, as at boot\n"
        "uv = logging.getLogger('uvicorn')\n"
        "assert uv.handlers and uv.propagate is False  # precondition: owns plain handler, no root propagation\n"
        "import app.main  # noqa: F401 — the import-time reroute must fire here\n"
        "print(json.dumps({'propagate': uv.propagate, 'handlers': len(uv.handlers)}))\n"
    )
    env = {**os.environ, "ENV": "prod", "APP_ENV": "production"}
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["propagate"] is True, (
        "importing app.main under ENV=prod must reroute the uvicorn logger to root"
    )
    assert result["handlers"] == 0, "uvicorn's plain handler must be cleared at import"


def test_auth_failure_increments_auth_counter_not_request_counter(monkeypatch) -> None:
    """Regression: a rejected (401) request was invisible to metrics — only
    authenticated/served requests bumped a counter. A bad key must bump the
    dedicated rag_auth_failures_total counter and must NOT be counted as a
    served request in rag_requests_total."""
    from prometheus_client import REGISTRY

    monkeypatch.setattr(main.settings, "api_key", "s3cret")

    def _auth_failures() -> float:
        return REGISTRY.get_sample_value("rag_auth_failures_total") or 0.0

    def _index_served() -> float:
        return (
            REGISTRY.get_sample_value("rag_requests_total", {"endpoint": "index"})
            or 0.0
        )

    auth_before = _auth_failures()
    served_before = _index_served()
    r = client.post(
        "/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "wrong"}
    )
    assert r.status_code == 401
    assert _auth_failures() == auth_before + 1.0, (
        "a 401 must bump rag_auth_failures_total"
    )
    assert _index_served() == served_before, (
        "a rejected request must not count as a served /index in rag_requests_total"
    )
