"""Integration tests for the reference service (no network, no API key).

Covers the happy path plus the error surface and edge cases the service must
handle: readiness before indexing, query-before-index, input validation
(empty corpus, non-positive k), corpus-replace semantics, the re-index
torn-read regression, and optional API-key auth on the destructive write.
"""

import threading
from typing import Callable

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
    assert r.json() == {"indexed": 2, "chunks": 2}
    body = client.post(
        "/query", json={"query": "vector similarity search", "k": 1}
    ).json()
    assert body["grounded"] is True
    assert [hit["text"] for hit in body["retrieved"]] == [
        "FAISS in-process vector similarity search"
    ]
    # The answer must carry its evidence, not merely assert that it has some.
    assert "FAISS" in body["answer"]


def test_query_before_index_returns_409() -> None:
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 409
    assert r.json()["error"] == "index documents first"


def test_query_409_path_is_observed_in_latency_histogram() -> None:
    """Observed in a finally, so the handler's 409 and error exits are timed
    and not only its success tail, which would understate latency and hide a
    slow failure. Rejections before the body (401, 413, 422) never reach it."""
    from prometheus_client import REGISTRY

    def _count() -> float:
        return REGISTRY.get_sample_value("rag_query_latency_seconds_count") or 0.0

    before = _count()
    r = client.post("/query", json={"query": "anything"})  # 409; no index
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
    # The old corpus is gone entirely.
    assert [hit["text"] for hit in body["retrieved"]] == ["second corpus beta"]


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
    assert [hit["text"] for hit in r.json()["retrieved"]] == [
        "only one doc about vectors"
    ]


def _corpus(n: int) -> list[str]:
    """Documents a query can rank apart, with the best match at the HIGHEST index.

    Ranking matters to what this test can detect. If every document scores the
    same, the store returns ties in ascending index order, so the top hit is
    always index 0; which is in range for any corpus, and a query that read a
    larger store than its document list would still find something to return.
    Weighting the match toward the end makes the returned index large, so pairing
    a big store with a small document list resolves out of range instead.
    """
    return [f"document {i} " + ("vectors " * (i + 1)) for i in range(n)]


def test_concurrent_reindex_and_query_never_5xx() -> None:
    # Exercises the two paths against each other under real threads. It cannot
    # guarantee it lands in the one-bytecode window a torn publish would open;
    # the structural check below is what proves that window does not exist; so
    # what this rules out is the broader class: a handler that errors when the
    # corpus changes size beneath it.
    # raise_server_exceptions=False so a handler error becomes a 500 this thread
    # can record, instead of an exception the default client re-raises inside it.
    observer = TestClient(app, raise_server_exceptions=False)
    observer.post("/index", json={"documents": _corpus(10)})
    errors: list[int] = []

    def reindexer() -> None:
        for i in range(40):
            observer.post("/index", json={"documents": _corpus(2 if i % 2 else 10)})

    def querier() -> None:
        for _ in range(40):
            r = observer.post("/query", json={"query": "vectors", "k": 5})
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
    """Every APP_ENV the deploy files set must be a valid Settings.env value:
    env is a Literal, so a typo in one of them is an import-time crash in the
    deployed pod rather than a warning anywhere."""
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
        # A probe key, because production now refuses to boot without one; the
        # assertion here is only that the env LITERAL is valid.
        Settings(env=val, api_key="env-literal-probe")


def test_index_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    assert client.post("/index", json={"documents": ["x doc"]}).status_code == 401
    assert (
        client.post(
            "/index", json={"documents": ["x doc"]}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    # A non-ASCII guess must be a clean 401, not a 500; the constant-time
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
    spends LLM budget) must require the key too; not only the destructive
    /index write. A shared deployment that guards /index but leaves /query open
    lets anyone read the indexed corpus and burn the LLM allowance."""
    monkeypatch.setattr(main.settings, "api_key", "s3cret")
    # Seed a corpus (with the key) so a served /query would be a 200; proving
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
    """The key comparison in app/main.py must go through secrets.compare_digest.

    SCOPE, stated because it bounds what this can promise: the walk starts at
    the guard and follows BARE-NAME calls (`helper(...)`) whose name it can
    look up DIRECTLY IN THE MODULE NAMESPACE and find a function defined in
    app/main.py. That is all it follows. A comparison reached any other way is
    outside it: through an attribute (`obj.helper(...)`, a bound method, a
    module alias), through a locally rebound name (`check = helper` then
    `check(...)`; the lookup sees no `check` on the module), or in another
    module. The limit is deliberate; resolving arbitrary call graphs is not a
    test's job; and it is stated so the gate never promises coverage it lacks.

    The body is parsed with the docstring removed, because reading raw source
    would let the prose describing the property satisfy the check for it.
    """
    import ast
    import inspect
    import textwrap

    def statements(func: Callable[..., object]) -> list[ast.stmt]:
        parsed = ast.parse(textwrap.dedent(inspect.getsource(func)))
        node = parsed.body[0]
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        return [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]

    # The guard plus what its bare-name calls reach at module level here. Scope
    # is in the docstring: attribute calls and other modules are out of range.
    bodies: list[ast.stmt] = []
    pending: list[Callable[..., object]] = [main.require_api_key]
    seen: set[Callable[..., object]] = set()
    while pending:
        func = pending.pop()
        if func in seen:
            continue
        seen.add(func)
        reached = statements(func)
        bodies += reached
        for statement in reached:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", "")
                found = getattr(main, name, None) if name else None
                if (
                    callable(found)
                    and getattr(found, "__module__", "") == main.__name__
                ):
                    pending.append(found)

    tree = ast.parse("\n".join(ast.unparse(statement) for statement in bodies))

    # A branch on a literal is unreachable, so a compare_digest call parked inside
    # one would satisfy a presence check while never running.
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant)
    ], "unreachable branch in the key check"

    # The digest comparison must be somewhere on the reachable path. Demanding it
    # sit in a particular `if` would reject a correct helper that returns its
    # result; the very refactor this walk exists to allow.
    assert [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None))
        == "compare_digest"
    ], "the key must be compared with secrets.compare_digest"

    # Membership and prefix tests short-circuit exactly like equality does.
    equality: list[ast.AST] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops
        )
    ]
    equality += [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) in {"startswith", "endswith"}
    ]
    assert not equality, (
        f"short-circuiting comparison in the key check: {[ast.unparse(n) for n in equality]}"
    )


def test_startup_emits_structured_config_summary(caplog) -> None:
    """Startup emits a config summary through the app logger, which is what
    makes the structured-logging claim answerable: without a single log
    statement the claim is about a formatter nothing reaches."""
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
    """Both write and read paths must leave an INFO trail (counts only; never
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


def _permitted(field: str) -> tuple[str, ...]:
    """The values a backend setting accepts, read from its own annotation.

    A hand-written list here would not cover the next value added to the
    Literal.
    """
    from typing import get_args

    from app.config import Settings

    return get_args(Settings.model_fields[field].annotation)


def test_the_backend_settings_expose_the_values_this_file_checks() -> None:
    """Pins the advertised set, so adding a backend is a deliberate act and the
    parametrised test below cannot silently run zero cases."""
    assert _permitted("vector_backend") == ("numpy", "faiss", "qdrant")
    assert _permitted("llm_backend") == ("mock", "openai")
    assert _permitted("embedding_backend") == ("hash", "semantic")


@pytest.mark.parametrize(
    "field,value",
    [
        # Embedding backends have no constructor; test_embedder.py holds them.
        (field, value)
        for field in ("llm_backend", "vector_backend")
        for value in _permitted(field)
    ],
)
def test_every_permitted_backend_starts_or_names_its_missing_extra(
    field: str, value: str
) -> None:
    """Each Literal value must construct the backend it names, or boot must
    refuse naming that backend's own extra. Reaching neither leaves a value
    selectable and never exercised."""
    from app.config import Settings

    settings = Settings.model_validate({field: value})
    try:
        main._require_backend_packages(settings)
    except RuntimeError as missing:
        assert f"production-rag-platform[{value}]" in str(missing), str(missing)
        return
    # Built the way main builds it, so qdrant is handed the collection name the
    # library requires instead of a call shape only this test makes.
    built = (
        main._vector_store(settings)
        if field == "vector_backend"
        else main.get_llm(value)
    )
    # A factory ignoring its argument satisfies a bare `assert .backend_name`.
    assert built.backend_name == value


def test_pyproject_declares_every_nondefault_backend_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selectable non-default backend must be pip-installable via an extra,
    so the boot guard's `pip install …[extra]` hint actually resolves. Read off
    config.py: a hand-written list here cannot cover the next value added."""
    import pathlib
    import re

    from app.config import Settings

    pyproject = (
        pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text()
    extras_block = re.search(
        r"\[project\.optional-dependencies\]\n((?:.*\n)+?)\n?\[", pyproject
    )
    assert extras_block, "expected an optional-dependencies table"
    # Read the hint from the guard, not from what this environment installed.
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda name: None)
    checked = 0
    for field in ("llm_backend", "vector_backend", "embedding_backend"):
        for value in _permitted(field):
            if value == Settings.model_fields[field].default:
                continue
            checked += 1
            with pytest.raises(RuntimeError) as refused:
                main._require_backend_packages(Settings.model_validate({field: value}))
            extra = (
                str(refused.value).split("production-rag-platform[")[1].split("]")[0]
            )
            assert re.search(rf"(?m)^{re.escape(extra)}\s*=", extras_block.group(1)), (
                f"backend '{extra}' is selectable in config.py but has no install extra"
            )
    assert checked, "no non-default backend derived"


# Names the same capability goes by elsewhere in the README, so an abbreviation
# cannot smuggle a private-only capability into the hook.
_ALIASES = {"otel": ("opentelemetry",), "rate": ("slowapi",)}
# Words that describe rather than name, so banning them would reject ordinary prose.
_GENERIC = {"output", "limiting", "validation"}


def test_readme_hook_claims_only_tech_that_runs_here() -> None:
    """The hook must not claim a capability the README itself calls private-only.

    The banned set is read out of the boundary table, so a capability added there
    is covered without anyone remembering to extend a list here.

    LIMITATION: this is name-based. It flags the words the table uses. Redis,
    arq, OTel; and a paraphrase that never says them ("hot answers served from
    an in-memory store") passes. It catches drift, not deliberate rewording.
    """
    import pathlib
    import re

    readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8"
    )
    # The table is quoted with "> ", so the row does not start at the line edge.
    row = re.search(
        r"^>?\s*\|\s*\*\*Includes\*\*\s*\|.*\|(.*)\|\s*$", readme, flags=re.M
    )
    assert row, "expected an 'Includes' row in the public/private boundary table"
    private_only = [cell.strip() for cell in row.group(1).split("·") if cell.strip()]
    assert len(private_only) >= 4, private_only

    hook = readme.split("\n---", 1)[0].lower()
    # Claimed nowhere in this codebase at all, so it belongs in no section.
    banned = {"langchain"}
    for capability in private_only:
        # Every naming word of the cell, not just the first: "OTel tracing" and
        # "tracing via OTel" must ban the same thing.
        for word in re.findall(r"[A-Za-z]+", capability.lower()):
            if word in _GENERIC or len(word) < 3:
                continue
            banned.add(word)
            banned.update(_ALIASES.get(word, ()))
    for term in sorted(banned):
        assert not re.search(rf"\b{re.escape(term)}\b", hook), (
            f"README hook claims '{term}', which does not run in this repo"
        )


def _without_comments(text: str) -> str:
    """Drop `#` comment lines so a claim in prose cannot satisfy a check on code."""
    import re

    return "\n".join(re.sub(r"#.*$", "", line) for line in text.splitlines())


def test_default_helm_image_tag_is_published_by_ci() -> None:
    """The tag a bare `helm install` resolves must be a tag CI actually pushes.

    The Deployment falls back to the chart's appVersion when image.tag is empty,
    so if CI publishes only :latest and :<sha> that default install references a
    tag GHCR does not have and the pod cannot pull. Both halves are read with
    comments stripped, and the CI half is matched inside the loop that pushes;
    a mention anywhere else in the file is not a push.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    deployment = _without_comments(
        (root / "deploy" / "helm" / "templates" / "deployment.yaml").read_text()
    )
    image = re.search(r"^\s*image:\s*(.+)$", deployment, flags=re.M)
    assert image and ".Chart.AppVersion" in image.group(1), (
        f"the image tag must default to the chart appVersion, got {image and image.group(1)}"
    )

    ci = _without_comments((root / ".github" / "workflows" / "ci.yml").read_text())
    assert "deploy/helm/Chart.yaml" in ci, (
        "CI must read the appVersion from the chart itself"
    )
    # The tag set the push loop iterates, not a mention anywhere in the file.
    pushed = re.search(r"^\s*for tag in (.+); do$", ci, flags=re.M)
    assert pushed, "expected a loop naming the tags that get pushed"
    tags = pushed.group(1)
    for required in ("latest", "GITHUB_SHA", "steps.chart.outputs.app_version"):
        assert required in tags, (required, tags)
    assert re.search(r'^\s*docker push "\$IMAGE:\$tag"$', ci, flags=re.M), (
        "the loop must push each tag it names"
    )


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
        "ingress must default to disabled; a default-on Ingress publishes the "
        "unauthenticated data-plane to the internet"
    )
    template = (root / "deploy" / "helm" / "templates" / "ingress.yaml").read_text()
    assert "if .Values.ingress.enabled" in template, (
        "the Ingress must be guarded by ingress.enabled so default-false renders nothing"
    )


def test_helm_deploy_activates_json_logging() -> None:
    """rag-llm-infra keys its JSON formatter on ENV=prod, which is a different
    knob from the app's own APP_ENV. The values must carry both, and the
    template must render every key under .Values.env; setting only APP_ENV
    leaves the shipped deploy emitting human-readable logs."""
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
    """Under ENV=prod every uvicorn.* logger must reach the root JSON handler.
    uvicorn holds plain-text handlers on uvicorn and uvicorn.access with
    propagate=False, and uvicorn.error propagates into the first of those and
    stops, so without the reroute prod stdout mixes both formats."""
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
    """Under ENV=prod the reroute must run at import, not only in the lifespan.

    uvicorn configures logging, imports the app, then logs its two boot banners;
    so a lifespan-only reroute leaves those first lines in plain text. Runs in a
    subprocess because re-importing app.main re-registers its Prometheus
    collectors.
    """
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
        "import app.main  # noqa: F401; the import-time reroute must fire here\n"
        "print(json.dumps({'propagate': uv.propagate, 'handlers': len(uv.handlers)}))\n"
    )
    # The probe key keeps the production boot refusal out of this test's way;
    # what is under test here is only the logging reroute.
    env = {**os.environ, "ENV": "prod", "APP_ENV": "production", "APP_API_KEY": "k"}
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
    """A 401 never reaches a route body, so it bumps its own counter rather
    than the served-request one: counted as served it would overstate traffic,
    counted nowhere it would hide a credential-stuffing spike."""
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


def test_every_served_endpoint_increments_its_request_counter() -> None:
    """The counter must count. Asserting the metric NAME appears in /metrics is
    satisfied by the HELP and TYPE lines, which are emitted with zero samples."""
    from prometheus_client import REGISTRY

    def _served(endpoint: str) -> float:
        return (
            REGISTRY.get_sample_value("rag_requests_total", {"endpoint": endpoint})
            or 0.0
        )

    before = {name: _served(name) for name in ("health", "ready", "index", "query")}
    client.get("/health")
    client.get("/ready")
    client.post("/index", json={"documents": ["a doc about vectors"]})
    client.post("/query", json={"query": "vectors", "k": 1})
    for name, was in before.items():
        assert _served(name) == was + 1.0, (
            f"rag_requests_total{{endpoint={name}}} did not move"
        )


def test_the_index_is_published_and_read_in_one_step() -> None:
    """The snapshot invariant, checked structurally rather than by racing.

    A thread test can only catch an interleave it happens to hit, and the window
    between two writes is a single bytecode wide; the concurrency test above
    exercises the paths but cannot be relied on to land inside it. What makes the
    torn read impossible is the shape of the code: one name, published in one
    assignment, read once per request. That is checkable exactly.
    """
    import ast
    import inspect
    import textwrap

    def assignments(func: Callable[..., object], name: str) -> int:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            for target in (
                [node.targets] if isinstance(node, ast.Assign) else [[node.target]]
            )
            for element in target
            if isinstance(element, ast.Name) and element.id == name
        )

    def reads(func: Callable[..., object], name: str) -> int:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        return sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        )

    assert assignments(main.index, "_index") == 1, (
        "the corpus must be published in a single assignment; two writes leave a "
        "window in which a reader sees one of them"
    )
    assert reads(main.query, "_index") == 1, (
        "the corpus must be read once per request; two reads can straddle a write"
    )
    assert getattr(main._Index, "__dataclass_params__").frozen, (
        "the published snapshot must be immutable, or it can be edited after publication"
    )
