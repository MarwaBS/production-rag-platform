"""The deployment's stated hardening must be enforced, not just described.

Some of these the chart argues for at length in its own comments; others it does
not mention at all. Both kinds share the same defect: nothing fails when they go
away. A safeguard whose only witness is the paragraph beside it is a paragraph,
and one with no witness at all is a hope.

How each file is read, because it bounds what an assertion can promise:
`values.yaml` carries no template directives, so it is PARSED and its assertions
are structural — an empty `limits:` is a null value, not a line a pattern can
skim past into the next block. The files under `templates/` are Go templates and
are not valid YAML until Helm renders them, so those assertions read text with
comments stripped. That is weaker, and what closes the gap is rendering the chart
in CI, which is where a rendered chart can be validated; until then they
catch deletion, not every malformed edit.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
HELM = ROOT / "deploy" / "helm"


def _values() -> dict[str, Any]:
    return yaml.safe_load((HELM / "values.yaml").read_text(encoding="utf-8"))


def _template(name: str) -> str:
    raw = (HELM / "templates" / name).read_text(encoding="utf-8")
    return "\n".join(re.sub(r"#.*$", "", line) for line in raw.splitlines())


def _probe_block(deployment: str, probe: str) -> str:
    """The indented body of one probe, so a per-probe assertion cannot be
    satisfied by a sibling probe's settings."""
    found = re.search(rf"^(\s+){probe}:\n((?:\1\s+.*\n)+)", deployment, flags=re.M)
    assert found, f"expected a {probe} block"
    return found.group(2)


def test_replica_count_stays_one() -> None:
    """The index lives in the process, so a second pod answers from an empty one."""
    assert _values()["replicaCount"] == 1, (
        "replicaCount must stay 1 while the corpus is held in process"
    )


@pytest.mark.parametrize("bound", ["cpu", "memory"])
def test_the_container_declares_resource_limits(bound: str) -> None:
    """An unbounded pod competes with its neighbours for the node, and a limit is
    what turns a runaway request into one dead pod rather than a dead node."""
    resources = _values().get("resources") or {}
    limits = resources.get("limits")
    assert limits, f"no resource limits are declared: {resources}"
    assert limits.get(bound), f"no {bound} limit is declared: {limits}"
    assert ".Values.resources" in _template("deployment.yaml"), (
        "the Deployment must render the declared resources"
    )


def test_the_container_runs_as_a_named_non_root_user() -> None:
    """The image must end as an account the image itself created.

    Docker honours the LAST USER, a bare uid has no passwd entry, and root is the
    thing the claim exists to deny — so the final instruction is what counts.
    """
    raw = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    dockerfile = "\n".join(re.sub(r"#.*$", "", line) for line in raw.splitlines())
    users = re.findall(r"^USER\s+(\S+)", dockerfile, flags=re.M)
    assert users, "the image must switch to a non-root user; the README claims it does"
    final = users[-1]
    assert final not in {"root", "0"}, f"the image ends as root (USER {final})"
    assert not final.isdigit(), (
        f"USER {final} is a bare uid with no passwd entry; the account the image "
        "creates is what makes the non-root claim checkable"
    )
    assert re.search(rf"useradd[^\n]*\b{re.escape(final)}\b", dockerfile), (
        f"USER {final} is never created in the image"
    )


@pytest.mark.parametrize(
    "field, expected",
    [
        ("runAsNonRoot", "true"),
        ("readOnlyRootFilesystem", "true"),
        ("allowPrivilegeEscalation", "false"),
    ],
)
def test_restricted_pod_security_fields_are_set(field: str, expected: str) -> None:
    deployment = _template("deployment.yaml")
    found = re.search(rf"^\s*{field}:\s*(\S+)", deployment, flags=re.M)
    assert found and found.group(1) == expected, (
        f"{field} must be {expected} for the 'restricted' posture the chart claims"
    )


def test_all_capabilities_are_dropped() -> None:
    assert re.search(r"drop:\s*\[\s*[\"']ALL[\"']\s*\]", _template("deployment.yaml"))


def test_the_service_account_token_is_not_mounted() -> None:
    """The app never calls the Kubernetes API, so the token is only a credential
    waiting to be stolen."""
    rendered = _template("deployment.yaml") + _template("serviceaccount.yaml")
    assert re.search(r"automountServiceAccountToken:\s*false", rendered), (
        "a pod that never calls the API server must not mount its token"
    )


def test_probes_target_health_and_not_ready() -> None:
    """Pod readiness gated on /ready deadlocks: the Service sends no traffic to an
    un-indexed pod, so it can never receive the /index call that readies it."""
    deployment = _template("deployment.yaml")
    for probe in ("readinessProbe", "livenessProbe"):
        path = re.search(r"path:\s*(\S+)", _probe_block(deployment, probe))
        assert path and path.group(1) == "/health", (
            f"{probe} must target /health, got {path and path.group(1)}"
        )


@pytest.mark.parametrize(
    "field", ["timeoutSeconds", "periodSeconds", "failureThreshold"]
)
def test_probes_declare_their_timing(field: str) -> None:
    """Left unset, the cluster defaults apply — a 1s timeout that a large request
    can already exceed."""
    deployment = _template("deployment.yaml")
    for probe in ("readinessProbe", "livenessProbe"):
        assert re.search(
            rf"^\s*{field}:", _probe_block(deployment, probe), flags=re.M
        ), f"{probe} must set {field} rather than inherit the cluster default"


def test_the_secret_is_optional_when_the_chart_does_not_create_it() -> None:
    """The chart documents `secrets.create: false` for externally managed secrets.

    With an unconditional envFrom and no Secret rendered, that documented path
    produces a pod that cannot start.
    """
    deployment = _template("deployment.yaml")
    # Scoped to the envFrom block: a guard anywhere else in the file does not
    # make THIS reference survive a Secret that was never rendered.
    env_from = re.search(r"^(\s+)envFrom:\n((?:\1\s+.*\n)+)", deployment, flags=re.M)
    assert env_from, "expected an envFrom block"
    block = env_from.group(2)
    # No template control-flow inside the block: `optional: true` behind an
    # `{{- if }}` is absent on the very path this gate protects. A condition,
    # if the chart ever needs one, wraps the whole block — inside it,
    # conditional and deleted are the same thing.
    controlled = [
        line.strip()
        for line in block.splitlines()
        if re.search(r"\{\{-?\s*(if|else|end|range|with)\b", line)
    ]
    assert not controlled, (
        f"control-flow inside envFrom makes the secretRef conditional: {controlled}"
    )
    # The secretRef ENTRY alone. A sibling configMapRef carrying optional: true
    # would otherwise satisfy this while the Secret reference stays required.
    entry_lines = block.splitlines()
    heads = [i for i, line in enumerate(entry_lines) if line.strip() == "- secretRef:"]
    assert heads, "expected an envFrom secretRef"
    head = heads[0]
    indent = len(entry_lines[head]) - len(entry_lines[head].lstrip())
    entry = [entry_lines[head]]
    for line in entry_lines[head + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        entry.append(line)
    block = chr(10).join(entry)
    # `optional: true` only. Wrapping the reference in `if .Values.secrets.create`
    # also stops the pod wedging — by dropping the env entirely, so it starts with
    # no API key and serves an unauthenticated data-plane on the very path the
    # chart documents for externally managed secrets.
    assert "optional: true" in block, (
        "with secrets.create=false no Secret is rendered, so the reference must be "
        "optional or the pod wedges in CreateContainerConfigError"
    )


def test_the_secret_is_rendered_only_when_the_chart_creates_it() -> None:
    """`secrets.create: false` exists so an out-of-band Secret survives upgrades.

    Without the guard the chart renders its own (empty) Secret unconditionally,
    and every `helm upgrade` resets the externally managed data — the exact
    defect the create flag exists to prevent.
    """
    secret = _template("secret.yaml")
    directives = re.findall(r"\{\{-?(.*?)-?\}\}", secret, flags=re.S)
    assert any(
        re.search(r"\bif\b.*\.Values\.secrets\.create", d) for d in directives
    ), "templates/secret.yaml must be guarded by .Values.secrets.create"


def test_the_chart_ships_a_scrape_object() -> None:
    """The retrieval-health metrics exist so Prometheus can read them; without
    a ServiceMonitor the chart exports series nothing ever scrapes."""
    monitor = _template("servicemonitor.yaml")
    assert "kind: ServiceMonitor" in monitor
    directives = re.findall(r"\{\{-?(.*?)-?\}\}", monitor, flags=re.S)
    assert any(
        re.search(r"\bif\b.*\.Values\.monitoring\.enabled", d) for d in directives
    ), (
        "the ServiceMonitor needs the prometheus-operator CRDs, so it must be "
        "guarded by monitoring.enabled rather than break every bare install"
    )
    assert isinstance(_values()["monitoring"]["enabled"], bool)


def test_the_chart_ships_an_alert_rule_on_metrics_the_app_exports() -> None:
    """An alert on a metric the app does not export can never fire — the rule
    must name real series, checked against the code that registers them."""
    rule = _template("prometheusrule.yaml")
    assert "kind: PrometheusRule" in rule
    directives = re.findall(r"\{\{-?(.*?)-?\}\}", rule, flags=re.S)
    assert any(
        re.search(r"\bif\b.*\.Values\.monitoring\.enabled", d) for d in directives
    ), "the PrometheusRule needs the CRDs, so it shares the monitoring.enabled guard"
    exported = set(
        re.findall(
            r"[\"'](rag_[a-z_]+)[\"']",
            (ROOT / "app" / "main.py").read_text(encoding="utf-8"),
        )
    )
    referenced = set(re.findall(r"\brag_[a-z_]+", rule))
    assert referenced, "the rule alerts on none of this service's own series"

    def is_exported(name: str) -> bool:
        # Histograms export derived _bucket/_count/_sum series; a rule may
        # legitimately reference those under the registered base name.
        candidates = {name}
        for suffix in ("_bucket", "_count", "_sum"):
            candidates.add(name.removesuffix(suffix))
        return bool(candidates & exported)

    ghosts = {name for name in referenced if not is_exported(name)}
    assert not ghosts, f"the rule references series the app never exports: {ghosts}"


def test_the_ingress_is_disabled_by_default() -> None:
    """A bare install must publish nothing: the data-plane is only authenticated
    when an API key is set."""
    assert _values()["ingress"]["enabled"] is False


def test_enabling_the_ingress_alone_cannot_publish_plaintext() -> None:
    """The chart must refuse the unsafe combination, not merely warn about it.

    values.yaml and the README both say the Ingress may only be enabled together
    with an API key and TLS. The guard is read as VALUE PATHS: a message naming
    the prerequisites is prose, and prose is what this rejects.
    """
    template = _template("ingress.yaml")
    directives = re.findall(r"\{\{-?(.*?)-?\}\}", template, flags=re.S)
    # A branch on a literal never depends on the install, so a guard parked
    # inside one is decoration — the same unreachable-branch rule the key-check
    # gate applies. Literals only: evaluating expressions is the renderer's
    # job, and rendering the chart in CI is what closes that residual.
    dead = [
        d.strip()
        for d in directives
        if re.match(r"""^if\s+(false|true|0|1|""|'')\s*$""", d.strip())
    ]
    assert not dead, f"constant-condition branch in the ingress template: {dead}"
    refusing = [d for d in directives if re.search("required|fail", d)]
    assert refusing, (
        "the chart must refuse to render a public Ingress without its stated "
        "prerequisites rather than documenting them"
    )
    # Strip the human-readable messages; only the values a guard consults count.
    # `{{- if ... }}{{- fail }}{{- end }}` is as real a refusal as `required` with
    # the path inline, so the whole template's directives are read, not just the
    # one carrying the keyword.
    paths = " ".join(re.sub(r"\"[^\"]*\"|'[^']*'", " ", d) for d in directives).lower()
    for prerequisite, path in (
        ("TLS", ".values.ingress.tls"),
        ("an API key", ".values.secrets"),
    ):
        assert path in paths, (
            f"the Ingress guard never consults {path}, so nothing stops it "
            f"rendering without {prerequisite}: {refusing}"
        )
