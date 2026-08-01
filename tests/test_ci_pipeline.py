"""The CI gates the README advertises must exist in the pipeline definition.

These read the workflow file; they do not run it. What they can establish is
that each gate is present, that its command is exactly what it claims to be, and
that nothing in the file reduces what it examines. What no reading of the file
can establish is that the runner behaved — only the pipeline executing does
that, which is why the deploy-posture file makes the same trade.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The whole run line, not a substring of it. `: python scripts/...` contains the
# substring and executes nothing; the checker takes no argument so that the
# assertion below can be an equality rather than a search.
PROVE_SEMANTIC = "python scripts/check_semantic_report.py"


def _ci() -> Dict[Any, Any]:
    # Not str-keyed: YAML 1.1 reads the bare `on` trigger key as a boolean.
    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )


def _runs(job: Dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_ci_enforces_the_coverage_floor() -> None:
    """A floor that is not in the test command is a number in a document; the
    gate must be able to fail the build when covered code loses its tests."""
    test_job = _runs(_ci()["jobs"]["test"])
    assert "--cov=app" in test_job, "the test job never measures coverage"
    assert "--cov-fail-under=85" in test_job, (
        "the test job measures coverage but no floor can fail the build"
    )


def test_ci_audits_the_python_dependencies() -> None:
    """Trivy scans the built image's installed libraries; nothing audits the
    source tree's dependency set on pull requests, where a vulnerable pin
    should be caught before an image is ever built."""
    everything = "\n".join(_runs(job) for job in _ci()["jobs"].values())
    # An invocation line, not a mention: `pip install pip-audit` alone would
    # download the auditor and never run it.
    lines = [line.strip() for line in everything.splitlines()]
    assert any(
        line == "pip-audit" or line.startswith("pip-audit ") for line in lines
    ), "no run line invokes pip-audit"


def _top_level(listing: str) -> set[str]:
    """What each tracked path must be reachable as on a type-check command line:
    its package for a file in one, the file itself for one at the root."""
    paths = [pathlib.PurePosixPath(path) for path in listing.split("\0") if path]
    return {path.parts[0] if len(path.parts) > 1 else path.name for path in paths}


def test_the_type_check_argument_reader_does_not_overlook_a_root_level_file() -> None:
    """Today every tracked module sits in a package, so a reader that silently
    dropped root-level files would look right up until one appeared."""
    listing = "app/main.py\0conftest.py\0tests/test_x.py\0a dir/mod.py\0"
    assert _top_level(listing) == {"app", "conftest.py", "tests", "a dir"}


def test_ci_type_checks_every_directory_that_ships_python() -> None:
    """The gate that fails the build on an unproven semantic run now lives in
    scripts/, which the type-check argument list did not name. Naming the
    directories that exist beats naming the three that existed when it was
    written."""
    everything = "\n".join(_runs(job) for job in _ci()["jobs"].values())
    invocation = [
        line.strip()
        for line in everything.splitlines()
        if line.strip().startswith("mypy ")
    ]
    assert invocation, "no run line invokes mypy"
    checked = {argument for line in invocation for argument in line.split()[1:]}
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], capture_output=True, text=True, cwd=str(ROOT)
    )
    shipping = _top_level(tracked.stdout)
    assert shipping, "fixture: git listed no tracked python files"
    assert shipping <= checked, sorted(shipping - checked)


WORKFLOW_KEYS = {"name", True, "jobs"}
JOB_KEYS = {"runs-on", "steps", "needs", "permissions", "env"}
JOB_ENV = {"IMAGE"}
STEP_KEYS = {"name", "run", "uses", "with"}
PUBLISH_STEP_KEYS = STEP_KEYS | {"if", "id"}


def _unvetted_keys(workflow: Dict[Any, Any], job: Dict[str, Any]) -> List[str]:
    """Keys in this job that nobody has read and accepted.

    Listing the keys that can neuter a step is a blacklist over a set that keeps
    producing new members — shell, then defaults, then continue-on-error, then
    working-directory, then a step-level env that rebinds the image being
    scanned. The keys GitHub defines are a closed set, so the sound direction is
    the other one: allow what is used here and refuse the rest, which forces the
    next key to be read before it is adopted.
    """
    problems = [f"workflow: {key}" for key in workflow if key not in WORKFLOW_KEYS]
    problems += [f"job: {key}" for key in job if key not in JOB_KEYS]
    # Job-level env reaches every step in the job and is strictly stronger than
    # the step-level form refused below: PYTEST_ADDOPTS=--no-cov set here
    # removes the coverage floor while its run line still reads correctly.
    problems += [
        f"job env: {key}" for key in job.get("env") or {} if key not in JOB_ENV
    ]
    for step in job.get("steps", []):
        allowed = PUBLISH_STEP_KEYS if step.get("if") else STEP_KEYS
        where = step.get("name") or step.get("uses") or step.get("run")
        problems += [f"{where}: {key}" for key in step if key not in allowed]
    return problems


def test_ci_is_triggered_by_the_events_that_deliver_code() -> None:
    """The most complete mask is not a key on a step. A workflow no push and no
    pull request triggers never runs at all, and every gate in this file still
    passes. YAML reads a bare `on` as the boolean it spells."""
    triggers = _ci()[True]
    assert "pull_request" in triggers, sorted(triggers)
    # Naming the events is not enough: the filter keys under them are a closed
    # set too, and any one can narrow the trigger down to nothing. `branches` is
    # the exception only where the branch it must keep is checked as well.
    filters = {
        "branches-ignore",
        "paths",
        "paths-ignore",
        "tags",
        "tags-ignore",
        "types",
    }
    assert not set(triggers.get("pull_request") or {}) & (filters | {"branches"}), (
        f"pull requests are narrowed by {sorted(triggers['pull_request'])}"
    )
    push = triggers.get("push") or {}
    assert not set(push) & filters, f"push is narrowed by {sorted(set(push) & filters)}"
    branches = push.get("branches", [])
    # A scalar `branches: mainline` would satisfy a substring test and narrow
    # the trigger to a branch that does not exist.
    assert isinstance(branches, list) and "main" in branches, push
    # A negative pattern excludes the branch that the line above still finds.
    assert not [pattern for pattern in branches if pattern.startswith("!")], branches


def test_the_key_check_refuses_every_key_it_has_not_read() -> None:
    """It returns nothing on the shipped file — which is also what a check
    looking for keys that cannot occur returns. Each of these neuters a gate
    without touching its run line, and none had to be foreseen."""
    clean: Dict[str, Any] = {"steps": [{"run": PROVE_SEMANTIC}]}
    assert _unvetted_keys({"name": "CI", "jobs": {}}, clean) == []
    assert _unvetted_keys({"name": "CI", "jobs": {}, "defaults": {}}, clean)
    for key in ("defaults", "if", "continue-on-error", "strategy"):
        assert _unvetted_keys({}, {**clean, key: "x"}), key
    for key in ("shell", "continue-on-error", "working-directory", "env"):
        assert _unvetted_keys({}, {"steps": [{"run": PROVE_SEMANTIC, key: "x"}]}), key
    # A conditional step may carry `if` and `id`; what it may not be is a gate,
    # which is asserted where the conditional tail is.
    assert _unvetted_keys({}, {"steps": [{"run": "x", "if": "y", "shell": "z"}]})


def test_ci_gates_the_paraphrase_eval_under_the_semantic_backend() -> None:
    """Without a job installing the extra and running the semantic-marked tests,
    the paraphrase floor is deselected everywhere and can never fail."""
    workflow = _ci()
    jobs = workflow["jobs"]
    semantic_jobs = [job for job in jobs.values() if "semantic]" in str(job)]
    assert semantic_jobs, "no CI job installs the 'semantic' extra"
    for job in semantic_jobs:
        runs = [step.get("run", "").strip() for step in job["steps"]]
        assert PROVE_SEMANTIC in runs, runs
    assert (ROOT / "scripts" / "check_semantic_report.py").exists()
    assert "semantic" in jobs["docker"].get("needs", []), (
        "the image publish does not wait for the semantic gate"
    )


def test_no_job_carries_a_key_nobody_has_read() -> None:
    """Every job, not the ones some other job waits on: scoping this by `needs`
    left the publish job — the one holding the image scan — unexamined."""
    workflow = _ci()
    for name, job in workflow["jobs"].items():
        assert _unvetted_keys(workflow, job) == [], (
            f"{name}: {_unvetted_keys(workflow, job)}"
        )


PUBLISH_CONDITION = "github.event_name == 'push' && github.ref == 'refs/heads/main'"

# What the README's CI section sells. A claim in prose that no step implements
# fails nothing when it disappears, which is what this file exists to catch.
ADVERTISED_COMMANDS = (
    "ruff check .",
    "ruff format --check .",
    "mypy ",
    "pip-audit",
    "--cov-fail-under=85",
    "python -m evals",
    "helm lint",
    "helm template",
    PROVE_SEMANTIC,
)
# What each gate step runs, line for line. Asking whether a command APPEARS
# admits every way of having it present and inert: quoted in a comment, echoed,
# fenced in a heredoc, or followed by an excuse like `|| true`. These bodies are
# ours, so they are a closed set and can be pinned rather than searched.
GATE_RUNS = {
    "Lint": ("ruff check .",),
    "Format check": ("ruff format --check .",),
    "Type-check": ("mypy app evals scripts tests",),
    "Audit Python dependencies (pip-audit)": (
        "pip install pip-audit -c constraints-dev.txt",
        "pip-audit",
    ),
    "Integration tests (coverage floor enforced)": (
        "pytest -q --cov=app --cov-fail-under=85",
    ),
    "Retrieval eval (recall gate enforced in tests/test_eval.py; print the numbers)": (
        "python -m evals",
    ),
    "Paraphrase floor + floor-derivation reproduce (semantic-marked gates)": (
        PROVE_SEMANTIC,
    ),
    "Helm lint + render": (
        "helm lint deploy/helm",
        "helm template release deploy/helm > /dev/null",
    ),
}

# Every input every action may carry. Restricting only the scanners left the
# builder free to publish a different stage, and checkout free to fetch a
# different tree — neither needs a key anyone had thought to forbid.
ACTION_INPUTS = {
    "actions/checkout": set(),
    "actions/setup-python": {"python-version"},
    "actions/cache": {"path", "key"},
    "azure/setup-helm": set(),
    "docker/setup-buildx-action": set(),
    "docker/login-action": {"registry", "username", "password"},
    "docker/build-push-action": {"context", "file", "load", "push", "tags"},
}
# Config files these tools read from the repo root without being told to. An
# ignore file excuses findings with the workflow untouched.
TOOL_IGNORE_FILES = (
    ".trivyignore",
    ".trivyignore.yaml",
    ".hadolint.yaml",
    ".hadolint.yml",
)
VETTED_INPUTS = {
    "aquasecurity/trivy-action": {
        "image-ref",
        "format",
        "exit-code",
        "severity",
        "ignore-unfixed",
        "vuln-type",
    },
    "anchore/sbom-action": {"image", "format", "output-file"},
    "actions/upload-artifact": {"name", "path"},
    "hadolint/hadolint-action": {"dockerfile", "failure-threshold"},
}
ADVERTISED_ACTIONS = tuple(VETTED_INPUTS)
ACTION_INPUTS.update(VETTED_INPUTS)


def _steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _action(workflow: Dict[Any, Any], name: str) -> List[Dict[str, Any]]:
    return [s for s in _steps(workflow) if s.get("uses", "").split("@")[0] == name]


def _uncommented(run: str) -> List[str]:
    """A gate quoted inside a shell comment is not a gate that runs."""
    return [line for line in run.splitlines() if not line.strip().startswith("#")]


def _advertised(step: Dict[str, Any]) -> bool:
    lines = _uncommented(step.get("run", ""))
    return (
        step.get("uses", "").split("@")[0] in ADVERTISED_ACTIONS
        or bool((step.get("with") or {}).get("load"))
        or any(command in line for line in lines for command in ADVERTISED_COMMANDS)
    )


def _gate_steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Steps whose outcome decides a verdict.

    Excluding the publish by a `with.push` key let any step opt out of being a
    gate by setting it, and YAML reads `push: "false"` as a truthy string. What
    identifies the publish is that it publishes: it carries the merge
    condition, which nothing that decides a verdict may do."""
    return [s for s in _steps(workflow) if not s.get("if") and _advertised(s)]


def _conditional_gates(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Advertised gates wearing the merge condition — how one leaves the set
    above while remaining the thing the README sells."""
    return [s for s in _steps(workflow) if s.get("if") and _advertised(s)]


def test_every_command_the_readme_advertises_runs_in_the_pipeline() -> None:
    """Each is sold as a gate. Presence is not the property — a command can be
    echoed, fenced in a heredoc or excused with `|| true` and still be there —
    so each gate step's body has to be exactly what it is supposed to run."""
    named = {step.get("name"): step for step in _steps(_ci())}
    for name, expected in GATE_RUNS.items():
        step = named.get(name)
        assert step, f"the {name} step is gone"
        body = tuple(
            line.strip() for line in _uncommented(step.get("run", "")) if line.strip()
        )
        assert body == expected, (name, body)


def test_every_action_the_readme_advertises_is_present_and_can_fail() -> None:
    """Present is half of it: a scan that reports and exits zero, or one told to
    skip every directory it would have looked in, is decoration with a tick."""
    workflow = _ci()
    for step in _steps(workflow):
        action = step.get("uses", "").split("@")[0]
        if not action:
            continue
        allowed = ACTION_INPUTS.get(action)
        assert allowed is not None, f"{action} has no vetted input list"
        assert set(step.get("with") or {}) <= allowed, (action, step.get("with"))
    for ignore_file in TOOL_IGNORE_FILES:
        assert not (ROOT / ignore_file).exists(), ignore_file
    for name, vetted in VETTED_INPUTS.items():
        steps = _action(workflow, name)
        # Exactly one: a second, later step of the same action overwrites what
        # the first produced, and every read below takes the first.
        assert len(steps) == 1, (name, len(steps))
        for step in steps:
            # An exact version, so a re-pointed tag cannot change the verdict
            # without changing this file. The actions not listed here still
            # float on major tags, which is a weaker position, not a defended
            # one — their inputs are pinned above but their code is not.
            assert re.fullmatch(r"v\d+\.\d+\.\d+", step["uses"].split("@")[1]), step
            assert set(step["with"]) <= vetted, set(step["with"]) - vetted
    scan = _action(workflow, "aquasecurity/trivy-action")[0]["with"]
    # The values, not just the keys: os-only drops every library CVE, and
    # CRITICAL-only drops the HIGH the step's own name promises.
    assert scan["exit-code"] == "1", scan
    assert scan["severity"] == "HIGH,CRITICAL", scan
    assert scan["vuln-type"] == "os,library", scan
    lint = _action(workflow, "hadolint/hadolint-action")[0]["with"]
    assert lint["failure-threshold"] == "error", lint
    assert (ROOT / lint["dockerfile"]).is_file(), lint


def test_what_gets_scanned_is_what_gets_built_and_what_gets_pushed() -> None:
    """Comparing the two expressions is not comparing the two images: a
    step-level env rebinds what they expand to, and the push is a second build
    whose inputs were never tied to the one that was scanned."""
    workflow = _ci()
    loaded = [s for s in _steps(workflow) if (s.get("with") or {}).get("load")]
    assert len(loaded) == 1, loaded
    build = loaded[0]["with"]
    assert _action(workflow, "aquasecurity/trivy-action")[0]["with"]["image-ref"] == (
        build["tags"].strip()
    )
    assert _action(workflow, "anchore/sbom-action")[0]["with"]["image"] == (
        build["tags"].strip()
    )
    published = [s for s in _steps(workflow) if (s.get("with") or {}).get("push")]
    assert len(published) == 1, published
    for shared in ("context", "file"):
        assert published[0]["with"][shared] == build[shared], shared
    # The chart's appVersion tag must be among them, or a bare `helm install`
    # resolves a tag that was never pushed.
    tags = published[0]["with"]["tags"].split()
    assert any("app_version" in tag for tag in tags), tags
    assert any(tag.endswith(":latest") for tag in tags), tags
    assert any("github.sha" in tag for tag in tags), tags


def test_the_sbom_that_is_uploaded_is_the_sbom_that_was_generated() -> None:
    """Two steps joined by a filename nobody compared: the artifact can hold
    anything while the README sells a CycloneDX bill of materials."""
    workflow = _ci()
    sbom = _action(workflow, "anchore/sbom-action")[0]["with"]
    upload = _action(workflow, "actions/upload-artifact")[0]["with"]
    assert sbom["format"] == "cyclonedx-json", sbom
    assert upload["path"] == sbom["output-file"], (upload, sbom)


def test_the_publish_waits_on_every_other_job() -> None:
    """Waiting on some of them publishes while the rest are still failing."""
    jobs = _ci()["jobs"]
    assert set(jobs["docker"]["needs"]) == set(jobs) - {"docker"}, jobs["docker"]


def test_no_gate_step_is_conditional_and_publishing_comes_last() -> None:
    """A condition on a gate is invisible in a green run: the step is skipped,
    not failed. Publishing is rightly conditional, so the conditional steps must
    be the tail of their job and carry the one condition that means this is a
    merge — and every gate must be unconditional, which is what puts it before
    them. Position alone does not: a gate that is also conditional sits happily
    in that tail, running after the image it was meant to vet has been pushed."""
    workflow = _ci()
    assert _conditional_gates(workflow) == [], _conditional_gates(workflow)
    for step in _gate_steps(workflow):
        assert "if" not in step, step.get("name") or step.get("uses")
    for name, job in workflow["jobs"].items():
        assert "if" not in job, f"{name} is switched off wholesale"
        steps = job.get("steps", [])
        conditional = [index for index, step in enumerate(steps) if "if" in step]
        tail = list(range(len(steps) - len(conditional), len(steps)))
        assert conditional == tail, f"{name}: {conditional} is not the tail"
        for index in conditional:
            assert steps[index]["if"] == PUBLISH_CONDITION, steps[index]["if"]


def _report(cases: str) -> str:
    return f"<testsuites><testsuite>{cases}</testsuite></testsuites>"


def _junitxml(argv: List[str]) -> pathlib.Path:
    option = next(arg for arg in argv if arg.startswith("--junitxml="))
    return pathlib.Path(option.split("=", 1)[1])


def _case(name: str, outcome: str = "") -> str:
    module, bare = name.split("::")
    body = f"<{outcome}/>" if outcome else ""
    return f'<testcase classname="{module}" name="{bare}">{body}</testcase>'


def test_the_report_check_covers_every_semantic_gate_in_the_suite() -> None:
    """pytest is the authority on which tests carry the marker, so ask it rather
    than a second copy of the same guess: a gate added later must be covered."""
    from scripts.check_semantic_report import required_tests

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "semantic", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert collected.returncode == 0, collected.stdout[-400:]
    marked = {
        line.strip().replace("/", ".").replace(".py::", "::")
        for line in collected.stdout.splitlines()
        if "::" in line
    }
    assert marked, "pytest collected no marked gates to check against"
    assert required_tests() == marked


def test_a_report_missing_a_gate_is_not_evidence_that_it_ran() -> None:
    """Collect-only, deselect, a swallowed exit status: each ends in a report
    that cannot account for one of the tests the job claims to have run."""
    from scripts.check_semantic_report import verify

    required = {"tests.test_a::test_alpha", "tests.test_b::test_beta"}
    both = _case("tests.test_a::test_alpha") + _case("tests.test_b::test_beta")
    assert verify(_report(both), required) == []
    assert verify(_report(""), required) == [
        "tests.test_a::test_alpha: never ran",
        "tests.test_b::test_beta: never ran",
    ]
    assert verify(_report(_case("tests.test_a::test_alpha")), required) == [
        "tests.test_b::test_beta: never ran"
    ]
    for outcome in ("failure", "error", "skipped"):
        report = _report(
            _case("tests.test_a::test_alpha", outcome)
            + _case("tests.test_b::test_beta")
        )
        assert verify(report, required) == [f"tests.test_a::test_alpha: {outcome}"]


def test_a_failed_gate_is_not_masked_by_a_same_named_test_elsewhere() -> None:
    """Two modules may each define a test of the same name. Keyed on the bare
    name, whichever the report lists last stands in for the other."""
    from scripts.check_semantic_report import verify

    required = {"tests.test_a::test_shared"}
    report = _report(
        _case("tests.test_a::test_shared", "failure")
        + _case("tests.test_b::test_shared")
    )
    assert verify(report, required) == ["tests.test_a::test_shared: failure"]


def test_the_checker_runs_the_marked_gates_and_records_them() -> None:
    """The command is the only part of the run the checker chooses; selecting
    nothing, or recording nowhere, leaves it proving something else."""
    from scripts.check_semantic_report import _pytest_command

    report = pathlib.Path("somewhere") / "report.xml"
    argv = _pytest_command(report)
    assert argv[0] == sys.executable and argv[2] == "pytest", argv
    options = argv[3:]
    assert options[options.index("-m") + 1] == "semantic", argv
    assert f"--junitxml={report}" in options, argv


def test_the_checker_reads_only_a_report_the_run_it_started_wrote() -> None:
    """A report the checker is handed can be committed, or written by a step
    that ran no tests, and copied into place. One it opens a private path for
    and then reads back cannot be."""
    from scripts.check_semantic_report import check, required_tests

    seen: Dict[str, Any] = {}

    def runner(argv: List[str], cwd: str) -> int:
        target = _junitxml(argv)
        seen["path"] = target
        seen["existed"] = target.exists()
        seen["cwd"] = cwd
        target.write_text(
            _report("".join(_case(name) for name in required_tests())),
            encoding="utf-8",
        )
        return 0

    assert check(runner=runner) == []
    assert seen["existed"] is False, (
        "the run was pointed at a file that already existed"
    )
    assert ROOT not in seen["path"].parents, seen["path"]
    assert pathlib.Path(seen["cwd"]) == ROOT, seen["cwd"]

    # A run that writes nothing is the collection-only case: no report, no proof.
    assert check(runner=lambda argv, cwd: 1) == [
        "the run wrote no report (pytest exited 1)"
    ]


def test_an_empty_marker_set_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing required means every report satisfies it — a gate that certifies
    an empty suite is the vacuous case this whole check exists to reject. The
    run has to write its report, or this passes on the missing file instead."""
    from scripts import check_semantic_report as checker

    def runner(argv: List[str], cwd: str) -> int:
        _junitxml(argv).write_text(_report(""), encoding="utf-8")
        return 0

    monkeypatch.setattr(checker, "required_tests", set)
    assert checker.check(runner=runner)


def test_the_checker_fails_the_build_when_the_gates_are_not_proven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list of problems is a build failure only once something exits on it."""
    from scripts import check_semantic_report as checker

    monkeypatch.setattr(
        checker, "check", lambda: ["tests.test_a::test_alpha: never ran"]
    )
    with pytest.raises(SystemExit) as failure:
        checker.main()
    assert failure.value.code, "an unproven gate did not fail the build"
    assert "test_alpha" in str(failure.value.code)
    monkeypatch.setattr(checker, "check", list)
    checker.main()  # a proven run must not fail the build
