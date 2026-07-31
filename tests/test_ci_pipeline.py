"""The CI gates the README advertises must exist in the pipeline definition.

Text gates on config, the same trade the deploy-posture file states: they catch
deletion and renaming, not every malformed edit; the real pipeline executing is
what closes the residual. A coverage floor or audit step that only ever existed
in prose fails nothing when it silently disappears.
"""

from __future__ import annotations

import pathlib
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


def _masking_keys(workflow: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
    """Keys that can mask a step's status without editing its run line.

    Enumerating them is sound only because GitHub defines them — unlike shell
    spellings, the set is closed. `shell` is a step key and a defaults key; at
    job scope it does not exist, so asserting its absence there proves nothing.
    """
    problems = []
    for scope, where in ((workflow, "workflow"), (job, "job")):
        if "defaults" in scope:
            problems.append(f"{where} defaults can redefine every shell")
    for key in ("if", "continue-on-error"):
        if key in job:
            problems.append(f"the job carries {key}")
    for step in job.get("steps", []):
        for key in ("if", "continue-on-error", "shell"):
            if key in step:
                problems.append(f"a step carries {key}")
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


def test_the_masking_key_check_rejects_every_key_it_claims_to_cover() -> None:
    """It returns nothing on the shipped file — which is also what a check
    looking at keys that cannot occur returns."""
    clean: Dict[str, Any] = {"steps": [{"run": PROVE_SEMANTIC}]}
    assert _masking_keys({}, clean) == []
    shell = {"run": {"shell": "bash -c :"}}
    assert _masking_keys({"defaults": shell}, clean)
    assert _masking_keys({}, {**clean, "defaults": shell})
    for key in ("if", "continue-on-error"):
        assert _masking_keys({}, {**clean, key: "x"})
        assert _masking_keys({}, {"steps": [{"run": PROVE_SEMANTIC, key: "x"}]})
    assert _masking_keys({}, {"steps": [{"run": PROVE_SEMANTIC, "shell": "bash -c :"}]})


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


def test_no_job_the_publish_waits_on_can_mask_a_failure() -> None:
    """The jobs the image publish waits on are exactly the ones whose verdicts
    decide anything, so those are the ones no masking key may sit on. Reading
    them off `needs` covers a gate job added later; naming the semantic job
    covered the one this check was written for."""
    workflow = _ci()
    jobs = workflow["jobs"]
    gate_jobs = jobs["docker"].get("needs", [])
    assert gate_jobs, "the image publish waits on nothing"
    for name in gate_jobs:
        assert _masking_keys(workflow, jobs[name]) == [], (
            f"{name}: {_masking_keys(workflow, jobs[name])}"
        )


def test_nothing_anywhere_is_allowed_to_fail_without_failing() -> None:
    """Tolerating an error has no honest use in this workflow, at any scope."""
    for name, job in _ci()["jobs"].items():
        for scope in (job, *job.get("steps", [])):
            assert "continue-on-error" not in scope, name


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
ADVERTISED_ACTIONS = (
    "docker/build-push-action",
    "aquasecurity/trivy-action",
    "anchore/sbom-action",
    "actions/upload-artifact",
    "hadolint/hadolint-action",
)


def _steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _action(workflow: Dict[Any, Any], name: str) -> List[Dict[str, Any]]:
    return [s for s in _steps(workflow) if s.get("uses", "").split("@")[0] == name]


def _gate_steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Steps that decide a verdict, as against the step that publishes."""
    gates = []
    for step in _steps(workflow):
        if (step.get("with") or {}).get("push"):
            continue
        run, action = step.get("run", ""), step.get("uses", "").split("@")[0]
        if action in ADVERTISED_ACTIONS or any(c in run for c in ADVERTISED_COMMANDS):
            gates.append(step)
    return gates


def test_every_command_the_readme_advertises_runs_in_the_pipeline() -> None:
    """Each is sold as a gate, and deleting any of their steps was silent."""
    everything = "\n".join(_runs(job) for job in _ci()["jobs"].values())
    for command in ADVERTISED_COMMANDS:
        assert command in everything, command


def test_every_action_the_readme_advertises_is_present_and_can_fail() -> None:
    """Present is half of it: a scan that reports and exits zero, or a linter
    whose threshold excuses everything, is decoration with a green tick."""
    workflow = _ci()
    for name in ADVERTISED_ACTIONS:
        assert _action(workflow, name), name
    for step in _steps(workflow):
        # A moving ref changes the gate without changing the file.
        assert not step.get("uses", "").endswith(("@master", "@main")), step["uses"]
    scan = _action(workflow, "aquasecurity/trivy-action")[0]["with"]
    assert scan["exit-code"] == "1", scan
    assert "CRITICAL" in scan["severity"], scan
    lint = _action(workflow, "hadolint/hadolint-action")[0]["with"]
    assert lint["failure-threshold"] == "error", lint
    assert (ROOT / lint["dockerfile"]).is_file(), lint


def test_what_gets_scanned_is_what_gets_built() -> None:
    """The scan names an image by tag. Any other tag scans something else and
    passes, while the image that ships was never looked at."""
    workflow = _ci()
    loaded = [s for s in _steps(workflow) if (s.get("with") or {}).get("load")]
    assert len(loaded) == 1, loaded
    tag = loaded[0]["with"]["tags"].strip()
    assert _action(workflow, "aquasecurity/trivy-action")[0]["with"]["image-ref"] == tag
    assert _action(workflow, "anchore/sbom-action")[0]["with"]["image"] == tag


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
