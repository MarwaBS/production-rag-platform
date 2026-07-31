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


def _ci() -> Dict[str, Any]:
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
        ["git", "ls-files", "*.py"], capture_output=True, text=True, cwd=str(ROOT)
    )
    shipping = {path.split("/")[0] for path in tracked.stdout.split() if "/" in path}
    assert shipping, "fixture: git listed no tracked python packages"
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
        assert _masking_keys(workflow, job) == [], _masking_keys(workflow, job)
    assert (ROOT / "scripts" / "check_semantic_report.py").exists()
    assert "semantic" in jobs["docker"].get("needs", []), (
        "the image publish does not wait for the semantic gate"
    )


def _report(cases: str) -> str:
    return f"<testsuites><testsuite>{cases}</testsuite></testsuites>"


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
        target = pathlib.Path(
            next(arg for arg in argv if arg.startswith("--junitxml=")).split("=", 1)[1]
        )
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
    an empty suite is the vacuous case this whole check exists to reject."""
    from scripts import check_semantic_report as checker

    monkeypatch.setattr(checker, "required_tests", set)
    assert checker.check(runner=lambda argv, cwd: 0)


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
