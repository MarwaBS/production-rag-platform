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
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ci() -> dict[str, Any]:
    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )


def _runs(job: dict[str, Any]) -> str:
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


def test_ci_gates_the_paraphrase_eval_under_the_semantic_backend() -> None:
    """Without a job installing the extra and selecting the semantic-marked
    tests, the paraphrase floor is deselected everywhere and can never fail."""
    jobs = _ci()["jobs"]
    semantic_jobs = [
        job
        for job in jobs.values()
        if "semantic]" in _runs(job) or "semantic]" in str(job)
    ]
    assert semantic_jobs, "no CI job installs the 'semantic' extra"
    assert any("-m semantic" in _runs(job) for job in semantic_jobs), (
        "the semantic extra is installed but the semantic-marked gates never run"
    )
    for job in semantic_jobs:
        # Exit status is not evidence — a shell can be told to lie in more ways
        # than a gate can enumerate. The job must emit a report and verify it,
        # so success requires the gates to appear in it as having passed.
        runs = _runs(job)
        assert "--junitxml=semantic-report.xml" in runs, (
            "the semantic job leaves no record of what it ran"
        )
        assert "scripts/check_semantic_report.py semantic-report.xml" in runs, (
            "nothing checks that the semantic gates appear in the report"
        )
    assert (ROOT / "scripts" / "check_semantic_report.py").exists()
    assert "semantic" in jobs["docker"].get("needs", []), (
        "the image publish does not wait for the semantic gate"
    )


def _report(cases: str) -> str:
    return f"<testsuites><testsuite>{cases}</testsuite></testsuites>"


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
        line.split("::")[-1].strip()
        for line in collected.stdout.splitlines()
        if "::" in line
    }
    assert marked, "pytest collected no marked gates to check against"
    assert required_tests() == marked


def test_a_report_missing_a_gate_is_not_evidence_that_it_ran() -> None:
    """Collect-only, deselect, a swallowed exit status: each ends in a report
    that cannot account for one of the tests the job claims to have run."""
    from scripts.check_semantic_report import verify

    required = {"test_alpha", "test_beta"}
    passed = _report('<testcase name="test_alpha"/><testcase name="test_beta"/>')
    assert verify(passed, required) == []
    assert verify(_report(""), required) == [
        "test_alpha: never ran",
        "test_beta: never ran",
    ]
    assert verify(_report('<testcase name="test_alpha"/>'), required) == [
        "test_beta: never ran"
    ]
    for outcome in ("failure", "error", "skipped"):
        report = _report(
            f'<testcase name="test_alpha"><{outcome}/></testcase>'
            '<testcase name="test_beta"/>'
        )
        assert verify(report, required) == [f"test_alpha: {outcome}"]
