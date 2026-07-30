"""The CI gates the README advertises must exist in the pipeline definition.

Text gates on config, the same trade the deploy-posture file states: they catch
deletion and renaming, not every malformed edit; the real pipeline executing is
what closes the residual. A coverage floor or audit step that only ever existed
in prose fails nothing when it silently disappears.
"""

from __future__ import annotations

import pathlib
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
