"""Run the semantic gates and prove every one of them passed.

An exit status is not evidence: a collection-only run, a deselect or a swallowed
status all exit zero. A report handed in is not evidence either — it can be
committed, or written by a step that ran nothing, and copied into place. So this
starts the run itself, into a directory it creates outside the tree, and reads
back only what that run wrote there.

Run: python scripts/check_semantic_report.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Callable
from xml.etree import ElementTree

REPO = pathlib.Path(__file__).resolve().parent.parent
_MARKED = re.compile(r"@pytest\.mark\.semantic\s*\ndef (test_\w+)", re.MULTILINE)


def required_tests() -> set[str]:
    """Every semantic-marked test in the suite, keyed the way the report keys it
    — on a bare name, a test of the same name elsewhere stands in for this one."""
    return {
        f"{path.relative_to(REPO).with_suffix('').as_posix().replace('/', '.')}::{name}"
        for path in (REPO / "tests").glob("test_*.py")
        for name in _MARKED.findall(path.read_text(encoding="utf-8"))
    }


def _pytest_command(report: pathlib.Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "semantic",
        f"--junitxml={report}",
    ]


def verify(report: str, required: set[str]) -> list[str]:
    """Reasons the report fails to prove the gates ran; empty means proven."""
    cases = {
        f"{case.get('classname')}::{case.get('name')}": case
        for case in ElementTree.fromstring(report).iter("testcase")
    }
    problems = []
    for name in sorted(required):
        case = cases.get(name)
        if case is None:
            problems.append(f"{name}: never ran")
            continue
        for outcome in ("failure", "error", "skipped"):
            if case.find(outcome) is not None:
                problems.append(f"{name}: {outcome}")
    return problems


def check(runner: Callable[..., int] = subprocess.call) -> list[str]:
    """Run the marked gates and report whatever that run leaves unproven."""
    required = required_tests()
    if not required:
        return ["no semantic-marked test is defined: there is nothing to prove"]
    with tempfile.TemporaryDirectory() as scratch:
        # A path opened by this process outside the tree: whatever the repo or an
        # earlier step may hold, it is not what gets read back here.
        report = pathlib.Path(scratch) / "semantic-report.xml"
        status = runner(_pytest_command(report), cwd=str(REPO))
        if not report.exists():
            return [f"the run wrote no report (pytest exited {status})"]
        return verify(report.read_text(encoding="utf-8"), required)


def main() -> None:
    problems = check()
    if problems:
        sys.exit("the semantic gates are unproven:\n  " + "\n  ".join(problems))
    sys.stdout.write(f"{len(required_tests())} semantic gate(s) ran and passed\n")


if __name__ == "__main__":
    main()
