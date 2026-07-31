"""Prove the semantic gates ran, rather than trusting the job's exit status.

A collection-only run, a swallowed exit code, a deselect or a tolerated failure
all let the job report success; none of them can produce a report in which every
semantic-marked test is present and passed. This checks the report instead.

Run: python scripts/check_semantic_report.py <junit-xml>
"""

from __future__ import annotations

import pathlib
import re
import sys
from xml.etree import ElementTree

REPO = pathlib.Path(__file__).resolve().parent.parent
_MARKED = re.compile(r"@pytest\.mark\.semantic\s*\ndef (test_\w+)", re.MULTILINE)


def required_tests() -> set[str]:
    """Every semantic-marked test in the suite — read from the tests, so a new
    gate is covered the moment it is written."""
    return {
        name
        for path in (REPO / "tests").glob("test_*.py")
        for name in _MARKED.findall(path.read_text(encoding="utf-8"))
    }


def verify(report: str, required: set[str]) -> list[str]:
    """Reasons the report fails to prove the gates ran; empty means proven."""
    cases = {
        case.get("name"): case
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


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: check_semantic_report.py <junit-xml>")
    required = required_tests()
    if not required:
        sys.exit("no semantic-marked tests found: the report proves nothing")
    problems = verify(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), required)
    if problems:
        sys.exit("the semantic gates did not run:\n  " + "\n  ".join(problems))
    sys.stdout.write(f"{len(required)} semantic gate(s) ran and passed\n")


if __name__ == "__main__":
    main()
