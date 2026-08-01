"""Run the semantic gates and prove every one of them passed.

Run: python -m scripts.check_semantic_report
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import List, Set

from scripts.gate_report import REPO, Runner, prove

_MARKED = re.compile(r"@pytest\.mark\.semantic\s*\ndef (test_\w+)", re.MULTILINE)


def required_tests() -> Set[str]:
    """Every semantic-marked test in the suite, keyed the way the report keys it."""
    return {
        f"{path.relative_to(REPO).with_suffix('').as_posix().replace('/', '.')}::{name}"
        for path in (REPO / "tests").glob("test_*.py")
        for name in _MARKED.findall(path.read_text(encoding="utf-8"))
    }


def _pytest_command(report: pathlib.Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "semantic",
        f"--junitxml={report}",
    ]


def check(runner: Runner = subprocess.call) -> List[str]:
    return prove(required_tests(), _pytest_command, runner)


def main() -> None:
    problems = check()
    if problems:
        sys.exit("the semantic gates are unproven:\n  " + "\n  ".join(problems))
    sys.stdout.write(f"{len(required_tests())} semantic gate(s) ran and passed\n")


if __name__ == "__main__":
    main()
