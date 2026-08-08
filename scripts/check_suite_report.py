"""Run the whole suite and prove every test in it passed.

A green run cannot say which tests it ran. A conftest that marks every item
skipped leaves the files collected, the run green, and the coverage floor met on
whatever still executes; a deselect does the same more quietly. So the required
set is read from the source of the tracked test files, and the report of the run
has to account for every name in it.

Run: python -B -m scripts.check_suite_report
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import List, Set

from scripts.check_semantic_report import required_tests as semantic_tests
from scripts.gate_report import Runner, keyed, prove, tracked_test_files

# Column zero: a test defined inside a class is reported under the class, and
# this key would not match it. The suite holds itself to the form this reads.
_DEFINED = re.compile(r"^def (test_\w+)", re.MULTILINE)

# The floor the README sells. Set below measured coverage rather than at it,
# so an unrelated change does not trip it; pytest-cov rounds to whole percent.
COVERAGE_FLOOR = "93"


def defined_tests() -> Set[str]:
    return {
        keyed(path, name)
        for path in tracked_test_files()
        for name in _DEFINED.findall(path.read_text(encoding="utf-8"))
    }


def required_tests() -> Set[str]:
    """What the default run selects: the semantic gates are deselected there and
    proven by their own job, against a report of the same shape."""
    return defined_tests() - semantic_tests()


def _pytest_command(report: pathlib.Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        # The settings this run gets are the ones read above, not whatever
        # file happens to be found first.
        "-c",
        "pyproject.toml",
        # Named, because the run loads no plugin it was not told to load.
        "-p",
        "pytest_cov",
        "--cov=app",
        f"--cov-fail-under={COVERAGE_FLOOR}",
        f"--junitxml={report}",
    ]


def check(runner: Runner = subprocess.call) -> List[str]:
    return prove(required_tests(), _pytest_command, runner)


def main() -> None:
    problems = check()
    if problems:
        sys.exit("the suite is unproven:\n  " + "\n  ".join(problems))
    sys.stdout.write(
        f"{len(required_tests())} test(s) defined in the suite ran and passed\n"
    )


if __name__ == "__main__":
    main()
