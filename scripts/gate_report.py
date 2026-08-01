"""Prove a named set of tests ran and passed, from a report their own run wrote.

An exit status is not evidence: a collection-only run, a deselect, a conftest
that marks every item skipped, or a swallowed status all end at zero. A report
handed in is not evidence either — it can be committed, or written by a step
that ran nothing, and copied into place. So a check built on this starts the run
itself, into a directory it creates outside the tree, and reads back only what
that run wrote there.

Which leaves what the run itself can do to that report. Anything loaded into the
run is handed the path it writes and the status it exits on, so the run starts
with nothing loaded into it that its own command did not name.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
from typing import Callable, Dict, List, Set
from xml.etree import ElementTree

REPO = pathlib.Path(__file__).resolve().parent.parent

Command = Callable[[pathlib.Path], List[str]]
Runner = Callable[..., int]

_OUTCOMES = ("error", "failure", "skipped")

# Names that need no naming: import happens because the file is there.
_IMPORTED_ON_SIGHT = ("conftest.py", "sitecustomize.py", "usercustomize.py")

# Variables that add code or options to a run that did not ask for them.
_UNSET = (
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
)


def verify(report: str, required: Set[str]) -> List[str]:
    """Reasons the report fails to prove those tests passed; empty means proven.

    Keyed on module and name together, or a test of the same name in another
    module stands in for this one. A parameter set shares one name, so every
    case listed under it has to have passed.
    """
    cases: Dict[str, List[ElementTree.Element]] = {}
    for case in ElementTree.fromstring(report).iter("testcase"):
        bare = str(case.get("name")).split("[")[0]
        cases.setdefault(f"{case.get('classname')}::{bare}", []).append(case)
    problems = []
    for name in sorted(required):
        listed = cases.get(name, [])
        if not listed:
            problems.append(f"{name}: never ran")
            continue
        problems.extend(
            f"{name}: {outcome}"
            for outcome in _OUTCOMES
            if any(case.find(outcome) is not None for case in listed)
        )
    return problems


def tracked_test_files() -> List[pathlib.Path]:
    """The test files git carries — what a reviewer of this repo is shown, and
    the one listing both required sets are taken from."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "tests/test_*.py"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    return [REPO / name for name in listing.stdout.split("\0") if name]


def keyed(path: pathlib.Path, name: str) -> str:
    """A test keyed the way a report keys it: module path, then bare name."""
    module = path.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")
    return f"{module}::{name}"


def environment() -> Dict[str, str]:
    """The run's environment with every route that loads code the command did
    not name closed: installed plugins register themselves through entry points,
    and two variables add plugins and options to any run in the process."""
    settings = dict(os.environ)
    settings["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for variable in _UNSET:
        settings.pop(variable, None)
    return settings


def auto_loaded() -> List[str]:
    """Files in the tree that the run imports for being there and nothing else:
    pytest reads a conftest from any directory on the way to a test, and the
    interpreter imports these two before it reads the command at all."""
    return sorted(
        str(path.relative_to(REPO))
        for path in REPO.rglob("*")
        if path.name in _IMPORTED_ON_SIGHT and ".venv" not in path.parts
    )


def prove(required: Set[str], command: Command, runner: Runner) -> List[str]:
    """Run those tests and report whatever that run leaves unproven."""
    if not required:
        return ["no test is required: there is nothing to prove"]
    # Read here rather than asked of the run: a conftest is loaded into the run,
    # is handed the path it reports to, and can write whatever it likes there.
    loaded = auto_loaded()
    if loaded:
        return [
            f"{name}: loaded into the run that would be reporting" for name in loaded
        ]
    with tempfile.TemporaryDirectory() as scratch:
        # A path opened by this process outside the tree: whatever the repo or an
        # earlier step may hold, it is not what gets read back here.
        report = pathlib.Path(scratch) / "report.xml"
        status = runner(command(report), cwd=str(REPO), env=environment())
        if not report.exists():
            return [f"the run wrote no report (pytest exited {status})"]
        problems = verify(report.read_text(encoding="utf-8"), required)
        if status:
            # A floor the report does not carry — coverage — fails only here.
            problems.append(f"pytest exited {status}")
        return problems
