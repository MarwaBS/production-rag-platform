"""Prove a named set of tests ran and passed, from a report their own run wrote.

An exit status is not evidence: a collection-only run, a deselect, a conftest
that marks every item skipped, or a swallowed status all end at zero. A report
handed in is not evidence either — it can be committed, or written by a step
that ran nothing, and copied into place. So a check built on this starts the run
itself, into a directory it creates outside the tree, and reads back only what
that run wrote there.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Callable, Dict, List, Set
from xml.etree import ElementTree

REPO = pathlib.Path(__file__).resolve().parent.parent

Command = Callable[[pathlib.Path], List[str]]
Runner = Callable[..., int]

_OUTCOMES = ("error", "failure", "skipped")


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


def prove(required: Set[str], command: Command, runner: Runner) -> List[str]:
    """Run those tests and report whatever that run leaves unproven."""
    if not required:
        return ["no test is required: there is nothing to prove"]
    with tempfile.TemporaryDirectory() as scratch:
        # A path opened by this process outside the tree: whatever the repo or an
        # earlier step may hold, it is not what gets read back here.
        report = pathlib.Path(scratch) / "report.xml"
        status = runner(command(report), cwd=str(REPO))
        if not report.exists():
            return [f"the run wrote no report (pytest exited {status})"]
        problems = verify(report.read_text(encoding="utf-8"), required)
        if status:
            # A floor the report does not carry — coverage — fails only here.
            problems.append(f"pytest exited {status}")
        return problems
