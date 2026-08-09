"""What the checkers read before they will start a run, and what a report of
one has to say for the run to count as proven.

None of this reads the workflow file. These cover `scripts/gate_report.py` and
the two checkers over it: the tree a run may start from, the settings and
environment it is given, and whether the report it wrote accounts for every
test that was required of it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any, Dict, List

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def accepted_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """For the tests about what a report proves: the tree reading is a separate
    property, and while the suite runs the tree is whatever is being edited."""
    from scripts import gate_report

    monkeypatch.setattr(gate_report, "unaccepted_tree", list)


def _report(cases: str) -> str:
    return f"<testsuites><testsuite>{cases}</testsuite></testsuites>"


def _junitxml(argv: List[str]) -> pathlib.Path:
    option = next(arg for arg in argv if arg.startswith("--junitxml="))
    return pathlib.Path(option.split("=", 1)[1])


def _case(name: str, outcome: str = "") -> str:
    module, bare = name.split("::")
    body = f"<{outcome}/>" if outcome else ""
    return f'<testcase classname="{module}" name="{bare}">{body}</testcase>'


def _collected(*selection: str) -> set[str]:
    """What pytest itself says it would run, keyed as a report keys it."""
    listing = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *selection],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert listing.returncode == 0, listing.stdout[-400:]
    return {
        line.strip()
        .replace(chr(92), "/")
        .split("[")[0]
        .replace("/", ".")
        .replace(".py::", "::")
        for line in listing.stdout.splitlines()
        if "::" in line
    }


def test_the_report_check_covers_every_semantic_gate_in_the_suite() -> None:
    """pytest is the authority on which tests carry the marker, so ask it rather
    than a second copy of the same guess: a gate added later must be covered."""
    from scripts.check_semantic_report import required_tests

    marked = _collected("-m", "semantic")
    assert marked, "pytest collected no marked gates to check against"
    assert required_tests() == marked


def test_a_report_missing_a_gate_is_not_evidence_that_it_ran() -> None:
    """Collect-only, deselect, a swallowed exit status: each ends in a report
    that cannot account for one of the tests the job claims to have run."""
    from scripts.gate_report import verify

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
    from scripts.gate_report import verify

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
    assert options[options.index("-c") + 1] == "pyproject.toml", argv
    assert options[options.index("-m") + 1] == "semantic", argv
    assert f"--junitxml={report}" in options, argv


def test_the_checker_reads_only_a_report_the_run_it_started_wrote(
    accepted_tree: None,
) -> None:
    """A report the checker is handed can be committed, or written by a step
    that ran no tests, and copied into place. One it opens a private path for
    and then reads back cannot be."""
    from scripts.check_semantic_report import check, required_tests

    seen: Dict[str, Any] = {}

    def runner(argv: List[str], cwd: str, env: Dict[str, str]) -> int:
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
    assert check(runner=lambda argv, cwd, env: 1) == [
        "the run wrote no report (pytest exited 1)"
    ]


def test_the_run_is_started_with_nothing_loaded_its_command_did_not_name(
    monkeypatch: pytest.MonkeyPatch, accepted_tree: None
) -> None:
    """Reading back only what the run wrote settles who handed the report over,
    not what the run put in it. An installed plugin registers itself through an
    entry point and is given the same path to write and the same status to exit
    on; two variables add plugins and options to any run in the process."""
    from scripts.check_semantic_report import check, required_tests
    from scripts.gate_report import environment

    added = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "COVERAGE_RCFILE")
    for variable in added:
        monkeypatch.setenv(variable, "anything")
    cache = pathlib.Path("somewhere") / "bytecode"
    settings = environment(cache)
    assert settings["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1", settings
    assert settings["PYTHONNOUSERSITE"] == "1", settings
    # Bytecode left in the tree is read in place of the source beside it, so
    # the run is given a cache of its own outside the tree.
    assert settings["PYTHONPYCACHEPREFIX"] == str(cache), settings
    # Built from a named set, so what has to be listed is what survives.
    assert not [name for name in added if name in settings], sorted(settings)

    seen: Dict[str, Any] = {}

    def runner(argv: List[str], cwd: str, env: Dict[str, str]) -> int:
        seen["env"] = env
        _junitxml(argv).write_text(
            _report("".join(_case(name) for name in required_tests())), encoding="utf-8"
        )
        return 0

    assert check(runner=runner) == []
    assert seen["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1", "the run never got it"


def test_a_second_place_to_take_settings_from_is_read_before_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-p` in a settings file loads a plugin by name, which autoload being off
    does not touch, into the position a conftest holds; over the suite whose
    tests would otherwise be the thing objecting. So the settings are read from
    the tree too, and only from the file the command names."""
    from scripts import gate_report

    config = tmp_path / "pyproject.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setattr(gate_report, "REPO", tmp_path)
    monkeypatch.setattr(gate_report, "CONFIG", config)
    assert gate_report.unapproved_settings() == [
        "the settings the run would be given are not the pinned ones"
    ]

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "tox.ini").write_text("", encoding="utf-8")
    pinned = gate_report.PYTEST_CONFIG
    config.write_text(
        f"""[tool.pytest.ini_options]
addopts = {pinned["addopts"]!r}
markers = {pinned["markers"]!r}
""",
        encoding="utf-8",
    )
    assert gate_report.unapproved_settings() == [
        "pkg/tox.ini: a second place to take settings from"
    ]

    # The direction that carries the weight: present, and saying something else.
    # A settings file the run is given is refused for what it says, not for
    # existing, or the one option that loads a plugin would go through unread.
    (tmp_path / "pkg" / "tox.ini").unlink()
    poisoned = f"{pinned['addopts']} -p anything"
    config.write_text(
        f"""[tool.pytest.ini_options]
addopts = {poisoned!r}
markers = {pinned["markers"]!r}
""",
        encoding="utf-8",
    )
    assert gate_report.unapproved_settings() == [
        "the settings the run would be given are not the pinned ones"
    ]


def test_a_file_whose_content_left_the_index_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`status` trusts a stat cache, so an edit of equal size with the recorded
    time put back leaves it silent. What the index records is a hash, and the
    file on disk either hashes to it or does not."""
    from scripts import gate_report

    answers = {
        ("ls-files", "-s"): "100644 aaaa 0\tapp/main.py\n100644 bbbb 0\tapp/config.py"
    }
    monkeypatch.setattr(
        gate_report,
        "_git",
        lambda *arguments, **_: (
            # Hashed in sorted order: config.py, whose digest the index does
            # not record, then main.py, whose digest it does.
            "cccc aaaa" if arguments[0] == "hash-object" else answers.get(arguments, "")
        ),
    )
    monkeypatch.setattr(gate_report, "MANIFEST", frozenset())
    problems = gate_report.unaccepted_tree()
    assert "app/config.py: its content is not what the index records" in problems
    assert "app/main.py: its content is not what the index records" not in problems


def test_a_filter_between_a_file_and_its_hash_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashing applies whatever the repo has been told to apply, so a filter
    would sit between a file and the hash it is compared to. An attribute can
    select one from four places, two of them outside the repository as well as
    outside the tree, so git is asked which files carry one rather than the
    places an answer could have come from."""
    from scripts import gate_report

    monkeypatch.setattr(gate_report, "tracked_files", lambda: {"a.py", "b.py"})
    unspecified = "\0".join(
        ["a.py", "filter", "unspecified", "b.py", "filter", "unspecified"]
    )
    monkeypatch.setattr(gate_report, "_git", lambda *a, **k: unspecified)
    assert gate_report._filtered() == []

    selected = "\0".join(["a.py", "filter", "zap", "b.py", "filter", "unspecified"])
    monkeypatch.setattr(gate_report, "_git", lambda *a, **k: selected)
    assert gate_report._filtered() == ["a.py"]

    # And that the reading is wired into what refuses a run, not just readable.
    monkeypatch.setattr(gate_report, "_filtered", lambda: ["a.py"])
    monkeypatch.setattr(gate_report, "_changed_since_the_index", list)
    assert "a.py: a filter stands between it and the hash it is compared to" in (
        gate_report.unaccepted_tree()
    )


def test_a_file_the_index_stopped_looking_at_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`git status` honours a flag that asks it to ignore a tracked file, so a
    modified file can leave it silent. The manifest and the walk compare names,
    and the name does not change, which leaves the index's own view of each
    file as the only thing that still says the tree is the commit."""
    from scripts import gate_report

    flagged = ["h app/chunking.py", "H app/main.py"]
    answers = {("ls-files", "-v"): "\n".join(flagged)}
    monkeypatch.setattr(
        gate_report, "_git", lambda *arguments, **_: answers.get(arguments, "")
    )
    monkeypatch.setattr(gate_report, "MANIFEST", frozenset())
    problems = gate_report.unaccepted_tree()
    assert "app/chunking.py: the index was told to stop looking at it" in problems
    assert "app/main.py: the index was told to stop looking at it" not in problems


def test_what_the_tree_holds_is_read_before_the_run_it_would_report_on_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suite cannot police this one: a file that empties collection also
    removes the test that would have objected. Read from outside and before the
    run, what it costs is the run never starting."""
    from scripts import gate_report

    started: List[Any] = []

    def runner(argv: List[str], cwd: str, env: Dict[str, str]) -> int:
        started.append(argv)
        return 0

    monkeypatch.setattr(gate_report, "unaccepted_tree", lambda: ["conftest.py: x"])
    problems = gate_report.prove(
        {"tests.test_a::test_alpha"}, lambda report: ["pytest"], runner
    )
    assert problems == ["conftest.py: x"]
    assert started == [], "the run was started anyway"


def test_a_file_nobody_read_stops_the_run_whatever_it_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both directions have to bite: a file the repo carries that the manifest
    does not list, and a manifest entry the repo no longer carries."""
    from scripts import gate_report

    monkeypatch.setattr(
        gate_report, "MANIFEST", gate_report.MANIFEST | {"tests/__init__.py"}
    )
    assert "tests/__init__.py: in the manifest and no longer carried" in (
        gate_report.unaccepted_tree()
    )
    monkeypatch.setattr(gate_report, "MANIFEST", gate_report.MANIFEST - {"app/main.py"})
    assert (
        "app/main.py: carried and not in the manifest" in gate_report.unaccepted_tree()
    )


def test_a_file_the_repo_was_told_to_ignore_is_still_in_the_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git status` is silent about an ignored file, and the import system is
    not: one named in .gitignore sits on the same path as everything else. The
    walk runs over a tree of its own, so an interrupted run leaves nothing in
    this one for the next checker run to refuse."""
    from scripts import gate_report

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "ignored.key").write_text("", encoding="utf-8")
    # A cache name below the root is a directory that only looks like a cache,
    # so the names are pruned where caches live and not wherever they appear.
    (tmp_path / "pkg" / ".ruff_cache").mkdir()
    (tmp_path / "pkg" / ".ruff_cache" / "probe").write_text("", encoding="utf-8")
    # The same for a file: an ignored name below the root is still in the tree.
    (tmp_path / "pkg" / ".coverage").write_text("", encoding="utf-8")
    # A link is stepped over rather than into, so whatever is behind one would
    # be in the tree and unread. The answer is stubbed because creating one
    # needs a privilege the runner may not have.
    (tmp_path / "linked").mkdir()
    monkeypatch.setattr(gate_report, "_is_link", lambda path: path.name == "linked")
    monkeypatch.setattr(gate_report, "REPO", tmp_path)
    monkeypatch.setattr(gate_report, "tracked_files", set)
    monkeypatch.setattr(gate_report, "MANIFEST", frozenset())
    problems = gate_report.unaccepted_tree()
    assert "pkg/ignored.key: in the tree and not carried" in problems
    assert "pkg/.ruff_cache/probe: in the tree and not carried" in problems
    assert "pkg/.coverage: in the tree and not carried" in problems
    assert "linked: in the tree and not carried" in problems


def test_bytecode_left_in_the_tree_stops_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache entry whose header matches the source is loaded instead of it and
    runs whatever it was compiled from, with the source and every listing git
    keeps unchanged. Skipping the directory is what let it sit there."""
    from scripts import gate_report

    (tmp_path / "pkg" / gate_report._BYTECODE).mkdir(parents=True)
    monkeypatch.setattr(gate_report, "REPO", tmp_path)
    monkeypatch.setattr(gate_report, "tracked_files", set)
    monkeypatch.setattr(gate_report, "MANIFEST", frozenset())
    assert (
        f"pkg/{gate_report._BYTECODE}: bytecode read in place of the source beside it"
        in (gate_report.unaccepted_tree())
    )


def test_a_tracked_file_that_changed_stops_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest compares names, and a modified file keeps its name. This is
    the only reading that sees one, so it is the only thing behind the claim
    that the tree is the commit."""
    from scripts import gate_report

    monkeypatch.setattr(
        gate_report,
        "_git",
        lambda *arguments: " M app/main.py\n" if arguments[0] == "status" else "",
    )
    monkeypatch.setattr(gate_report, "MANIFEST", frozenset())
    assert "M app/main.py: in the tree and not in the commit" in (
        gate_report.unaccepted_tree()
    )


def test_a_startup_module_stops_the_run_before_a_command_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """These are imported by the interpreter itself, so the tree is not the only
    place one can come from; the environment the tools are installed into is."""
    from scripts import gate_report

    monkeypatch.setattr(gate_report, "_STARTUP_MODULES", ("os",))
    # The message names where it was found, so a refusal is diagnosable.
    assert any(
        problem.startswith("os: importable from ")
        for problem in gate_report.unaccepted_tree()
    )


def test_an_empty_marker_set_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing required means every report satisfies it; a gate that certifies
    an empty suite is the vacuous case this whole check exists to reject. The
    run has to write its report, or this passes on the missing file instead."""
    from scripts import check_semantic_report as checker

    def runner(argv: List[str], cwd: str, env: Dict[str, str]) -> int:
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


def test_the_suite_checker_requires_every_test_the_default_run_collects() -> None:
    """The checker reads its required set out of the source, so a test written
    in a form that reader does not recognise; inside a class, generated at
    import; would be absent from both sides and prove itself. pytest is the
    authority on what the run contains, so the two have to agree."""
    from scripts.check_suite_report import required_tests

    collected = _collected()
    assert collected, "pytest collected nothing to check against"
    assert required_tests() == collected


def test_between_them_the_two_checkers_require_every_test_the_repo_defines() -> None:
    """The suite checker subtracts the semantic gates because their own job
    proves them. Subtracting a name neither job requires would retire it."""
    from scripts.check_semantic_report import required_tests as semantic
    from scripts.check_suite_report import defined_tests, required_tests

    assert semantic(), "fixture: no semantic gate to subtract"
    assert required_tests() | semantic() == defined_tests()
    assert required_tests() & semantic() == set()


def test_a_parameter_set_is_proven_only_when_every_case_in_it_passed() -> None:
    """Parameter sets share one name in the source and appear once per case in
    the report. Matching the first case found lets the rest fail unnoticed."""
    from scripts.gate_report import verify

    required = {"tests.test_a::test_alpha"}
    passed = _case("tests.test_a::test_alpha[0]") + _case("tests.test_a::test_alpha[1]")
    assert verify(_report(passed), required) == []
    one_bad = _case("tests.test_a::test_alpha[0]") + _case(
        "tests.test_a::test_alpha[1]", "failure"
    )
    assert verify(_report(one_bad), required) == ["tests.test_a::test_alpha: failure"]


def test_a_run_that_passed_every_test_and_still_exited_nonzero_is_not_proven(
    accepted_tree: None,
) -> None:
    """The coverage floor fails the run without failing a test: every case in
    the report passed and pytest exits one. The report alone calls that proven."""
    from scripts.check_suite_report import check, required_tests

    def runner(argv: List[str], cwd: str, env: Dict[str, str]) -> int:
        _junitxml(argv).write_text(
            _report("".join(_case(name) for name in required_tests())), encoding="utf-8"
        )
        return 1

    assert check(runner=runner) == ["pytest exited 1"]


def test_the_suite_checker_runs_the_whole_suite_under_the_coverage_floor() -> None:
    """The command is the only part of the run the checker chooses. Selecting a
    marker narrows it; dropping the floor leaves the coverage claim ungated."""
    from scripts.check_suite_report import _pytest_command

    report = pathlib.Path("somewhere") / "report.xml"
    argv = _pytest_command(report)
    assert argv[0] == sys.executable and argv[2] == "pytest", argv
    options = argv[3:]
    assert "--cov=app" in options and "--cov-fail-under=93" in options, argv
    assert f"--junitxml={report}" in options, argv
    assert "-m" not in options, argv
    # Autoload is off for the run, so what it measures with has to be named,
    # and so does the file its settings come from.
    assert options[options.index("-p") + 1] == "pytest_cov", argv
    assert options[options.index("-c") + 1] == "pyproject.toml", argv


def test_the_readme_sells_the_floor_the_gate_enforces() -> None:
    """The constant and the README sentence must carry the same number."""
    import re

    from scripts.check_suite_report import COVERAGE_FLOOR

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # `an?` for the article (an 85, a 93); the anchor stops 93 matching in 193.
    assert re.search(rf"under an? {COVERAGE_FLOOR}% coverage floor", readme), (
        COVERAGE_FLOOR
    )
