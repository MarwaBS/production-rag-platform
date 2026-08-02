"""Prove a named set of tests ran and passed, from a report their own run wrote.

An exit status is not evidence: a collection-only run, a deselect, a conftest
that marks every item skipped, or a swallowed status all end at zero. A report
handed in is not evidence either — it can be committed, or written by a step
that ran nothing, and copied into place. So a check built on this starts the run
itself, into a directory it creates outside the tree, and reads back only what
that run wrote there.

A run begins only from the files the repo carries, the settings it is given and
the variables it starts with, which leaves the code the tests import — the thing
under review.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import tempfile
import tomllib
from typing import Callable, Dict, List, Set, Tuple
from xml.etree import ElementTree

REPO = pathlib.Path(__file__).resolve().parent.parent

Command = Callable[[pathlib.Path], List[str]]
Runner = Callable[..., int]

_OUTCOMES = ("error", "failure", "skipped")

CONFIG = REPO / "pyproject.toml"

# Every file the repo carries. A run can execute anything here and nothing
# that is not, so naming the whole set closes the names nobody has thought of.
MANIFEST = frozenset(
    {
        ".github/workflows/ci.yml",
        ".gitignore",
        "LICENSE",
        "README.md",
        "app/__init__.py",
        "app/chunking.py",
        "app/config.py",
        "app/embedder.py",
        "app/main.py",
        "chunking_derivation.json",
        "constraints-dev.txt",
        "deploy/Dockerfile",
        "deploy/docker-compose.yml",
        "deploy/helm/Chart.yaml",
        "deploy/helm/templates/deployment.yaml",
        "deploy/helm/templates/hpa.yaml",
        "deploy/helm/templates/ingress.yaml",
        "deploy/helm/templates/pdb.yaml",
        "deploy/helm/templates/prometheusrule.yaml",
        "deploy/helm/templates/secret.yaml",
        "deploy/helm/templates/service.yaml",
        "deploy/helm/templates/serviceaccount.yaml",
        "deploy/helm/templates/servicemonitor.yaml",
        "deploy/helm/values.yaml",
        "docs/ci-cd-pipeline.yml",
        "docs/decisions/001-faiss-over-managed-vector-db.md",
        "docs/decisions/002-pre-grounding-over-post-filtering.md",
        "docs/decisions/003-circuit-breaker-for-llm-resilience.md",
        "docs/decisions/004-vendor-neutral-llm-protocol.md",
        "eval_floors_derivation.json",
        "evals/__init__.py",
        "evals/__main__.py",
        "evals/harness.py",
        "pyproject.toml",
        "scale_cliff_derivation.json",
        "scripts/__init__.py",
        "scripts/check_semantic_report.py",
        "scripts/check_suite_report.py",
        "scripts/derive_chunking.py",
        "scripts/derive_eval_floors.py",
        "scripts/derive_scale_cliff.py",
        "scripts/gate_report.py",
        "tests/test_app.py",
        "tests/test_chunking.py",
        "tests/test_ci_pipeline.py",
        "tests/test_gate_report.py",
        "tests/test_deploy_posture.py",
        "tests/test_embedder.py",
        "tests/test_eval.py",
        "tests/test_generation_safety.py",
        "tests/test_grounding_is_gated.py",
        "tests/test_input_bounds.py",
        "tests/test_observability.py",
        "tests/test_posture.py",
        "tests/test_retrieval_honesty.py",
        "tests/test_scale.py",
    }
)

# Git's own store and what the toolchain writes AT THE ROOT, which is where
# they live. Pruning the names at any depth hides a directory that only looks
# like one of them; bytecode is refused outright rather than skipped anywhere.
_BYTECODE = "__pycache__"
_NOT_SOURCE = frozenset(
    {".coverage", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
)

# Where pytest would take its settings from if the command named nowhere.
_OTHER_SETTINGS = ("pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg")

# What the file the command does name has to say. Read here and not by a test,
# because what a settings file loads through `-p` is a plugin, holding the
# position a conftest holds, over the suite that test would have been in.
PYTEST_CONFIG = {
    "addopts": "-ra -m 'not semantic'",
    "markers": [
        "semantic: needs a semantic embedder; not satisfiable by the shipped one"
    ],
}

# Built, not filtered: a variable that survives is one somebody read. PATH
# resolves the git every reading runs, TEMP and TMP decide where the report is
# written, four resolve the home holding the cached model, three start a shell.
# Ten names, and the list holds ten.
_INHERITED = (
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
)
_IMPOSED = {
    # Entry points are how an installed distribution loads itself into a run.
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    # The user site directory is where a usercustomize module would come from.
    "PYTHONNOUSERSITE": "1",
    # Docstrings and assertion text carry em dashes, which the console
    # default on this platform cannot encode.
    "PYTHONIOENCODING": "utf-8",
}

# Modules the interpreter imports at startup, before it reads any command.
_STARTUP_MODULES = ("sitecustomize", "usercustomize")


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


def _git(*arguments: str, stdin: str = "") -> str:
    return subprocess.run(
        ["git", *arguments],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    ).stdout


def _filtered() -> List[str]:
    """Tracked files a filter would be applied to when they are hashed.

    Asked of git rather than of the places an attribute can be written: there
    are four, two of them outside both the tree and the repository, and naming
    them is how the last three attempts at this were walked around."""
    names = sorted(tracked_files())
    if not names:
        return []
    answer = _git(
        "check-attr", "--stdin", "-z", "filter", stdin="\0".join(names) + "\0"
    ).split("\0")
    # Triples of path, attribute, value; anything but unspecified is a filter.
    return sorted(
        {
            answer[start]
            for start in range(0, len(answer) - 2, 3)
            if answer[start + 2] != "unspecified"
        }
    )


def _changed_since_the_index() -> List[str]:
    """Tracked files whose content differs from what the index records.

    Compared by hash and not by `status`, which trusts a stat cache: an edit of
    equal size with the recorded time put back leaves it silent."""
    recorded = {}
    for line in _git("ls-files", "-s").splitlines():
        details, _, name = line.partition("\t")
        fields = details.split()
        if not name or len(fields) < 2:
            return [f"the index listing is not what it should be: {line[:40]!r}"]
        recorded[name] = fields[1]
    names = sorted(recorded)
    if not names:
        return []
    # Hashed the way the index was written, line endings and all. What could
    # stand between the two is a clean filter, refused rather than bypassed.
    hashed = _git("hash-object", "--stdin-paths", stdin="\n".join(names)).split()
    if len(hashed) != len(names):
        # Pairing a short list against a long one compares a file to another
        # file's hash, which reads as a difference where there is none.
        return [f"{len(names)} files listed and {len(hashed)} hashed"]
    return [name for name, digest in zip(names, hashed) if digest != recorded[name]]


def tracked_files() -> Set[str]:
    """Every path the repo carries, as git spells it."""
    return {name for name in _git("ls-files", "-z").split("\0") if name}


def tracked_test_files() -> List[pathlib.Path]:
    """The test files git carries — the one listing both required sets are
    taken from, so neither can be narrowed without the other noticing."""
    return sorted(
        REPO / name for name in tracked_files() if name.startswith("tests/test_")
    )


def _is_link(path: pathlib.Path) -> bool:
    """Whether os.walk will step over this rather than into it."""
    return path.is_symlink() or path.is_junction()


def _walk_tree() -> Tuple[List[str], List[str]]:
    """The files under the repo, the links among them, and the bytecode
    directories, all spelled the way git spells a path. Pruning rather than
    filtering, because the installed environment alone holds tens of thousands
    of files."""
    files: List[str] = []
    bytecode: List[str] = []
    linked: List[str] = []
    for directory, names, found in os.walk(REPO):
        here = pathlib.Path(directory).relative_to(REPO)
        bytecode += [(here / name).as_posix() for name in names if name == _BYTECODE]
        root = here == pathlib.Path(".")
        # os.walk neither descends a link nor lists it as a file, so whatever
        # is on the far side of one would be in the tree and unread.
        linked += [
            (here / name).as_posix()
            for name in names
            if _is_link(pathlib.Path(directory) / name)
        ]
        names[:] = [
            name
            for name in names
            if name != _BYTECODE and not (root and name in _NOT_SOURCE)
        ]
        files += [
            (here / name).as_posix()
            for name in found
            if not (root and name in _NOT_SOURCE)
        ]
    return sorted(files + linked), sorted(bytecode)


def keyed(path: pathlib.Path, name: str) -> str:
    """A test keyed the way a report keys it: module path, then bare name."""
    module = path.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")
    return f"{module}::{name}"


def environment(cache: pathlib.Path) -> Dict[str, str]:
    """The environment the run starts with, built rather than filtered.

    The cache directory is the checker's own, so the run neither reads bytecode
    left in the tree nor leaves any there for the next one to read."""
    settings = {name: os.environ[name] for name in _INHERITED if name in os.environ}
    settings.update(_IMPOSED)
    settings["PYTHONPYCACHEPREFIX"] = str(cache)
    return settings


def unaccepted_tree() -> List[str]:
    """Whatever the run could execute that nobody has read.

    The tree is the commit and the commit is the manifest, so a file reaches a
    run only by appearing in a diff — whatever it is called, and whether it is
    imported at startup, collected, or loaded as a plugin."""
    problems = [
        f"{line.strip()}: in the tree and not in the commit"
        for line in _git("status", "--porcelain").splitlines()
        if line.strip()
    ]
    # `status` honours the index flags that ask it to stop looking at a file,
    # so the flags are read too: anything but H is a file it stopped tracking.
    problems += [
        f"{line[2:]}: the index was told to stop looking at it"
        for line in _git("ls-files", "-v").splitlines()
        if line and line[0] != "H"
    ]
    problems += [
        f"{name}: a filter stands between it and the hash it is compared to"
        for name in _filtered()
    ]
    problems += [
        f"{name}: its content is not what the index records"
        for name in _changed_since_the_index()
    ]
    tracked = tracked_files()
    problems += [
        f"{name}: carried and not in the manifest"
        for name in sorted(tracked - MANIFEST)
    ]
    problems += [
        f"{name}: in the manifest and no longer carried"
        for name in sorted(MANIFEST - tracked)
    ]
    files, bytecode = _walk_tree()
    problems += [
        f"{name}: in the tree and not carried" for name in files if name not in tracked
    ]
    problems += [
        f"{name}: importable from {found.origin}, and imported before any command"
        for name in _STARTUP_MODULES
        for found in [importlib.util.find_spec(name)]
        if found is not None
    ]
    # Bytecode whose header matches the source is loaded instead of it, and
    # carries whatever it was compiled from. The run is pointed at a cache
    # outside the tree; what is left here would still reach this process.
    problems += [
        f"{name}: bytecode read in place of the source beside it" for name in bytecode
    ]
    return problems


def unapproved_settings() -> List[str]:
    """Anything deciding how the run behaves other than its own command and the
    one file that command names."""
    problems = [
        f"{name}: a second place to take settings from"
        for name in _walk_tree()[0]
        if pathlib.PurePosixPath(name).name in _OTHER_SETTINGS
    ]
    settings = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    if settings.get("tool", {}).get("pytest", {}).get("ini_options") != PYTEST_CONFIG:
        problems.append("the settings the run would be given are not the pinned ones")
    return problems


def prove(required: Set[str], command: Command, runner: Runner) -> List[str]:
    """Run those tests and report whatever that run leaves unproven."""
    if not required:
        return ["no test is required: there is nothing to prove"]
    # Read here and not asked of the run: each of these decides what the run
    # loads, and what it loads is handed the report the run is judged by.
    refusals = unaccepted_tree() + unapproved_settings()
    if refusals:
        return refusals
    with tempfile.TemporaryDirectory() as scratch:
        # A path opened by this process outside the tree: whatever the repo or an
        # earlier step may hold, it is not what gets read back here.
        report = pathlib.Path(scratch) / "report.xml"
        cache = pathlib.Path(scratch) / "bytecode"
        status = runner(command(report), cwd=str(REPO), env=environment(cache))
        if not report.exists():
            return [f"the run wrote no report (pytest exited {status})"]
        problems = verify(report.read_text(encoding="utf-8"), required)
        if status:
            # A floor the report does not carry — coverage — fails only here.
            problems.append(f"pytest exited {status}")
        return problems
