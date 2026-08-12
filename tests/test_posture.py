"""A configuration the documentation calls unsafe must not start.

The chart's own comments say the data-plane is only authenticated when APP_API_KEY
is set. Leaving that to prose means the unsafe deployment is one forgotten value
away and nothing objects. The service already refuses to boot on a missing backend
package; an unauthenticated production deploy deserves the same treatment.
"""

from __future__ import annotations

import ast
from typing import Literal

import pytest

from app.config import Settings

from scripts.gate_report import SUBPROCESS_TIMEOUT_S


def test_production_without_an_api_key_refuses_to_start() -> None:
    with pytest.raises(Exception, match="API_KEY"):
        Settings(env="production", api_key="")


def test_production_with_an_api_key_starts() -> None:
    assert Settings(env="production", api_key="a-key").env == "production"


@pytest.mark.parametrize("env", ["development", "staging"])
def test_non_production_environments_stay_open_for_the_local_run(
    env: Literal["development", "staging"],
) -> None:
    assert Settings(env=env, api_key="").api_key == ""


@pytest.mark.parametrize("supplied", ["a-key\n", " a-key", "a-key\t", "\ra-key "])
def test_the_key_is_taken_without_the_whitespace_around_it(supplied: str) -> None:
    """Compared raw, a key carrying an invisible newline matches nothing a
    client can send, so every request 401s against a key correct in both."""
    assert Settings(env="development", api_key=supplied).api_key == "a-key"


@pytest.mark.parametrize("env", ["development", "staging", "production"])
@pytest.mark.parametrize(
    "blank",
    [
        "   \n",  # the ASCII spaces HTTP itself trims
        "\xa0",  # a pasted non-breaking space
        "　",  # an ideographic space
        " ",  # a figure space
        "\x0b",  # a vertical tab
        "\x00",  # a NUL, which no strip() treats as whitespace
    ],
)
def test_a_key_with_nothing_visible_in_it_is_refused_everywhere(
    env: Literal["development", "staging", "production"], blank: str
) -> None:
    """Read as no-auth-configured it would open the data-plane, and an operator
    reading the config sees a key. Refused in all three environments: the two
    that need no key are exactly where a blank one goes unnoticed."""
    with pytest.raises(Exception, match="no visible character"):
        Settings(env=env, api_key=blank)


def test_a_space_that_is_not_a_header_space_stays_part_of_the_key() -> None:
    """Only what HTTP itself trims is removed. Stripping every unicode space
    would make two distinct configured keys the same key."""
    assert Settings(env="development", api_key="\xa0secret").api_key == "\xa0secret"


def test_production_does_not_serve_the_interactive_docs() -> None:
    """/docs, /redoc and /openapi.json hand any ingress visitor the full API
    map. They are development conveniences; production must not mount them,
    while /metrics stays up for the in-cluster scrape. Runs in a subprocess
    because the app is built from settings at import time."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    child = (
        "import json\n"
        "import app.main\n"
        "paths = {r.path for r in app.main.app.routes}\n"
        "print(json.dumps(sorted(paths & {'/docs', '/redoc', '/openapi.json', '/metrics'})))\n"
    )
    env = {**os.environ, "APP_ENV": "production", "APP_API_KEY": "posture-probe-key"}
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr}"
    mounted = json.loads(proc.stdout.strip().splitlines()[-1])
    assert mounted == ["/metrics"], (
        f"production mounts {mounted}; only /metrics belongs there"
    )


def test_development_keeps_the_interactive_docs() -> None:
    """The control for the gate above: losing /docs everywhere would satisfy a
    production-only assertion by accident."""
    from app.main import app

    assert "/docs" in {getattr(route, "path", "") for route in app.routes}


def _spellings(tree: ast.Module, exported: set[str]) -> set[str]:
    """How this file can name those members. Read from its own imports, so an
    aliased module or a bare `from subprocess import run` is not missed.

    Seeded with the plain name so the dotted spelling stays covered whether or
    not this file's own import is the one that bound it.
    """
    modules: set[str] = {"subprocess"}
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "subprocess"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            bare.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in exported
            )
    return {f"{module}.{name}" for module in modules for name in exported} | bare


def _has_deadline(call: ast.Call) -> bool:
    """`timeout=None` restores the unbounded default and `timeout=0` is a
    deadline no child can meet, so neither is a real bound and the
    keyword's presence is not the property wanted.

    A `**kwargs` splat carries no `arg`, so a spawn is flagged even when the
    mapping holds a timeout; write the deadline at the call site."""
    for word in call.keywords:
        if word.arg == "timeout":
            return not (isinstance(word.value, ast.Constant) and not word.value.value)
    return False


def _names_bound_to(tree: ast.Module, spawns: set[str]) -> set[str]:
    """Local names holding a spawn, from a parameter default or an assignment.

    A bare `from subprocess import run` binds an `ast.Name`, not an
    `ast.Attribute`, so both spellings are read.
    """
    held = (ast.Attribute, ast.Name)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            taking = node.args.posonlyargs + node.args.args
            given = node.args.defaults
            pairs = list(zip(taking[len(taking) - len(given) :], given, strict=True))
            pairs += [
                (arg, default)
                for arg, default in zip(
                    node.args.kwonlyargs, node.args.kw_defaults, strict=True
                )
                if default is not None
            ]
            names.update(
                arg.arg
                for arg, default in pairs
                if isinstance(default, held) and ast.unparse(default) in spawns
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # An annotated binding is a different node, and reads identically.
            bound = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, held) and ast.unparse(node.value) in spawns:
                names.update(t.id for t in bound if isinstance(t, ast.Name))
    return names


def test_every_subprocess_call_is_bounded_by_a_timeout() -> None:
    """A child with no deadline hangs the job to its six-hour ceiling.

    Of the seven spawns `subprocess` exports, `getoutput`, `getstatusoutput`
    and `Popen` accept no `timeout`, so they are refused rather than waved
    through by a rule they cannot satisfy. A spawn reaching its call site as a
    value is read through the name it is bound to; one handed to another
    module is not, and `prove`'s runner is held by the fakes instead.
    """
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent

    assert SUBPROCESS_TIMEOUT_S > 0, SUBPROCESS_TIMEOUT_S
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    takes_timeout = {"run", "call", "check_call", "check_output"}
    takes_none = {"getoutput", "getstatusoutput", "Popen"}
    unbounded: list[str] = []
    refused: list[str] = []
    for name in [n for n in listing.stdout.split("\0") if n]:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        bounded = _spellings(tree, takes_timeout)
        unboundable = _spellings(tree, takes_none)
        aliases = _names_bound_to(tree, bounded)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if (callee in bounded or callee in aliases) and not _has_deadline(node):
                    unbounded.append(f"{name}:{node.lineno} {callee}")
            # Reading the name rather than the call covers a refused spawn
            # whether it is called or handed on, and reports it once.
            elif isinstance(node, (ast.Name, ast.Attribute)):
                if ast.unparse(node) in unboundable:
                    refused.append(f"{name}:{node.lineno} {ast.unparse(node)}")
    assert not unbounded, unbounded
    assert not refused, refused
