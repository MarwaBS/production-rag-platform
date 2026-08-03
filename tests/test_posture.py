"""A configuration the documentation calls unsafe must not start.

The chart's own comments say the data-plane is only authenticated when APP_API_KEY
is set. Leaving that to prose means the unsafe deployment is one forgotten value
away and nothing objects. The service already refuses to boot on a missing backend
package; an unauthenticated production deploy deserves the same treatment.
"""

from __future__ import annotations

from typing import Literal

import pytest

from app.config import Settings


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
def test_a_key_of_only_whitespace_is_refused_in_every_environment(
    env: Literal["development", "staging", "production"],
) -> None:
    """Stripped to empty it would read as no-auth-configured and open the
    data-plane. Refused in all three: the two that need no key are exactly
    where a blank goes unnoticed."""
    with pytest.raises(Exception, match="only whitespace"):
        Settings(env=env, api_key="   \n")


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
