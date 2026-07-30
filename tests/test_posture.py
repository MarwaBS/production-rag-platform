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

    assert "/docs" in {route.path for route in app.routes}
