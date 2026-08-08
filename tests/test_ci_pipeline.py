"""The CI gates the README advertises must exist in the pipeline definition.

These read the workflow file; they do not run it. What they establish is that
the file describes exactly the steps below and no others. What no reading of the
file can establish is that the runner behaved, or that the tools those steps
invoke behave as their names suggest — only the pipeline executing does that,
which is why the deploy-posture file makes the same trade.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import pytest
import yaml

from scripts.gate_report import PYTEST_CONFIG

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The whole run line, not a substring of it. `: python scripts/...` contains the
# substring and executes nothing; the checker takes no argument so that the
# assertion below can be an equality rather than a search.
PROVE_SEMANTIC = "python -B -m scripts.check_semantic_report"
PROVE_SUITE = "python -B -m scripts.check_suite_report"
PROVE_BACKENDS = (
    "tests/test_app.py::test_every_permitted_backend_starts_or_names_its_missing_extra"
)


def _ci() -> Dict[Any, Any]:
    # Not str-keyed: YAML 1.1 reads the bare `on` trigger key as a boolean.
    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )


def _runs(job: Dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


# What the gates read besides the workflow: an exclude in one of these, or an
# addopts deselect, leaves the pinned command unchanged and looking at nothing.
TOOL_CONFIG = {
    # Pinned where the checker reads it before starting a run, not in a copy
    # that a plugin loaded by the value being pinned would have deleted.
    "pytest": {"ini_options": PYTEST_CONFIG},
    "hatch": {"build": {"targets": {"wheel": {"packages": ["app"]}}}},
    "mypy": {
        "python_version": "3.12",
        "warn_unused_configs": True,
        "check_untyped_defs": True,
        "overrides": [
            {
                "module": ["rag_llm_infra.*", "sentence_transformers.*"],
                "ignore_missing_imports": True,
            }
        ],
    },
    "ruff": {
        "target-version": "py312",
        "lint": {
            "select": ["F", "E"],
            "ignore": ["E501"],
            "per-file-ignores": {"tests/**": ["F401", "F841"]},
        },
    },
}


def test_the_tools_the_gates_run_are_configured_as_pinned() -> None:
    """A pinned command is only as strong as what it is pointed at."""
    import tomllib

    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tools = settings["tool"]
    for name, expected in TOOL_CONFIG.items():
        assert tools.get(name) == expected, (name, tools.get(name))
    assert set(tools) == set(TOOL_CONFIG), sorted(tools)
    # A distribution that registers one is a plugin pytest loads by name.
    declared = {"entry-points", "scripts", "gui-scripts"} & set(settings["project"])
    assert declared == set(), sorted(declared)
    # pip imports the backend named here, and runs its hooks, before any gate.
    assert settings["build-system"] == {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    }, settings["build-system"]


def test_no_file_in_the_tree_reconfigures_a_tool_a_gate_runs() -> None:
    """The pinned sections are only the config the tools would read if these
    were absent. Each of them either outranks a pinned section or replaces it,
    and none is named on any command line, so none shows up in the workflow.
    Ruff resolves per file up the directory chain, so which pyproject.toml this
    is matters as much as the name: only the root one is the pinned one."""
    present = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.name in TOOL_CONFIG_FILES
        and path.relative_to(ROOT).parts[0] != ".venv"
        and path != ROOT / "pyproject.toml"
    ]
    assert present == [], present


# Every place a tracked file tells a gate to look away, one form per gate that
# reads one. Adding a place is how a live finding leaves the report with the
# command and the config both still correct, so the places are pinned. Not all
# of them are comments: the type checker takes a decorator too.
DIRECTIVES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"#\s*noqa(:\s*[A-Z0-9, ]+)?",
        r"#\s*type:\s*ignore(\[[\w, -]+\])?",
        r"#\s*(ruff|flake8|mypy)\s*:\s*[^\n]*",
        r"#\s*fmt:\s*(off|on|skip)",
        r"#\s*pragma[:\s]\s*no\s*\w+",
    )
)

# Every decorator the repo uses. A waiver can be spelled as any name that
# resolves to one, and after `@` python takes any expression at all, so the
# names are read from the parse tree rather than matched against a surface form.
DECORATORS = frozenset(
    {
        "@app.get",
        "@app.post",
        "@asynccontextmanager",
        "@dataclass",
        "@model_validator",
        "@pytest.fixture",
        "@pytest.mark.parametrize",
        "@pytest.mark.semantic",
    }
)


def _decorators() -> set[str]:
    """Every decorator applied anywhere in tracked python, named the way it is
    written. A call is reduced to what is being called, so a parametrised mark
    reads as the mark."""
    applied = set()
    for name in _tracked("*.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            for applied_to in node.decorator_list:
                target = (
                    applied_to.func if isinstance(applied_to, ast.Call) else applied_to
                )
                applied.add(f"@{ast.unparse(target)}")
    return applied


SUPPRESSIONS = {
    "scripts/derive_chunking.py": ("noqa:E402",),
    "scripts/derive_eval_floors.py": ("noqa:E402", "noqa:E402"),
    "scripts/derive_scale_cliff.py": ("noqa:E402",),
    "tests/test_app.py": ("noqa:F401",),
    "tests/test_embedder.py": ("type:ignore[arg-type]",),
    "tests/test_retrieval_honesty.py": ("noqa:E731",),
}


def _tracked(pattern: str) -> List[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z", pattern],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    return [name for name in listing.stdout.split("\0") if name]


def _suppressions() -> Dict[str, Any]:
    found = {}
    for name in _tracked("*.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        told = tuple(
            sorted(
                "".join(match.group(0).lstrip("#").split())
                for pattern in DIRECTIVES
                for match in pattern.finditer(text)
            )
        )
        if told:
            found[name] = told
    return found


def test_every_spelling_a_gate_accepts_is_one_this_reader_matches() -> None:
    """Both of these are honoured — the linter takes the waiver in either case,
    and coverage takes the pragma with the colon left out. Neither shows up in
    the pinned set unless the patterns above are as wide as the tools are."""
    # Built rather than written out: a waiver spelled here would be a waiver
    # this file is holding, and the pinned set above would have to carry it.
    waived = ("NOQA", "noqa: F401", "pragma no cover", "PRAGMA: NO COVER")
    for text in waived:
        assert any(pattern.search(f"# {text}") for pattern in DIRECTIVES), text


def test_every_decorator_in_the_tree_is_one_that_has_been_read() -> None:
    """A decorator waives what a comment does and leaves no comment behind, and
    the name it is spelled with need not be the name it resolves to. Reading
    the names that are here beats matching the spellings a reader thought of."""
    found = _decorators()
    assert found == DECORATORS, sorted(found ^ DECORATORS)


def test_the_places_a_gate_is_told_to_look_away_are_the_pinned_ones() -> None:
    """A directive in the file beats every check on the command and the config:
    one line at the top of a module and its whole file stops being judged, with
    the linter still opening it and the type checker still counting it."""
    assert _suppressions() == SUPPRESSIONS, _suppressions()


def test_the_linter_opens_every_python_file_the_repo_tracks() -> None:
    """Pinning a config section proves what it says, not what the file list the
    run walks turns out to be: a file dropped from that list is one `ruff check
    .` reports nothing about. Ask ruff for the list. What this does not answer
    is whether every file on it is judged — a per-file ignore leaves a file
    listed and unjudged, which is why the config sections are pinned too."""
    listed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--show-files"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert listed.returncode == 0, listed.stderr[-400:]
    examined = {
        pathlib.Path(line.strip()).resolve()
        for line in listed.stdout.splitlines()
        if line.strip()
    }
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], capture_output=True, text=True, cwd=str(ROOT)
    )
    expected = {(ROOT / name).resolve() for name in tracked.stdout.split("\0") if name}
    assert expected, "fixture: git listed no tracked python files"
    assert expected <= examined, sorted(str(path) for path in expected - examined)


def test_the_type_checker_still_reports_an_error_under_the_config_it_resolves() -> None:
    """The same for mypy, which has no list of files to ask for. A config that
    silences every error leaves the pinned command exiting zero over code it has
    stopped judging, and mypy resolves that config from the directory it runs
    in, where a file outranks the pinned section. So give it one that must
    fail."""
    with tempfile.TemporaryDirectory() as scratch:
        probe = pathlib.Path(scratch) / "probe.py"
        probe.write_text(
            "def probe() -> int:\n    return 'not an int'\n", encoding="utf-8"
        )
        judged = subprocess.run(
            [sys.executable, "-m", "mypy", str(probe)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
    assert judged.returncode != 0, judged.stdout[-400:]
    assert "return-value" in judged.stdout, judged.stdout[-400:]


def test_no_stub_stands_in_for_a_module_the_type_checker_would_judge() -> None:
    """mypy reads a .pyi in place of the module beside it, so an error in that
    module stops being reported with nothing in it changed and no directive in
    either file. This project declares its types where the code is."""
    stubs = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.pyi")
        if path.relative_to(ROOT).parts[0] != ".venv"
    ]
    assert stubs == [], stubs


def test_every_tracked_test_file_is_collected() -> None:
    """The suite that runs has to be the suite that exists. A conftest
    naming files in collect_ignore drops them with nothing else changing: the
    run is green and the coverage floor is still met on what remains."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "tests/test_*.py"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    expected = {path for path in tracked.stdout.split(chr(0)) if path}
    assert expected, "fixture: git listed no tracked test files"
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert collected.returncode == 0, collected.stdout[-400:]
    seen = {
        line.split("::")[0].replace(chr(92), "/")
        for line in collected.stdout.splitlines()
        if "::" in line
    }
    assert expected <= seen, sorted(expected - seen)


def _top_level(listing: str) -> set[str]:
    """What each tracked path must be reachable as on a type-check command line:
    its package for a file in one, the file itself for one at the root."""
    paths = [pathlib.PurePosixPath(path) for path in listing.split("\0") if path]
    return {path.parts[0] if len(path.parts) > 1 else path.name for path in paths}


def test_the_type_check_argument_reader_does_not_overlook_a_root_level_file() -> None:
    """Today every tracked module sits in a package, so a reader that silently
    dropped root-level files would look right up until one appeared."""
    listing = "app/main.py\0conftest.py\0tests/test_x.py\0a dir/mod.py\0"
    assert _top_level(listing) == {"app", "conftest.py", "tests", "a dir"}


def test_ci_type_checks_every_directory_that_ships_python() -> None:
    """Naming the directories that ship python beats naming the ones that
    happened to exist when the command line was written."""
    everything = "\n".join(_runs(job) for job in _ci()["jobs"].values())
    invocation = [
        line.strip()
        for line in everything.splitlines()
        if line.strip().startswith("mypy ")
    ]
    assert invocation, "no run line invokes mypy"
    checked = {argument for line in invocation for argument in line.split()[1:]}
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], capture_output=True, text=True, cwd=str(ROOT)
    )
    shipping = _top_level(tracked.stdout)
    assert shipping, "fixture: git listed no tracked python files"
    assert shipping <= checked, sorted(shipping - checked)


WORKFLOW_KEYS = {"name", True, "jobs"}
JOB_KEYS = {"runs-on", "steps", "needs", "permissions", "env"}
JOB_ENV = {"IMAGE"}
STEP_KEYS = {"name", "run", "uses", "with"}
PUBLISH_STEP_KEYS = STEP_KEYS | {"if", "id"}


def _unvetted_keys(workflow: Dict[Any, Any], job: Dict[str, Any]) -> List[str]:
    """Keys in this job that nobody has read and accepted.

    Listing the keys that can neuter a step is a blacklist over a set with no
    edge: a shell, a default, a condition, a working directory and a step-level
    env each do it differently. The keys GitHub defines are a closed set, so the
    sound direction is the other one: allow what is used here and refuse the
    rest, which forces the next key to be read before it is adopted.
    """
    problems = [f"workflow: {key}" for key in workflow if key not in WORKFLOW_KEYS]
    problems += [f"job: {key}" for key in job if key not in JOB_KEYS]
    # Job-level env reaches every step in the job and is strictly stronger than
    # the step-level form refused below. It cannot touch the checker's own run,
    # which is handed a built environment — it reaches the steps beside it.
    problems += [
        f"job env: {key}" for key in job.get("env") or {} if key not in JOB_ENV
    ]
    for step in job.get("steps", []):
        allowed = PUBLISH_STEP_KEYS if step.get("if") else STEP_KEYS
        where = step.get("name") or step.get("uses") or step.get("run")
        problems += [f"{where}: {key}" for key in step if key not in allowed]
    return problems


def test_ci_is_triggered_by_the_events_that_deliver_code() -> None:
    """The most complete mask is not a key on a step. A workflow no push and no
    pull request triggers never runs at all, and every gate in this file still
    passes. YAML reads a bare `on` as the boolean it spells."""
    triggers = _ci()[True]
    assert "pull_request" in triggers, sorted(triggers)
    # Naming the events is not enough: the filter keys under them are a closed
    # set too, and any one can narrow the trigger down to nothing. `branches` is
    # the exception only where the branch it must keep is checked as well.
    filters = {
        "branches-ignore",
        "paths",
        "paths-ignore",
        "tags",
        "tags-ignore",
        "types",
    }
    assert not set(triggers.get("pull_request") or {}) & (filters | {"branches"}), (
        f"pull requests are narrowed by {sorted(triggers['pull_request'])}"
    )
    push = triggers.get("push") or {}
    assert not set(push) & filters, f"push is narrowed by {sorted(set(push) & filters)}"
    branches = push.get("branches", [])
    # A scalar `branches: mainline` would satisfy a substring test and narrow
    # the trigger to a branch that does not exist.
    assert isinstance(branches, list) and "main" in branches, push
    # A negative pattern excludes the branch that the line above still finds.
    assert not [pattern for pattern in branches if pattern.startswith("!")], branches


def test_the_key_check_refuses_every_key_it_has_not_read() -> None:
    """It returns nothing on the shipped file — which is also what a check
    looking for keys that cannot occur returns. Each of these neuters a gate
    without touching its run line, and none had to be foreseen."""
    clean: Dict[str, Any] = {"steps": [{"run": PROVE_SEMANTIC}]}
    assert _unvetted_keys({"name": "CI", "jobs": {}}, clean) == []
    assert _unvetted_keys({"name": "CI", "jobs": {}, "defaults": {}}, clean)
    for key in ("defaults", "if", "continue-on-error", "strategy"):
        assert _unvetted_keys({}, {**clean, key: "x"}), key
    for key in ("shell", "continue-on-error", "working-directory", "env"):
        assert _unvetted_keys({}, {"steps": [{"run": PROVE_SEMANTIC, key: "x"}]}), key
    # A conditional step may carry `if` and `id`; what it may not be is a gate,
    # which is asserted where the conditional tail is.
    assert _unvetted_keys({}, {"steps": [{"run": "x", "if": "y", "shell": "z"}]})


def test_ci_gates_the_paraphrase_eval_under_the_semantic_backend() -> None:
    """Without a job installing the extra and running the semantic-marked tests,
    the paraphrase floor is deselected everywhere and can never fail."""
    workflow = _ci()
    jobs = workflow["jobs"]
    semantic_jobs = [job for job in jobs.values() if "semantic]" in str(job)]
    assert semantic_jobs, "no CI job installs the 'semantic' extra"
    for job in semantic_jobs:
        runs = [step.get("run", "").strip() for step in job["steps"]]
        assert PROVE_SEMANTIC in runs, runs
    assert (ROOT / "scripts" / "check_semantic_report.py").exists()
    assert (ROOT / "scripts" / "check_suite_report.py").exists()
    assert "semantic" in jobs["docker"].get("needs", []), (
        "the image publish does not wait for the semantic gate"
    )


def test_no_job_carries_a_key_nobody_has_read() -> None:
    """Every job, not the ones some other job waits on: the publish job holds
    the image scan and nothing waits on it."""
    workflow = _ci()
    for name, job in workflow["jobs"].items():
        assert _unvetted_keys(workflow, job) == [], (
            f"{name}: {_unvetted_keys(workflow, job)}"
        )


PUBLISH_CONDITION = "github.event_name == 'push' && github.ref == 'refs/heads/main'"

# What the README's CI section sells. A claim in prose that no step implements
# fails nothing when it disappears, which is what this file exists to catch.
ADVERTISED_COMMANDS = (
    "ruff check .",
    "ruff format --check .",
    "mypy ",
    "pip-audit",
    # The coverage floor moved into the checker below with the run it fails on,
    # and is pinned where it now lives.
    PROVE_SUITE,
    PROVE_BACKENDS,
    "python -m evals",
    "helm lint",
    "helm template",
    "docker run -d --name smoke",
    PROVE_SEMANTIC,
    # The SBOM is sold as a gate, so the step that reads the document back is
    # one: generating and uploading a file cannot fail on what the file says.
    "sbom.cyclonedx.json",
)
# Every step this pipeline runs, in order, by job. Pinning the gate bodies left
# the step list open, and a step that adds nothing to a gate can still take one
# away: a shim earlier on PATH, a variable exported into every later step.
PIPELINE_STEPS = {
    "test": (
        (None, "actions/checkout@11d5960a326750d5838078e36cf38b85af677262", ()),
        (None, "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", ()),
        (None, None, ("python -m pip install --upgrade pip",)),
        (
            "Install (pulls rag-llm-infra from PyPI; dev toolchain pinned)",
            None,
            ('pip install -e ".[dev]" -c constraints-dev.txt',),
        ),
        ("Lint", None, ("ruff check .",)),
        ("Format check", None, ("ruff format --check .",)),
        ("Type-check", None, ("mypy app evals scripts tests",)),
        (
            "Audit Python dependencies (pip-audit)",
            None,
            ("pip install pip-audit -c constraints-dev.txt", "pip-audit"),
        ),
        (
            "Integration tests (the run has to account for every test it selects; coverage floor enforced)",
            None,
            (PROVE_SUITE,),
        ),
        (
            "Retrieval eval (recall gate enforced in tests/test_eval.py; print the numbers)",
            None,
            ("python -m evals",),
        ),
    ),
    "semantic": (
        (None, "actions/checkout@11d5960a326750d5838078e36cf38b85af677262", ()),
        (None, "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", ()),
        (
            "Cache the embedding model",
            "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830",
            (),
        ),
        (None, None, ("python -m pip install --upgrade pip",)),
        (
            "Install with the semantic extra",
            None,
            ('pip install -e ".[dev,semantic]" -c constraints-dev.txt',),
        ),
        (
            "Paraphrase floor + floor-derivation reproduce (semantic-marked gates)",
            None,
            (PROVE_SEMANTIC,),
        ),
    ),
    "backends": (
        (None, "actions/checkout@11d5960a326750d5838078e36cf38b85af677262", ()),
        (None, "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065", ()),
        (None, None, ("python -m pip install --upgrade pip",)),
        (
            "Install every optional backend",
            None,
            ('pip install -e ".[dev,faiss,qdrant,openai]" -c constraints-dev.txt',),
        ),
        (
            "Prove the extras landed",
            None,
            ('python -c "import faiss, qdrant_client, openai"',),
        ),
        (
            "Every advertised backend constructs",
            None,
            (
                "python -B -m pytest -q \\",
                f"{PROVE_BACKENDS} \\",
                "tests/test_app.py::"
                "test_the_backend_settings_expose_the_values_this_file_checks",
            ),
        ),
    ),
    "iac": (
        (None, "actions/checkout@11d5960a326750d5838078e36cf38b85af677262", ()),
        (None, "azure/setup-helm@1a275c3b69536ee54be43f2070a358922e12c8d4", ()),
        (
            "Helm lint + render (defaults, then every opt-in template)",
            None,
            (
                "helm lint deploy/helm",
                "helm template release deploy/helm > /dev/null",
                "helm template release deploy/helm \\",
                "--set ingress.enabled=true \\",
                "--set ingress.tls.enabled=true \\",
                "--set ingress.tls.clusterIssuer=letsencrypt-prod \\",
                "--set monitoring.enabled=true \\",
                "--set secrets.data.APP_API_KEY=c2VjcmV0 > /dev/null",
            ),
        ),
        (
            "Lint Dockerfile (hadolint)",
            "hadolint/hadolint-action@54c9adbab1582c2ef04b2016b760714a4bfde3cf",
            (),
        ),
    ),
    "docker": (
        (None, "actions/checkout@11d5960a326750d5838078e36cf38b85af677262", ()),
        (
            "Set up Buildx",
            "docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
            (),
        ),
        (
            "Build image (load locally for scan + SBOM)",
            "docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
            (),
        ),
        (
            "Trivy image scan (fail on fixable HIGH/CRITICAL)",
            "aquasecurity/trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25",
            (),
        ),
        (
            "Run the built image and exercise its API",
            None,
            (
                'docker run -d --name smoke -p 8000:8000 "$IMAGE:ci"',
                "for _ in $(seq 30); do",
                "if curl -fsS localhost:8000/health > /dev/null; then serving=1; break; fi",
                "sleep 1",
                "done",
                "docker logs smoke",
                'test -n "${serving:-}"',
                "curl -fsS -XPOST localhost:8000/index -H 'content-type: application/json' \\",
                '-d \'{"documents":["a document the smoke test can retrieve"]}\'',
                "curl -fsS -XPOST localhost:8000/query -H 'content-type: application/json' \\",
                '-d \'{"query":"retrieve","k":1}\'',
                "docker rm -f smoke",
            ),
        ),
        (
            "Generate CycloneDX SBOM",
            "anchore/sbom-action@e22c389904149dbc22b58101806040fa8d37a610",
            (),
        ),
        (
            "Check the SBOM inventories the image",
            None,
            (
                'test "$(jq -r \'.bomFormat\' sbom.cyclonedx.json)" = "CycloneDX"',
                "components=\"$(jq '.components | length' sbom.cyclonedx.json)\"",
                'test "$components" -gt 0',
                'echo "SBOM inventories $components components"',
            ),
        ),
        (
            "Upload SBOM artifact",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
            (),
        ),
        (
            "Log in to GHCR",
            "docker/login-action@c94ce9fb468520275223c153574b00df6fe4bcc9",
            (),
        ),
        (
            "Read chart appVersion (the tag a bare `helm install` resolves)",
            None,
            (
                "v=\"$(awk -F'\"' '/^appVersion:/ {print $2}' deploy/helm/Chart.yaml)\"",
                'test -n "$v"  # fail loudly rather than pushing a malformed empty tag',
                'echo "app_version=$v" >> "$GITHUB_OUTPUT"',
            ),
        ),
        (
            "Push the scanned image to GHCR (latest + commit SHA + chart appVersion)",
            None,
            (
                'for tag in latest "$GITHUB_SHA" "${{ steps.chart.outputs.app_version }}"; do',
                'docker tag "$IMAGE:ci" "$IMAGE:$tag"',
                'docker push "$IMAGE:$tag"',
                "done",
            ),
        ),
    ),
}

# The steps the README sells as gates. What each one runs is pinned above.
ADVERTISED_STEPS = (
    "Lint",
    "Format check",
    "Type-check",
    "Audit Python dependencies (pip-audit)",
    "Integration tests (the run has to account for every test it selects; coverage floor enforced)",
    "Retrieval eval (recall gate enforced in tests/test_eval.py; print the numbers)",
    "Paraphrase floor + floor-derivation reproduce (semantic-marked gates)",
    "Helm lint + render (defaults, then every opt-in template)",
    "Check the SBOM inventories the image",
)

# Every input every action may carry. Restricting only the scanners left the
# builder free to publish a different stage, and checkout free to fetch a
# different tree — neither needs a key anyone had thought to forbid.
ACTION_INPUTS = {
    "actions/checkout": set(),
    "actions/setup-python": {"python-version"},
    "actions/cache": {"path", "key"},
    "azure/setup-helm": set(),
    "docker/setup-buildx-action": set(),
    "docker/login-action": {"registry", "username", "password"},
    "docker/build-push-action": {"context", "file", "load", "push", "tags"},
}
# What each tool loads on its own, without a flag naming it: a ruff.toml
# replaces the pinned sections outright, a mypy.ini silences every error. Closed
# by each tool's documented discovery order, not by the one example.
TOOL_CONFIG_FILES = (
    "pyproject.toml",
    ".helmignore",
    ".trivyignore",
    ".trivyignore.yaml",
    ".trivyignore.yml",
    "trivy.yaml",
    "trivy.yml",
    ".hadolint.yaml",
    ".hadolint.yml",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
    "pytest.ini",
    ".pytest.ini",
    "tox.ini",
    "setup.cfg",
    ".coveragerc",
)
VETTED_INPUTS = {
    "aquasecurity/trivy-action": {
        "image-ref",
        "format",
        "exit-code",
        "severity",
        "ignore-unfixed",
        "vuln-type",
    },
    "anchore/sbom-action": {"image", "format", "output-file"},
    "actions/upload-artifact": {"name", "path"},
    "hadolint/hadolint-action": {"dockerfile", "failure-threshold"},
}
ADVERTISED_ACTIONS = tuple(VETTED_INPUTS)
ACTION_INPUTS.update(VETTED_INPUTS)


def _steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _action(workflow: Dict[Any, Any], name: str) -> List[Dict[str, Any]]:
    return [s for s in _steps(workflow) if s.get("uses", "").split("@")[0] == name]


def _uncommented(run: str) -> List[str]:
    """A gate quoted inside a shell comment is not a gate that runs."""
    return [line for line in run.splitlines() if not line.strip().startswith("#")]


def _uncommented_text(run: str) -> str:
    return chr(10).join(_uncommented(run))


def _advertised(step: Dict[str, Any]) -> bool:
    lines = _uncommented(step.get("run", ""))
    return (
        step.get("uses", "").split("@")[0] in ADVERTISED_ACTIONS
        or bool((step.get("with") or {}).get("load"))
        or any(command in line for line in lines for command in ADVERTISED_COMMANDS)
    )


def _gate_steps(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Steps whose outcome decides a verdict.

    What identifies the publish is that it publishes: it carries the merge
    condition, which nothing that decides a verdict may do. Identifying it by a
    `with.push` key would let any step opt out by setting one, and YAML reads
    `push: "false"` as a truthy string."""
    return [s for s in _steps(workflow) if not s.get("if") and _advertised(s)]


def _conditional_gates(workflow: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Advertised gates wearing the merge condition — how one leaves the set
    above while remaining the thing the README sells."""
    return [s for s in _steps(workflow) if s.get("if") and _advertised(s)]


def _signature(step: Dict[str, Any]) -> Any:
    return (
        step.get("name"),
        # The version too: the action name alone leaves a re-pointed or bumped
        # ref free to change what the step does with the file untouched.
        step.get("uses") or None,
        tuple(
            line.strip() for line in _uncommented(step.get("run", "")) if line.strip()
        ),
    )


def test_the_pipeline_runs_these_steps_and_no_others() -> None:
    """Pinning what the gates run leaves what runs BESIDE them open, and a step
    that touches no gate can still disarm one: a shim earlier on PATH, an
    exclusion appended to the config a gate reads, a variable exported into
    every later step. Nothing is subtracted in any of those, so every check
    that asks what a gate says still passes. The steps are ours; this is all
    of them."""
    actual = {
        job: tuple(_signature(step) for step in body["steps"])
        for job, body in _ci()["jobs"].items()
    }
    assert actual == PIPELINE_STEPS


def test_every_step_the_readme_sells_as_a_gate_is_one_of_them() -> None:
    """The pinned list above is what runs; this is which of it the README sells
    — and a name occurring twice would make the pinning ambiguous."""
    pinned = [step for steps in PIPELINE_STEPS.values() for step in steps]
    names = [name for name, _, _ in pinned if name]
    assert len(names) == len(set(names)), sorted(names)
    for advertised in ADVERTISED_STEPS:
        assert advertised in names, advertised


def test_every_action_the_readme_advertises_is_present_and_can_fail() -> None:
    """Present is half of it: a scan that reports and exits zero, or one told to
    skip every directory it would have looked in, is decoration with a tick."""
    workflow = _ci()
    for step in _steps(workflow):
        action = step.get("uses", "").split("@")[0]
        if not action:
            continue
        allowed = ACTION_INPUTS.get(action)
        assert allowed is not None, f"{action} has no vetted input list"
        assert set(step.get("with") or {}) <= allowed, (action, step.get("with"))
    for name, vetted in VETTED_INPUTS.items():
        steps = _action(workflow, name)
        # Exactly one: a second, later step of the same action overwrites what
        # the first produced, and every read below takes the first.
        assert len(steps) == 1, (name, len(steps))
        for step in steps:
            # A commit, not a tag: an exact version tag is still movable.
            assert re.fullmatch(r"[0-9a-f]{40}", step["uses"].split("@")[1]), step
            assert set(step["with"]) <= vetted, set(step["with"]) - vetted
    scan = _action(workflow, "aquasecurity/trivy-action")[0]["with"]
    # The values, not just the keys: os-only drops every library CVE, and
    # CRITICAL-only drops the HIGH the step's own name promises.
    assert scan["exit-code"] == "1", scan
    assert scan["severity"] == "HIGH,CRITICAL", scan
    assert scan["vuln-type"] == "os,library", scan
    lint = _action(workflow, "hadolint/hadolint-action")[0]["with"]
    # error alone admits every warning the linter has, which is most of them.
    assert lint["failure-threshold"] == "warning", lint
    assert (ROOT / lint["dockerfile"]).is_file(), lint


def test_what_gets_scanned_is_what_gets_built_and_what_gets_pushed() -> None:
    """One build, and the publish carries its tags rather than repeating it.
    A second build cannot be tied to the first by comparing their inputs: equal
    context and Dockerfile is not equal bytes, so the only way the published
    image is the vetted one is for nothing to build it twice."""
    workflow = _ci()
    builds = _action(workflow, "docker/build-push-action")
    assert len(builds) == 1, [step.get("name") for step in builds]
    build = builds[0]["with"]
    assert build.get("load"), build
    assert not build.get("push"), build
    image = build["tags"].strip()
    assert (
        _action(workflow, "aquasecurity/trivy-action")[0]["with"]["image-ref"] == image
    )
    assert _action(workflow, "anchore/sbom-action")[0]["with"]["image"] == image

    published = [
        step
        for step in _steps(workflow)
        if "docker push" in _uncommented_text(step.get("run", ""))
    ]
    assert len(published) == 1, published
    body = _uncommented_text(published[0]["run"])
    # Pushed by re-tagging the local image the scan read, and the ids compared
    # before each push: a `docker build` here would publish unvetted bytes.
    assert "docker build" not in body, body
    assert 'docker tag "$IMAGE:ci"' in body, body
    # The chart's appVersion tag must be among them, or a bare `helm install`
    # resolves a tag that was never pushed.
    assert "app_version" in body, body
    assert "latest" in body and "GITHUB_SHA" in body, body


def test_the_sbom_that_is_uploaded_is_the_sbom_that_was_generated() -> None:
    """Two steps joined by a filename nobody compared: the artifact can hold
    anything while the README sells a CycloneDX bill of materials."""
    workflow = _ci()
    sbom = _action(workflow, "anchore/sbom-action")[0]["with"]
    upload = _action(workflow, "actions/upload-artifact")[0]["with"]
    assert sbom["format"] == "cyclonedx-json", sbom
    assert upload["path"] == sbom["output-file"], (upload, sbom)


def test_the_publish_waits_on_every_other_job() -> None:
    """Waiting on some of them publishes while the rest are still failing."""
    jobs = _ci()["jobs"]
    assert set(jobs["docker"]["needs"]) == set(jobs) - {"docker"}, jobs["docker"]


def test_the_readme_names_the_jobs_the_publish_actually_waits_on() -> None:
    """That sentence is the only prose copy of the wait list. Names, not count:
    a swap keeps the count."""
    jobs = _ci()["jobs"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sentence = next(
        (line for line in readme.splitlines() if "The image publish waits on" in line),
        "",
    )
    assert sentence, "README no longer carries the wait-list sentence"
    named = {job for job in jobs if re.search(rf"\b{re.escape(job)}\b", sentence, re.I)}
    assert named == set(jobs["docker"]["needs"]), (named, jobs["docker"]["needs"])


def test_no_gate_step_is_conditional_and_publishing_comes_last() -> None:
    """A condition on a gate is invisible in a green run: the step is skipped,
    not failed. Publishing is rightly conditional, so the conditional steps must
    be the tail of their job and carry the one condition that means this is a
    merge — and every gate must be unconditional, which is what puts it before
    them. Position alone does not: a gate that is also conditional sits happily
    in that tail, running after the image it was meant to vet has been pushed."""
    workflow = _ci()
    assert _conditional_gates(workflow) == [], _conditional_gates(workflow)
    for step in _gate_steps(workflow):
        assert "if" not in step, step.get("name") or step.get("uses")
    for name, job in workflow["jobs"].items():
        assert "if" not in job, f"{name} is switched off wholesale"
        steps = job.get("steps", [])
        conditional = [index for index, step in enumerate(steps) if "if" in step]
        tail = list(range(len(steps) - len(conditional), len(steps)))
        assert conditional == tail, f"{name}: {conditional} is not the tail"
        for index in conditional:
            assert steps[index]["if"] == PUBLISH_CONDITION, steps[index]["if"]
