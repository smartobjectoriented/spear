"""Run a task's oracle tests against the workspace a run left behind.

Two properties make this an oracle rather than a formality:

* the tests are the TASK's, not the workspace's.  A test file the model could
  edit is a test file the model can make pass; the same reasoning already puts
  the project acceptance benches in ``benches/`` rather than in the trees they
  judge.  The sources live in the task definition and are written into a copy
  the run never saw.
* the code they exercise was written by the model, so it runs under the same
  Bubblewrap confinement and the same resource contract as everything the
  harness runs -- not in a bare subprocess of the scorer.

pytest is not a dependency of this checkout and the fixtures are plain
``test_*`` functions, so the child runner written beside the tests is the
whole of what they need.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from tool_runtime import (
    BubblewrapSandbox, Capability, CommandRunner, ExecutionProfile, Workspace,
)

#: Where the oracle lands inside the copy.  Nothing the model wrote shares the
#: name, and whatever did is removed before the tests are written.
ORACLE_DIR = ".spear-oracle"

_CHILD_SOURCE = Path(__file__).resolve().parent / "_acceptance_child.py"
_IGNORED = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache")

# One sandbox for the whole scoring process: preflight happens once instead of
# once per task.
_RUNNERS: dict[int, CommandRunner] = {}


def _runner(timeout: int) -> CommandRunner:
    if timeout not in _RUNNERS:
        _RUNNERS[timeout] = CommandRunner(
            sandbox=BubblewrapSandbox(timeout_seconds=timeout),
            timeout_seconds=timeout)

    return _RUNNERS[timeout]


def _stage_oracle(workspace: Path, tests: dict) -> list[str]:
    """Write the child runner and the task's tests into the copy."""

    oracle = workspace / ORACLE_DIR

    if oracle.exists():
        shutil.rmtree(oracle)

    oracle.mkdir(parents=True)
    (oracle / "run_tests.py").write_text(
        _CHILD_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    names = []

    for name, source in tests.items():
        (oracle / name).write_text(source, encoding="utf-8")
        names.append(f"{ORACLE_DIR}/{name}")

    return sorted(names)


def run_acceptance_tests(root: str | Path, tests: dict, *,
                         timeout: int = 60,
                         runner: CommandRunner | None = None) -> tuple[bool, tuple[dict, ...]]:
    """(all passed, per-file reports) for the task's oracle tests.

    A sandbox that cannot be established is reported as ``not_run`` and is
    NOT a pass: an oracle that could not be evaluated has decided nothing,
    and the harness fails closed everywhere else for the same reason.
    """

    if not tests:
        return True, ()

    root = Path(root)

    with tempfile.TemporaryDirectory(prefix="spear-accept-") as directory:
        workspace = Path(directory) / "workspace"
        shutil.copytree(root, workspace, ignore=_IGNORED)
        names = _stage_oracle(workspace, tests)
        runner = runner or _runner(timeout)
        profile = ExecutionProfile.from_capabilities(
            [Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE])

        try:
            result = runner.run_sandboxed(
                Workspace.from_path(workspace),
                ["python3", f"{ORACLE_DIR}/run_tests.py", *names], profile)
        except (OSError, ValueError) as exc:
            return False, ({"path": ", ".join(names), "status": "not_run",
                            "error": f"{type(exc).__name__}: {exc}"[:240]},)

        if result.status == "cancelled" or (
                not result.stdout and result.status != "ok" and result.exit_code is None):
            # The sandbox itself refused or never started the command: no test
            # ran, so nothing was decided.

            return False, ({"path": ", ".join(names), "status": "not_run",
                            "error": (result.summary or result.stderr)[:240]},)

        try:
            reports = tuple(json.loads(result.stdout))
        except (json.JSONDecodeError, TypeError, ValueError):
            return False, ({"path": ", ".join(names), "status": "unreadable",
                            "error": (result.stderr or result.stdout or
                                      result.summary)[-240:]},)

    # One report per file, or the child did not run what it was given: an
    # empty list would otherwise pass `all()` and read as a clean sweep.

    if len(reports) != len(names):
        return False, (reports or ({"path": ", ".join(names), "status": "unreadable",
                                    "error": "incomplete report"},))

    return (all(item.get("status") == "passed" for item in reports)
            and result.exit_code in (None, 0)), reports
