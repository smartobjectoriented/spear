"""The project's own checks: its bench, build verification, normative check."""

import os
import subprocess
from harness.tool_runtime import (
    BubblewrapSandbox, CommandRunner, ExecutionProfile, Capability,
    PathPolicyError, Workspace, shell_argv,
)
from runtime.tracing import EventStatus, EventType, new_action_id, new_task_id
from runtime.agent_runtime import AgentRuntime
from evidence import project_build
from cli import session_workspace
from cli.chat_settings import TRACE, resource_dir
from cli.corpus_registry import load_projects
from cli.session_workspace import COMMAND_RUNNER


# Acceptance benches belong to the harness, not to the trees they judge: a
# project should not have to carry the test rig that rates the assistant, and
# a bench a project owns is a bench the assistant can edit.

BENCH_DIR = resource_dir("SPEAR_BENCH_DIR", "benches")


def project_bench():
    """The acceptance command this project declares, if any (projects.json)."""
    return (load_projects().get(session_workspace.PROJECT) or {}).get("bench")


def _as_list(value):
    """One declared command, several, or none — always a list."""

    if isinstance(value, str):
        return [value]

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return []


def bench_command():
    """Resolve the declared bench to something runnable.

    A bare name is looked up in BENCH_DIR and becomes its absolute path, so
    projects.json says `"bench": "sye_sol-ls.sh"` and nothing has to live in
    the project. Anything else is passed through as a shell command, which
    keeps a project-owned script possible for whoever wants one.
    """
    declared = project_bench()

    if not declared:
        return None

    candidate = os.path.join(BENCH_DIR, declared)

    return candidate if os.path.isfile(candidate) else declared


# The project's own build and tests exercise code the model has just written,
# so they run where the model's own commands run. Their own runner because
# their own ceiling: a build is minutes, not the 45 seconds a tool call gets.

PROJECT_VERIFY_TIMEOUT = int(os.environ.get("SPEAR_PROJECT_VERIFY_TIMEOUT", "900"))

# Whether a `tests/` package may be taken as "this is how the tree is tested".
# Off for real trees -- a project says how it is verified, it is not guessed
# at -- and on for the benchmark fixtures, which have no owner to ask.

INFER_TEST_COMMAND = os.environ.get(
    "SPEAR_INFER_TEST_COMMAND", "").strip().lower() in {"1", "true", "yes", "on"}
PROJECT_VERIFY_RUNNER = CommandRunner(
    sandbox=BubblewrapSandbox(timeout_seconds=PROJECT_VERIFY_TIMEOUT),
    timeout_seconds=PROJECT_VERIFY_TIMEOUT)


# Running the project's verification on the host is the operator's call to
# make, in advance and in writing. It is never a fallback: a sandbox that
# cannot be established is a verification that did not happen, and degrading
# it into an unconfined run of code the model just wrote is the one outcome
# worse than not verifying at all.

PROJECT_VERIFY_ON_HOST = os.environ.get(
    "SPEAR_PROJECT_VERIFY_ON_HOST", "").strip().lower() in {"1", "true", "yes", "on"}


def verify_project_command(command):
    """Run one of the project's own verification commands, confined.

    Returns (status, detail) with status "passed", "failed" or "not_run".
    "not_run" is the closed door: nothing ran, nothing is claimed, and the
    turn is left unjudged rather than told its build is broken.
    """

    if not command:
        return "passed", ""

    if PROJECT_VERIFY_ON_HOST:
        ok, output = project_build.run(command, session_workspace.PROJECT_ROOT,
                                       PROJECT_VERIFY_TIMEOUT)

        return ("passed" if ok else "failed"), output

    if session_workspace.WORKSPACE is None:
        return "not_run", ("no workspace boundary for project verification "
                           "(set SPEAR_PROJECT_VERIFY_ON_HOST=1 to accept "
                           "running it unconfined)")

    profile = ExecutionProfile.from_capabilities(
        {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
         Capability.SHELL_COMPLEX})
    availability = PROJECT_VERIFY_RUNNER.ensure_sandbox(session_workspace.WORKSPACE, profile)

    if not availability.ok:
        return "not_run", (f"{availability.summary} — project verification did "
                           f"not run (set SPEAR_PROJECT_VERIFY_ON_HOST=1 to "
                           f"accept running it unconfined)")

    result = PROJECT_VERIFY_RUNNER.run_sandboxed(
        session_workspace.WORKSPACE, shell_argv(command), profile, availability=availability)

    if result.status == "cancelled":
        return "not_run", result.summary

    if result.status == "ok" and result.exit_code in (None, 0):
        return "passed", ""

    return "failed", project_build.summarize(
        (result.stdout or "") + (result.stderr or "") or result.summary)


def run_normative_check(command):
    """Run a project check bound to a normative provision, confined.

    The same boundary as the project's own verification, with the outcome
    kept apart from the infrastructure: (status, exit code, output), status
    PASSED, FAILED, NOT_RUN or ERROR. A command that could not be found or
    started, or that timed out, is an ERROR -- it says nothing about the
    provision.
    """

    if PROJECT_VERIFY_ON_HOST:
        try:
            done = subprocess.run(["bash", "-lc", command], cwd=session_workspace.PROJECT_ROOT,
                                  timeout=PROJECT_VERIFY_TIMEOUT, capture_output=True,
                                  text=True)
        except subprocess.TimeoutExpired:
            return "ERROR", None, f"timed out after {PROJECT_VERIFY_TIMEOUT}s"
        except (OSError, subprocess.SubprocessError) as exc:
            return "ERROR", None, f"{type(exc).__name__}: {exc}"

        output = (done.stdout or "") + (done.stderr or "")

        if done.returncode == 0:
            return "PASSED", 0, output

        return ("ERROR" if done.returncode in (126, 127) else "FAILED"), done.returncode, output

    if session_workspace.WORKSPACE is None:
        return "NOT_RUN", None, "no workspace boundary for project checks"

    profile = ExecutionProfile.from_capabilities(
        {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
         Capability.SHELL_COMPLEX})
    availability = PROJECT_VERIFY_RUNNER.ensure_sandbox(session_workspace.WORKSPACE, profile)

    if not availability.ok:
        return "NOT_RUN", None, availability.summary

    result = PROJECT_VERIFY_RUNNER.run_sandboxed(
        session_workspace.WORKSPACE, shell_argv(command), profile, availability=availability)
    output = (result.stdout or "") + (result.stderr or "")

    if result.status == "cancelled":
        return "NOT_RUN", None, result.summary

    if result.status == "ok" and result.exit_code in (None, 0):
        return "PASSED", 0, output

    if result.status == "failed" and result.exit_code not in (None, 126, 127):
        return "FAILED", result.exit_code, output

    return "ERROR", result.exit_code, output or result.summary


def run_project_bench(agent_context=None):
    """Run the project's acceptance command. True only on a clean exit.

    Run BY THE HARNESS, never offered to the model: it is the judge, so it
    must not be something the model can talk its way past, mistake for its own
    work, or edit. Returns None when the project declares no bench — no
    verdict, rather than a fabricated pass.
    """
    cmd = bench_command()

    if not cmd or session_workspace.WORKSPACE is None or COMMAND_RUNNER is None:
        return None

    task_id = agent_context.task_id if agent_context is not None else new_task_id()
    trace = agent_context.trace if agent_context is not None else TRACE
    action_id = new_action_id("project_bench")
    span = trace.start_span(
        EventType.VERIFICATION_STARTED,
        EventType.VERIFICATION_FINISHED,
        EventType.VERIFICATION_FINISHED,
        task_id,
        action_id=action_id,
        metadata={"kind": "project_bench", "command_configured": True},
    )

    # The bench lives outside the workspace, so it must be mounted for the run
    # or the sandbox cannot see it. Read-only would be cleaner still, but the
    # workspace itself must stay writable for a bench that builds.

    workspace = session_workspace.WORKSPACE

    if os.path.isdir(BENCH_DIR):
        try:
            workspace = Workspace.from_path(
                session_workspace.WORKSPACE.root,
                allow_absolute_paths=session_workspace.WORKSPACE.allow_absolute_paths,
                extra_roots=tuple(session_workspace.WORKSPACE.extra_roots) + (BENCH_DIR,))
        except (PathPolicyError, OSError):
            span.finish(
                status=EventStatus.FAILED,
                error_category="bench_workspace",
                error_summary="could not prepare bench workspace",
                metadata={"kind": "project_bench", "verdict": None},
            )

            if agent_context is not None:
                AgentRuntime._record_verification(
                    agent_context,
                    agent_context.verification_policy.project_bench_evidence(
                        agent_context.working_state, executed=False, passed=None,
                        action_id=action_id,
                        summary="bench workspace could not be prepared",
                    ),
                )

            return None

    profile = ExecutionProfile.from_capabilities(
        {Capability.FILESYSTEM_READ, Capability.WORKSPACE_WRITE,
         Capability.SHELL_COMPLEX})
    availability = COMMAND_RUNNER.ensure_sandbox(workspace, profile)

    if not availability.ok:
        span.finish(
            status=EventStatus.FAILED,
            error_category="sandbox_unavailable",
            error_summary=availability.summary,
            metadata={"kind": "project_bench", "verdict": None},
        )

        if agent_context is not None:
            AgentRuntime._record_verification(
                agent_context,
                agent_context.verification_policy.project_bench_evidence(
                    agent_context.working_state, executed=False, passed=None,
                    action_id=action_id, summary="sandbox unavailable",
                ),
            )

        return None

    result = COMMAND_RUNNER.run_sandboxed(
        workspace, shell_argv(cmd), profile, availability=availability)
    verdict = result.status == "ok" and result.exit_code in (None, 0)
    span.finish(
        status=EventStatus.OK if verdict else EventStatus.FAILED,
        error_category=None if verdict else result.status,
        error_summary=None if verdict else result.summary,
        metadata={
            "kind": "project_bench",
            "verdict": verdict,
            "exit_code": result.exit_code,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
        },
    )

    if agent_context is not None:
        AgentRuntime._record_verification(
            agent_context,
            agent_context.verification_policy.project_bench_evidence(
                agent_context.working_state, executed=True, passed=verdict,
                action_id=action_id,
                summary="project bench passed" if verdict else "project bench failed",
            ),
        )

    return verdict


def _record_bench_answer(text):
    """Append this turn's final answer to the benchmark's answer sink.

    Off unless SPEAR_BENCH_ANSWER_FILE names a file, which only the benchmark
    runner does. It exists so an informative task can be scored on what the
    run actually ANSWERED: reading the right files and then naming the wrong
    function is a failure, and nothing in the workspace shows it.
    """

    path = os.environ.get("SPEAR_BENCH_ANSWER_FILE")

    if not path or not text:
        return

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
    except OSError:
        # Losing the sink costs the benchmark an oracle, never the turn.

        pass
