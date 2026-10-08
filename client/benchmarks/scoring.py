"""Objective, model-independent benchmark scoring and trace summarization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .acceptance import run_acceptance_tests
from .tasks import BenchmarkTask, fragments

# Artefacts of running the tests, not work the model did.  They are excluded
# from the walk so a run that tried its own tests does not report a tree full
# of changed files.

_IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class TaskScore:
    """Two verdicts, because they are two different facts.

    ``functionally_correct`` is what the workspace and the hidden oracle say:
    the file the task was about ends up right.  ``task_success`` is whether
    the RUN did the task -- and a task that says "run the test" is not done by
    a run that never ran it, however correct the file it left behind.  The
    benchmark reported 100% for a run that tried `python3 -c` twice, gave up,
    and concluded "UNVERIFIED — main.py changed without sufficient current
    verification". It was right about itself; the score was not.
    """

    task_success: bool
    functionally_correct: bool
    runtime_verified: bool | None
    acceptance_tests_passed: bool | None
    expected_files_ok: bool
    forbidden_content_absent: bool
    required_reads_ok: bool
    forbidden_files_unchanged: bool
    answer_ok: bool
    changed_files: tuple[str, ...]
    oracles: tuple[str, ...] = ()
    acceptance_reports: tuple[dict, ...] = ()
    failure_reason: str | None = None

    def to_dict(self):
        return asdict(self)


def _answer_hits(answer: str | None, expectation) -> bool:
    """True when the answer contains any spelling of one expectation.

    Case is ignored: these are things the run NAMED -- a file, a function --
    and a benchmark that failed a correct answer over a capital letter would
    be measuring typography.
    """

    if not answer:
        return False

    lowered = answer.lower()

    return any(item.lower() in lowered for item in fragments(expectation))


def _ignored(path: Path, root: Path) -> bool:
    return bool(set(path.relative_to(root).parts) & _IGNORED_PARTS)


def score_task(task: BenchmarkTask, root: str | Path, initial: Mapping[str, str],
               trace_metrics: Mapping[str, object] | None = None, *,
               answer: str | None = None,
               acceptance_timeout: int = 60) -> TaskScore:
    """Score one run from the workspace it left behind, never from what it said.

    The workspace decides: the expected content is present, what the task ruled
    out is absent, the files it was told not to touch are untouched, and any
    reads the task required actually happened.  A task declaring acceptance
    tests has them RUN -- the task's own tests, in a sandboxed copy of the
    finished workspace -- because "the expected string is in the file" is not
    the same claim as "the code works".

    A task that says to run or test what it changed is scored on that too:
    the hidden oracle proves the file is right, not that the run did the job.

    A task that only REPORTS leaves nothing in the workspace, so those are
    scored on `answer` as well: the reads prove the run looked at the right
    files, and the answer oracle is what distinguishes looking from knowing.

    A task that declares no oracle at all cannot be passed.  Four tasks used
    to have none, and every run of them was recorded as a success: nothing was
    being measured, and the suite's success rate was that much fiction.
    """

    root = Path(root)
    oracles: list[str] = []

    # Fragment containment rather than an exact match: the task states what the
    # result must contain, not how the whole file has to be written.  Several
    # alternatives may be given, and any one of them satisfies the check.

    expected_ok = True

    for relative, expectation in dict(task.expected).items():
        oracles.append(f"expected:{relative}")
        path = root / relative
        text = (path.read_text(encoding="utf-8", errors="replace")
                if path.is_file() else None)

        if text is None or not any(item in text for item in fragments(expectation)):
            expected_ok = False

    # The other half of a content oracle: what the task says must be gone.
    # Without it, "rename host to endpoint" passed on a file that had both.

    forbidden_content_ok = True

    for relative, expectation in dict(task.forbidden_content).items():
        oracles.append(f"forbidden_content:{relative}")
        path = root / relative
        text = (path.read_text(encoding="utf-8", errors="replace")
                if path.is_file() else "")

        if any(item in text for item in fragments(expectation)):
            forbidden_content_ok = False

    changed = []
    current = {}

    for path in root.rglob("*"):
        if path.is_file() and not _ignored(path, root):
            relative = str(path.relative_to(root))
            current[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

            if initial.get(relative) != current[relative]:
                changed.append(relative)

    # Deletions count as changes too, and only appear by their absence from the
    # walk above.

    changed.extend(relative for relative in initial if relative not in current)
    changed = tuple(sorted(set(changed)))

    forbidden_ok = not (set(changed) & set(task.forbidden_paths))

    # Grounded reads come from the trace, so a task requiring evidence cannot
    # be passed by an answer that happened to be right.

    read_paths = set(trace_metrics.get("read_paths", ())) if trace_metrics else set()
    reads_ok = set(task.required_reads) <= read_paths

    if task.required_reads:
        oracles.append("required_reads")

    # What the run SAID, for the tasks whose whole product is what they said.

    answer_ok = True

    for expectation in task.expected_answer:
        oracles.append("expected_answer")

        if not _answer_hits(answer, expectation):
            answer_ok = False

    for expectation in task.forbidden_answer:
        oracles.append("forbidden_answer")

        if _answer_hits(answer, expectation):
            answer_ok = False

    # Run the tests last: everything above reads the workspace as the model
    # left it, and the run happens in a copy so it stays that way.

    acceptance_passed: bool | None = None
    reports: tuple[dict, ...] = ()

    if task.acceptance:
        oracles.append("acceptance")
        acceptance_passed, reports = run_acceptance_tests(
            root, dict(task.acceptance), timeout=acceptance_timeout)

    # Did the RUN verify its own change? Only asked of tasks that say to.
    # "The run" means everything inside it: the commands the model chose AND
    # the project's own suite run deterministically by the harness. Both are
    # the complete agent at work. The benchmark's hidden oracle is not -- it
    # runs out here, after the fact, and never appears in the trace.

    runtime_verified: bool | None = None

    if task.requires_agent_verification:
        oracles.append("runtime_verification")
        runtime_verified = bool(trace_metrics
                                and trace_metrics.get("runtime_verifications"))

    has_oracle = bool(oracles)
    acceptance_ok = acceptance_passed in (None, True)
    not_run = tuple(item for item in reports
                    if item.get("status") in {"not_run", "unreadable"})
    functionally_correct = (has_oracle and expected_ok and forbidden_content_ok
                            and reads_ok and forbidden_ok and answer_ok
                            and acceptance_ok)
    success = functionally_correct and runtime_verified is not False

    if success:
        reason = None
    elif functionally_correct:
        reason = ("the task asked for the change to be tested and no "
                  "verification covering it was run")
    elif not has_oracle:
        reason = "task declares no oracle, so nothing about it can be scored"
    elif not expected_ok:
        reason = "expected content missing"
    elif not forbidden_content_ok:
        reason = "content the task rules out is still present"
    elif not reads_ok:
        reason = "required grounded reads missing"
    elif not answer_ok:
        reason = ("the run gave no final answer" if not answer
                  else "the final answer does not satisfy the task")
    elif not_run:
        # Never confused with a test that ran and failed: an oracle that could
        # not be evaluated has decided nothing about the run.

        reason = f"acceptance tests could not be run: {not_run[0].get('error', '')}"[:200]
    elif not acceptance_ok:
        reason = "acceptance tests failed"
    else:
        reason = "forbidden file changed"

    return TaskScore(
        task_success=success, functionally_correct=functionally_correct,
        runtime_verified=runtime_verified, acceptance_tests_passed=acceptance_passed,
        expected_files_ok=expected_ok, forbidden_content_absent=forbidden_content_ok,
        required_reads_ok=reads_ok, forbidden_files_unchanged=forbidden_ok,
        answer_ok=answer_ok, changed_files=changed, oracles=tuple(oracles),
        acceptance_reports=reports, failure_reason=reason)


def summarize_trace(path: str | Path) -> dict[str, object]:
    """Aggregate one run's trace into the counters the benchmark report uses."""

    counts: dict[str, int] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "context_tokens": 0}
    files_read: set[str] = set()
    files_modified: set[str] = set()
    actions: set[str] = set()
    runtime_verifications: list[str] = []

    # Tracing is off by default, so an absent trace is an ordinary outcome and
    # is reported as unavailable rather than raised.

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"events": 0, "trace_available": False}

    for line in lines:
        # A torn line from an interrupted run costs one event, not the report.

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = str(event.get("event_type", ""))
        counts[kind] = counts.get(kind, 0) + 1

        for field_name, key in (("input_tokens", "input_tokens"),
                                ("output_tokens", "output_tokens"),
                                ("context_tokens_estimate", "context_tokens")):
            if isinstance(event.get(field_name), int):
                totals[key] += event[field_name]

        metadata = event.get("metadata") or {}

        if kind == "file_read":
            files_read.add(str(metadata.get("path", metadata.get("file", "unknown"))))

        if kind == "file_modified":
            files_modified.add(str(metadata.get("path", metadata.get("file", "unknown"))))

        if event.get("action_id"):
            actions.add(str(event["action_id"]))

        # A verification performed BY THE RUN that actually covers the
        # change: full coverage and conclusive. The harness running the
        # project's own suite counts -- it is part of what the complete agent
        # does, and a run that has been shown a green suite must not be made
        # to run it again to prove what was already observed. What does not
        # count is the benchmark's own hidden oracle, which is external to the
        # run and never reaches its trace at all.

        if (kind == "verification_passed"
                and metadata.get("coverage") == "full"
                and metadata.get("proves_change") is not False):
            runtime_verifications.append(str(metadata.get("category", "unknown")))

    # Every action emits at least a started and a finished event, so counting
    # events that share an action id counted each action as a repeat of
    # itself -- the normal life of an action, reported as a defect.  The
    # harness detects the real thing and says so: `repeated_action_detected`
    # is emitted when a tool call is answered from cache because the same call
    # was made again.

    repeated = counts.get("repeated_action_detected", 0)

    # Verification counters come from the classified outcome, not from the
    # event name alone: an observation that proves nothing about the change is
    # traced as classified, and only the rest can pass or fail.

    return {"events": len(lines), "trace_available": True, "event_counts": counts,
            **totals, "files_read": len(files_read), "distinct_files_read": len(files_read),
            "read_paths": sorted(files_read), "modified_paths": sorted(files_modified),
            "files_modified": len(files_modified), "repeated_actions": repeated,
            "distinct_actions": len(actions),
            "model_calls": sum(counts.get(kind, 0)
                               for kind in ("model_call_finished", "model_call_failed")),
            "tool_calls": sum(counts.get(kind, 0)
                              for kind in ("tool_call_finished", "tool_call_failed")),
            "verifications_passed": counts.get("verification_passed", 0),
            "runtime_verifications": len(runtime_verifications),
            "runtime_verification_categories": sorted(set(runtime_verifications)),
            "verifications_failed": counts.get("verification_failed", 0),
            "verifications_inconclusive": counts.get("verification_classified", 0),
            "compactions": counts.get("compaction_finished", 0),
            "exploration_calls": counts.get("exploration_finished", 0),
            "review_calls": counts.get("review_finished", 0)}
