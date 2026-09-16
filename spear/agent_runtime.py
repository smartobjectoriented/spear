"""Reusable provider-neutral execution core for one SPEAR agent task.

The runtime owns model/tool iteration and task-scoped context state.  It knows
nothing about terminal rendering, project selection, persistence, or concrete
tool dispatch.  The legacy dispatcher is injected through ToolExecutor until
the later ToolRegistry/ToolRouter phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
import re
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Callable, ContextManager, Mapping, Protocol, Sequence

from cancellation import CancellationToken, NEVER_CANCELLED, OperationCancelled
from checkpoint import (
    Checkpoint, CheckpointManager, CheckpointStatus, RollbackResult,
    RollbackStatus,
)
from failure_policy import Failure, FailureKind, RetryPolicy, classify_tool_failure
from progress_monitor import ProgressMonitor
from session_store import (
    InFlightOperation, SessionError, SessionEventType, SessionHandle, SessionSnapshot,
)

from compaction import (
    CompactionArtifact, CompactionPolicy, CompactionRequest,
    CompactionService, ModelBackendSummarizer,
)
from context_engine import (
    ContextEngine, ContextItem, ContextLayer, ContextRequest, ContextSnapshot,
    Freshness, working_state_context_item,
)
from model_backend import (
    ConversationMessage, ModelBackend, ModelTurn, StopReason, TextBlock,
    ToolDefinition, ToolResultBlock, ToolUseBlock,
)
from tracing import EventStatus, EventType, TraceEmitter, new_action_id
from tool_router import ToolResultEnvelope, ToolResultStatus
import conformance_mode
import project_build
import request_intent
import work_order
from verification import (
    CompletionVerificationStatus, VerificationCategory, VerificationCoverage,
    VerificationEvidence, VerificationPolicy,
)
from budgets import BudgetExceeded, BudgetKind, BudgetManager
from working_state import (
    ActionKind, StateEvent, StateEventType, StateSource, TerminalStatus,
    VerificationOutcome, WorkingState,
)


class RuntimeTerminalReason(StrEnum):
    COMPLETED = "completed"
    REFUSAL = "refusal"
    MODEL_FAILURE = "model_failure"
    INVALID_TURN = "invalid_turn"
    ROUND_BUDGET_EXHAUSTED = "round_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    INTERRUPTED = "interrupted"
    RUNTIME_FAILURE = "runtime_failure"
    STALLED = "stalled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ToolExecutor(Protocol):
    def __call__(
        self, context: "AgentContext", tool_call_id: str, name: str,
        arguments: Mapping[str, object], cache: dict[Any, Any],
    ) -> ToolResultEnvelope: ...


class RuntimeObserver(Protocol):
    def model_activity(
        self, label: str,
    ) -> ContextManager[Callable[[], None] | None]: ...

    def intermediate_text(self, text: str) -> None: ...

    def notice(self, kind: str, metadata: Mapping[str, object]) -> None: ...


class NullRuntimeObserver:
    def model_activity(self, label: str):
        return nullcontext(None)

    def intermediate_text(self, text: str) -> None:
        return None

    def notice(self, kind: str, metadata: Mapping[str, object]) -> None:
        return None


@dataclass
class AgentContext:
    working_state: WorkingState
    backend: ModelBackend
    context_engine: ContextEngine
    trace: TraceEmitter
    system_prompt: str
    context_items: tuple[ContextItem, ...]
    conversation: list[ConversationMessage]
    tools: tuple[ToolDefinition, ...]
    tool_executor: ToolExecutor
    max_model_rounds: int
    max_tool_actions: int
    context_limit: int
    compaction_policy: CompactionPolicy = field(default_factory=CompactionPolicy)
    compaction_artifact: CompactionArtifact | None = None
    provider: str | None = None
    model: str | None = None
    output_reserve: int | None = None
    safety_margin: int = 768
    observer: RuntimeObserver = field(default_factory=NullRuntimeObserver)
    task_trace_metadata: Mapping[str, object] = field(default_factory=dict)
    trace_started: bool = False
    session: SessionHandle | None = None
    cancellation: CancellationToken = field(default_factory=lambda: NEVER_CANCELLED)

    # Whether the harness may ASK the model what the operator meant, when its
    # own verb list does not recognise the request. One short call per turn,
    # and only when the pattern is silent.
    #
    # Off by default, and switched on by the CLI. A behaviour that spends a
    # model call is opted into, never inherited: left on by default it
    # silently ate a scripted turn in every suite that builds a context of
    # its own, and those tests then measured the classifier instead of the
    # loop they were written for.
    judge_intent: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    progress_monitor: ProgressMonitor = field(default_factory=ProgressMonitor)
    result_references: set[str] = field(default_factory=set)
    session_tool_log: list[str] = field(default_factory=list)
    session_trajectory: list[Mapping[str, object]] = field(default_factory=list)
    verification_policy: VerificationPolicy = field(default_factory=VerificationPolicy)
    checkpoint_manager: CheckpointManager | None = None
    checkpoint: Checkpoint | None = None
    role: str = "main"
    exploration_reports: list[Mapping[str, object]] = field(default_factory=list)
    review_results: list[Mapping[str, object]] = field(default_factory=list)
    review_required: bool = False
    budget_manager: BudgetManager | None = None
    delegation_results: list[Mapping[str, object]] = field(default_factory=list)
    tool_exposure: Mapping[str, object] = field(default_factory=dict)

    # Optional training observer.  It is derived evidence only and never owns
    # runtime state or participates in resume.

    training_recorder: object | None = None

    # Optional task-scoped normative identity. Domain behavior remains in
    # registered tools and TaskController; the runtime only persists it.

    standard_binding: Mapping[str, object] | None = None

    #: The request forbade changing anything ("without editing files", "do
    #: not modify the code", "read-only"). Set by the controller from the
    #: tool view, and read by the command boundary: the write TOOLS are not
    #: offered, and bash runs without workspace write, so the prohibition
    #: cannot be routed around with `sed -i`, `cp` or a redirection.
    read_only: bool = False

    #: How this tree builds and tests itself, and where it lives. A writing
    #: turn is not finished until these pass; without them the harness can
    #: only report that it does not know.
    project_commands: object | None = None
    project_root: str = ""

    #: Runs one of those commands and returns (ok, what went wrong). Supplied
    #: by the caller so the project's own build and tests run where the
    #: model's commands run: the command is the project's, but the code it
    #: exercises was written by the model this turn. Without one the gate
    #: falls back to running it on the host, which is what callers with no
    #: sandbox have always done.
    project_verifier: object | None = None

    #: Filled in by the runtime on the way out: the clauses this turn read.
    clauses_read: tuple = ()

    #: Clauses the PREVIOUS turn of this session retrieved. The reasoning a
    #: change needs was already done when the question was answered; carrying
    #: it forward is what stops the next turn from starting over on the wrong
    #: sections.
    prior_clauses: tuple = ()

    #: The previous turn's answer, when it was bound. On a two-prompt
    #: shape it is the specification the change has to meet.
    prior_answer: str = ""
    standard_source_ids_used: set[str] = field(default_factory=set)

    # The per-turn deterministic policy, attached by the runtime when the turn
    # is bound to a standard. Turn-scoped state only: never persisted, never
    # resumed, and None on every non-standard turn.

    standard_policy: object | None = None

    def __post_init__(self) -> None:
        if self.max_model_rounds < 1:
            raise ValueError("max_model_rounds must be positive")

        if self.max_tool_actions < 1:
            raise ValueError("max_tool_actions must be positive")

        if self.context_limit < 1:
            raise ValueError("context_limit must be positive")

        if not self.role or not self.role.replace("_", "").isalnum():
            raise ValueError("role must be a simple non-empty identifier")

        # Named for tracing when the caller did not say; the backend knows what
        # it is, and an unlabelled trace is much harder to read afterwards.

        self.provider = self.provider or type(self.backend).__name__
        self.model = self.model or getattr(self.backend, "model", None)

        if self.budget_manager is None:
            self.budget_manager = BudgetManager.from_legacy(
                component=self.role, model_turns=self.max_model_rounds,
                tool_calls=self.max_tool_actions,
            )

    @classmethod
    def from_session_snapshot(
        cls, snapshot: SessionSnapshot, *, session: SessionHandle,
        backend: ModelBackend, context_engine: ContextEngine, trace: TraceEmitter,
        tools: tuple[ToolDefinition, ...], tool_executor: ToolExecutor,
        context_limit: int, output_reserve: int | None = None,
        observer: RuntimeObserver | None = None,
        compaction_policy: CompactionPolicy | None = None,
        verification_policy: VerificationPolicy | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> "AgentContext":
        """Rehydrate provider-neutral runtime dependencies around durable state."""

        if snapshot.session_id != session.session_id:
            raise ValueError("snapshot and session handle do not match")

        progress = (ProgressMonitor.from_dict(snapshot.progress_state)
                    if snapshot.progress_state else ProgressMonitor())

        max_rounds = snapshot.working_state.max_model_rounds or 30
        max_tools = snapshot.working_state.max_tool_actions or 60
        items = snapshot.context_items

        # A checkpoint is only reattached when the caller supplied a manager to
        # load it with; without one the task resumes unable to roll back.

        checkpoint = None

        if snapshot.checkpoint_id and checkpoint_manager is not None:
            checkpoint = checkpoint_manager.load(
                snapshot.checkpoint_id, task_id=snapshot.task_id,
                session_id=snapshot.session_id,
            )

        return cls(
            working_state=snapshot.working_state, backend=backend,
            context_engine=context_engine, trace=trace,
            system_prompt="".join(item.content for item in items),
            context_items=items, conversation=list(snapshot.conversation),
            tools=tools, tool_executor=tool_executor,
            max_model_rounds=max_rounds, max_tool_actions=max_tools,
            context_limit=context_limit, output_reserve=output_reserve,
            compaction_policy=compaction_policy or CompactionPolicy(),
            compaction_artifact=snapshot.compaction_artifact,
            observer=observer or NullRuntimeObserver(), session=session,
            progress_monitor=progress,
            result_references=set(snapshot.result_references),
            session_tool_log=list(snapshot.tool_log),
            session_trajectory=list(snapshot.trajectory),
            task_trace_metadata={"resumed": True, "raw_content_recorded": False},
            verification_policy=verification_policy or VerificationPolicy(),
            checkpoint_manager=checkpoint_manager, checkpoint=checkpoint,
            exploration_reports=list(snapshot.exploration_reports),
            review_results=list(snapshot.review_results),
            budget_manager=(BudgetManager.from_dict(snapshot.budget_state)
                            if snapshot.budget_state else None),
            delegation_results=list(snapshot.delegation_results),
            tool_exposure=dict(snapshot.tool_exposure),
            standard_binding=(dict(snapshot.standard_binding)
                              if snapshot.standard_binding else None),
            standard_source_ids_used=set(snapshot.standard_source_ids_used),
        )

    @property
    def task_id(self) -> str:
        return self.working_state.task_id

    @property
    def session_id(self) -> str | None:
        return self.session.session_id if self.session else None

    def apply_state_event(
        self, event_type: StateEventType,
        source: StateSource = StateSource.HARNESS, **data: Any,
    ) -> StateEvent:
        event = StateEvent.create(event_type, self.task_id, source, **data)

        # Applied first: the working state validates the transition, so a
        # rejected event is never traced or persisted as if it had happened.

        self.working_state.apply(event)

        self.trace.emit(
            EventType.WORKING_STATE_UPDATED,
            self.task_id,
            status=EventStatus.OK,
            action_id=data.get("action_id"),
            metadata={
                "state_event_type": event_type.value,
                "source": source.value,
                "current_round": self.working_state.current_round,
                "action_count": len(self.working_state.actions),
                "unresolved_failure_count": len(
                    self.working_state.unresolved_failures
                ),
                "modified_file_count": len(
                    self.working_state.modified_files
                    | self.working_state.created_files
                ),
                "terminal_status": self.working_state.terminal_status.value,
                "raw_content_recorded": False,
            },
        )

        if self.session is not None:
            self.session.append(
                SessionEventType.WORKING_STATE_TRANSITION, self.task_id,
                {"state_event_id": event.event_id,
                 "state_event_type": event_type.value,
                 "state_event": event.to_dict()},
            )

        return event

    def rollback_checkpoint(self) -> RollbackResult:
        """Rollback through the task context; never bypass task/session truth."""

        if self.checkpoint_manager is None or self.checkpoint is None:
            raise RuntimeError("no active task checkpoint")

        action_id = new_action_id("rollback")

        self.trace.emit(
            EventType.ROLLBACK_STARTED, self.task_id,
            session_id=self.session_id, status=EventStatus.STARTED,
            action_id=action_id,
            metadata={"checkpoint_id": self.checkpoint.checkpoint_id},
        )

        result = self.checkpoint_manager.rollback(self.checkpoint)

        self.apply_state_event(
            StateEventType.CHECKPOINT_STATUS_UPDATED,
            checkpoint_id=self.checkpoint.checkpoint_id,
            status=self.checkpoint.status.value,
        )

        # A rollback that did not fully succeed is itself a failed action, so
        # the working state carries it rather than only the trace.

        if result.status in {RollbackStatus.CONFLICT, RollbackStatus.FAILED,
                             RollbackStatus.PARTIAL}:
            self.apply_state_event(
                StateEventType.ACTION_FAILED, source=StateSource.HARNESS,
                action_id=action_id, kind=ActionKind.TOOL.value,
                name="checkpoint_rollback", observed_status="failed",
                category=(FailureKind.CHECKPOINT_CONFLICT.value
                          if result.status == RollbackStatus.CONFLICT
                          else FailureKind.RUNTIME_ERROR.value),
                summary=f"checkpoint rollback {result.status.value}",
                round_number=self.working_state.current_round,
            )

        event_type = (EventType.ROLLBACK_CONFLICT
                      if result.status == RollbackStatus.CONFLICT
                      else EventType.ROLLBACK_FINISHED)

        self.trace.emit(
            event_type, self.task_id, session_id=self.session_id,
            status=(EventStatus.OK if result.status == RollbackStatus.SUCCESS
                    else EventStatus.FAILED), action_id=action_id,
            metadata={"checkpoint_id": self.checkpoint.checkpoint_id,
                      "rollback_status": result.status.value,
                      "path_count": len(result.paths)},
        )

        AgentRuntime._persist_session(
            self, SessionEventType.ROLLBACK_RECORDED,
            {"checkpoint_id": self.checkpoint.checkpoint_id,
             "status": result.status.value,
             "path_results": [item.status.value for item in result.paths]},
        )

        return result


@dataclass
class AgentResult:
    task_id: str
    terminal_status: TerminalStatus
    terminal_reason: RuntimeTerminalReason
    final_response: str
    working_state: WorkingState
    rounds_consumed: int
    model_calls: int
    tool_actions: int
    verification_outcome: VerificationOutcome
    conversation: tuple[ConversationMessage, ...]
    transcript: tuple[str, ...]
    tool_log: tuple[str, ...]
    trajectory: tuple[Mapping[str, object], ...]
    did_modify: bool
    budget_exhausted: bool
    verification_action_id: str | None = None
    error_category: str | None = None
    error_summary: str | None = None
    failure_kind: FailureKind | None = None
    execution_cache: dict[Any, Any] = field(default_factory=dict, repr=False)
    completion_deferred: bool = False

    # Whether the project's own build and tests passed after this turn wrote.
    # None when nothing judged it: the turn changed nothing, or the tree
    # declares no build. That third state is the point -- a corpus that
    # cannot tell "failed" from "never checked" teaches the difference as if
    # it were noise, and until now every unjudged turn was filed as
    # "unrated" alongside turns the project had actually verified.
    project_build_ok: bool | None = None


_SYSTEM_CONTEXT_LAYERS = {
    ContextLayer.SYSTEM_RULES,
    ContextLayer.PROJECT_RULES,
    ContextLayer.DURABLE_MEMORY,
    ContextLayer.TASK_WORKING_STATE,
    ContextLayer.CONVERSATION_SUMMARY,
    ContextLayer.RETRIEVED_CONTEXT,
}
_WRITE_TOOLS = ("edit_file", "write_file", "append_file")


# A read that answers the same thing again has told the turn nothing. Said
# at the second, refused at the fourth: the first repeat can be a coincidence
# of two honest questions, the fourth is a loop.
_REPEAT_NOTICE = 2

# Refused at the third, not the fourth. Watched live: told twice that a
# result was one it already had, a turn asked a third time and kept going to
# a hundred and twenty-seven calls. Two warnings are enough warning.
_REPEAT_REFUSE = 3

# Below this a result is too small to identify: "0", "" and a one-line count
# repeat for perfectly good reasons.
_REPEAT_MIN_CHARS = 200


def _repeated_result(seen, envelope) -> int:
    """How many times this exact result has come back this turn.

    Mutations are exempt: writing the same content twice is idempotent and
    fine. Failures are exempt too -- the same error twice is the tree
    telling the truth twice, and the repair loop depends on hearing it.
    """
    text = envelope.text or ""

    if envelope.mutation or not envelope.success or len(text) < _REPEAT_MIN_CHARS:
        return 0

    key = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    seen[key] = seen.get(key, 0) + 1

    return seen[key]


def _with_repeat_note(envelope, count):
    """The same result, carrying the fact that it is the same.

    Refused outright past the limit: a turn that has been told three times
    and asked a fourth is not reading the answer, and handing it the wall of
    text again is how the loop is fed.
    """
    if count >= _REPEAT_REFUSE:
        return replace(
            envelope,
            model_content=(
                f"REFUSED: this call returns a result you have already been "
                f"given {count - 1} times. Nothing about it has changed. Use "
                f"what you have, or do something different -- and if the task "
                f"asks you to change the code, change it."),
        )

    return replace(
        envelope,
        model_content=(
            f"{envelope.model_content}\n\n[the same result as "
            f"{count - 1} earlier call(s) this turn: reading it again will "
            f"not add anything]"),
    )


def _narrowed(tools, names):
    """The tools offered this round, when the round is only allowed one thing.

    Telling a model "fix it now, do not read anything else first" is a
    request, and requests have been measured ineffective here nine times
    over. A build that fails three times running, answered three times with
    another `sed`, is the same measurement again. What the harness controls
    is what it OFFERS: for the one round after a failed build, the only tools
    on the table are the ones that change a file. Reading is not forbidden by
    instruction -- it is simply not available, for that round, and it comes
    back immediately afterwards.
    """
    if not names:
        return tools

    wanted = set(names)
    kept = tuple(tool for tool in tools if tool.name in wanted)

    return kept or tools

# How many times a turn may be sent back to an unfinished work order.
# Bounded because a redirect costs a round; more than one because a
# multi-section order is never finished by a single extra one.
ORDER_REDIRECT_LIMIT = int(os.environ.get("SPEAR_ORDER_REDIRECTS", "4"))

# How many times a turn may be sent back to a build it broke. A build
# failure is concrete and usually one edit away, so this is worth more
# rounds than a vaguer redirect -- but not unbounded: a model that cannot
# fix it in three will not fix it in ten.
BUILD_REPROMPT_LIMIT = int(os.environ.get("SPEAR_BUILD_RETRIES", "3"))

# How many times a turn may be sent back to a change it was asked for and has
# not made. One was the old cap, and one is measurably not enough: a turn was
# told "nothing was changed — make the edit", read for another twenty rounds,
# looped, and ended with an empty diff and nothing left to ask it. Bounded
# like the work-order redirect, and for the same reason: a redirect costs a
# round, and a redirect that changes nothing twice will not change anything
# on the third.
WRITE_REDIRECT_LIMIT = int(os.environ.get("SPEAR_WRITE_REDIRECTS", "3"))

# Rounds a turn gets to itself between two of those redirects. Raising the
# cap from one to three without this made the harness impatient rather than
# persistent: the model answered its first two rounds with "let me examine
# the source files", the preamble guard absorbed two, and the three
# redirects then fired in three consecutive rounds -- all of them spent
# before the turn had read anything at all. Asking again is only useful
# after the turn has had room to act on the last ask.
WRITE_REDIRECT_SPACING = int(os.environ.get("SPEAR_WRITE_REDIRECT_SPACING", "6"))

# How many times a turn may be sent back to the clauses its own session
# established and its diff does not account for.
CLAUSE_REDIRECT_LIMIT = int(os.environ.get("SPEAR_CLAUSE_REDIRECTS", "3"))
_INTENT_RE = re.compile(
    r"\b(let me|let's|let us|i'?ll|i will|i'?m going to|going to|"
    r"first,? (?:let|i|read|check)|now (?:let|i)|here'?s what)\b", re.I,
)
_FABRICATION_RE = re.compile(r"^\s*\[tool\].*$", re.M)


def strip_fabrications(text: str) -> str:
    return _FABRICATION_RE.sub("", text)


from standard_answer_policy import STOPPED_BY_BUDGET


# What the sandbox refused rather than ran: the tool never executed, so the
# round produced nothing the model could use.
_REFUSED = frozenset({ToolResultStatus.DENIED, ToolResultStatus.UNKNOWN_TOOL,
                      ToolResultStatus.INVALID_ARGUMENTS})


def no_conclusion_note(tool_log) -> str:
    """What a turn that did work and said nothing hands back.

    One wording, used by the loop's own fallback and by the exit-point
    guarantee behind it, so the two cannot drift apart.
    """
    return (
        "I stopped after " + str(len(tool_log)) + " tool call(s) without "
        "reaching a conclusion. What was observed:\n\n"
        + "\n\n".join(tool_log[-3:])[:1500]
        + "\n\nTell me which part to dig into, or narrow the request."
    )


# Reading rounds a turn gets before it is asked to conclude. Scaled to the
# budget, but CAPPED -- and the cap is the point. It used to be a plain
# fraction, which was fine at a 60-round ceiling and became absurd when the
# ceiling rose to 500: the nudge moved to a hundred rounds, and a master run
# read for thirty minutes with nothing to interrupt it and wrote nothing. A
# bigger budget should buy a turn more room to WORK, not more room to read
# before it starts.
INVESTIGATION_CEILING = int(os.environ.get("SPEAR_MAX_READING_ROUNDS", "40"))

# Rounds between two compactions, and the pressure that overrides the wait.
COMPACTION_COOLDOWN_ROUNDS = int(
    os.environ.get("SPEAR_COMPACTION_COOLDOWN", "4"))
COMPACTION_URGENT = float(os.environ.get("SPEAR_COMPACTION_URGENT", "0.92"))


def _tool_activity(name: str) -> str:
    """The status label for a running tool: what it is, in the same words.

    Same indicator, same colours, same clock as a model call -- only the
    label changes. Nothing here reformats the line: a tool that takes four
    seconds should look like work in progress, not like a different program.
    """

    return f"{name}…"


def _investigation_ceiling(round_limit) -> int:
    return min(INVESTIGATION_CEILING, max(12, round_limit // 5))


def _past_wall_fraction(context, fraction) -> bool:
    """Has the turn spent this much of its wall-clock budget?

    Read rather than counted: the manager updates the elapsed figure on
    every charge, so this needs no clock of its own and cannot drift from
    the limit that will actually end the turn.
    """
    manager = getattr(context, "budget_manager", None)

    if manager is None:
        return False

    limit = manager.limits.get(BudgetKind.WALL_TIME_MS)

    if limit is None or limit.amount <= 0:
        return False

    return manager.consumed.get(BudgetKind.WALL_TIME_MS, 0) >= limit.amount * fraction


def _round_was_all_refused(envelopes) -> bool:
    """Did the sandbox refuse every tool call of that round?

    A refusal is the harness's answer, not the turn's work, and a round made
    only of refusals gave the model nothing to go on.
    """
    if not envelopes:
        return False

    return all(envelope.status in _REFUSED for envelope in envelopes)


def _asked(conversation) -> str:
    """The operator's own words for this turn, which the policy routes on.

    Harness-authored messages are skipped. They carry the user role because
    the API has no other one, and reading them back as the request made the
    harness answer itself: a nudge that offers "make ONE small edit_file
    change" matched the write-request detector, so a turn that had only been
    asked a question was told it had changed nothing and demanded an edit.
    The model, correctly, refused to invent an old_text for a file it had
    never opened -- and the session's first prompt ended in an argument
    instead of an answer.
    """
    for message in reversed(list(conversation or ())):
        if getattr(message, "role", None) != "user":
            continue

        if getattr(message, "authored_by", "operator") != "operator":
            continue

        parts = [getattr(block, "text", "") for block in message.content
                 if getattr(block, "text", "")]

        if parts:
            return "\n".join(parts)

    return ""


def standard_policy_for(context: AgentContext, *, repair_ask=None):
    """The deterministic standard-bound policy for this turn, if it is one.

    Activation is the session's binding, never the user's phrasing, and never
    anything the model says. A turn with no bound standard gets no policy and
    behaves exactly as it did before -- the runtime stays generic and the
    normative reasoning stays in one place.

    `repair_ask` is the one tool-less question the policy may put back to the
    model when the guards reject an answer the evidence could still support.
    It travels from here because this is where the runtime's model access
    lives; without it the repair silently never ran in an interactive
    session, only in the evaluation harness.
    """
    if not context.standard_binding:
        return None

    import standard_answer_policy

    return standard_answer_policy.policy_for(context.standard_binding,
                                             _asked(context.conversation),
                                             repair_ask=repair_ask)


def looks_like_preamble(text: str) -> bool:
    value = (text or "").strip()
    return 0 < len(value) <= 400 and bool(_INTENT_RE.search(value))


def changed_files(tool_log: Sequence[str]) -> list[str]:
    result = []

    for entry in tool_log:
        head, _, output = entry.partition("\n")

        if head.split(" ", 1)[0] in _WRITE_TOOLS:
            match = re.search(r'"path"\s*:\s*"([^"]+)"', head)

            if match and output.startswith("OK"):
                result.append(match.group(1))

    return list(dict.fromkeys(result))


# What actually establishes that a change works. `ls`, `cat` and `git status`
# are not on this list: a turn that edited four files, ran `ls`, `cat
# CMakeLists.txt` and `cmake -S . -B build`, and stopped, had "run something
# after the change" and had verified nothing. It left the tree not compiling.
_VERIFYING = frozenset({
    VerificationCategory.BUILD,
    VerificationCategory.UNIT_TEST,
    VerificationCategory.INTEGRATION_TEST,
    VerificationCategory.LINT,
    VerificationCategory.STATIC_ANALYSIS,
    VerificationCategory.EXECUTION,
})

# rag_chat appends this to a command result whose status was not zero.
_EXIT_CODE_RE = re.compile(r"\(exit (\d+)\)")


def _verification_runs(tool_log: Sequence[str], policy=None) -> list[tuple[str, bool]]:
    """Every command after the last write that verifies anything, and whether
    it succeeded. Configuring a build is not building it: `cmake -S . -B
    build` classifies as BUILD and would pass for verification on its name
    alone, so a run counts only when its own output carries no failing exit
    code.

    What changed is passed to the classifier, because a command that
    exercises none of it verifies none of it: a turn that edited a Python
    file and compiled a standalone C program was, without this, a turn with a
    passing build behind it.
    """
    policy = policy or VerificationPolicy()
    last_change, runs = -1, []
    changed = changed_files(tool_log)

    for index, entry in enumerate(tool_log):
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name in _WRITE_TOOLS and output.startswith("OK"):
            last_change, runs = index, []
        elif name == "bash" and last_change >= 0:
            # The command itself, not the JSON around it: classify_command
            # anchors on the start of the string, and `./test_command` inside
            # {"command": "..."} matches nothing.
            found = re.search(r'"command"\s*:\s*"(.*)"\s*\}?\s*$', head)
            command = found.group(1) if found else ""
            category, _ = policy.classify_command(command, changed_paths=changed)

            if category in _VERIFYING:
                found = _EXIT_CODE_RE.search(output)
                runs.append((command, found is None or found.group(1) == "0"))

    return runs


def unverified_change(tool_log: Sequence[str]) -> bool:
    """A change with nothing at all run after it.

    Deliberately looser than _verification_runs: this drives the mid-turn
    nudge, which asks the model to go and verify, and a turn that ran
    something has at least been pointed at the question. What that something
    established is settled at the end, by unverified_write_note, where the
    answer is already written and the judgement costs no round.
    """
    last_change, ran_after = -1, False

    for index, entry in enumerate(tool_log):
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name in _WRITE_TOOLS and output.startswith("OK"):
            last_change, ran_after = index, False
        elif name == "bash" and last_change >= 0:
            ran_after = True

    return last_change >= 0 and not ran_after


def unverified_write_note(tool_log: Sequence[str], project_runs=()) -> str:
    """What to append when a turn changed files and never showed they work.

    The sibling of unsupported_change_claim, for the opposite fault. That one
    refuses to hand over a claim the tool log contradicts; this one refuses to
    hand over a change the tool log never stands behind.

    A turn edited four C files, spent its last rounds configuring a build,
    and was cut by the round budget on "The changes have been made. Let me
    verify the build works". The build did not work -- a declaration had been
    added without the old one being removed, and the tree no longer compiled.
    Nothing in the answer said so, and the nudge that exists for this could
    not fire, because `cmake` after an edit counted as having run something.

    So the last word on a writing turn is deterministic and comes from the
    tool log: what changed, whether anything verified it, and what that
    verification said. It fires on every exit path, including the budget's,
    which is the one that produced this.
    """
    # Anchored on files the log actually names. A mutation this cannot name
    # is one it cannot describe either, and a warning that says "something
    # changed" helps nobody.

    changed = changed_files(tool_log)

    if not changed:
        return ""

    # The harness's own run of the project's build and tests counts here too.
    # Without this the note said "nothing was run afterwards" to a turn whose
    # tree had just been built and tested by the harness -- and a warning the
    # turn can see is false is a warning it learns to ignore. A command that
    # FAILED is reported by project_build.note, in its own words, so this one
    # stays quiet rather than saying the same thing less precisely.

    if any(status in {"passed", "failed"} for _, status, _ in project_runs):
        return ""

    runs = _verification_runs(tool_log)
    files = ", ".join(changed)

    if not runs:
        return (f"\n\n⚠ {files} changed, and nothing was run afterwards that "
                f"could show the change works — no build, no test, no lint. "
                f"Treat the change above as unverified.")

    # Order decides. A turn began with `make clean`, which failed because the
    # project has no clean target, and then compiled every edited file
    # successfully. Reporting the earlier failure called a working change
    # broken. What stands is the last verification: a failure still counts
    # when nothing after it passed.

    if runs[-1][1]:
        return ""

    failed = runs[-1][0]

    return (f"\n\n⚠ {files} changed, and the last verification to run did "
            f"not pass: `{failed[:120]}`. The change above is not shown to "
            f"work as written.")


def verify_demand(tool_log: Sequence[str]) -> str:
    files = ", ".join(changed_files(tool_log)) or "a file"
    return (f"You changed {files} and ran nothing afterwards, so nothing shows "
            f"the change works. Verify it now with bash: build it, and run it "
            f"if it can run here — including the cases that could reasonably "
            f"fail, not only the one the user named. Then conclude, naming what "
            f"you actually observed. If it genuinely cannot be built or run "
            f"here, say that plainly instead of reporting it as working.")


# "I have created /path/x.rst" -- a claim of a completed write, in the perfect
# or the passive. Intent ("I'll create") is deliberately excluded: saying what
# you are about to do is not a false report.

# The verbs a finished write is reported with. "implemented", "completed",
# "done" and "applied" were missing, and they are the ones a TASK is reported
# with rather than a file: on a fresh checkout, having read the task file and
# grepped for the enum it asks for -- exit 1, "not yet defined" -- the model
# answered "The task in doc/ack-task.md has already been implemented, adding
# the CmdAckKind enum and updating all relevant files". Nothing was written.
_DONE_VERB = (r"(?:created|written|added|updated|modified|fixed|implemented"
              r"|completed|applied|done|made)")

_CHANGE_CLAIM_RE = re.compile(
    r"\b(?:i (?:have |'ve )?" + _DONE_VERB
    # "has ALREADY been implemented": an adverb between the auxiliary and the
    # participle is the ordinary way to say it, and it defeated `has been`.
    + r"|(?:has|have|had|was|were|is|are)\s+(?:\w+\s+){0,2}been\s+" + _DONE_VERB
    # "was successfully applied", with no "been" at all. Past tense only:
    # "is implemented in command_wire.c" describes code that exists, which is
    # not a report of work done this turn.
    + r"|(?:was|were)\s+(?:\w+\s+){0,2}" + _DONE_VERB
    + r")\b",
    re.I)
_FILE_TOKEN_RE = re.compile(r"[\w./-]+\.[A-Za-z]\w{0,4}\b")

# The mirror image of _CHANGE_CLAIM_RE: not "I updated the file" but "I'll
# update the file". One reports a write that did not happen; the other
# promises one and stops.
_ANNOUNCED_CHANGE_RE = re.compile(
    r"\b(?:i'?(?:ll|m going to)|i will|let me|now i'?ll|next,? i'?ll|"
    r"i can|i'?m about to|going to)\s+"
    r"(?:\w+\s+){0,4}?"
    r"(?:implement|make|apply|add|create|write|update|modify|change|fix|"
    r"edit|introduce|refactor|rename|remove|delete)\b", re.I)


def announced_but_unmade_change(text: str, tool_log: Sequence[str],
                                did_modify: bool) -> bool:
    """Did the answer promise an edit the turn then never made?

    A bound turn read twelve windows of C, worked out the change, and ended
    on "I'll implement the change by adding the CmdAckKind enum and modifying
    the encode function". Nothing was written. The task file had asked for an
    implementation, so a plan is not a smaller answer -- it is no answer, and
    the next turn starts from nothing and pays for the reading again.

    The harness already refuses to hand over a claim of a write that did not
    happen (unsupported_change_claim). This is the same evidence read the
    other way: the write was announced, the tool log holds no write, so the
    turn is not finished. It costs one more round, which is what the nudge to
    conclude costs in the opposite direction.
    """
    if did_modify or not text:
        return False

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return False

    return bool(_ANNOUNCED_CHANGE_RE.search(text)
                and _FILE_TOKEN_RE.search(text))


# A turn whose request is a write, in the user's own words. Not a guess about
# intent: these are imperatives naming an edit.
_WRITE_REQUEST_RE = re.compile(
    r"\b(?:do|make|apply|carry\s+out|perform)\s+(?:the\s+|these\s+|those\s+|"
    r"all\s+(?:the\s+)?)?(?:modifications?|changes?|edits?|task|work|fix(?:es)?)"
    r"|\b(?:implement|write|edit|patch|refactor|rename|add|create|update|fix|"
    r"modify|remove|delete|adapt|adjust|amend|revise|rework|correct)\b"
    r"|\b(?:fais|faire|applique|implémente|implementer|implémenter|corrige|"
    r"modifie|modifier|adapte|adapter|ajuste|ajuster|ajoute|écris|ecris)\b",
    re.I)


def wants_write(context, question) -> bool:
    """The turn's answer to "was I asked to change the code", decided once.

    Cached on the context because every gate below asks it and the answer
    cannot change inside a turn -- and because the model-side half costs a
    call, which is worth paying once and not once per round.

    An explicit prohibition is checked FIRST, before the cache and before any
    reading of the words: it is the strongest statement of intent there is,
    and the pattern below cannot see past a single token. Asked "Answer which
    file defines add without editing files", `\badd\b` matched the FUNCTION'S
    NAME, the turn was classified as a writing request, and every gate that
    asks this then spent rounds demanding an edit the user had forbidden.
    """
    if getattr(context, "read_only", False):
        # Written through, so the cache and the decision cannot disagree
        # later in the turn.

        context._write_request = False

        return False

    cached = getattr(context, "_write_request", None)

    if cached is not None:
        return cached

    pattern = is_write_request(question)
    model = None

    if not pattern and getattr(context, "judge_intent", False):
        # Only asked when the pattern is silent: the floor is already
        # decided, and a call that cannot change the answer is a call not
        # worth making.
        model = request_intent.judge(
            getattr(context, "backend", None), question,
            budget_manager=getattr(context, "budget_manager", None),
            budget_kind=BudgetKind.AUXILIARY_MODEL_CALLS)

        if model is not None:
            context.observer.notice(
                "intent_judged", {"write": model, "asked": question[:80]})

    answer = request_intent.resolve(pattern, model)
    context._write_request = answer

    return answer


def is_write_request(question: str) -> bool:
    """Did the user ask for the tree to change, in so many words?"""
    return bool(_WRITE_REQUEST_RE.search(question or ""))


def may_demand_write(context) -> bool:
    """Whether this turn is allowed to be sent back to the code at all.

    One question, asked by every gate that would inject a write demand,
    reopen the write window, or narrow the tools to the writing ones. A turn
    the user told not to change anything is finished by an answer, and a
    harness that keeps asking for an edit is asking it to disobey.
    """

    return not getattr(context, "read_only", False)


def carried_obligations(context, question) -> tuple:
    """The clauses an earlier turn established that THIS turn owes work on.

    Clauses carried across turns are EVIDENCE, not a checklist. The carry
    exists for one shape -- "how should X work?", then "change the code" --
    where the reasoning was done in the first prompt and the second would
    otherwise start over on the wrong sections. It says what a change has to
    satisfy; it does not say that a change was asked for.

    Nothing enforced that. The completion gate read `prior_clauses` alone,
    so a session that had established ten clauses turned the NEXT question
    -- "is this explicitly required by the standard, and which clauses say
    so?" -- into an implementation audit: the answer was complete and
    correct, and the harness then demanded file-and-line for eight clauses
    the user had not asked about, which the turn duly supplied by editing
    the customer's tree, building it and running its tests.

    So the obligation is decided by THIS turn's request, by the same test
    that decides whether the clauses are put in front of the model at all.
    A gate may only ask for what the turn was actually given to do: if the
    request is a question, the carried clauses are context for answering it
    and the turn is finished when it is answered.

    The execution mode is deliberately absent. `--auto` says which actions
    may proceed without confirmation; it has never had a say in which
    actions are in scope, and a gate that widened the task because
    confirmation was cheap would be reading it as permission to do more.
    """
    prior = tuple(getattr(context, "prior_clauses", ()) or ())

    if not prior:
        return ()

    # A turn told not to change anything cannot be sent back to the code,
    # whatever its words otherwise match.

    if not may_demand_write(context):
        return ()

    return prior if is_write_request(question) else ()


def conclude_demand(question: str, is_write: bool | None = None) -> str:
    """The nudge that ends a turn which has only been reading.

    It used to offer two branches -- answer the question, or make the edit --
    and let the model pick. Asked "please do the modifications", a turn that
    had read eight files and knew the enum was missing took the first branch
    and re-emitted the previous turn's prose, word for word. Nothing was
    written, and the answer closed by saying the task "has already been
    implemented".

    A request to change the tree does not have a question branch. When the
    user's own words are an imperative to edit, the only way out of the nudge
    is the edit.

    ``is_write`` is the turn's already-resolved answer -- prohibition,
    pattern, and the model judgement if one was made. Re-reading the words
    here made this the last place that could still call "Answer which file
    defines add" an instruction to edit, after every other gate had agreed it
    was not. The pattern stays as the fallback for callers with no turn to
    ask (the tests, and nothing else).
    """
    if is_write_request(question) if is_write is None else is_write:
        return (
            "You have investigated enough. The user asked you to change the "
            "code, so this turn is not finished until a file changes. Make "
            "ONE small edit_file change now: the single most important edit, "
            "with a SHORT old_text (the few exact lines to replace, copied "
            "verbatim from what you read) and the new_text. Call NO other "
            "tool first: no shell command and no retrieval of any kind. "
            "Told only to stop reading files, a turn spent its last seven "
            "rounds on retrieval instead and wrote nothing. Do NOT rewrite "
            "whole files, do NOT summarise what you read, and do NOT report "
            "the task as already done -- nothing has been written yet. You "
            "can make more small edits after this one."
        )

    return (
        "You have investigated enough. Conclude now, with no further "
        "bash/grep/find/cat/read commands. If the user asked a question, "
        "answer it from what you found, citing the exact paths. If a change "
        "was requested, make ONE small edit_file change: the single most "
        "important fix, with a SHORT old_text (the few exact lines to "
        "replace) and the new_text — do NOT rewrite the whole file, keep it "
        "under ~15 lines so it completes. You can make more small edits after."
    )


def make_it_demand(text: str) -> str:
    """The one message that turns an announcement into the edit itself."""
    return (
        "You said you would make this change and then stopped, so the turn "
        "wrote nothing. Make it now: call edit_file with a SHORT old_text "
        "(the few exact lines to replace, copied verbatim from the file you "
        "read) and the new_text. Do not re-read, do not restate the plan, do "
        "not rewrite whole files. If the change spans several files, make the "
        "single most important edit now; you can make the others after."
    )


def unsupported_change_claim(text: str, tool_log: Sequence[str],
                             did_modify: bool) -> str:
    """A note to append when the answer claims a write the turn never made.

    A cut turn asked to conclude produced, verbatim, the previous turn's
    summary from history: "I have created the documentation chapter at
    .../user_space_ls.rst", plus an account of a sed refusal that happened in
    another turn. Nothing had been written -- the tool log held four reads.

    The harness cannot stop a model from claiming that. It can refuse to hand
    the claim over as the turn's result while holding the evidence that it is
    false, which is the same rule it applies to sandboxes and budgets: what is
    reported is what was observed.
    """

    if did_modify or not text:
        return ""

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return ""

    if not _CHANGE_CLAIM_RE.search(text) or not _FILE_TOKEN_RE.search(text):
        return ""

    return ("\n\n⚠ Nothing was written this turn: no file was created or "
            "modified, and the tool log holds only reads. Any claim above that "
            "a file was created or updated describes work that did not happen "
            "here — treat it as a plan, not a result.")


def turn_evidence(tool_log: Sequence[str]) -> str:
    if not tool_log:
        return ""

    commands = failures = 0
    modified = []

    for entry in tool_log:
        head, _, output = entry.partition("\n")
        name = head.split(" ", 1)[0]

        if name == "bash":
            commands += 1

            if output.startswith("ERROR:") or re.search(r"\(exit [1-9]", output):
                failures += 1
        elif name in _WRITE_TOOLS:
            match = re.search(r'"path"\s*:\s*"([^"]+)"', head)

            if match and output.startswith("OK"):
                modified.append(match.group(1))

    files = ", ".join(dict.fromkeys(modified)) if modified else "none"

    return (f"\n\nFacts from this turn's tool log — {commands} command(s) run, "
            f"{failures} of them exited non-zero; files changed: {files}. "
            f"Base your summary on tool results, not on what you meant to do. "
            f"If you report behaviour as working, name the cases whose output "
            f"you actually saw, and say which cases you did not test. A command "
            f"that printed nothing did not confirm anything.")


def validate_agent_turn(turn: ModelTurn) -> str:
    if turn.stop_reason == StopReason.TOOL_USE:
        if not turn.tool_calls:
            raise RuntimeError("provider returned TOOL_USE without tool_calls")

        return "tool_use"

    if turn.tool_calls:
        raise RuntimeError(
            f"provider returned tool_calls with terminal stop_reason {turn.stop_reason}"
        )

    if turn.stop_reason == StopReason.END_TURN:
        return "end_turn"

    if turn.stop_reason == StopReason.REFUSAL:
        return "refusal"

    raise RuntimeError(turn.error or f"model backend stopped with {turn.stop_reason}")


def _message_context_content(message: ConversationMessage) -> tuple[str, bool]:
    parts = []
    text_only = True

    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            text_only = False
            parts.append(block.content)
        elif isinstance(block, ToolUseBlock):
            text_only = False
            parts.append(block.name + " " + json.dumps(
                dict(block.arguments), ensure_ascii=False, sort_keys=True,
            ))

    return "\n".join(parts), text_only


def _conversation_context_items(
    conversation: Sequence[ConversationMessage],
) -> tuple[ContextItem, ...]:
    items = []
    call_groups: dict[str, str] = {}
    plain_group = None
    count = len(conversation)

    for index, message in enumerate(conversation):
        content, text_only = _message_context_content(message)
        calls = [block for block in message.content if isinstance(block, ToolUseBlock)]
        results = [block for block in message.content if isinstance(block, ToolResultBlock)]

        # The eviction group is what keeps a call and its results together: drop
        # one without the other and the conversation stops being well-formed.

        if calls:
            group = "tool-exchange:" + ",".join(block.id for block in calls)

            for block in calls:
                call_groups[block.id] = group
        elif results:
            groups = sorted({call_groups.get(
                block.tool_call_id, f"tool-result:{block.tool_call_id}"
            ) for block in results})
            group = "+".join(groups)
        else:
            # A plain exchange is grouped from its user message through to the
            # assistant reply that closes it.

            if message.role == "user":
                plain_group = f"conversation-exchange:{index}"

            group = plain_group or f"conversation-message:{index}"

            if message.role == "assistant":
                plain_group = None

        distance = count - index
        freshness = (Freshness.CURRENT if distance <= 3 else
                     Freshness.RECENT if index >= count // 2 else Freshness.STALE)
        is_tool = bool(calls or results)

        items.append(ContextItem(
            item_id=f"conversation:{index}",
            layer=(ContextLayer.TOOL_EVIDENCE if is_tool
                   else ContextLayer.RECENT_CONVERSATION),
            source="tool_protocol" if is_tool else "conversation_history",
            content=content,
            priority=100 if index == count - 1 else min(85, 35 + index),
            freshness=freshness,
            protected=distance <= 3,
            token_overhead=4,
            inclusion_reason=("preserve tool exchange" if is_tool
                              else "recent conversational continuity"),
            reference=(message, content, text_only),
            eviction_group=group,
            truncatable=text_only,
        ))

    return tuple(items)


def _render_context_snapshot(
    snapshot: ContextSnapshot,
) -> tuple[str, tuple[ConversationMessage, ...]]:
    system = "".join(
        item.content for item in snapshot.selected_items
        if item.layer in _SYSTEM_CONTEXT_LAYERS and item.reference is None
    )
    conversation: list[ConversationMessage] = []

    for item in snapshot.selected_items:
        if item.reference is None:
            continue

        message, original_content, text_only = item.reference

        if text_only and item.content != original_content:
            message = ConversationMessage(message.role, (TextBlock(item.content),),
                                          authored_by=message.authored_by)

        # Merged only within one author. Folding a harness nudge into the
        # operator's question would put the nudge's words back into what the
        # routing reads as the request, which is the whole thing authored_by
        # exists to prevent.
        if (conversation and conversation[-1].role == message.role
                and conversation[-1].authored_by == message.authored_by
                and all(isinstance(block, TextBlock)
                        for block in conversation[-1].content)
                and all(isinstance(block, TextBlock) for block in message.content)):
            prior = "".join(block.text for block in conversation[-1].content)
            current = "".join(block.text for block in message.content)
            conversation[-1] = ConversationMessage(
                message.role, (TextBlock(prior + "\n\n" + current),),
                authored_by=message.authored_by,
            )
        else:
            conversation.append(message)

    return system, tuple(conversation)


# The completion states a turn may still be argued out of: either nothing
# covered the change, or what covered it did not pass.

_NEEDS_VERIFICATION = frozenset({
    CompletionVerificationStatus.UNVERIFIED,
    CompletionVerificationStatus.PARTIALLY_VERIFIED,
    CompletionVerificationStatus.FAILED,
})

_PROJECT_OUTCOMES = {
    "passed": VerificationOutcome.PASSED,
    "failed": VerificationOutcome.FAILED,
    "not_run": VerificationOutcome.NOT_RUN,
}


def _record_project_verification(context, commands, runs):
    """Record what the project's own build and tests just said about this turn.

    Full coverage, because these are the commands the project declared as its
    verification -- not a guess about which files a command happened to reach.
    A run that could not be performed is recorded as NOT_RUN, which is not a
    pass and never becomes one.
    """

    if not runs or not getattr(commands, "configured", False):
        return

    policy = getattr(context, "verification_policy", None)
    state = getattr(context, "working_state", None)

    if policy is None or state is None:
        return

    # A turn that has already ended -- interrupted, failed, out of budget --
    # accepts no further state events. The verification still happened; there
    # is simply no longer a completion for it to settle.

    if state.terminal_status != TerminalStatus.RUNNING:
        return

    # Recorded once per generation, like the run itself. The completion gate
    # asks for this before deciding whether to demand a verification, and the
    # end of the turn asks again; both see the same single run, and the
    # working state must not carry it twice.

    if getattr(context, "_project_verification_recorded", None) == state.mutation_generation:
        return

    try:
        context._project_verification_recorded = state.mutation_generation
    except (AttributeError, TypeError):
        pass

    for command, status, text in runs:
        category = (VerificationCategory.UNIT_TEST if project_build.kind(command) == "test"
                    else VerificationCategory.BUILD)
        outcome = _PROJECT_OUTCOMES.get(status, VerificationOutcome.NOT_RUN)
        AgentRuntime._record_verification(context, VerificationEvidence(
            "verification_" + uuid.uuid4().hex, category,
            VerificationCoverage.FULL, status != "not_run", outcome,
            context.working_state.mutation_generation,
            action_id=new_action_id("project_verify"), command=command,
            summary=f"project {project_build.kind(command)} {status}"
                    + (f": {text.splitlines()[0][:120]}" if text else ""),
        ))


def project_build_runs(context, tool_log):
    """The project's own verification, run once per generation of changes.

    Each call used to re-run the tree's build and tests, and the gate is
    consulted up to six times in a turn: a project whose suite takes a minute
    paid six. Nothing can have changed between two calls at the same mutation
    generation, so the first answer is the answer until something is written.

    Returns one (command, status, output) per command, status being "passed",
    "failed" or "not_run" -- and "not_run" is not "passed": a verification
    that could not be performed decided nothing.
    """
    commands = getattr(context, "project_commands", None)
    root = getattr(context, "project_root", "") or "."
    verifier = getattr(context, "project_verifier", None)

    if commands is None or not changed_files(tool_log):
        return ()

    state = getattr(context, "working_state", None)
    generation = getattr(state, "mutation_generation", 0)
    cached = getattr(context, "_project_verification", None)

    if cached is not None and cached[0] == generation:
        return cached[1]

    runs = []

    for command in commands.verifies():
        if verifier is not None:
            status, output = verifier(command)
        else:
            # No verifier: the caller has no execution boundary to offer, so
            # nothing is run rather than run somewhere unexamined.

            status, output = "not_run", "no confined runner for project verification"

        runs.append((command, status, output))

        if status != "passed":
            # `verifies()` is ordered: the tests do not run against a tree
            # that did not build.

            break

    runs = tuple(runs)

    try:
        context._project_verification = (generation, runs)
    except (AttributeError, TypeError):
        pass

    return runs


def project_build_gap(context, tool_log):
    """The first of the project's own commands that did not pass, if any.

    A change that has not been built is not a change that works. Reporting
    that was the first step; handing the round back is the one that finishes
    the turn. Nothing here is task-specific: it is the tree's own build and
    its own tests, and a turn that never wrote is never asked.
    """
    for command, status, output in project_build_runs(context, tool_log):
        if status != "passed":
            return command, output

    return "", ""


def asked_to_write_and_did_not(question, tool_log, did_modify,
                               *, is_write=None) -> bool:
    """The user asked for the code to change, and nothing changed.

    Narrower than it looks, and it has to be: the two other gates need the
    answer to say something first -- one catches a claim of a write, the
    other a promise of one. A turn that reads for sixty rounds and then
    answers in prose says neither, so both stay quiet and the turn is handed
    back with an empty diff. Asked "can you validate and make changes in the
    code accordingly", that is not a smaller answer; it is the wrong one.

    The request is the user's own words, and the evidence is the tool log.
    Neither is a guess about intent.
    """
    if did_modify or changed_files(tool_log):
        return False

    if any(entry.split(" ", 1)[0] in _WRITE_TOOLS for entry in tool_log):
        return False

    # `is_write` is the turn's decided answer -- pattern plus reading -- and
    # the pattern alone is only the fallback for callers that have no
    # context to decide from (the tests, and nothing else).
    return is_write_request(question or "") if is_write is None else is_write


def write_demand(question: str) -> str:
    """The one message that sends an empty writing turn back to the code."""
    return (
        "You were asked to change the code and this turn has not changed "
        "anything: the tool log holds reads only. Make the edit now, with "
        "edit_file and a SHORT old_text copied verbatim from a file you have "
        "read. Do not read anything else first -- no shell command and no "
        "retrieval. If after looking you believe no change is needed, say "
        "exactly which file and which lines already do what was asked, and "
        "why; an answer that describes the change instead of making it is "
        "not an answer to this request."
    )


def work_order_gap(context, tool_log, cache):
    """The sections this turn has not finished, and the order they are in.

    An order that supplies acceptance checks is judged by running them --
    file granularity marked a section done because a neighbouring section
    shares its file. One that supplies none falls back to the diff.
    """
    order = work_order.find(context.conversation)

    if not order:
        return "", []

    checks = work_order.acceptance(order)

    if not checks:
        return order, work_order.unaddressed(
            order, changed_files(tool_log))

    outputs = {}

    def run(command):
        try:
            envelope = context.tool_executor(
                context, f"acceptance-{abs(hash(command)) % 10**8}",
                "bash", {"command": command}, cache,
            )
        except Exception:                                   # noqa: BLE001
            return False

        # rag_chat appends "(exit N)" to a command that did not exit
        # zero, and nothing to one that did.
        text = envelope.text or ""
        found = re.search(r"\(exit (\d+)\)", text)

        ok = bool(envelope.success) and (found is None
                                         or found.group(1) == "0")

        if not ok:
            # What the check PRINTED, not only what it was. Told a section
            # failed `test $(... | grep -c ...) -eq 1`, a turn could not see
            # that the count was 2 and that the fix was a deletion; the
            # output says so in one number.
            outputs[command] = " ".join(text.split())[:200]

        return ok

    missing = work_order.unmet(order, run)

    for item in missing:
        item["output"] = outputs.get(item["check"], "")

    return order, missing


class AgentRuntime:
    """Reusable execution lifecycle. Instances contain no task state."""

    def run(self, context: AgentContext, *, defer_completion: bool = False) -> AgentResult:
        transcript: list[str] = []
        tool_log = context.session_tool_log
        trajectory = context.session_trajectory
        cache: dict[Any, Any] = {}
        response = ""

        # One deterministic policy per standard-bound turn. None for every
        # other turn, which then takes exactly the path it took before.

        policy = standard_policy_for(
            context, repair_ask=lambda text: self._repair_ask(context, text))
        context.standard_policy = policy

        investigate_rounds = preamble_reprompts = repeats = make_reprompts = 0
        scope_final = False
        order_reprompts = 0
        order_remaining = None
        build_reprompts = 0
        clause_reprompts = 0
        did_modify = nudged = verify_reprompts = budget_event_emitted = False
        only_tools: tuple[str, ...] | None = None
        wrap_up_warned = False
        seen_results: dict[str, int] = {}
        last_write_redirect = -WRITE_REDIRECT_SPACING
        last_clause_redirect = -WRITE_REDIRECT_SPACING
        last_build_output = ""
        round_envelopes: list = []
        rounds_since_nudge = 0
        verification_action_id = None
        reason = RuntimeTerminalReason.COMPLETED

        # Which stage the loop is in, so an exception below can name what was
        # being done rather than reporting an undifferentiated runtime failure.

        failure_stage = "runtime"

        try:
            self._start_task(context)
            context.cancellation.raise_if_cancelled()

            # A bound turn reads the standard before it reads anything else.
            # The boundary bootstrap already made this call, but only once the
            # model had finished: a turn that spent twelve rounds in the C
            # sources got its one normative search delivered into a context
            # that already held twelve greps and a written answer, and
            # concluded the code compliant anyway. The same call placed here
            # is a source rather than a rebuttal.

            if policy is not None:
                opening = policy.opening()

                if opening is not None:
                    self._inject_evidence_call(context, policy, opening, cache,
                                               tool_log)

            # A resumed task continues from the round it reached, so the range
            # starts at the working state rather than at zero.

            # The bound is a variable, not a literal range: a turn that ends
            # with the tree not compiling has not finished, and the one thing
            # worth spending a round beyond the budget on is making it build
            # again. Raised at most BUILD_REPROMPT_LIMIT times, by two rounds
            # each -- enough to read an error and make one edit.

            round_limit = context.max_model_rounds
            round_index = context.working_state.current_round - 1

            while round_index + 1 < round_limit:
                round_index += 1
                context.apply_state_event(
                    StateEventType.ROUND_STARTED, round_number=round_index + 1,
                )

                # "Conclude now, with no further commands" was advisory, and a
                # model that ignored it kept investigating for another eight
                # tool calls. It now costs exactly one more round: enough to
                # make the single small edit the nudge also asks for, and not
                # enough to resume grepping. Anything else makes the promise in
                # that message false, which is worse than not making it.

                # `round_envelopes` still holds the PREVIOUS round here: it is
                # reset further down, after this check.
                if nudged and not _round_was_all_refused(round_envelopes):
                    rounds_since_nudge += 1

                # A round the sandbox refused outright is not a round the
                # turn was given. The nudge grants exactly one round with
                # tools before the window closes, and twice in a row a turn
                # spent it on a no-op the allowlist denied -- `true`, then
                # `echo skip` -- and was then asked to conclude with no
                # tools and nothing done. It answered with nothing at all.
                # Refusing a call and charging for it is the harness taking
                # back what it just offered.

                # Two different things, and conflating them cost a user their
                # answer: a BUDGET stop is a failure the controller reports as
                # such, while CONCLUDING is the normal end of a turn that was
                # told to wrap up. Both close the tool window; only the first
                # is an exhausted budget. Routing the nudge through the budget
                # branch marked the task BUDGET_EXHAUSTED, and the CLI then
                # dropped the model's conclusion -- after its edit had landed.

                # Against the live bound, not the original one: a turn
                # granted extra rounds to repair a build would otherwise
                # spend them with its tool window already closed.
                budget_final = (len(tool_log) >= context.max_tool_actions
                                or round_index == round_limit - 1)
                concluding = rounds_since_nudge > 1

                # The budget is about to close the tool window. If the turn
                # wrote and the tree no longer builds, that is not a finished
                # turn, and the boundary check below will never be reached:
                # a turn that runs out of rounds never stops making tool
                # calls, so it never passes through the place where the
                # answer is judged. Repair is decided here instead, where the
                # window actually closes.

                if (budget_final and build_reprompts < BUILD_REPROMPT_LIMIT
                        and may_demand_write(context)
                        and changed_files(tool_log)):
                    broken, output = project_build_gap(context, tool_log)

                    if broken:
                        build_reprompts += 1
                        round_limit = round_index + 3
                        budget_final = False
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(project_build.demand(
                                broken, output, last_build_output)),),
                            authored_by="harness"))
                        last_build_output = output
                        only_tools = _WRITE_TOOLS
                        context.observer.notice(
                            "project_build_failed",
                            {"command": broken, "retry": build_reprompts},
                        )

                # A second attempt to change a file the user said not to
                # change. The first refusal named the scope and said what to
                # do instead; a turn that goes looking for another syntax is
                # not going to be argued out of it by a third. The tool
                # window closes and the answer is asked for.

                if max((envelope.metadata.get("read_only_violations", 0)
                        for envelope in round_envelopes), default=0) >= 2:
                    scope_final = True

                    context.observer.notice(
                        "read_only_violation",
                        {"attempts": max(envelope.metadata.get(
                            "read_only_violations", 0)
                            for envelope in round_envelopes)},
                    )

                force_final = budget_final or concluding or scope_final

                if budget_final and not budget_event_emitted:
                    budget_event_emitted = True

                    budget_reason = ("command_budget" if
                                     len(tool_log) >= context.max_tool_actions
                                     else "round_budget")
                    reason = (RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED
                              if budget_reason == "command_budget" else
                              RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED)

                    context.apply_state_event(
                        StateEventType.BUDGET_REACHED, reason=budget_reason,
                    )
                    context.trace.emit(
                        EventType.BUDGET_EXHAUSTED, context.task_id,
                        status=EventStatus.EXHAUSTED,
                        metadata={
                            "round": round_index + 1,
                            "tool_call_count": len(tool_log),
                            "max_tool_rounds": context.max_model_rounds,
                            "max_commands": context.max_tool_actions,
                            "reason": budget_reason,
                        },
                    )

                # Said once, at four fifths of the wall-clock budget: stop
                # opening new ground and land what you have. A hard stop that
                # arrives with no warning cuts a turn mid-thought; a warning
                # gives it the chance to finish deliberately.

                if not wrap_up_warned and _past_wall_fraction(context, 0.8):
                    wrap_up_warned = True
                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            "Time budget nearly spent. Stop starting new "
                            "investigation. Finish the change you are on, "
                            "verify it, and report -- or say plainly what is "
                            "left undone."),), authored_by="harness"))
                    context.observer.notice("wall_time_wrap_up", {})

                label = "Thinking…" if round_index == 0 else "Analyzing results…"

                failure_stage = "model"

                # Consumed here and cleared immediately: it narrows exactly
                # one round, the one that was just told its change broke the
                # build.
                narrowed, only_tools = only_tools, None

                with context.observer.model_activity(label) as on_token:
                    turn = self._complete_with_retry(
                        context, use_tools=not force_final, on_token=on_token,
                        only_tools=narrowed,
                    )

                context.cancellation.raise_if_cancelled()

                failure_stage = "turn_validation"

                action = validate_agent_turn(turn)
                text = strip_fabrications(turn.text).strip()
                calls = list(turn.tool_calls)

                if action == "refusal":
                    reason = RuntimeTerminalReason.REFUSAL
                    response = text or "The model declined this request."

                    break

                if not calls:
                    # A turn that only narrates what it is about to do has not
                    # done it. Re-prompting is bounded, and never on the final
                    # round, where tools are closed and narration is all there is.

                    if (not force_final and preamble_reprompts < 2
                            and looks_like_preamble(text)):
                        preamble_reprompts += 1

                        context.apply_state_event(
                            StateEventType.RETRY_RECORDED,
                            reason="action_preamble_without_tool_call",
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "action_preamble_without_tool_call",
                                      "retry": preamble_reprompts},
                        )
                        self._show_intermediate(context, transcript, text)

                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(
                                "Do it now — actually CALL the tool (bash, "
                                "read_file, edit_file, write_file). Don't just say "
                                "what you will do."
                            ),), authored_by="harness"))

                        continue

                    # A turn that changed files and stopped without running
                    # anything gets exactly one demand to verify its own work.

                    verification_state = context.verification_policy.evaluate_completion(
                        context.working_state,
                    )
                    native_mutation = any(
                        item.name in _WRITE_TOOLS and item.status.value == "succeeded"
                        for item in context.working_state.actions
                    )

                    # The project's own suite runs BEFORE the demand that the
                    # model go and verify -- because it is the answer to the
                    # question that demand is about to ask. Asking first cost
                    # a model call and a retry to learn what the harness can
                    # observe for free: a run fixed main.py, ran a targeted
                    # test, was told to verify anyway, and only then did the
                    # deterministic build/full/passed evidence appear.
                    #
                    # It is cached per generation, so this is the same single
                    # execution the end of the turn would have performed, just
                    # at the moment its answer is worth something.

                    if (native_mutation
                            and verification_state.status in _NEEDS_VERIFICATION):
                        _record_project_verification(
                            context, getattr(context, "project_commands", None),
                            project_build_runs(context, tool_log))
                        verification_state = (
                            context.verification_policy.evaluate_completion(
                                context.working_state))

                    if (not force_final and not verify_reprompts
                            and native_mutation
                            and verification_state.status in {
                                CompletionVerificationStatus.UNVERIFIED,
                                CompletionVerificationStatus.PARTIALLY_VERIFIED,
                                CompletionVerificationStatus.FAILED,
                            }):
                        verify_reprompts += 1

                        context.apply_state_event(
                            StateEventType.VERIFICATION_REPROMPT_RECORDED,
                            reason="unverified_change",
                        )
                        context.apply_state_event(
                            StateEventType.RETRY_RECORDED,
                            reason="unverified_change",
                        )

                        verification_action_id = new_action_id("verification_nudge")

                        context.trace.emit(
                            EventType.VERIFICATION_STARTED, context.task_id,
                            status=EventStatus.STARTED,
                            action_id=verification_action_id,
                            metadata={"kind": "verification_nudge",
                                      "changed_file_count": len(changed_files(tool_log))},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=verification_action_id,
                            metadata={"reason": "unverified_change", "retry": 1},
                        )
                        self._show_intermediate(context, transcript, text)

                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(verify_demand(tool_log)),), authored_by="harness"))

                        context.observer.notice(
                            "verification_nudge",
                            {"changed_file_count": len(changed_files(tool_log))},
                        )

                        continue

                    # The model is about to answer a bound normative question
                    # having looked at nothing, or having read an empty
                    # structure registry and stopped there. One deterministic
                    # retrieval, once per turn, then it decides again. It is
                    # not asked to and never sees that this happened.

                    if policy is not None:
                        injected = policy.answer_boundary(text)

                        if injected is not None and self._inject_evidence_call(
                                context, policy, injected, cache, tool_log):
                            continue

                    # The project's own build and tests, on a turn that
                    # wrote. This needs no task file and no acceptance block:
                    # a tree states how it is built, and a change that does
                    # not survive that is not finished. Bounded, and never
                    # once the real budget has closed the tool window.

                    # Not gated on the budget, unlike every other redirect.
                    # A round budget says "stop investigating"; it does not
                    # say "leave the tree broken". Asked to change the code,
                    # a turn spent sixty rounds reading, wrote once, and
                    # handed back a tree that did not compile -- with this
                    # gate disarmed by the very budget that ended it.

                    if (build_reprompts < BUILD_REPROMPT_LIMIT
                            and may_demand_write(context)):
                        broken, output = project_build_gap(context, tool_log)

                        if broken:
                            build_reprompts += 1

                            if round_index + 1 >= round_limit:
                                round_limit = round_index + 3

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(project_build.demand(
                                    broken, output, last_build_output)),),
                                authored_by="harness"))
                            last_build_output = output
                            only_tools = _WRITE_TOOLS
                            context.observer.notice(
                                "project_build_failed",
                                {"command": broken, "retry": build_reprompts},
                            )
                            context.trace.emit(
                                EventType.RETRY, context.task_id,
                                status=EventStatus.DETECTED,
                                metadata={"reason": "project_build_failed",
                                          "retry": build_reprompts,
                                          "arguments_recorded": False},
                            )

                            # Keep what it just said: if the extra
                            # round produces nothing, the turn still
                            # has a conclusion to hand back.
                            response = text

                            continue

                    # Every clause this session established, accounted for.
                    # The specification is carried into the turn and the turn
                    # is asked to say what the code does about each rule;
                    # nothing checked it, and four runs in a row answered
                    # about two clauses out of ten, changed one line, and
                    # stopped -- each of them building and passing its tests.

                    # Same two brakes as the write redirect had, and the
                    # same evidence against them. `concluding` suppressed
                    # this on the turn that most needs it -- a session
                    # established twelve clauses, the turn added one field
                    # to one struct, said it was done, and nothing asked for
                    # the other eleven. And once is not enough for a change
                    # that spans five files; spaced, like the write ask, so
                    # each one has room to be acted on.

                    # ...and only for a turn that was asked to change the
                    # code. The clauses a previous turn established are
                    # evidence for this one, not work it owes: read alone,
                    # they turned a normative question into an
                    # implementation audit of someone else's tree.

                    carried = list(carried_obligations(
                        context, _asked(context.conversation)))

                    if (clause_reprompts < CLAUSE_REDIRECT_LIMIT
                            and round_index - last_clause_redirect
                                >= WRITE_REDIRECT_SPACING
                            and carried):
                        skipped = conformance_mode.unaddressed_clauses(
                            text, carried)

                        if skipped:
                            clause_reprompts += 1
                            last_clause_redirect = round_index

                            # Everything except the standard. The clauses
                            # this asks about are already established and
                            # carried in the turn's own context; watched
                            # live, the model answered "let me work through
                            # each clause systematically" and spent the rest
                            # of the turn re-fetching sections it had
                            # already been given, at twenty seconds of
                            # compaction a round. What it needs here is the
                            # code, not the rulebook.

                            only_tools = tuple(
                                tool.name for tool in context.tools
                                if not tool.name.startswith("standard."))

                            if round_index + 1 >= round_limit:
                                round_limit = round_index + 3

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    conformance_mode.clause_demand(
                                        skipped, carried)),), authored_by="harness"))
                            context.observer.notice(
                                "clauses_unaddressed",
                                {"missing": skipped[:8],
                                 "carried": len(carried)},
                            )
                            response = text

                            continue

                    # A work order with sections the turn never wrote to. The
                    # model does the mechanical part -- the signature, until
                    # it compiles -- and reports the task done; the sections
                    # that need a decision are the ones it skips.

                    # Gated on the BUDGET, not on force_final. The nudge to
                    # conclude sets force_final two rounds after it fires, and
                    # a turn that had done six of seven sections was let go
                    # because of it: the tally named the seventh, and nothing
                    # had asked for it. A round budget is a real limit; being
                    # told to wrap up is not, and finishing the work order is
                    # exactly what wrapping up should mean.

                    # Redirected more than once, while it keeps working. A
                    # seven-section order is not finished by one extra round,
                    # and capping at one let a turn hand back six of seven
                    # with the seventh named in the tally and nobody asking
                    # for it. The stopping rule is the monitor's own: a
                    # redirect that does not shrink the failing set is a loop,
                    # and the next one would be too.

                    if not budget_final and order_reprompts < ORDER_REDIRECT_LIMIT:
                        order, missing = work_order_gap(
                            context, tool_log, cache)
                        labels = {item["label"] for item in missing}
                        progressed = (order_remaining is None
                                      or labels < order_remaining)

                        if missing and progressed:
                            order_remaining = labels
                            order_reprompts += 1

                            self._show_intermediate(context, transcript, text)
                            context.conversation.append(ConversationMessage(
                                "assistant", (TextBlock(turn.text),)))
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    work_order.demand(missing)),), authored_by="harness"))
                            context.observer.notice(
                                "work_order_sections_skipped",
                                {"sections": [item["label"]
                                              for item in missing]},
                            )

                            # Keep what it just said: if the extra
                            # round produces nothing, the turn still
                            # has a conclusion to hand back.
                            response = text

                            continue

                    # An announced edit that was never made. Once per turn,
                    # and never when the budget has already closed the tool
                    # window -- asking for an edit that cannot be executed
                    # would spend the last round on a refusal.

                    # A writing request answered with an empty diff. Like
                    # the build gate, this outlives the round budget: "stop
                    # investigating" is not "hand back nothing".

                    # Past the round budget AND past the nudge. Concluding
                    # used to suppress this, on the reasoning that a model
                    # told to conclude, which concluded, has answered the
                    # instruction it was given. That is true of a question
                    # and false of a write: the nudge's own words on a write
                    # request are "this turn is not finished until a file
                    # changes", so a conclusion with an empty diff has not
                    # answered it. Two master runs ended exactly there --
                    # correct diagnosis, correct plan, the sentence "Proposed
                    # fix (not yet applied)", and nothing written in sixteen
                    # minutes. The gate is the request's own nature, which is
                    # what asked_to_write_and_did_not already reads; and the
                    # conclusion is kept either way (response = text below),
                    # so nothing is discarded by asking once more.

                    if (make_reprompts < WRITE_REDIRECT_LIMIT
                            and may_demand_write(context)
                            and round_index - last_write_redirect
                                >= WRITE_REDIRECT_SPACING
                            and asked_to_write_and_did_not(
                                _asked(context.conversation), tool_log,
                                did_modify,
                                is_write=wants_write(
                                    context, _asked(context.conversation)))):
                        make_reprompts += 1

                        # Reopen the tool window. force_final is
                        # `budget_final or concluding`, and use_tools is its
                        # negation, so a redirect issued while concluding
                        # asked for an edit in a round that offered no tools
                        # at all. The model answered, accurately, "I did not
                        # call edit_file this turn" -- it could not have. A
                        # demand the harness makes impossible is worse than
                        # no demand: it reads as the model refusing.

                        nudged = False
                        rounds_since_nudge = 0
                        investigate_rounds = 0

                        if round_index + 1 >= round_limit:
                            round_limit = round_index + 3

                        self._show_intermediate(context, transcript, text)
                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(write_demand(
                                _asked(context.conversation))),), authored_by="harness"))
                        # Same lever as the build repair, for the same
                        # reason and on the same evidence: told three times
                        # "make the edit now, read nothing first", a turn
                        # read three more times and ended with an empty
                        # diff. The demand is a request; the tool list is
                        # not. For this one round the only tools are the
                        # ones that change a file.
                        only_tools = _WRITE_TOOLS
                        last_write_redirect = round_index
                        context.observer.notice(
                            "write_request_unanswered",
                            {"tool_calls": len(tool_log)},
                        )

                        # Every relaunch of a completion says so in the trace,
                        # with its reason. A notice alone lives in the
                        # observer and nowhere a trace reader can see, which
                        # made the extra model calls of a turn impossible to
                        # account for afterwards.

                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "write_request_unanswered",
                                      "retry": make_reprompts,
                                      "tool_calls": len(tool_log)},
                        )

                        response = text

                        continue

                    # An announcement is the model's own words, not the
                    # user's, so this gate never consulted the request at
                    # all -- and a turn forbidden to write that said "I will
                    # fix this" was then told to go and do it.

                    if (not force_final and not make_reprompts
                            and may_demand_write(context)
                            and announced_but_unmade_change(text, tool_log,
                                                            did_modify)):
                        make_reprompts += 1

                        self._show_intermediate(context, transcript, text)
                        context.conversation.append(ConversationMessage(
                            "assistant", (TextBlock(turn.text),)))
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(make_it_demand(text)),), authored_by="harness"))
                        context.observer.notice(
                            "announced_change_not_made",
                            {"tool_calls": len(tool_log)},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "announced_change_not_made",
                                      "retry": 1},
                        )

                        response = text

                        continue

                    response = text

                    break

                self._show_intermediate(context, transcript, text)

                context.conversation.append(ConversationMessage(
                    "assistant", (TextBlock(turn.text),) + tuple(
                        ToolUseBlock(call.id, call.name, call.arguments)
                        for call in calls
                    ),
                ))

                result_blocks = []
                round_envelopes = []

                failure_stage = "tool"

                for call_index, call in enumerate(calls):
                    arguments = dict(call.arguments)

                    # A cancellation mid-round still owes the provider one
                    # result block per call it was given, or the conversation
                    # is malformed for every backend that checks the pairing.

                    if context.cancellation.is_cancelled:
                        for pending in calls[call_index:]:
                            result_blocks.append(ToolResultBlock(
                                pending.id,
                                "ERROR: task cancellation prevented this tool from starting.",
                                is_error=True,
                            ))

                        context.conversation.append(ConversationMessage(
                            "user", tuple(result_blocks),
                        ))
                        context.cancellation.raise_if_cancelled()

                    # The command budget is only checked here; it is consumed
                    # after the call, so a cached result costs nothing.

                    if context.budget_manager is not None:
                        context.budget_manager.consume(BudgetKind.TOOL_CALLS)

                        if call.name == "bash":
                            context.budget_manager.ensure_available(
                                BudgetKind.COMMAND_EXECUTIONS
                            )

                    self._persist_session(
                        context, SessionEventType.TOOL_ACTION_STARTED,
                        {"tool_call_id": call.id, "tool_name": call.name,
                         "mutating": call.name in _WRITE_TOOLS},
                        in_flight=InFlightOperation(
                            "tool", call.id, call.name,
                            call.name in _WRITE_TOOLS,
                        ),
                    )

                    # A tool that raises instead of returning is still a result
                    # the model must see, so the exception becomes an envelope.
                    #
                    # Behind the same indicator as a model call, because the
                    # operator is waiting the same way. Only the model call
                    # had one, so when a tool ran the status line kept the
                    # finished call's last frame -- label, token count and
                    # all -- and stopped moving: a tool that prints nothing
                    # while it works, which is every standard.* tool, looked
                    # like a freeze. Measured on one turn: 85 of 145 seconds
                    # spent in tool calls, 25 of 30 of them over two seconds.
                    # The clock is the TURN's, so it carries straight on
                    # across the handover instead of restarting.

                    try:
                        with context.observer.model_activity(_tool_activity(call.name)):
                            envelope = context.tool_executor(
                                context, call.id, call.name, arguments, cache,
                            )
                    except Exception as exc:
                        text = f"ERROR: tool '{call.name}' failed: {exc}"
                        envelope = ToolResultEnvelope(
                            call.id, new_action_id(call.name), call.name, False,
                            (ToolResultStatus.CANCELLED
                             if isinstance(exc, OperationCancelled)
                             else ToolResultStatus.FAILED),
                            text, text, "other", 0.0,
                            len(text.encode("utf-8")),
                            error_category=(FailureKind.INTERRUPTED.value
                                            if isinstance(exc, OperationCancelled)
                                            else type(exc).__name__),
                            error_summary=str(exc)[:240] or type(exc).__name__,
                        )
                        context.observer.notice(
                            "tool_exception", {
                                "tool_name": call.name,
                                "error_summary": text,
                            },
                        )

                    training_generation_before = context.working_state.mutation_generation

                    self._record_tool_result(context, envelope)

                    if policy is not None and call.name in policy.STANDARD_TOOLS:
                        policy.observe_tool_result(call.name,
                                                   envelope.model_content)

                    # Every call, for the record of which source files the
                    # turn read. A conformance answer that read none says so.

                    if policy is not None:
                        policy.observe_code_read(call.name, call.arguments,
                                                 envelope.model_content)

                    if (context.budget_manager is not None and call.name == "bash"
                            and envelope.status != ToolResultStatus.CACHED):
                        context.budget_manager.consume(BudgetKind.COMMAND_EXECUTIONS)

                    evidence = context.verification_policy.evidence_for_tool(
                        envelope, arguments, context.working_state,
                    )

                    if evidence is not None:
                        self._record_verification(context, evidence)

                    # Training capture is derived evidence: a failure there
                    # detaches the recorder and never disturbs the turn.

                    if context.training_recorder is not None:
                        try:
                            context.training_recorder.record_tool_result(
                                envelope,
                                mutation_generation_before=training_generation_before,
                                mutation_generation_after=(
                                    context.working_state.mutation_generation
                                ),
                                verification_generation=(
                                    evidence.mutation_generation if evidence else None
                                ),
                            )
                        except Exception as exc:
                            context.observer.notice("training_capture_error", {
                                "stage": "tool_result",
                                "error_category": type(exc).__name__,
                            })
                            context.training_recorder = None

                    round_envelopes.append(envelope)
                    result = envelope.text

                    if envelope.status == ToolResultStatus.CACHED:
                        repeats += 1

                    # Repetition judged by what came BACK, not by what was
                    # asked. The cache above compares arguments, and a turn
                    # ran twelve greps that differed only in their --include
                    # glob, each returning the same wall of matches, and
                    # nothing counted them as repeats. What tells a loop from
                    # progress is that the answer stopped changing.

                    same = _repeated_result(seen_results, envelope)

                    if same >= _REPEAT_NOTICE:
                        repeats += 1
                        envelope = _with_repeat_note(envelope, same)

                        if same == _REPEAT_NOTICE:
                            context.observer.notice(
                                "identical_result", {"tool": call.name,
                                                     "count": same})

                    # The log is the short form the nudges quote back at the
                    # model; the trajectory keeps the fuller record.

                    tool_log.append(
                        f"{call.name} "
                        f"{json.dumps(arguments, ensure_ascii=False)[:200]}\n"
                        f"{result[:400]}"
                    )
                    trajectory.append({
                        "tool": call.name, "arguments": arguments,
                        "result": result[:4000],
                        "action_id": envelope.action_id,
                        "status": envelope.status.value,
                        "result_reference": envelope.result_reference,
                    })

                    if envelope.result_reference:
                        context.result_references.add(envelope.result_reference)

                    result_blocks.append(ToolResultBlock(
                        call.id, envelope.model_content,
                        is_error=not envelope.success,
                    ))

                    # A result already seen carries no evidence, whatever
                    # paths the call happens to name. The monitor judges
                    # progress by new evidence, and a fetch of a section the
                    # turn has already been given looked like progress every
                    # time because the source id was in its arguments. That
                    # is how a turn reached a hundred and twenty-seven calls
                    # with the stall detector never firing: each repeat
                    # presented itself as new ground.

                    observation = context.progress_monitor.observe_action(
                        call.name, arguments,
                        evidence=tuple(envelope.read_paths)
                        + tuple(envelope.affected_paths),
                        mutation=envelope.mutation,
                        fruitless=same >= _REPEAT_NOTICE,
                    )

                    # A cached result is already counted as a repeat above, so
                    # recording it again here would double-count the same call.

                    if (observation.repeated
                            and envelope.status != ToolResultStatus.CACHED):
                        context.apply_state_event(
                            StateEventType.REPEATED_ACTION_RECORDED,
                            source=StateSource.HARNESS,
                            action_id=envelope.action_id,
                        )

                    self._persist_session(
                        context, SessionEventType.TOOL_ACTION_COMPLETED,
                        {"tool_call_id": call.id, "action_id": envelope.action_id,
                         "status": envelope.status.value,
                         "result_reference": envelope.result_reference},
                        conversation=tuple(context.conversation) + (
                            ConversationMessage("user", tuple(result_blocks)),
                        ),
                    )

                    if context.cancellation.is_cancelled:
                        for pending in calls[call_index + 1:]:
                            result_blocks.append(ToolResultBlock(
                                pending.id,
                                "ERROR: task cancellation prevented this tool from starting.",
                                is_error=True,
                            ))

                        context.conversation.append(ConversationMessage(
                            "user", tuple(result_blocks),
                        ))
                        context.cancellation.raise_if_cancelled()

                    if observation.stalled:
                        reason = RuntimeTerminalReason.STALLED

                        context.trace.emit(
                            EventType.STALLED_DETECTED, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=envelope.action_id,
                            metadata={"repeat_count": observation.consecutive_without_progress,
                                      "arguments_recorded": False},
                        )

                    # An identical failing call the router has already stopped,
                    # asked for again. The tool result it just read said that
                    # repeating it changes nothing and named the tools that
                    # act on this workspace; asking once more after that is a
                    # loop. The turn ends here, with whatever it has, instead
                    # of spending the rest of its budget on the same refusal --
                    # a run made the same fetch_url call twelve times.

                    # The router says when a repetition has become terminal:
                    # it told the model once, in the tool result the model
                    # just read, and the model asked for the same thing
                    # again. Failed or merely uninformative, the answer is
                    # the same -- this turn is not going anywhere.

                    if envelope.metadata.get("terminal_repeat"):
                        suppressed = envelope.metadata.get("repeat_count", 0)
                        reason = RuntimeTerminalReason.STALLED

                        context.trace.emit(
                            EventType.STALLED_DETECTED, context.task_id,
                            status=EventStatus.DETECTED,
                            action_id=envelope.action_id,
                            metadata={"kind": envelope.metadata.get(
                                          "repeat_kind", "repeated_action"),
                                      "tool_name": envelope.tool_name,
                                      "repeat_count": suppressed,
                                      "arguments_recorded": False},
                        )

                context.conversation.append(ConversationMessage(
                    "user", tuple(result_blocks),
                ))

                # One round of retrieval is over: did it establish anything the
                # session had not already read?

                if policy is not None:
                    policy.close_round(round_index + 1)

                if reason == RuntimeTerminalReason.STALLED:
                    # A stall is a loop, and a loop on a task that is not
                    # finished is the moment to say what is missing rather
                    # than to give up: told to stop reading files, a turn
                    # looped on a retrieval tool instead and ended here
                    # having written nothing, five of seven sections
                    # untouched.

                    order, missing = work_order_gap(
                        context, tool_log, cache)

                    # A writing turn that loops is not a finished turn.
                    # The work-order redirect below needs an order to name
                    # sections from; without one -- the ordinary case, two
                    # prompts and no task file -- a stall simply dropped the
                    # turn wherever it stood, once with an empty diff and
                    # once seventeen calls into re-grepping the same header.

                    # A broken tree comes first, and it was reachable from
                    # neither of the other two gates: they run at the answer
                    # boundary and at the budget, and a stall passes through
                    # neither. Measured -- a turn changed a declaration in a
                    # header, left the definition in the .c, looped, and
                    # ended stalled. The tree no longer compiled, the turn
                    # was never told, and the operator found out from the
                    # bench afterwards.

                    if (build_reprompts < BUILD_REPROMPT_LIMIT
                            and may_demand_write(context)):
                        broken, output = project_build_gap(context, tool_log)

                        if broken:
                            build_reprompts += 1
                            reason = RuntimeTerminalReason.COMPLETED
                            context.progress_monitor.record_progress(
                                "build repair")
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(project_build.demand(
                                    broken, output, last_build_output)),),
                                authored_by="harness"))
                            last_build_output = output
                            only_tools = _WRITE_TOOLS
                            context.observer.notice(
                                "project_build_failed",
                                {"command": broken, "retry": build_reprompts,
                                 "after": "stall"},
                            )
                            context.trace.emit(
                                EventType.RETRY, context.task_id,
                                status=EventStatus.DETECTED,
                                metadata={"reason": "project_build_failed",
                                          "retry": build_reprompts,
                                          "after": "stall",
                                          "arguments_recorded": False},
                            )

                            continue

                    # ...and nothing was written. The other site checks
                    # that through asked_to_write_and_did_not; this one
                    # checked only what was asked, so a turn that HAD edited
                    # the file and then looped was told "nothing was
                    # changed" -- false, and it sends the model looking for
                    # work it has already done.

                    if (make_reprompts < WRITE_REDIRECT_LIMIT
                            and may_demand_write(context)
                            and not did_modify
                            and not changed_files(tool_log)
                            and wants_write(
                                context, _asked(context.conversation))):
                        make_reprompts += 1
                        reason = RuntimeTerminalReason.COMPLETED
                        context.progress_monitor.record_progress(
                            "write redirect")
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(write_demand(
                                _asked(context.conversation))),), authored_by="harness"))
                        only_tools = _WRITE_TOOLS
                        last_write_redirect = round_index
                        context.observer.notice(
                            "write_request_unanswered",
                            {"tool_calls": len(tool_log), "after": "stall"},
                        )
                        context.trace.emit(
                            EventType.RETRY, context.task_id,
                            status=EventStatus.DETECTED,
                            metadata={"reason": "write_request_unanswered",
                                      "retry": make_reprompts, "after": "stall",
                                      "tool_calls": len(tool_log)},
                        )

                        continue

                    # The clauses this session established, on the path
                    # that had no way of asking for them. A turn wrote the
                    # one field the change needs, stalled, and ended -- the
                    # tree compiled, so the build gate was silent; something
                    # had been written, so the write redirect was silent;
                    # and the clause redirect lives only at the answer
                    # boundary, which a stall never reaches. Fourth time a
                    # gate has turned out to be unreachable from one exit.

                    carried = list(carried_obligations(
                        context, _asked(context.conversation)))

                    if clause_reprompts < CLAUSE_REDIRECT_LIMIT and carried:
                        # Everything the turn has said so far, not the
                        # final answer -- at a stall there is no final answer
                        # yet, and judging against an empty one declares
                        # every clause unaddressed and fires on a turn that
                        # had accounted for all of them.

                        said = response or "\n".join(transcript)
                        skipped = conformance_mode.unaddressed_clauses(
                            said, carried)

                        if skipped:
                            clause_reprompts += 1
                            last_clause_redirect = round_index
                            reason = RuntimeTerminalReason.COMPLETED
                            context.progress_monitor.record_progress(
                                "clause redirect")
                            context.conversation.append(ConversationMessage(
                                "user", (TextBlock(
                                    conformance_mode.clause_demand(
                                        skipped, carried)),),
                                authored_by="harness"))
                            only_tools = tuple(
                                tool.name for tool in context.tools
                                if not tool.name.startswith("standard."))
                            context.observer.notice(
                                "clauses_unaddressed",
                                {"missing": skipped, "carried": len(carried),
                                 "after": "stall"},
                            )

                            continue

                    labels = {item["label"] for item in missing}
                    progressed = (order_remaining is None
                                  or labels < order_remaining)

                    if (missing and progressed
                            and order_reprompts < ORDER_REDIRECT_LIMIT):
                        order_remaining = labels
                        order_reprompts += 1
                        reason = RuntimeTerminalReason.COMPLETED

                        # The loop that produced the stall is over, and the
                        # turn has been given a new instruction. Leaving the
                        # counter where it is makes the very next action stall
                        # again, which turned the redirect into a formality.

                        context.progress_monitor.record_progress(
                            "work-order redirect")
                        context.conversation.append(ConversationMessage(
                            "user", (TextBlock(work_order.demand(missing)),), authored_by="harness"))
                        context.observer.notice(
                            "work_order_sections_skipped",
                            {"sections": [item["label"] for item in missing],
                             "after": "stall"},
                        )

                        continue

                    # Nothing above took the turn back. Say so, and say what
                    # was true when the decision was made: a turn that ends
                    # on a loop, having been offered three ways out and taken
                    # none, is exactly the case an operator cannot diagnose
                    # from "stalled" alone.

                    context.observer.notice("stalled_without_redirect", {
                        "tool_calls": len(tool_log),
                        "wrote": bool(changed_files(tool_log)) or did_modify,
                        "write_request": wants_write(
                            context, _asked(context.conversation)),
                        "redirects_spent": {"write": make_reprompts,
                                            "build": build_reprompts,
                                            "order": order_reprompts},
                        "asked": _asked(context.conversation)[:80],
                    })

                    break

                names = [call.name for call in calls]

                if any(envelope.mutation for envelope in round_envelopes):
                    did_modify = True

                    # A turn that is writing is not the failure this nudge
                    # exists to stop, and its clock must not keep running. The
                    # nudge's own words are "you can make more small edits
                    # after this one"; with the clock left running the window
                    # shut on the very next round, so a master run made the
                    # first edit of a five-file change -- adding the field the
                    # rest of the change needs -- and was cut off before
                    # threading it anywhere. The promise was false, and the
                    # tree was left half-changed.

                    nudged = False
                    rounds_since_nudge = 0
                    investigate_rounds = 0
                elif names:
                    investigate_rounds += 1

                # Reading forever without changing anything is the failure mode
                # this catches: a round that only investigates counts towards a
                # single nudge to conclude, which is issued at most once.
                #
                # The count is a fraction of the budget, not a constant. Twelve
                # was calibrated against a sixty-round turn; a write request now
                # gets twice that, and the flat twelve cut off a turn eleven
                # reads into a five-file change -- before its first edit, with
                # nine tenths of its budget unspent. A turn given more room to
                # work is allowed more room to look.

                if (not did_modify and not nudged and not force_final
                        and (repeats >= 2
                             or investigate_rounds >= _investigation_ceiling(
                                 round_limit))):
                    nudged = True

                    context.conversation.append(ConversationMessage(
                        "user", (TextBlock(
                            conclude_demand(
                                _asked(context.conversation),
                                wants_write(context, _asked(context.conversation)))
                            + turn_evidence(tool_log)
                        ),), authored_by="harness"))
                    context.trace.emit(
                        EventType.RETRY, context.task_id,
                        status=EventStatus.DETECTED,
                        metadata={"reason": "investigation_nudge", "retry": 1,
                                  "investigate_rounds": investigate_rounds},
                    )
                    context.observer.notice(
                        "investigation_nudge",
                        {"investigate_rounds": investigate_rounds,
                         "repeated_actions": repeats},
                    )

            # The loop can run out of rounds with work done but nothing said.
            # One tool-free turn turns that into an answer for the user.

            if not response and (transcript or tool_log):
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(
                        "Now summarize your findings and give the final answer / fix. "
                        "Do not call any tool." + turn_evidence(tool_log)
                    ),), authored_by="harness"))

                failure_stage = "model"

                with context.observer.model_activity("Concluding…") as on_token:
                    final_turn = self._complete_with_retry(
                        context, use_tools=False, on_token=on_token,
                    )

                failure_stage = "turn_validation"

                validate_agent_turn(final_turn)
                response = strip_fabrications(final_turn.text).strip()

                # Even the conclusion can come back empty; the tool log is then
                # the only honest thing left to hand the user.

                if not response:
                    context.observer.notice(
                        "empty_final_synthesis", {"tool_call_count": len(tool_log)},
                    )
                    response = no_conclusion_note(tool_log)
        except BudgetExceeded as exc:
            budget_event_emitted = True

            if context.working_state.terminal_status == TerminalStatus.RUNNING:
                context.apply_state_event(
                    StateEventType.BUDGET_EXHAUSTED,
                    summary=str(exc), reason=exc.kind.value,
                )

            context.trace.emit(
                EventType.BUDGET_EXHAUSTED, context.task_id,
                session_id=context.session_id, status=EventStatus.EXHAUSTED,
                metadata={"component": exc.component, "reason": exc.kind.value,
                          "raw_content_recorded": False},
            )

            self._persist_session(
                context, SessionEventType.BUDGET_UPDATED,
                {"component": exc.component, "reason": exc.kind.value,
                 "exhausted": True},
            )

            # A tree left broken is worse than a budget overrun by one
            # round. The build gate hangs off the answer boundary and off
            # the stall, and a budget exit passes through neither: measured
            # today, a master run made the right edit at minute nineteen of
            # thirty, left a symbol undefined, and the budget ended the turn
            # with ten minutes to spare and nothing asking it to finish.
            #
            # So the repair gets the same courtesy the summary below gets --
            # one round, outside the budget, with only the writing tools on
            # the table. Once. A budget that has already ended the turn is
            # not an invitation to keep going.

            broken, output = project_build_gap(context, tool_log)

            if broken and build_reprompts < BUILD_REPROMPT_LIMIT:
                build_reprompts += 1
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(project_build.demand(
                        broken, output, last_build_output)),),
                    authored_by="harness"))
                context.observer.notice(
                    "project_build_failed",
                    {"command": broken, "retry": build_reprompts,
                     "after": "budget"},
                )

                try:
                    with context.observer.model_activity("Repairing the build…") as tick:
                        repair = self.complete_model_turn(
                            context, use_tools=True, on_token=tick,
                            only_tools=_WRITE_TOOLS, grace=True)

                    for call in repair.tool_calls:
                        envelope = context.tool_executor(
                            context, call.id, call.name, call.arguments, cache)
                        tool_log.append(f"{call.name} {json.dumps(call.arguments)[:200]}")

                        if envelope.mutation:
                            did_modify = True
                except Exception:
                    # The repair is a courtesy too; its failure must not
                    # replace the budget's own account of the turn.
                    pass

            # A budget that ends a turn must not also erase it. One
            # tool-less call, outside the budget, asking the model to say
            # what it did -- the alternative, measured today, is a turn that
            # edited five files and handed back "I stopped after 125 tool
            # calls without reaching a conclusion".

            if not (response or "").strip() and tool_log:
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(
                        "You have run out of room for this turn. Do not call "
                        "any tool. Say what you changed, what you verified, "
                        "and what remains -- briefly, and only what actually "
                        "happened." + turn_evidence(tool_log)),),
                    authored_by="harness"))

                try:
                    with context.observer.model_activity("Wrapping up…") as tick:
                        summary = self.complete_model_turn(
                            context, use_tools=False, on_token=tick, grace=True)

                    response = strip_fabrications(summary.text).strip() or response
                except Exception:
                    # The grace call is a courtesy; its failure must not
                    # replace the budget's own account of the turn.
                    pass

            return self._result(
                context, RuntimeTerminalReason.BUDGET_EXHAUSTED, response,
                transcript, tool_log, trajectory, did_modify, True,
                verification_action_id, cache,
                error_category="budget_exhausted", error_summary=str(exc),
            )
        except (KeyboardInterrupt, OperationCancelled) as exc:
            summary = (exc.reason if isinstance(exc, OperationCancelled)
                       else "interrupted by user")

            try:
                context.apply_state_event(
                    StateEventType.TASK_INTERRUPTED, summary=summary,
                )
            except SessionError:
                # Cancellation remains truthful even if durable persistence is
                # unavailable; the failure is observable in the returned result.

                context.session = None

                if context.working_state.terminal_status == TerminalStatus.RUNNING:
                    context.apply_state_event(
                        StateEventType.TASK_INTERRUPTED, summary=summary,
                    )

            context.trace.emit(
                EventType.TASK_INTERRUPTED, context.task_id,
                status=EventStatus.CANCELLED,
                provider=context.provider, model=context.model,
                session_id=context.session_id,
                metadata={"tool_call_count": len(tool_log)},
            )
            context.trace.emit(
                EventType.CANCELLATION_OBSERVED, context.task_id,
                session_id=context.session_id, status=EventStatus.CANCELLED,
                error_category=FailureKind.INTERRUPTED.value,
                error_summary=summary,
                metadata={"scope": (exc.scope.value if isinstance(exc, OperationCancelled)
                                    else "task")},
            )

            self._persist_session(
                context, SessionEventType.TASK_INTERRUPTED,
                {"reason": summary},
            )

            return self._result(
                context, RuntimeTerminalReason.INTERRUPTED, response,
                transcript, tool_log, trajectory, did_modify,
                budget_event_emitted, verification_action_id, cache,
                error_category=FailureKind.INTERRUPTED.value,
                error_summary=summary,
            )
        except Exception as exc:
            # Persistence failures are different from tracing failures: they
            # are surfaced as runtime failure. Detach the failed handle so the
            # in-memory task can be terminated truthfully without recursively
            # attempting the same broken store operation.

            if isinstance(exc, SessionError):
                context.session = None

            # An exception with no message -- raise SomeError() -- used to
            # reach the user as the bare words "runtime failure": the class
            # was recorded, and the one line that showed it was not. Name the
            # class and where it was raised, so a failed turn is diagnosable
            # from the terminal alone. Tracing is off by default, and a
            # persistence failure is exactly the case that also loses the
            # session record, so this string is sometimes the only evidence.

            detail = str(exc).strip()

            if not detail:
                frames = traceback.extract_tb(exc.__traceback__)

                # rsplit rather than os.path.basename: this module imports no
                # os and touches no filesystem, and one file name is not a
                # reason to give it either.

                where = (f" at {frames[-1].filename.rsplit('/', 1)[-1]}:"
                         f"{frames[-1].lineno}" if frames else "")
                detail = f"{type(exc).__name__}{where}"

            detail = detail[:240]

            if context.working_state.terminal_status == TerminalStatus.RUNNING:
                context.apply_state_event(
                    StateEventType.TASK_FAILED, summary=detail,
                )

            # The stage recorded as the loop advanced says whether the model,
            # its answer, or the runtime itself is what actually broke.

            reason = (RuntimeTerminalReason.MODEL_FAILURE if failure_stage == "model"
                      else RuntimeTerminalReason.INVALID_TURN
                      if failure_stage == "turn_validation"
                      else RuntimeTerminalReason.RUNTIME_FAILURE)

            context.trace.emit(
                EventType.TASK_FINISHED, context.task_id,
                status=EventStatus.FAILED,
                provider=context.provider, model=context.model,
                error_category=type(exc).__name__,
                error_summary=detail,
                metadata={"tool_call_count": len(tool_log),
                          "failure_stage": failure_stage},
            )

            self._persist_session(
                context, SessionEventType.TASK_FAILED,
                {"failure_stage": failure_stage,
                 "error_category": type(exc).__name__},
            )

            return self._result(
                context, reason, response, transcript, tool_log, trajectory,
                did_modify, budget_event_emitted, verification_action_id, cache,
                error_category=type(exc).__name__, error_summary=detail,
            )

        # A stall leaves the loop normally, so the task is failed here rather
        # than in an except branch it never reached.

        if (reason == RuntimeTerminalReason.STALLED
                and context.working_state.terminal_status == TerminalStatus.RUNNING):
            context.apply_state_event(
                StateEventType.TASK_FAILED,
                summary="stalled after repeated actions without grounded progress",
            )
            self._persist_session(
                context, SessionEventType.TASK_FAILED,
                {"failure_kind": FailureKind.STALLED.value},
            )

        result = self._result(
            context, reason, response, transcript, tool_log, trajectory,
            did_modify, budget_event_emitted, verification_action_id, cache,
            completion_deferred=defer_completion,
        )

        # A deferred completion leaves the task and its checkpoint open for the
        # controller to review, repair, or roll back before closing them.

        if not defer_completion:
            self.complete(context, result)

        return result

    def _inject_evidence_call(self, context: AgentContext, policy,
                              injected, cache, tool_log) -> bool:
        """Run one policy-issued evidence call and hand the result back.

        Appended to the conversation the way any tool result is, so the model
        resumes from a turn that looks exactly like one it made itself. The
        answer it was about to give never stands; the trace keeps it.
        """
        call_id = f"policy-{injected.origin.lower()}-{len(tool_log)}"

        try:
            envelope = context.tool_executor(
                context, call_id, injected.tool, dict(injected.arguments), cache,
            )
        except Exception as exc:                                # noqa: BLE001
            context.observer.notice(
                "standard_policy_call_failed",
                {"tool_name": injected.tool, "error_summary": str(exc)[:240]},
            )

            return False

        policy.observe_tool_result(injected.tool, envelope.model_content,
                                   origin=injected.origin)
        context.conversation.append(ConversationMessage(
            "assistant", (ToolUseBlock(call_id, injected.tool,
                                       dict(injected.arguments)),),
        ))
        context.conversation.append(ConversationMessage(
            "user", (ToolResultBlock(call_id, envelope.model_content,
                                     is_error=not envelope.success),),
        ))
        context.trace.emit(
            EventType.TOOL_CALL_FINISHED, context.task_id,
            session_id=context.session_id, status=EventStatus.OK,
            metadata={"component": "standard_answer_policy",
                      "origin": injected.origin, "tool_name": injected.tool},
        )

        # The MODEL is not told this happened — that is the design, and it
        # stands. The USER is: the answer about to be given cited the standard
        # having consulted nothing, and whether the citation that finally
        # appears is grounded or invented is the one thing a reader of the
        # transcript cannot otherwise tell.

        context.observer.notice(
            "standard_evidence_injected",
            {"tool_name": injected.tool, "origin": injected.origin},
        )

        return True


    def _complete_with_retry(self, context: AgentContext, **kwargs: Any) -> ModelTurn:
        # `only_tools` travels with the call rather than living on the context:
        # it applies to ONE round and must not leak into the next.
        attempts = 0

        while True:
            context.cancellation.raise_if_cancelled()

            self._persist_session(
                context, SessionEventType.MODEL_TURN_STARTED,
                {"attempt": attempts + 1},
                in_flight=InFlightOperation("model", new_action_id("model_call"),
                                            "model_complete"),
            )

            try:
                turn = self.complete_model_turn(context, **kwargs)
                context.cancellation.raise_if_cancelled()

                self._persist_session(
                    context, SessionEventType.MODEL_TURN_COMPLETED,
                    {"stop_reason": turn.stop_reason.value},
                )

                return turn
            except OperationCancelled:
                # A cancellation is the user's decision, not a model failure,
                # and must never be retried.

                raise
            except BudgetExceeded:
                # Nor is a spent budget. BudgetExceeded is a RuntimeError, so
                # the handler below took it for a model failure and retried
                # it -- against a budget that raises on every charge. The
                # refund added beside it then handed the exhausted unit back
                # each time, and a turn that should have ended at its wall
                # clock ran for an hour past it. A budget says stop; retrying
                # a stop is not a retry.

                raise
            except Exception as exc:
                decision = context.retry_policy.decide(
                    Failure(FailureKind.MODEL_ERROR, str(exc)[:240]), attempts,
                )

                context.trace.emit(
                    EventType.RETRY_DECISION, context.task_id,
                    session_id=context.session_id,
                    status=(EventStatus.DETECTED if decision.should_retry
                            else EventStatus.FAILED),
                    error_category=FailureKind.MODEL_ERROR.value,
                    metadata={"decision": decision.action.value,
                              "attempt": decision.attempt,
                              "maximum_attempts": decision.maximum_attempts},
                )

                if not decision.should_retry:
                    raise

                attempts += 1

                # The attempt that just failed was charged to the turn's
                # budget and gave it nothing: no assistant message, no tool
                # call, nothing the next round can build on. Charging for it
                # lets a turn run out of room without ever having used it.
                # Refunded here, where the caller is the only one that knows
                # the attempt produced nothing.

                manager = getattr(context, "budget_manager", None)

                if manager is not None:
                    for kind in (BudgetKind.PRIMARY_MODEL_CALLS,
                                 BudgetKind.MODEL_TURNS):
                        manager.refund(kind)

                context.apply_state_event(
                    StateEventType.RETRY_RECORDED,
                    reason=FailureKind.MODEL_ERROR.value,
                )

    @staticmethod
    def _persist_session(
        context: AgentContext, event_type: SessionEventType,
        payload: Mapping[str, object], *,
        in_flight: InFlightOperation | None = None,
        conversation: Sequence[ConversationMessage] | None = None,
    ) -> None:
        if context.session is None:
            return

        context.session.append(event_type, context.task_id, payload)

        # The snapshot is written after every event, so a resume can rebuild
        # the whole runtime from the last one alone.

        snapshot = SessionSnapshot(
            context.session.session_id, context.task_id, context.session.sequence,
            context.session.configuration, context.working_state,
            tuple(conversation or context.conversation), context.context_items,
            context.compaction_artifact, tuple(sorted(context.result_references)),
            in_flight, tuple(context.session_tool_log),
            tuple(context.session_trajectory), context.progress_monitor.to_dict(),
            context.checkpoint.checkpoint_id if context.checkpoint else None,
            context.checkpoint.status.value if context.checkpoint else None,
            tuple(sorted(context.checkpoint.paths)) if context.checkpoint else (),
            tuple(context.exploration_reports),
            tuple(context.review_results),
            context.budget_manager.to_dict() if context.budget_manager else {},
            tuple(context.delegation_results), dict(context.tool_exposure),
            standard_binding=(dict(context.standard_binding)
                              if context.standard_binding else None),
            standard_source_ids_used=tuple(sorted(context.standard_source_ids_used)),
            standard_retrieval_cache_fingerprint=(
                str(context.standard_binding.get("retrieval_fingerprint")
                    or context.standard_binding.get("index_fingerprint"))
                if context.standard_binding else None),
        )

        context.session.save(snapshot)

        context.trace.emit(
            EventType.SESSION_SNAPSHOT, context.task_id,
            session_id=context.session.session_id, status=EventStatus.OK,
            metadata={"sequence": context.session.sequence,
                      "in_flight": in_flight.kind if in_flight else None,
                      "raw_content_recorded": False},
        )

    @staticmethod
    def persist_session_event(
        context: AgentContext, event_type: SessionEventType,
        payload: Mapping[str, object],
    ) -> None:
        """Persist an orchestration boundary through the canonical snapshot path."""
        AgentRuntime._persist_session(context, event_type, payload)

    @staticmethod
    def _record_tool_result(
        context: AgentContext, envelope: ToolResultEnvelope,
    ) -> None:
        generation_before = context.working_state.mutation_generation

        if envelope.status == ToolResultStatus.CANCELLED:
            event_type = StateEventType.ACTION_CANCELLED
        elif envelope.success:
            event_type = StateEventType.ACTION_SUCCEEDED
        else:
            event_type = StateEventType.ACTION_FAILED

        kind = (ActionKind.COMMAND if envelope.category == "command"
                else ActionKind.TOOL)
        failure_category = (None if envelope.success else classify_tool_failure(
            envelope.status.value, envelope.error_category, envelope.exit_code,
        ).value)

        context.apply_state_event(
            event_type,
            source=StateSource.TOOL_RUNTIME,
            action_id=envelope.action_id,
            kind=kind.value,
            name=envelope.tool_name,
            observed_status=("ok" if envelope.success else
                             "cancelled" if envelope.status == ToolResultStatus.CANCELLED
                             else "failed"),
            category=failure_category,
            summary=envelope.error_summary,
            round_number=context.working_state.current_round,
            output_chars=len(envelope.text),
            exit_code=envelope.exit_code,
        )

        if envelope.status == ToolResultStatus.CACHED:
            context.apply_state_event(
                StateEventType.REPEATED_ACTION_RECORDED,
                source=StateSource.TOOL_RUNTIME,
                action_id=envelope.action_id,
            )

        for path in envelope.read_paths:
            context.apply_state_event(
                StateEventType.FILE_READ,
                source=StateSource.TOOL_RUNTIME,
                path=path,
                action_id=envelope.action_id,
            )

        # The tool reports which of the paths it touched did not exist before,
        # which is the only thing distinguishing a creation from an edit here.

        created = frozenset(envelope.metadata.get("created_paths", ()))

        for path in envelope.affected_paths:
            context.apply_state_event(
                (StateEventType.FILE_CREATED if path in created
                 else StateEventType.FILE_MODIFIED),
                source=StateSource.TOOL_RUNTIME,
                path=path,
                action_id=envelope.action_id,
            )

        # A new generation means every verification recorded so far describes
        # code that has just changed, so it is announced as invalidated.

        if context.working_state.mutation_generation > generation_before:
            context.trace.emit(
                EventType.VERIFICATION_INVALIDATED, context.task_id,
                session_id=context.session_id, status=EventStatus.DETECTED,
                action_id=envelope.action_id,
                metadata={"previous_generation": generation_before,
                          "mutation_generation": context.working_state.mutation_generation,
                          "affected_path_count": len(envelope.affected_paths)},
            )

    @staticmethod
    def _record_verification(
        context: AgentContext, evidence: VerificationEvidence,
    ) -> None:
        context.apply_state_event(
            StateEventType.VERIFICATION_RECORDED,
            source=StateSource.TOOL_RUNTIME,
            verification_id=evidence.verification_id,
            kind=evidence.category.value, category=evidence.category.value,
            coverage=evidence.coverage.value, executed=evidence.executed,
            outcome=evidence.outcome.value, action_id=evidence.action_id,
            summary=evidence.summary,
            mutation_generation=evidence.mutation_generation,
            result_reference=evidence.result_reference,
            command=evidence.command, project_bench=evidence.project_bench,
        )

        # Only evidence that says something about the change may be traced as
        # passed or failed. A command nothing could classify was emitting
        # `verification_passed` with category=unknown and coverage=unknown
        # purely because it exited zero -- the completion gate ignored it, but
        # every reader of the trace counted it, and the benchmark counted it
        # as a verification the turn had performed. It is classified instead,
        # which is what it is.

        if not evidence.proves_change:
            event_type = EventType.VERIFICATION_CLASSIFIED
            status = EventStatus.DETECTED
        elif evidence.outcome == VerificationOutcome.PASSED:
            event_type = EventType.VERIFICATION_PASSED
            status = EventStatus.OK
        elif evidence.outcome == VerificationOutcome.FAILED:
            event_type = EventType.VERIFICATION_FAILED
            status = EventStatus.FAILED
        else:
            event_type = EventType.VERIFICATION_CLASSIFIED
            status = EventStatus.DETECTED

        context.trace.emit(
            event_type, context.task_id, status=status,
            action_id=evidence.action_id, session_id=context.session_id,
            metadata={"category": evidence.category.value,
                      "coverage": evidence.coverage.value,
                      "outcome": evidence.outcome.value,
                      "proves_change": evidence.proves_change,
                      "mutation_generation": evidence.mutation_generation,
                      "project_bench": evidence.project_bench,
                      "command_recorded": evidence.command is not None},
        )

        AgentRuntime._persist_session(
            context, SessionEventType.VERIFICATION_RECORDED,
            {"verification_id": evidence.verification_id,
             "category": evidence.category.value,
             "coverage": evidence.coverage.value,
             "outcome": evidence.outcome.value,
             "mutation_generation": evidence.mutation_generation,
             "result_reference": evidence.result_reference},
        )

    def complete(self, context: AgentContext, result: AgentResult) -> AgentResult:
        if result.task_id != context.task_id:
            raise ValueError("AgentResult does not belong to AgentContext")

        evaluation = context.verification_policy.evaluate_completion(
            context.working_state,
        )

        if context.review_required and (
            evaluation.status != CompletionVerificationStatus.VERIFIED
            or not context.working_state.current_review_accepted
        ):
            # Review is an additional current-generation gate.  Keep the task
            # and checkpoint open; the controller may repair, resume, or roll
            # back through its explicit boundaries.

            result.terminal_status = context.working_state.terminal_status
            result.verification_outcome = context.working_state.verification_outcome
            result.completion_deferred = True

            return result

        # A checkpoint is only finalized behind a verified completion; anything
        # less leaves it active so the work can still be rolled back.

        if (context.checkpoint is not None
                and context.checkpoint_manager is not None
                and context.checkpoint.status == CheckpointStatus.ACTIVE
                and evaluation.status == CompletionVerificationStatus.VERIFIED):
            context.checkpoint_manager.finalize(context.checkpoint)

            context.apply_state_event(
                StateEventType.CHECKPOINT_STATUS_UPDATED,
                checkpoint_id=context.checkpoint.checkpoint_id,
                status=context.checkpoint.status.value,
            )
            context.trace.emit(
                EventType.CHECKPOINT_FINALIZED, context.task_id,
                session_id=context.session_id, status=EventStatus.OK,
                metadata={"checkpoint_id": context.checkpoint.checkpoint_id,
                          "mutation_generation": context.working_state.mutation_generation},
            )

        if context.working_state.terminal_status == TerminalStatus.RUNNING:
            context.apply_state_event(
                StateEventType.TASK_COMPLETED, summary="task turn completed",
            )

        # The result was built before these last transitions, so the counters
        # it carries are refreshed from the working state that now owns them.

        result.terminal_status = context.working_state.terminal_status
        result.verification_outcome = context.working_state.verification_outcome
        result.tool_actions = context.working_state.tool_actions
        result.model_calls = context.working_state.model_rounds
        result.completion_deferred = False

        context.trace.emit(
            EventType.TASK_FINISHED, context.task_id,
            status=EventStatus.OK,
            provider=context.provider, model=context.model,
            metadata={
                "tool_call_count": len(result.tool_log),
                "changed_file_count": len(changed_files(result.tool_log)),
                "response_chars": len(result.final_response),
                "round_budget_exhausted": result.budget_exhausted,
            },
        )

        self._persist_session(
            context, SessionEventType.TASK_COMPLETED,
            {"terminal_reason": result.terminal_reason.value},
        )

        return result

    def _repair_ask(self, context: AgentContext, text: str) -> str:
        """One tool-less question, for the normative repair path.

        Deliberately the narrowest call the runtime can make: no tools, one
        message, the composed system context the turn already had. A repair
        that could reach a tool would be a second turn wearing the first
        one's clothes, which is the constraint the repair contract is built
        on -- so it is enforced here by what is not passed, not by asking.
        """
        from model_backend import ConversationMessage, TextBlock

        try:
            turn = self.complete_model_turn(
                context, use_tools=False,
                conversation=(ConversationMessage("user", (TextBlock(text),)),),
                grace=True)
        except Exception:
            return ""        # a failed repair withholds, exactly as before

        return (turn.text or "") if turn is not None else ""

    def complete_model_turn(
        self, context: AgentContext, *, use_tools: bool,
        conversation: Sequence[ConversationMessage] | None = None,
        on_token: Callable[[], None] | None = None,
        only_tools: Sequence[str] | None = None,
        grace: bool = False,
    ) -> ModelTurn:
        conversation = tuple(conversation or context.conversation)

        snapshot, system, selected_conversation = self.compose_context(
            context, conversation, use_tools=use_tools,
        )

        # Input tokens are charged from the composed snapshot rather than from
        # the provider's own count, which only arrives after the call.

        # `grace` is the one call a spent budget still allows: the tool-less
        # question "what did you just do", asked so a turn that ran out of
        # room hands back its work instead of a blank. Charging it would be
        # charging the budget for reporting that the budget ended.

        if context.budget_manager is not None and not grace:
            context.budget_manager.consume(BudgetKind.PRIMARY_MODEL_CALLS)
            context.budget_manager.consume(BudgetKind.MODEL_TURNS)
            context.budget_manager.consume(
                BudgetKind.INPUT_TOKENS, snapshot.estimated_input_tokens,
                approximate=True,
            )

        self._trace_context(context, snapshot, system, selected_conversation,
                            use_tools)

        action_id = new_action_id("model")

        # Held in a local as well: a capture failure detaches the recorder from
        # the context, and the rest of this call must then skip it too.

        recorder = context.training_recorder

        if recorder is not None:
            try:
                recorder.begin_model_turn(
                    system=system, conversation=selected_conversation,
                    tools=context.tools, role=context.role,
                    purpose="primary_agent" if context.role == "main" else context.role,
                    mutation_generation=context.working_state.mutation_generation,
                    memory_provenance_ids=tuple(
                        item.item_id for item in snapshot.selected_items
                        if item.layer == ContextLayer.DURABLE_MEMORY
                    ),
                    external_context=any(
                        item.source in {"external_web", "search_internet", "web"}
                        for item in snapshot.selected_items
                    ),
                    tools_enabled=use_tools,
                )
            except Exception as exc:
                context.observer.notice("training_capture_error", {
                    "stage": "model_input", "error_category": type(exc).__name__,
                })
                context.training_recorder = None
                recorder = None

        span = context.trace.start_span(
            EventType.MODEL_CALL_STARTED,
            EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED,
            context.task_id,
            provider=context.provider,
            model=context.model,
            context_tokens_estimate=snapshot.estimated_input_tokens,
            metadata={
                "purpose": "primary_agent",
                "conversation_message_count": len(selected_conversation),
                "tool_schema_count": len(context.tools) if use_tools else 0,
                "use_tools": use_tools,
            },
        )

        try:
            turn = context.backend.complete(
                system=system,
                conversation=selected_conversation,
                tools=_narrowed(context.tools, only_tools),
                use_tools=use_tools,
                on_token=on_token,
            )
        except Exception as exc:
            if recorder is not None:
                try:
                    recorder.fail_model_turn(
                        exc, model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
                except Exception:
                    context.training_recorder = None

            span.fail(exc, provider=context.provider, model=context.model)

            context.apply_state_event(
                StateEventType.ACTION_FAILED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="failed",
                category=type(exc).__name__,
                summary=str(exc)[:240] or type(exc).__name__,
                round_number=context.working_state.current_round,
            )

            raise

        usage = turn.usage or {}

        # The provider's own output count is preferred; without it the text is
        # charged at a rough four characters per token, flagged as approximate.

        if context.budget_manager is not None:
            output_tokens = _usage_token(usage, "output_tokens", "completion_tokens")

            context.budget_manager.consume(
                BudgetKind.OUTPUT_TOKENS,
                output_tokens if output_tokens is not None else len(turn.text) // 4 + 1,
                approximate=output_tokens is None,
            )

        fields = {
            "provider": context.provider,
            "model": context.model,
            "input_tokens": _usage_token(usage, "input_tokens", "prompt_tokens"),
            "output_tokens": _usage_token(usage, "output_tokens", "completion_tokens"),
            "context_tokens_estimate": snapshot.estimated_input_tokens,
            "metadata": {
                "purpose": "primary_agent",
                "stop_reason": str(turn.stop_reason),
                "tool_call_count": len(turn.tool_calls),
                "response_chars": len(turn.text),
            },
        }

        # A backend that reports an error in the turn rather than raising is
        # still a failed action, and is recorded as one.

        if turn.stop_reason == StopReason.ERROR:
            span.fail(turn.error or "model backend returned an error",
                      error_category="model_error", **fields)

            context.apply_state_event(
                StateEventType.ACTION_FAILED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="failed",
                category="model_error",
                summary=(turn.error or "model backend returned an error")[:240],
                round_number=context.working_state.current_round,
            )
        else:
            span.finish(**fields)

            context.apply_state_event(
                StateEventType.ACTION_SUCCEEDED,
                action_id=action_id,
                kind=ActionKind.MODEL.value,
                name="model_complete",
                observed_status="ok",
                summary=None,
                round_number=context.working_state.current_round,
            )

        if recorder is not None:
            try:
                if turn.stop_reason == StopReason.ERROR:
                    recorder.fail_model_turn(
                        RuntimeError(turn.error or "model backend returned an error"),
                        model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
                else:
                    recorder.complete_model_turn(
                        turn, model_action_id=action_id,
                        mutation_generation=context.working_state.mutation_generation,
                    )
            except Exception as exc:
                context.observer.notice("training_capture_error", {
                    "stage": "model_output", "error_category": type(exc).__name__,
                })
                context.training_recorder = None

        return turn

    def compose_context(
        self, context: AgentContext,
        conversation: Sequence[ConversationMessage], *, use_tools: bool,
    ) -> tuple[ContextSnapshot, str, tuple[ConversationMessage, ...]]:
        # The working-state projection is regenerated every turn, so the stale
        # copy carried in the context items is swapped out rather than added to.

        projection = working_state_context_item(context.working_state)

        items = []
        replaced = False

        for item in context.context_items:
            if item.layer == ContextLayer.TASK_WORKING_STATE:
                items.append(projection)
                replaced = True
            else:
                items.append(item)

        if not replaced:
            items.append(projection)

        items.extend(_conversation_context_items(conversation))

        # Room the reply will need, which the engine must not spend on input.

        output_reserve = context.output_reserve

        if output_reserve is None:
            value = getattr(context.backend, "max_tokens", None)
            output_reserve = value if isinstance(value, int) and value >= 0 else 8192

        request = ContextRequest(
            items=tuple(items),
            context_limit=context.context_limit,
            output_reserve=output_reserve,
            safety_margin=context.safety_margin,
            fixed_input_tokens=self._tool_schema_tokens(context.tools, use_tools) + 4,
        )

        request = self._compact_context(context, request)
        snapshot = context.context_engine.compose(request)
        system, selected = _render_context_snapshot(snapshot)

        return snapshot, system, selected

    def _compact_context(
        self, context: AgentContext, request: ContextRequest,
    ) -> ContextRequest:
        artifact = context.compaction_artifact
        service = CompactionService(
            engine=context.context_engine,
            summarizer=ModelBackendSummarizer(
                context.backend, trace=context.trace,
                provider=context.provider, model=context.model,
                budget_manager=context.budget_manager,
            ),
            policy=context.compaction_policy,
            trace=context.trace,
        )

        # The request with the existing summary already folded in: it is what
        # is returned if compaction fails, so a failure costs nothing gained.

        effective = service.apply_artifact(request, artifact) if artifact else request

        # A cooldown, because compaction has no hysteresis of its own. It
        # fires above a fraction of the window, and one round after a
        # successful compaction the context is back above it -- so a turn
        # that crosses the threshold once pays for a compaction EVERY round
        # for the rest of its life. Watched live: twenty seconds of
        # "Compacting context…" against twenty-five of work, round after
        # round, for nothing that had not just been summarised.
        #
        # Skipped only while there is room to skip it. Past the urgent mark
        # the call would be at risk of overflowing, and a compaction that
        # costs twenty seconds is cheaper than a turn that cannot continue.

        round_now = context.working_state.current_round
        since = round_now - getattr(context, "_last_compaction_round", -99)
        limit = max(1, getattr(effective, "context_limit", 0) or 1)
        pressure = context.context_engine.estimate_request_tokens(effective) / limit

        if since < COMPACTION_COOLDOWN_ROUNDS and pressure < COMPACTION_URGENT:
            return effective

        compaction_request = CompactionRequest(
            working_state=context.working_state,
            context_request=effective,
            existing_artifact=artifact,
            trigger_reason="automatic_context_pressure",
        )

        # The cooldown says compaction is ALLOWED now; it does not say the
        # context needs it. That distinction was missing, so the spinner was
        # opened first and the service declined inside it -- returning before
        # any model call, in a millisecond. `_last_compaction_round` starts at
        # -99, so round one always entered, then one round in four for the
        # rest of the turn: a session with 41 rounds and 0.42 of its window in
        # use showed "Compacting context…" about eleven times and compacted
        # nothing. The operator watched the interface alternate between
        # analysing and compacting for two minutes of work that was neither.
        #
        # Asked of the service, not re-derived here: `will_compact` is the
        # same question `compact` asks of the same numbers, so the label and
        # the work cannot disagree.

        if not service.will_compact(compaction_request):
            return effective

        context._last_compaction_round = round_now

        # Compaction is a model call -- up to three of them -- and it ran
        # behind no spinner and no notice at all. Under context pressure it
        # runs before nearly every round, so the turn showed a spinner, then
        # several silent seconds, then a fresh spinner starting again at 0s.
        # Work the operator waits for has to be work the operator can see.

        try:
            with context.observer.model_activity("Compacting context…"):
                result = service.compact(compaction_request)

            if result.succeeded:
                context.compaction_artifact = result.artifact

                self._persist_session(
                    context, SessionEventType.COMPACTION_COMPLETED,
                    {"generation": result.artifact.conversation_summary.generation,
                     "messages_compacted": result.artifact.messages_compacted},
                )

                return service.apply_artifact(request, result.artifact)
        except Exception as exc:
            context.trace.emit(
                EventType.COMPACTION_FAILED, context.task_id,
                status=EventStatus.FAILED,
                error_category=type(exc).__name__, error_summary=str(exc)[:240],
                metadata={"trigger_reason": "automatic_context_pressure",
                          "validation_result": "not_committed",
                          "raw_content_recorded": False},
            )

            # Said once per turn, not once per round. A summarizer that
            # cannot meet its bound fails on every attempt for the rest of
            # the turn -- one trace held twenty-one in a row -- and the
            # context it was called to relieve only grows. Announcing each
            # one would drown the turn; never announcing it hid the fact
            # that every later round was paying for it.

            if not getattr(context, "_compaction_failure_reported", False):
                context._compaction_failure_reported = True
                context.observer.notice(
                    "compaction_failed",
                    {"reason": str(exc)[:160], "trigger": "context_pressure"},
                )

        return effective

    @staticmethod
    def _tool_schema_tokens(
        tools: Sequence[ToolDefinition], use_tools: bool,
    ) -> int:
        if not use_tools:
            return 0

        chars = sum(
            len(tool.name) + len(tool.description)
            + len(json.dumps(tool.input_schema, ensure_ascii=False, sort_keys=True))
            for tool in tools
        )

        return chars // 4 + 1

    @staticmethod
    def _show_intermediate(
        context: AgentContext, transcript: list[str], text: str,
    ) -> None:
        if text:
            context.observer.intermediate_text(text)
            transcript.append(text)

    @staticmethod
    def _start_task(context: AgentContext) -> None:
        if context.trace_started:
            return

        context.trace_started = True

        metadata = dict(context.task_trace_metadata)
        metadata.setdefault("raw_content_recorded", False)

        context.trace.emit(
            EventType.TASK_STARTED, context.task_id,
            session_id=context.session_id,
            status=EventStatus.STARTED,
            provider=context.provider, model=context.model,
            metadata=metadata,
        )

        if context.session is not None:
            resumed = bool(context.task_trace_metadata.get("resumed"))

            # Sequence zero means nothing has ever been appended, which is the
            # one thing that distinguishes a new session from a resumed one.

            if context.session.sequence == 0:
                context.session.append(
                    SessionEventType.SESSION_STARTED, context.task_id,
                    {"provider": context.provider, "model": context.model},
                )

            context.trace.emit(
                EventType.SESSION_RESUMED if resumed else EventType.SESSION_STARTED,
                context.task_id, session_id=context.session.session_id,
                status=EventStatus.OK if resumed else EventStatus.STARTED,
                metadata={"raw_content_recorded": False},
            )

            AgentRuntime._persist_session(
                context, SessionEventType.TASK_STARTED,
                {"resumed": resumed},
            )

    @staticmethod
    def _trace_context(
        context: AgentContext, snapshot: ContextSnapshot, system: str,
        conversation: Sequence[ConversationMessage], use_tools: bool,
    ) -> None:
        layer_tokens = {
            layer.value: stats.estimated_tokens
            for layer, stats in snapshot.layer_statistics.items()
        }
        layer_items = {
            layer.value: stats.included_items
            for layer, stats in snapshot.layer_statistics.items()
        }

        # What was dropped and what was cut is traced by reason and count, not
        # by item: the content itself never goes into a trace.

        exclusions: dict[str, int] = {}

        for excluded in snapshot.excluded_items:
            exclusions[excluded.reason] = exclusions.get(excluded.reason, 0) + 1

        truncations: dict[str, int] = {}

        for decision in snapshot.truncation_decisions:
            truncations[decision.reason] = truncations.get(decision.reason, 0) + 1

        context.trace.emit(
            EventType.CONTEXT_COMPOSED, context.task_id,
            provider=context.provider, model=context.model,
            context_tokens_estimate=snapshot.estimated_input_tokens,
            metadata={
                "composition_stage": "model_request",
                "system_prompt_chars": len(system),
                "conversation_message_count": len(conversation),
                "tool_schema_count": len(context.tools) if use_tools else 0,
                "configured_context_limit": snapshot.context_limit,
                "configured_output_reserve": snapshot.output_reserve,
                "safety_margin_tokens": snapshot.safety_margin,
                "available_input_tokens": snapshot.available_input_tokens,
                "fixed_input_tokens": snapshot.fixed_input_tokens,
                "tokens_by_layer": layer_tokens,
                "items_by_layer": layer_items,
                "included_item_count": len(snapshot.selected_items),
                "excluded_item_count": len(snapshot.excluded_items),
                "exclusion_reasons": exclusions,
                "truncation_count": len(snapshot.truncation_decisions),
                "truncation_reasons": truncations,
                "overflow_tokens": snapshot.overflow_tokens,
                "token_estimate_approximate": snapshot.estimate_is_approximate,
                "raw_content_recorded": False,
            },
        )

    @staticmethod
    def _result(
        context: AgentContext, reason: RuntimeTerminalReason, response: str,
        transcript: list[str], tool_log: list[str],
        trajectory: list[Mapping[str, object]], did_modify: bool,
        budget_exhausted: bool, verification_action_id: str | None,
        cache: dict[Any, Any], *, error_category: str | None = None,
        error_summary: str | None = None, completion_deferred: bool = False,
    ) -> AgentResult:
        failure_kind = {
            RuntimeTerminalReason.MODEL_FAILURE: FailureKind.MODEL_ERROR,
            RuntimeTerminalReason.INVALID_TURN: FailureKind.INVALID_MODEL_TURN,
            RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED: FailureKind.BUDGET_EXHAUSTED,
            RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED: FailureKind.BUDGET_EXHAUSTED,
            RuntimeTerminalReason.INTERRUPTED: FailureKind.INTERRUPTED,
            RuntimeTerminalReason.STALLED: FailureKind.STALLED,
            RuntimeTerminalReason.RUNTIME_FAILURE: FailureKind.RUNTIME_ERROR,
        }.get(reason)

        # The deterministic normative boundary, on the way out and on every
        # return path. Same reasoning as the claim check below: the evidence
        # lives here, so an answer the evidence does not support must not reach
        # history either. Non-standard turns have no policy and are untouched.

        # A turn that did work and says nothing. The loop has its own
        # fallback for this, and on a bound turn it has been observed not to
        # reach here: three separate runs handed back an assessment record
        # with a blank body above it, and the reader could not tell whether
        # the model had said nothing or the harness had lost it. This is the
        # single exit every path goes through, so it is the one place the
        # guarantee can actually be made.

        if not (response or "").strip() and tool_log:
            response = no_conclusion_note(tool_log)
            context.observer.notice(
                "empty_final_synthesis", {"tool_call_count": len(tool_log)},
            )

        policy = getattr(context, "standard_policy", None)

        if policy is not None:
            stopped = (STOPPED_BY_BUDGET
                       if reason in (RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
                                     RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
                                     RuntimeTerminalReason.BUDGET_EXHAUSTED)
                       else None)
            response = policy.finalize(
                response, rounds=context.working_state.current_round,
                stopped_by=stopped)
            context.trace.emit(
                EventType.TOOL_CALL_FINISHED, context.task_id,
                session_id=context.session_id, status=EventStatus.OK,
                metadata={"component": "standard_answer_policy",
                          **{key: value
                             for key, value in policy.trace().items()
                             if key in ("bootstrap_triggered",
                                        "empty_structure_recovery_triggered",
                                        "exhaustion_triggered",
                                        "guard_replaced",
                                        "provenance_guard_triggered",
                                        # Without this the guard is invisible
                                        # in the only trace a diagnosis reads:
                                        # a removed requirement had to be
                                        # recovered by re-running the turn
                                        # under --record.
                                        "normative_precedence_triggered",
                                        "conformance_guard_triggered",
                                        "model_tool_calls", "stopped_by")}},
            )

        # Checked here rather than at the rendering layer: the evidence lives
        # here, and a claim contradicted by it must not reach history either.

        claim_note = unsupported_change_claim(response or "", tool_log, did_modify)

        if claim_note:
            response = (response or "") + claim_note

        # The last word on a turn that wrote. Deterministic, from the tool log,
        # and on every exit path -- the budget's included, which is the one
        # that handed back four edited files and a tree that did not compile.

        # The project's own verification first, because the note below has to
        # know whether it ran: it is cached per generation, so this is the
        # same single execution either way.

        runs = project_build_runs(context, tool_log)
        write_note = unverified_write_note(tool_log, runs)

        if write_note:
            response = (response or "") + write_note

        broken, output = next(((command, text) for command, status, text in runs
                               if status != "passed"), ("", ""))
        commands = getattr(context, "project_commands", None)

        # A verification that could not be performed leaves the turn UNJUDGED,
        # not failed: "not judged" and "judged and wrong" are different facts,
        # and a corpus that confuses them teaches the confusion.

        unjudged = any(status == "not_run" for _, status, _ in runs)
        build_ok = (None if commands is None or not changed_files(tool_log)
                    or unjudged else not broken)

        # The harness just ran the project's own build and tests against the
        # code this turn wrote. That IS a verification of this generation, and
        # recording it is what stops the turn from demanding the model run the
        # same suite a second time to prove what the harness already observed.
        # Only a CONFIGURED command counts: a probed `make` is a guess about
        # how to build a tree, not a certificate the operator asked for.

        _record_project_verification(context, commands, runs)

        if broken:
            response = (response or "") + project_build.note(broken, output)

        # What this turn read, so the next one starts from it rather than
        # from whatever a fresh search happens to surface.

        if policy is not None:
            context.clauses_read = tuple(sorted(policy.clauses.sections))

        # And the tally of the work order's own sections, when the turn was
        # working from one. Deterministic, from the diff.

        order, missing = work_order_gap(context, tool_log, cache)

        if order:
            response = (response or "") + work_order.record(
                order, changed_files(tool_log), missing)

        return AgentResult(
            task_id=context.task_id,
            terminal_status=context.working_state.terminal_status,
            terminal_reason=reason,
            final_response=response,
            working_state=context.working_state,
            rounds_consumed=context.working_state.current_round,
            model_calls=context.working_state.model_rounds,
            tool_actions=context.working_state.tool_actions,
            verification_outcome=context.working_state.verification_outcome,
            conversation=tuple(context.conversation),
            transcript=tuple(transcript),
            tool_log=tuple(tool_log),
            trajectory=tuple(trajectory),
            did_modify=did_modify,
            budget_exhausted=budget_exhausted,
            verification_action_id=verification_action_id,
            error_category=error_category,
            error_summary=error_summary,
            failure_kind=failure_kind,
            execution_cache=cache,
            completion_deferred=completion_deferred,
            project_build_ok=build_ok,
        )


def _usage_token(usage: Mapping[str, int], *names: str) -> int | None:
    """The first of ``names`` the provider reported, since each spells it its own way."""

    for name in names:
        value = usage.get(name)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None
