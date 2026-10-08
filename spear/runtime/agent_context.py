"""The task an agent runs: its context, its result, and what the runtime is given.

AgentContext carries everything task-scoped -- working state, conversation,
budgets, trace, session -- so that AgentRuntime itself holds no task state.
The tool executor and the observer are injected through the protocols here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, ContextManager, Mapping, Protocol

from runtime.cancellation import CancellationToken, NEVER_CANCELLED
from harness.checkpoint import (
    Checkpoint, CheckpointManager, RollbackResult, RollbackStatus,
)
from runtime.failure_policy import FailureKind, RetryPolicy
from runtime.progress_monitor import ProgressMonitor
from runtime.session_store import SessionEventType, SessionHandle, SessionSnapshot
from runtime.compaction import CompactionArtifact, CompactionPolicy
from context.context_engine import ContextEngine, ContextItem
from models.model_backend import ConversationMessage, ModelBackend, ToolDefinition
from runtime.tracing import EventStatus, EventType, TraceEmitter, new_action_id
from harness.tool_router import ToolResultEnvelope
from evidence.verification import VerificationPolicy
from runtime.budgets import BudgetManager
from runtime import work_phase
from runtime.working_state import (
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
    #: Tool results the conversation holds but the last request did not
    #: carry -- folded into a compaction summary or left out by the composer.
    #: The guards that say "you already have this" ask here, because the
    #: conversation keeps every block whatever the model was actually sent.
    dropped_tool_results: frozenset[str] = frozenset()
    #: Which runtime runs the turn: "coding" for the agent core (agent/),
    #: "legacy" for the loop that serves standard-bound turns.
    execution_core: str = "legacy"
    #: The agent core's host factory, (context, cache, record) -> Host, and
    #: the project context its system prompt carries. Set by the client.
    coding_host: Any = None
    coding_context: str = ""
    #: The agent core path's verdict, once decided (completion.Verdict).
    core_verdict: Any = None
    #: The agent core path's structured evidence (completion.Evidence), one
    #: per call. What changed is read from here, not from the tool log text.
    core_evidence: Any = None
    #: A MIXED turn's project-side normative configuration: the project spec
    #: holding `normative_checks` / `normative_applicability`, where it was
    #: declared, and the confined runner a bound check runs through,
    #: command -> (status, exit code, output).
    normative_project: Any = None
    normative_authority: str = ""
    normative_check_runner: Any = None
    #: Which context each pass of the turn was given (context_selection): the
    #: workspace, the selection per phase, and what each phase renders to. A
    #: MIXED pass that finds its phase here runs on that selection.
    workspace_context: Any = None
    context_selections: Any = None
    phase_contexts: Any = None
    #: The turn's tool window (Finalization), set when the turn starts.
    finalization: Any = None
    #: The control plane's door to the external capabilities this turn's
    #: workspace admits (capability_gateway.Gateway), or None.
    capability_gateway: Any = None
    #: The read-only door to this turn's workspace knowledge
    #: (workspace_knowledge.Door), or None.
    knowledge_door: Any = None
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
    #: Read-only because the request asked about a change rather than for
    #: one. Same boundary; the refusals say so instead of citing a
    #: prohibition the user never wrote.
    advisory: bool = False

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

    #: What the previous grounded turn established, when this turn pointed
    #: back at it. The lifecycle holds it to those requirements: each one
    #: needs a disposition before anything is written. None on every turn
    #: that set its own scope.
    carried_requirements: object | None = None

    #: INVESTIGATE / PLAN / EDIT / TEST / REVIEW, and the gap model behind
    #: them. Attached by the caller and engaged by the runtime for the one
    #: shape it governs -- an authoritative source to satisfy AND a request to
    #: change the code. On every other turn it stays disengaged and decides
    #: nothing, so a pure question and a pure code task take the path they
    #: always took. Turn-scoped, like the policy above.
    work_phase: object | None = None

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

        from runtime.agent_runtime import AgentRuntime

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

    # The files the turn changed, as established by the turn's evidence.
    changed_paths: tuple[str, ...] = ()
