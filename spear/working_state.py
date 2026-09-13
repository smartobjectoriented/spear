"""Provider-neutral, event-reduced truth for one SPEAR agent task.

The model does not own this state.  Grounded facts enter through explicit
events emitted by the harness or tool runtime.  Events are applied once:
duplicate event IDs are rejected instead of silently double-counting work.

Serialization stores the ordered transition log as schema version 1 and
reconstructs all derived fields by replay.  Unknown envelope/event fields are
ignored for additive compatibility; unknown schema versions and event kinds
are rejected because their semantics cannot be reconstructed safely.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1


class WorkingStateError(ValueError):
    pass


class DuplicateEventError(WorkingStateError):
    pass


class StateTransitionError(WorkingStateError):
    pass


class SerializationError(WorkingStateError):
    pass


class StateSource(StrEnum):
    USER = "user"
    HARNESS = "harness"
    TOOL_RUNTIME = "tool_runtime"
    RETRIEVAL = "retrieval"
    MODEL = "model"


class StateEventType(StrEnum):
    TASK_STARTED = "task_started"
    OBJECTIVE_RECORDED = "objective_recorded"
    CONSTRAINT_RECORDED = "constraint_recorded"
    ACCEPTANCE_CRITERION_RECORDED = "acceptance_criterion_recorded"
    PLAN_STEP_ADDED = "plan_step_added"
    PLAN_STEP_UPDATED = "plan_step_updated"
    PLAN_STEP_REVISED = "plan_step_revised"
    DISCOVERY_RECORDED = "discovery_recorded"
    DECISION_RECORDED = "decision_recorded"
    FILE_READ = "file_read"
    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_FAILED = "action_failed"
    ACTION_CANCELLED = "action_cancelled"
    FAILURE_RESOLVED = "failure_resolved"
    VERIFICATION_RECORDED = "verification_recorded"
    REVIEW_RECORDED = "review_recorded"
    CHECKPOINT_RECORDED = "checkpoint_recorded"
    CHECKPOINT_STATUS_UPDATED = "checkpoint_status_updated"
    ROUND_STARTED = "round_started"
    RETRY_RECORDED = "retry_recorded"
    REPEATED_ACTION_RECORDED = "repeated_action_recorded"
    VERIFICATION_REPROMPT_RECORDED = "verification_reprompt_recorded"
    BUDGET_REACHED = "budget_reached"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_INTERRUPTED = "task_interrupted"
    TASK_RESUMED = "task_resumed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class TerminalStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BUDGET_EXHAUSTED = "budget_exhausted"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ActionKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    COMMAND = "command"


class ActionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationOutcome(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


def _opaque_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _required_text(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)

    if not isinstance(value, str) or not value.strip():
        raise StateTransitionError(f"{name} must be a non-empty string")

    return value.strip()


def _optional_text(data: Mapping[str, Any], name: str) -> str | None:
    value = data.get(name)

    if value is None:
        return None

    if not isinstance(value, str):
        raise StateTransitionError(f"{name} must be a string or null")

    return value.strip() or None


def _required_int(data: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = data.get(name)

    # bool is an int subclass, so it has to be excluded explicitly.

    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise StateTransitionError(f"{name} must be an integer >= {minimum}")

    return value


def _string_tuple(data: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name, ())

    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise StateTransitionError(f"{name} must be a list of non-empty strings")

    return tuple(value)


@dataclass(frozen=True)
class StateEvent:
    event_type: StateEventType
    task_id: str
    source: StateSource
    data: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: _opaque_id("stateevt"))
    timestamp: str = field(default_factory=_timestamp)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, StateEventType):
            raise StateTransitionError("event_type must be a StateEventType")

        if not isinstance(self.source, StateSource):
            raise StateTransitionError("event source must be a StateSource")

        if not isinstance(self.task_id, str) or not self.task_id:
            raise StateTransitionError("event task_id must be a non-empty string")

        if not isinstance(self.event_id, str) or not self.event_id:
            raise StateTransitionError("event_id must be a non-empty string")

        if not isinstance(self.data, Mapping):
            raise StateTransitionError("event data must be a mapping")

        try:
            # JSON round-tripping gives the frozen event an independent,
            # JSON-compatible payload rather than a shallow view of mutable
            # caller data. NaN/Infinity are rejected because JSONL consumers
            # are not required to accept those non-standard values.

            data = json.loads(json.dumps(
                dict(self.data), ensure_ascii=False, allow_nan=False,
            ))
        except (TypeError, ValueError, OverflowError) as exc:
            raise StateTransitionError(f"event data is not JSON-compatible: {exc}") from exc

        object.__setattr__(self, "data", MappingProxyType(data))

    @classmethod
    def create(
        cls,
        event_type: StateEventType,
        task_id: str,
        source: StateSource,
        **data: Any,
    ) -> "StateEvent":
        return cls(event_type, task_id, source, data)

    def to_dict(self) -> dict[str, Any]:
        try:
            data = json.loads(json.dumps(dict(self.data), ensure_ascii=False))
        except (TypeError, ValueError, OverflowError) as exc:
            raise SerializationError(f"event data is not JSON-compatible: {exc}") from exc

        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "monotonic_ns": self.monotonic_ns,
            "data": data,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StateEvent":
        if not isinstance(raw, Mapping):
            raise SerializationError("event must be an object")

        try:
            event_type = StateEventType(raw["event_type"])
            source = StateSource(raw["source"])
            task_id = raw["task_id"]
            event_id = raw["event_id"]
            timestamp = raw["timestamp"]
            monotonic_ns = raw["monotonic_ns"]
            data = raw.get("data", {})
        except (KeyError, TypeError, ValueError) as exc:
            raise SerializationError(f"malformed state event: {exc}") from exc

        if not isinstance(timestamp, str) or not timestamp:
            raise SerializationError("event timestamp must be a non-empty string")

        if not isinstance(monotonic_ns, int) or isinstance(monotonic_ns, bool):
            raise SerializationError("event monotonic_ns must be an integer")

        # Construction re-validates the payload; a rejection here is a
        # deserialization failure, not a live transition failure.

        try:
            return cls(event_type, task_id, source, data, event_id, timestamp, monotonic_ns)
        except WorkingStateError as exc:
            raise SerializationError(str(exc)) from exc


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    evidence_action_ids: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    requires_evidence: bool = False


@dataclass(frozen=True)
class PlanRevision:
    step_id: str
    prior_description: str
    revised_description: str
    reason: str
    event_id: str
    timestamp: str


@dataclass(frozen=True)
class Discovery:
    summary: str
    file_path: str | None = None
    symbol: str | None = None
    action_id: str | None = None


@dataclass(frozen=True)
class Decision:
    summary: str
    rationale: str | None = None
    action_id: str | None = None


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    kind: ActionKind
    name: str
    status: ActionStatus
    summary: str | None
    round_number: int | None
    output_chars: int | None
    exit_code: int | None
    event_id: str
    timestamp: str


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    action_id: str | None
    category: str
    summary: str
    resolved: bool
    event_id: str
    timestamp: str
    resolved_by_event_id: str | None = None


@dataclass(frozen=True)
class VerificationRecord:
    verification_id: str
    kind: str
    executed: bool
    outcome: VerificationOutcome
    action_id: str | None
    summary: str | None
    event_id: str
    timestamp: str
    category: str = "unknown"
    coverage: str = "unknown"
    mutation_generation: int = 0
    result_reference: str | None = None
    command: str | None = None
    project_bench: bool = False


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    mutation_generation: int
    verdict: str
    blocking_count: int
    evidence_reference: str | None
    event_id: str
    timestamp: str


# Which event sources are trusted to assert each class of fact. The model is
# absent from all of them: it may propose, but only the harness and the tool
# runtime observe.

_GROUNDED_FILE_SOURCES = {StateSource.TOOL_RUNTIME}
_GROUNDED_FACT_SOURCES = {
    StateSource.HARNESS, StateSource.TOOL_RUNTIME, StateSource.RETRIEVAL,
}
_GROUNDED_ACTION_SOURCES = {StateSource.HARNESS, StateSource.TOOL_RUNTIME}
_GROUNDED_VERIFICATION_SOURCES = {StateSource.HARNESS, StateSource.TOOL_RUNTIME}


@dataclass
class WorkingState:
    task_id: str
    objective: str = ""
    user_constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    plan_steps: dict[str, PlanStep] = field(default_factory=dict)
    plan_revisions: list[PlanRevision] = field(default_factory=list)
    discoveries: list[Discovery] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    modified_files: set[str] = field(default_factory=set)
    created_files: set[str] = field(default_factory=set)
    deleted_files: set[str] = field(default_factory=set)
    actions: list[ActionRecord] = field(default_factory=list)
    failures: dict[str, FailureRecord] = field(default_factory=dict)
    verifications: list[VerificationRecord] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    current_round: int = 0
    model_rounds: int = 0
    tool_actions: int = 0
    command_actions: int = 0
    tool_budget_used: int = 0
    max_model_rounds: int | None = None
    max_tool_actions: int | None = None
    retry_count: int = 0
    repeated_action_count: int = 0
    verification_reprompt_count: int = 0
    mutation_generation: int = 0
    mutation_action_ids: set[str] = field(default_factory=set, repr=False)
    checkpoint_id: str | None = None
    checkpoint_status: str | None = None
    budget_reached: bool = False
    terminal_status: TerminalStatus = TerminalStatus.RUNNING
    outcome_summary: str | None = None
    started: bool = False
    events: list[StateEvent] = field(default_factory=list, repr=False)
    applied_event_ids: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise StateTransitionError("task_id must be a non-empty string")

    @classmethod
    def start(
        cls,
        task_id: str,
        objective: str,
        *,
        max_model_rounds: int | None = None,
        max_tool_actions: int | None = None,
    ) -> "WorkingState":
        state = cls(task_id)

        state.apply(StateEvent.create(
            StateEventType.TASK_STARTED,
            task_id,
            StateSource.USER,
            objective=objective,
            max_model_rounds=max_model_rounds,
            max_tool_actions=max_tool_actions,
        ))

        return state

    @property
    def completed_steps(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.plan_steps.values()
                     if step.status == PlanStepStatus.COMPLETED)

    @property
    def pending_steps(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.plan_steps.values()
                     if step.status == PlanStepStatus.PENDING)

    @property
    def blocked_steps(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.plan_steps.values()
                     if step.status == PlanStepStatus.BLOCKED)

    @property
    def current_step(self) -> PlanStep | None:
        return next((step for step in self.plan_steps.values()
                     if step.status == PlanStepStatus.ACTIVE), None)

    @property
    def unresolved_failures(self) -> tuple[FailureRecord, ...]:
        return tuple(failure for failure in self.failures.values() if not failure.resolved)

    @property
    def verification_outcome(self) -> VerificationOutcome:
        # Only verifications covering the current mutation generation count;
        # anything older describes code that has since changed.

        current = [item for item in self.verifications
                   if item.mutation_generation == self.mutation_generation
                   and item.coverage != "none"]

        return current[-1].outcome if current else VerificationOutcome.NOT_RUN

    @property
    def current_review(self) -> ReviewRecord | None:
        return next((item for item in reversed(self.reviews)
                     if item.mutation_generation == self.mutation_generation), None)

    @property
    def current_review_accepted(self) -> bool:
        review = self.current_review

        return bool(review and review.verdict == "accept" and review.blocking_count == 0)

    @property
    def relevant_files(self) -> frozenset[str]:
        paths = self.files_read | self.modified_files | self.created_files | self.deleted_files
        paths.update(item.file_path for item in self.discoveries if item.file_path)

        return frozenset(paths)

    def apply(self, event: StateEvent) -> None:
        if event.task_id != self.task_id:
            raise StateTransitionError("event task_id does not match WorkingState")

        if event.event_id in self.applied_event_ids:
            raise DuplicateEventError(f"duplicate state event: {event.event_id}")

        # A finished task accepts only the two events that can legitimately
        # follow it: resolving a recorded failure, and resuming an interrupt.

        if self.terminal_status != TerminalStatus.RUNNING and event.event_type not in {
            StateEventType.FAILURE_RESOLVED, StateEventType.TASK_RESUMED,
        }:
            raise StateTransitionError(
                f"cannot apply {event.event_type.value} after {self.terminal_status.value}"
            )

        # Reduce first: a rejected event must leave the log untouched.

        self._reduce(event)

        self.events.append(event)
        self.applied_event_ids.add(event.event_id)

    def _reduce(self, event: StateEvent) -> None:
        kind, data = event.event_type, event.data

        if kind == StateEventType.TASK_STARTED:
            if self.started:
                raise StateTransitionError("task has already started")

            if event.source != StateSource.USER:
                raise StateTransitionError("task definition must originate from the user")

            objective = _required_text(data, "objective")

            for name in ("max_model_rounds", "max_tool_actions"):
                value = data.get(name)

                if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                          or value < 1):
                    raise StateTransitionError(f"{name} must be null or a positive integer")

            self.objective = objective
            self.max_model_rounds = data.get("max_model_rounds")
            self.max_tool_actions = data.get("max_tool_actions")
            self.started = True

            return

        if kind == StateEventType.OBJECTIVE_RECORDED:
            if event.source != StateSource.USER:
                raise StateTransitionError("objective updates must originate from the user")

            self.objective = _required_text(data, "objective")

            return

        if kind == StateEventType.CONSTRAINT_RECORDED:
            if event.source != StateSource.USER:
                raise StateTransitionError("constraints must originate from the user")

            value = _required_text(data, "constraint")

            if value not in self.user_constraints:
                self.user_constraints.append(value)

            return

        if kind == StateEventType.ACCEPTANCE_CRITERION_RECORDED:
            if event.source != StateSource.USER:
                raise StateTransitionError("acceptance criteria must originate from the user")

            value = _required_text(data, "criterion")

            if value not in self.acceptance_criteria:
                self.acceptance_criteria.append(value)

            return

        if kind == StateEventType.PLAN_STEP_ADDED:
            if event.source not in {StateSource.USER, StateSource.HARNESS}:
                raise StateTransitionError(
                    "plans must originate from the user or harness validation"
                )

            step_id = _required_text(data, "step_id")

            if step_id in self.plan_steps:
                raise StateTransitionError(f"plan step already exists: {step_id}")

            # Steps are appended in order, so a dependency that is not already
            # present is either forward-looking or the step depending on itself.

            dependencies = _string_tuple(data, "dependencies")

            if step_id in dependencies or any(item not in self.plan_steps for item in dependencies):
                raise StateTransitionError("plan dependencies must reference earlier steps")

            criteria = _string_tuple(data, "completion_criteria")
            requires_evidence = data.get("requires_evidence", False)

            if not isinstance(requires_evidence, bool):
                raise StateTransitionError("requires_evidence must be boolean")

            self.plan_steps[step_id] = PlanStep(
                step_id, _required_text(data, "description"),
                dependencies=dependencies, completion_criteria=criteria,
                requires_evidence=requires_evidence,
            )

            return

        if kind == StateEventType.PLAN_STEP_UPDATED:
            self._update_plan_step(event)

            return

        if kind == StateEventType.PLAN_STEP_REVISED:
            if event.source != StateSource.HARNESS:
                raise StateTransitionError("plan revisions must originate from orchestration")

            step_id = _required_text(data, "step_id")
            step = self.plan_steps.get(step_id)

            if step is None:
                raise StateTransitionError(f"unknown plan step: {step_id}")

            description = _required_text(data, "description")
            reason = _required_text(data, "reason")
            status = PlanStepStatus(data.get("status", PlanStepStatus.PENDING.value))

            # A revision rewrites what the step means, so it cannot land on
            # COMPLETED: the new description has not been done by anyone yet.

            if status not in {PlanStepStatus.PENDING, PlanStepStatus.ACTIVE,
                              PlanStepStatus.BLOCKED, PlanStepStatus.SKIPPED}:
                raise StateTransitionError("revised plan step cannot remain completed")

            if status == PlanStepStatus.ACTIVE and any(
                item.status == PlanStepStatus.ACTIVE and item.step_id != step_id
                for item in self.plan_steps.values()
            ):
                raise StateTransitionError("only one plan step may be active")

            self.plan_revisions.append(PlanRevision(
                step_id, step.description, description,
                reason, event.event_id, event.timestamp,
            ))

            # The old evidence proved the old description; it is dropped.

            self.plan_steps[step_id] = replace(
                step, description=description, status=status,
                evidence_action_ids=(),
            )

            return

        if kind == StateEventType.DISCOVERY_RECORDED:
            if event.source not in _GROUNDED_FACT_SOURCES:
                raise StateTransitionError("discoveries require grounded harness/runtime evidence")

            self.discoveries.append(Discovery(
                _required_text(data, "summary"), _optional_text(data, "file_path"),
                _optional_text(data, "symbol"), _optional_text(data, "action_id"),
            ))

            return

        if kind == StateEventType.DECISION_RECORDED:
            # Unlike a discovery, a decision is a judgement rather than an
            # observation, so the model is allowed to be its source.

            if event.source not in {StateSource.HARNESS, StateSource.MODEL}:
                raise StateTransitionError("decisions must originate from the harness or model")

            self.decisions.append(Decision(
                _required_text(data, "summary"), _optional_text(data, "rationale"),
                _optional_text(data, "action_id"),
            ))

            return

        if kind in {
            StateEventType.FILE_READ, StateEventType.FILE_MODIFIED,
            StateEventType.FILE_CREATED, StateEventType.FILE_DELETED,
        }:
            self._record_file_event(event)

            return

        if kind in {
            StateEventType.ACTION_SUCCEEDED, StateEventType.ACTION_FAILED,
            StateEventType.ACTION_CANCELLED,
        }:
            self._record_action(event)

            return

        if kind == StateEventType.FAILURE_RESOLVED:
            failure_id = _required_text(data, "failure_id")
            failure = self.failures.get(failure_id)

            if failure is None:
                raise StateTransitionError(f"unknown failure: {failure_id}")

            if failure.resolved:
                raise StateTransitionError(f"failure already resolved: {failure_id}")

            self.failures[failure_id] = replace(
                failure, resolved=True, resolved_by_event_id=event.event_id,
            )

            return

        if kind == StateEventType.VERIFICATION_RECORDED:
            self._record_verification(event)

            return

        if kind == StateEventType.REVIEW_RECORDED:
            if event.source != StateSource.HARNESS:
                raise StateTransitionError("reviews must originate from orchestration")

            generation = event.data.get("mutation_generation")
            blocking = event.data.get("blocking_count")

            # A review of a generation that does not exist yet cannot be
            # honoured, so future generations are rejected along with garbage.

            if (not isinstance(generation, int) or isinstance(generation, bool)
                    or generation < 0 or generation > self.mutation_generation):
                raise StateTransitionError("invalid review mutation generation")

            if (not isinstance(blocking, int) or isinstance(blocking, bool)
                    or blocking < 0):
                raise StateTransitionError("invalid review blocking count")

            verdict = _required_text(event.data, "verdict")

            if verdict not in {"accept", "repair_required", "blocked"}:
                raise StateTransitionError("invalid review verdict")

            review_id = _required_text(event.data, "review_id")

            if any(item.review_id == review_id for item in self.reviews):
                raise StateTransitionError(f"review already recorded: {review_id}")

            if verdict == "accept" and blocking:
                raise StateTransitionError("accepted review cannot contain blockers")

            self.reviews.append(ReviewRecord(
                review_id, generation, verdict, blocking,
                _optional_text(event.data, "evidence_reference"),
                event.event_id, event.timestamp,
            ))

            return

        if kind == StateEventType.CHECKPOINT_RECORDED:
            if event.source != StateSource.HARNESS:
                raise StateTransitionError("checkpoint state must originate from harness")

            if self.checkpoint_id is not None:
                raise StateTransitionError("task already has a checkpoint")

            self.checkpoint_id = _required_text(data, "checkpoint_id")
            self.checkpoint_status = _required_text(data, "status")

            return

        if kind == StateEventType.CHECKPOINT_STATUS_UPDATED:
            if event.source != StateSource.HARNESS:
                raise StateTransitionError("checkpoint state must originate from harness")

            if _required_text(data, "checkpoint_id") != self.checkpoint_id:
                raise StateTransitionError("checkpoint ID does not match task")

            self.checkpoint_status = _required_text(data, "status")

            return

        if kind == StateEventType.ROUND_STARTED:
            round_number = _required_int(data, "round_number", minimum=1)

            if round_number <= self.current_round:
                raise StateTransitionError("round numbers must increase")

            self.current_round = round_number

            return

        if kind == StateEventType.RETRY_RECORDED:
            self.retry_count += 1

            return

        if kind == StateEventType.REPEATED_ACTION_RECORDED:
            self.repeated_action_count += 1

            return

        if kind == StateEventType.VERIFICATION_REPROMPT_RECORDED:
            self.verification_reprompt_count += 1

            return

        if kind == StateEventType.BUDGET_REACHED:
            self.budget_reached = True

            return

        if kind == StateEventType.TASK_RESUMED:
            if self.terminal_status != TerminalStatus.INTERRUPTED:
                raise StateTransitionError("only an interrupted task may be resumed")

            if event.source != StateSource.HARNESS:
                raise StateTransitionError("task resume must originate from the harness")

            self.terminal_status = TerminalStatus.RUNNING
            self.outcome_summary = None

            return

        if kind in {
            StateEventType.TASK_COMPLETED, StateEventType.TASK_FAILED,
            StateEventType.TASK_INTERRUPTED, StateEventType.BUDGET_EXHAUSTED,
        }:
            terminal = {
                StateEventType.TASK_COMPLETED: TerminalStatus.COMPLETED,
                StateEventType.TASK_FAILED: TerminalStatus.FAILED,
                StateEventType.TASK_INTERRUPTED: TerminalStatus.INTERRUPTED,
                StateEventType.BUDGET_EXHAUSTED: TerminalStatus.BUDGET_EXHAUSTED,
            }[kind]

            summary = _optional_text(data, "summary")

            self.terminal_status = terminal
            self.outcome_summary = summary

            return

        raise StateTransitionError(f"unsupported state event: {kind.value}")

    def _update_plan_step(self, event: StateEvent) -> None:
        data = event.data

        if event.source not in _GROUNDED_FACT_SOURCES:
            raise StateTransitionError("plan progress requires grounded harness/runtime evidence")

        step_id = _required_text(data, "step_id")
        step = self.plan_steps.get(step_id)

        if step is None:
            raise StateTransitionError(f"unknown plan step: {step_id}")

        try:
            status = PlanStepStatus(data.get("status"))
        except (TypeError, ValueError) as exc:
            raise StateTransitionError("invalid plan step status") from exc

        # COMPLETED and SKIPPED are terminal: a step cannot be reopened, which
        # is what keeps the evidence attached to it meaningful.

        allowed = {
            PlanStepStatus.PENDING: {PlanStepStatus.ACTIVE, PlanStepStatus.BLOCKED,
                                     PlanStepStatus.SKIPPED},
            PlanStepStatus.ACTIVE: {PlanStepStatus.COMPLETED, PlanStepStatus.BLOCKED,
                                    PlanStepStatus.SKIPPED},
            PlanStepStatus.BLOCKED: {PlanStepStatus.ACTIVE, PlanStepStatus.SKIPPED},
            PlanStepStatus.COMPLETED: set(),
            PlanStepStatus.SKIPPED: set(),
        }

        if status not in allowed[step.status]:
            raise StateTransitionError(
                f"invalid plan transition: {step.status.value} -> {status.value}"
            )

        if status == PlanStepStatus.ACTIVE and any(
            item.status == PlanStepStatus.ACTIVE and item.step_id != step_id
            for item in self.plan_steps.values()
        ):
            raise StateTransitionError("only one plan step may be active")

        evidence = data.get("evidence_action_ids", ())

        if not isinstance(evidence, (list, tuple)) or not all(
            isinstance(item, str) and item for item in evidence
        ):
            raise StateTransitionError("evidence_action_ids must be a list of strings")

        if status == PlanStepStatus.ACTIVE and any(
            self.plan_steps[item].status not in {
                PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED,
            }
            for item in step.dependencies
        ):
            raise StateTransitionError("plan step dependencies are incomplete")

        if status == PlanStepStatus.COMPLETED and step.requires_evidence:
            if event.source == StateSource.MODEL or not evidence:
                raise StateTransitionError("grounded plan completion requires evidence")

            # The only things that can serve as evidence are outcomes something
            # other than the model actually observed: a successful action, a
            # discovery it produced, a passing verification, an accepted review.

            grounded = ({item.action_id for item in self.actions
                         if item.status == ActionStatus.SUCCEEDED}
                        | {item.action_id for item in self.discoveries if item.action_id}
                        | {item.verification_id for item in self.verifications
                           if item.outcome == VerificationOutcome.PASSED}
                        | {item.action_id for item in self.verifications
                           if item.action_id and item.outcome == VerificationOutcome.PASSED}
                        | {item.review_id for item in self.reviews
                           if item.verdict == "accept" and item.blocking_count == 0})

            if not set(evidence) <= grounded:
                raise StateTransitionError("plan completion references ungrounded evidence")

        self.plan_steps[step_id] = replace(
            step, status=status, evidence_action_ids=tuple(evidence),
        )

    def _record_file_event(self, event: StateEvent) -> None:
        if event.source not in _GROUNDED_FILE_SOURCES:
            raise StateTransitionError("file state requires grounded tool-runtime evidence")

        path = _required_text(event.data, "path")
        action_id = _required_text(event.data, "action_id")

        # Every file fact has to name the action that produced it, so the file
        # set can never drift from the actions actually recorded.

        action = next(
            (item for item in self.actions if item.action_id == action_id), None
        )

        if action is None:
            raise StateTransitionError("file state must reference a recorded runtime action")

        if event.event_type != StateEventType.FILE_READ and (
            action.kind != ActionKind.TOOL or action.status != ActionStatus.SUCCEEDED
        ):
            raise StateTransitionError(
                "file mutation requires a successful grounded tool action"
            )

        # A path holds one status at a time: creating undoes a deletion, and
        # deleting withdraws the creation and any modification claimed for it.

        if event.event_type == StateEventType.FILE_READ:
            self.files_read.add(path)
        elif event.event_type == StateEventType.FILE_MODIFIED:
            self.modified_files.add(path)
        elif event.event_type == StateEventType.FILE_CREATED:
            self.created_files.add(path)
            self.deleted_files.discard(path)
        else:
            self.deleted_files.add(path)
            self.created_files.discard(path)
            self.modified_files.discard(path)

        # The generation counts mutating actions, not mutated paths, so one
        # action touching several files still advances it exactly once.

        if (event.event_type != StateEventType.FILE_READ
                and action_id not in self.mutation_action_ids):
            self.mutation_action_ids.add(action_id)
            self.mutation_generation += 1

    def _record_action(self, event: StateEvent) -> None:
        if event.source not in _GROUNDED_ACTION_SOURCES:
            raise StateTransitionError("actions require grounded harness/runtime evidence")

        action_id = _required_text(event.data, "action_id")

        if any(action.action_id == action_id for action in self.actions):
            raise StateTransitionError(f"action already recorded: {action_id}")

        try:
            action_kind = ActionKind(event.data.get("kind"))
        except (TypeError, ValueError) as exc:
            raise StateTransitionError("invalid action kind") from exc

        status = {
            StateEventType.ACTION_SUCCEEDED: ActionStatus.SUCCEEDED,
            StateEventType.ACTION_FAILED: ActionStatus.FAILED,
            StateEventType.ACTION_CANCELLED: ActionStatus.CANCELLED,
        }[event.event_type]

        # The event kind and the status the runtime actually observed must
        # agree, so a mislabelled event cannot record a success that failed.

        observed_status = event.data.get("observed_status")
        expected_observed = {
            ActionStatus.SUCCEEDED: "ok",
            ActionStatus.FAILED: "failed",
            ActionStatus.CANCELLED: "cancelled",
        }[status]

        if observed_status != expected_observed:
            raise StateTransitionError(
                f"{event.event_type.value} requires observed_status={expected_observed!r}"
            )

        round_number = event.data.get("round_number")

        if round_number is not None and (
            not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 0
        ):
            raise StateTransitionError("round_number must be null or a non-negative integer")

        output_chars = event.data.get("output_chars")

        if output_chars is not None and (
            not isinstance(output_chars, int) or isinstance(output_chars, bool) or output_chars < 0
        ):
            raise StateTransitionError("output_chars must be null or non-negative")

        exit_code = event.data.get("exit_code")

        if exit_code is not None and (
            not isinstance(exit_code, int) or isinstance(exit_code, bool)
        ):
            raise StateTransitionError("exit_code must be null or an integer")

        if (action_kind == ActionKind.COMMAND and status == ActionStatus.SUCCEEDED
                and exit_code not in (None, 0)):
            raise StateTransitionError("a nonzero command exit cannot be successful")

        record = ActionRecord(
            action_id, action_kind, _required_text(event.data, "name"), status,
            _optional_text(event.data, "summary"), round_number, output_chars, exit_code,
            event.event_id, event.timestamp,
        )

        failure_category = None

        if status == ActionStatus.FAILED:
            failure_category = _optional_text(event.data, "category") or "action_failed"

        self.actions.append(record)

        # Model turns and tool work draw on separate budgets; a command is a
        # tool action that additionally counts against the command tally.

        if action_kind == ActionKind.MODEL:
            self.model_rounds += 1
        else:
            self.tool_actions += 1
            self.tool_budget_used += 1

            if action_kind == ActionKind.COMMAND:
                self.command_actions += 1

        # A failed action opens a failure keyed by that action, so resolving it
        # later needs no separate bookkeeping to find it.

        if status == ActionStatus.FAILED:
            failure_id = f"failure:{action_id}"
            self.failures[failure_id] = FailureRecord(
                failure_id, action_id,
                failure_category,
                record.summary or f"{record.name} failed",
                False, event.event_id, event.timestamp,
            )

    def _record_verification(self, event: StateEvent) -> None:
        if event.source not in _GROUNDED_VERIFICATION_SOURCES:
            raise StateTransitionError("verification requires grounded harness/runtime evidence")

        executed = event.data.get("executed")

        if not isinstance(executed, bool):
            raise StateTransitionError("verification executed must be boolean")

        try:
            outcome = VerificationOutcome(event.data.get("outcome"))
        except (TypeError, ValueError) as exc:
            raise StateTransitionError("invalid verification outcome") from exc

        # "Did it run" and "what did it say" have to agree in both directions,
        # which is what stops an unrun check from being recorded as a pass.

        if not executed and outcome != VerificationOutcome.NOT_RUN:
            raise StateTransitionError("verification that did not execute cannot pass or fail")

        if executed and outcome == VerificationOutcome.NOT_RUN:
            raise StateTransitionError("executed verification needs a pass/fail outcome")

        verification_id = _required_text(event.data, "verification_id")

        if any(item.verification_id == verification_id for item in self.verifications):
            raise StateTransitionError(f"verification already recorded: {verification_id}")

        generation = event.data.get("mutation_generation", self.mutation_generation)

        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise StateTransitionError("verification mutation_generation must be non-negative")

        category = _optional_text(event.data, "category") or _required_text(
            event.data, "kind")
        coverage = _optional_text(event.data, "coverage") or "unknown"
        project_bench = event.data.get("project_bench", False)

        if not isinstance(project_bench, bool):
            raise StateTransitionError("verification project_bench must be boolean")

        record = VerificationRecord(
            verification_id=verification_id,
            kind=_required_text(event.data, "kind"), executed=executed,
            outcome=outcome, action_id=_optional_text(event.data, "action_id"),
            summary=_optional_text(event.data, "summary"), event_id=event.event_id,
            timestamp=event.timestamp, category=category, coverage=coverage,
            mutation_generation=generation,
            result_reference=_optional_text(event.data, "result_reference"),
            command=_optional_text(event.data, "command"),
            project_bench=project_bench,
        )

        self.verifications.append(record)

        # A failure that covered nothing is not evidence of a problem, so only
        # a failure with real coverage opens a failure record.

        if (outcome == VerificationOutcome.FAILED
                and coverage != "none"):
            failure_id = f"verification:{verification_id}"
            self.failures[failure_id] = FailureRecord(
                failure_id, record.action_id, "verification_failed",
                record.summary or f"verification {verification_id} failed",
                False, event.event_id, event.timestamp,
            )

    def to_dict(self) -> dict[str, Any]:
        # Only the event log is persisted; every derived field above is
        # reconstructed by replaying it in from_dict.

        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkingState":
        if not isinstance(raw, Mapping):
            raise SerializationError("working state must be an object")

        version = raw.get("schema_version")

        if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
            raise SerializationError(f"unsupported working state schema: {version!r}")

        task_id = raw.get("task_id")
        events = raw.get("events")

        if not isinstance(task_id, str) or not task_id:
            raise SerializationError("working state task_id must be a non-empty string")

        if not isinstance(events, list):
            raise SerializationError("working state events must be an array")

        state = cls(task_id)

        # Replay goes through apply(), so a stored log is held to exactly the
        # same invariants as a live one.

        try:
            for raw_event in events:
                state.apply(StateEvent.from_dict(raw_event))
        except (WorkingStateError, TypeError, ValueError) as exc:
            if isinstance(exc, SerializationError):
                raise

            raise SerializationError(f"could not restore working state: {exc}") from exc

        return state

    @classmethod
    def from_json(cls, value: str) -> "WorkingState":
        try:
            raw = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SerializationError(f"invalid working state JSON: {exc}") from exc

        return cls.from_dict(raw)


def replay_state(task_id: str, events: Iterable[StateEvent]) -> WorkingState:
    state = WorkingState(task_id)

    for event in events:
        state.apply(event)

    return state
