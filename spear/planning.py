"""Deterministic planning policy and grounded WorkingState plan operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from budgets import BudgetExceeded, BudgetKind, BudgetManager
from model_backend import ConversationMessage, StopReason, TextBlock
from tracing import EventStatus, EventType
from working_state import PlanStepStatus, StateEventType, StateSource


@dataclass(frozen=True)
class PlanningDecision:
    should_plan: bool
    reason: str


class PlanningPolicy:
    _COMPLEX = re.compile(
        r"\b(architecture|architectural|migration|migrate|refactor|rework|"
        r"multi[ -]?file|across .*files|build.*test|test.*fix|dependencies|"
        r"multiple steps|implement .* and .*test)\b", re.IGNORECASE | re.DOTALL,
    )
    _KNOWN_FILE = re.compile(r"(?:^|\s)[\w./-]+\.[A-Za-z0-9]{1,8}(?=\s|$|[:,])")

    def decide(
        self, objective: str, *, acceptance_criteria: Sequence[str] = (),
        exploration_required: bool = False, known_files: Sequence[str] = (),
    ) -> PlanningDecision:
        if exploration_required:
            return PlanningDecision(True, "exploration-dependent task")

        if len(acceptance_criteria) >= 2:
            return PlanningDecision(True, "multiple acceptance criteria")

        if self._COMPLEX.search(objective):
            return PlanningDecision(True, "ordered multi-step task")

        # Naming a file means the work has one known location, which needs no
        # plan; the default below is likewise not to plan.

        if known_files or self._KNOWN_FILE.search(objective):
            return PlanningDecision(False, "simple known-location task")

        return PlanningDecision(False, "no deterministic planning trigger")


@dataclass(frozen=True)
class PlanStepDefinition:
    step_id: str
    objective: str
    dependencies: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    requires_evidence: bool = True


@dataclass(frozen=True)
class PlanCreationResult:
    created: bool
    steps: tuple[PlanStepDefinition, ...]
    error: str | None = None


class PlanningService:
    """Uses the configured provider-neutral backend in a no-tool planning call."""

    def create(self, context, *, maximum_steps: int = 8) -> PlanCreationResult:
        if context.working_state.plan_steps:
            return PlanCreationResult(False, (), "plan already exists")

        # The call is charged before it is made, so an exhausted budget stops
        # planning rather than being noticed once the tokens are spent.

        budget: BudgetManager | None = context.budget_manager

        try:
            if budget is not None:
                budget.consume(BudgetKind.PRIMARY_MODEL_CALLS)
                budget.consume(BudgetKind.MODEL_TURNS)
        except BudgetExceeded as exc:
            context.trace.emit(
                EventType.BUDGET_EXHAUSTED, context.task_id,
                status=EventStatus.EXHAUSTED,
                metadata={"component": "planning", "reason": exc.kind.value,
                          "raw_content_recorded": False},
            )
            return PlanCreationResult(False, (), str(exc))

        context.trace.emit(
            EventType.PLAN_CREATED, context.task_id, status=EventStatus.STARTED,
            metadata={"phase": "planning", "tools_exposed": 0,
                      "raw_content_recorded": False},
        )

        prompt = {
            "objective": context.working_state.objective,
            "constraints": context.working_state.user_constraints,
            "acceptance_criteria": context.working_state.acceptance_criteria,
            "requirements": (
                "Return JSON only: {steps:[{step_id, objective, dependencies, "
                "completion_criteria, requires_evidence}]}. Use 2-8 ordered, "
                "small steps. Planning is read-only; do not claim work is complete."
            ),
        }

        # Planning runs with no tools at all: it may propose work, never do it.

        span = context.trace.start_span(
            EventType.MODEL_CALL_STARTED, EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED, context.task_id,
            provider=context.provider, model=context.model,
            metadata={"purpose": "planning", "use_tools": False,
                      "raw_content_recorded": False},
        )

        try:
            turn = context.backend.complete(
                system="Create a concise implementation plan without mutating anything.",
                conversation=(ConversationMessage("user", (TextBlock(
                    json.dumps(prompt, ensure_ascii=False, sort_keys=True)
                ),)),), tools=(), use_tools=False, on_token=None,
            )
        except Exception as exc:
            span.fail(exc, provider=context.provider, model=context.model)

            return PlanCreationResult(False, (), f"planning model failed: {type(exc).__name__}")

        usage = turn.usage or {}
        input_tokens = _usage(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage(usage, "output_tokens", "completion_tokens")

        try:
            if budget is not None:
                budget.consume(BudgetKind.INPUT_TOKENS,
                               input_tokens or len(json.dumps(prompt)) // 4 + 1,
                               approximate=input_tokens is None)
                budget.consume(BudgetKind.OUTPUT_TOKENS,
                               output_tokens or len(turn.text) // 4 + 1,
                               approximate=output_tokens is None)
        except BudgetExceeded as exc:
            span.fail(str(exc), error_category="budget_exhausted")

            return PlanCreationResult(False, (), str(exc))

        # The span is closed before the outcome is acted on, so a failed
        # planning call is still a complete, well-formed trace.

        if turn.stop_reason == StopReason.ERROR:
            span.fail(turn.error or "planning model failed", error_category="model_error")
        else:
            span.finish(
                provider=context.provider, model=context.model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                metadata={"purpose": "planning", "use_tools": False},
            )

        if turn.stop_reason == StopReason.ERROR:
            return PlanCreationResult(False, (), turn.error or "planning model failed")

        # A plan is applied only if it parses AND validates whole: a partially
        # applied plan would leave the working state describing nothing.

        try:
            raw = json.loads(turn.text)
            rows = raw["steps"]

            if not isinstance(rows, list) or not 1 < len(rows) <= maximum_steps:
                raise ValueError("plan must contain 2-8 steps")

            steps = tuple(PlanStepDefinition(
                str(row["step_id"]), str(row["objective"]),
                tuple(str(value) for value in row.get("dependencies", ())),
                tuple(str(value) for value in row.get("completion_criteria", ())),
                bool(row.get("requires_evidence", True)),
            ) for row in rows)

            self.apply(context, steps)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return PlanCreationResult(False, (), f"malformed plan: {exc}")

        context.trace.emit(
            EventType.PLAN_CREATED, context.task_id, status=EventStatus.OK,
            metadata={"step_count": len(steps), "raw_content_recorded": False},
        )

        from agent_runtime import AgentRuntime
        from session_store import SessionEventType

        AgentRuntime.persist_session_event(
            context, SessionEventType.PLAN_RECORDED,
            {"step_count": len(steps), "planning_mode": "read_only"},
        )

        return PlanCreationResult(True, steps)

    @staticmethod
    def apply(context, steps: Sequence[PlanStepDefinition]) -> None:
        """Validate the whole plan, then record it; never half of it."""

        # `known` grows as the plan is walked, so a dependency is only accepted
        # if it refers to a step already defined -- which also rules out cycles.

        known: set[str] = set()
        existing = set(context.working_state.plan_steps)

        for step in steps:
            if (not step.step_id.strip() or not step.objective.strip()
                    or step.step_id in known or step.step_id in existing):
                raise ValueError("plan step identifiers must be unique")

            if not set(step.dependencies) <= known:
                raise ValueError("plan dependencies must refer to earlier steps")

            if any(not value.strip() for value in step.completion_criteria):
                raise ValueError("plan completion criteria cannot be empty")

            known.add(step.step_id)

        # Only now, with the whole plan proven consistent, is anything applied.

        for step in steps:
            context.apply_state_event(
                StateEventType.PLAN_STEP_ADDED, source=StateSource.HARNESS,
                step_id=step.step_id, description=step.objective,
                dependencies=step.dependencies,
                completion_criteria=step.completion_criteria,
                requires_evidence=step.requires_evidence,
            )

    @staticmethod
    def start_step(context, step_id: str) -> None:
        context.apply_state_event(
            StateEventType.PLAN_STEP_UPDATED, source=StateSource.HARNESS,
            step_id=step_id, status=PlanStepStatus.ACTIVE.value,
        )

    @staticmethod
    def complete_step(context, step_id: str, evidence: Sequence[str]) -> None:
        context.apply_state_event(
            StateEventType.PLAN_STEP_UPDATED, source=StateSource.HARNESS,
            step_id=step_id, status=PlanStepStatus.COMPLETED.value,
            evidence_action_ids=tuple(evidence),
        )

    @staticmethod
    def revise_step(
        context, step_id: str, objective: str, reason: str, *,
        status: PlanStepStatus = PlanStepStatus.PENDING,
    ) -> None:
        context.apply_state_event(
            StateEventType.PLAN_STEP_REVISED, source=StateSource.HARNESS,
            step_id=step_id, description=objective, reason=reason,
            status=status.value,
        )
        context.trace.emit(
            EventType.PLAN_REVISED, context.task_id, status=EventStatus.OK,
            metadata={"step_id": step_id, "status": status.value,
                      "raw_content_recorded": False},
        )

    @staticmethod
    def start_next(context) -> str | None:
        """Activate the first pending step whose dependencies are all closed."""

        for step in context.working_state.plan_steps.values():
            if step.status != PlanStepStatus.PENDING:
                continue

            # A skipped dependency counts as closed: the step was deliberately
            # not done, which is a decision, not an obstacle.

            if all(context.working_state.plan_steps[item].status in {
                       PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED,
                   }
                   for item in step.dependencies):
                PlanningService.start_step(context, step.step_id)

                context.trace.emit(
                    EventType.PLAN_STEP_STARTED, context.task_id,
                    status=EventStatus.STARTED,
                    metadata={"step_id": step.step_id, "raw_content_recorded": False},
                )

                return step.step_id

        return None

    @staticmethod
    def advance(
        context, evidence_kind: str, evidence: Sequence[str],
    ) -> bool:
        """Close the active step if this evidence is the kind that step called for."""

        step = context.working_state.current_step

        if step is None:
            return False

        # The step's own wording decides what would complete it, so evidence of
        # the wrong kind advances nothing.

        patterns = {
            "exploration": r"locat|explor|inspect|discover|analy[sz]|understand",
            "mutation": r"implement|edit|modify|change|fix|refactor|migrat|write",
            "verification": r"test|verify|build|lint|check",
            "review": r"review|accept|final",
        }

        if not re.search(patterns[evidence_kind], step.description, re.IGNORECASE):
            return False

        PlanningService.complete_step(context, step.step_id, evidence)

        context.trace.emit(
            EventType.PLAN_STEP_COMPLETED, context.task_id, status=EventStatus.OK,
            metadata={"step_id": step.step_id, "evidence_kind": evidence_kind,
                      "evidence_count": len(evidence), "raw_content_recorded": False},
        )

        PlanningService.start_next(context)

        return True


def _usage(usage: Mapping[str, int], *names: str) -> int | None:
    """The first of ``names`` the provider reported, since each spells it its own way."""

    for name in names:
        value = usage.get(name)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None
