"""Transactional, provider-neutral semantic context compaction.

WorkingState remains authoritative.  This module derives a validated compact
projection from it and permits a summarizer to add only non-authoritative
conversation continuity.  Raw context is replaceable only after validation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol

from context_engine import (
    ContextEngine, ContextItem, ContextLayer, ContextRequest, Freshness,
)
from tracing import EventStatus, EventType, TraceEmitter, safe_value
from working_state import WorkingState


COMPACTION_SCHEMA_VERSION = 1


class CompactionError(ValueError):
    pass


class CompactionMode(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass(frozen=True)
class CompactPlanStep:
    step_id: str
    description: str
    status: str
    evidence_action_ids: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()
    requires_evidence: bool = False


@dataclass(frozen=True)
class CompactAction:
    action_id: str
    kind: str
    name: str
    status: str
    summary: str | None
    exit_code: int | None


@dataclass(frozen=True)
class CompactFailure:
    failure_id: str
    action_id: str | None
    category: str
    summary: str


@dataclass(frozen=True)
class CompactVerification:
    verification_id: str
    kind: str
    executed: bool
    outcome: str
    action_id: str | None
    summary: str | None
    category: str = "unknown"
    coverage: str = "unknown"
    mutation_generation: int = 0
    project_bench: bool = False


@dataclass(frozen=True)
class EvidenceReference:
    reference_id: str
    kind: str
    available: bool


@dataclass(frozen=True)
class StructuredCompactionState:
    """Model-context projection; never a second mutable task state."""

    schema_version: int
    task_id: str
    objective: str
    user_constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    plan: tuple[CompactPlanStep, ...]
    discoveries: tuple[tuple[str, str | None, str | None, str | None], ...]
    decisions: tuple[tuple[str, str | None, str | None], ...]
    important_actions: tuple[CompactAction, ...]
    unresolved_failures: tuple[CompactFailure, ...]
    verifications: tuple[CompactVerification, ...]
    files_read: tuple[str, ...]
    files_modified: tuple[str, ...]
    files_created: tuple[str, ...]
    files_deleted: tuple[str, ...]
    open_work: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]
    terminal_status: str

    @classmethod
    def from_working_state(cls, state: WorkingState) -> "StructuredCompactionState":
        if not isinstance(state, WorkingState):
            raise CompactionError("structured compaction requires WorkingState")

        # Everything that did not succeed is kept whatever its age, plus a
        # recent tail: an old failure still shapes the task, an old success
        # rarely does.

        important = [action for action in state.actions if action.status.value != "succeeded"]
        important_ids = {action.action_id for action in important}

        for action in state.actions[-12:]:
            if action.action_id not in important_ids:
                important.append(action)
                important_ids.add(action.action_id)

        # Every action id something still points at. These survive compaction
        # even when the action itself is dropped, so no citation is left dangling.

        action_refs = {action.action_id for action in state.actions}
        evidence_ids = set()

        for step in state.plan_steps.values():
            evidence_ids.update(step.evidence_action_ids)

        evidence_ids.update(
            item.action_id for item in state.discoveries if item.action_id
        )
        evidence_ids.update(
            item.action_id for item in state.decisions if item.action_id
        )
        evidence_ids.update(
            item.action_id for item in state.verifications if item.action_id
        )
        evidence_ids.update(
            item.action_id for item in state.unresolved_failures if item.action_id
        )

        # What the task still owes, which is what a compacted context most
        # needs to carry forward.

        open_work = [
            f"plan:{step.step_id}:{step.status.value}"
            for step in state.plan_steps.values()
            if step.status.value in {"pending", "active", "blocked"}
        ]

        open_work.extend(
            f"failure:{failure.failure_id}" for failure in state.unresolved_failures
        )

        return cls(
            COMPACTION_SCHEMA_VERSION,
            state.task_id,
            state.objective,
            tuple(state.user_constraints),
            tuple(state.acceptance_criteria),
            tuple(CompactPlanStep(
                step.step_id, step.description, step.status.value,
                tuple(step.evidence_action_ids),
                tuple(step.dependencies), tuple(step.completion_criteria),
                step.requires_evidence,
            ) for step in state.plan_steps.values()),
            tuple((item.summary, item.file_path, item.symbol, item.action_id)
                  for item in state.discoveries),
            tuple((item.summary, item.rationale, item.action_id)
                  for item in state.decisions),
            tuple(CompactAction(
                item.action_id, item.kind.value, item.name, item.status.value,
                item.summary, item.exit_code,
            ) for item in important),
            tuple(CompactFailure(
                item.failure_id, item.action_id, item.category, item.summary,
            ) for item in state.unresolved_failures),
            tuple(CompactVerification(
                item.verification_id, item.kind, item.executed,
                item.outcome.value, item.action_id, item.summary,
                item.category, item.coverage, item.mutation_generation,
                item.project_bench,
            ) for item in state.verifications),
            tuple(sorted(state.files_read)),
            tuple(sorted(state.modified_files)),
            tuple(sorted(state.created_files)),
            tuple(sorted(state.deleted_files)),
            tuple(open_work),
            tuple(EvidenceReference(
                action_id, "working_state_action", action_id in action_refs,
            ) for action_id in sorted(evidence_ids)),
            state.terminal_status.value,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)

        # These two are stored as positional tuples but shown to the model as
        # named fields, so it can read them without knowing the order.

        value["discoveries"] = [
            {"summary": item[0], "file_path": item[1], "symbol": item[2],
             "action_id": item[3]} for item in self.discoveries
        ]
        value["decisions"] = [
            {"summary": item[0], "rationale": item[1], "action_id": item[2]}
            for item in self.decisions
        ]

        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StructuredCompactionState":
        """Read back a projection a model claimed; every field is re-typed, not trusted."""

        if not isinstance(raw, Mapping):
            raise CompactionError("structured_state must be an object")

        if raw.get("schema_version") != COMPACTION_SCHEMA_VERSION:
            raise CompactionError("unsupported structured compaction schema")

        try:
            return cls(
                COMPACTION_SCHEMA_VERSION,
                _text(raw, "task_id"),
                _text(raw, "objective", allow_empty=True),
                _text_tuple(raw, "user_constraints"),
                _text_tuple(raw, "acceptance_criteria"),
                tuple(CompactPlanStep(
                    _text(item, "step_id"), _text(item, "description"),
                    _text(item, "status"), _text_tuple(item, "evidence_action_ids"),
                    _text_tuple(item, "dependencies"),
                    _text_tuple(item, "completion_criteria"),
                    bool(item.get("requires_evidence", False)),
                ) for item in _objects(raw, "plan")),
                tuple((_text(item, "summary"), _optional_text(item, "file_path"),
                       _optional_text(item, "symbol"), _optional_text(item, "action_id"))
                      for item in _objects(raw, "discoveries")),
                tuple((_text(item, "summary"), _optional_text(item, "rationale"),
                       _optional_text(item, "action_id"))
                      for item in _objects(raw, "decisions")),
                tuple(CompactAction(
                    _text(item, "action_id"), _text(item, "kind"),
                    _text(item, "name"), _text(item, "status"),
                    _optional_text(item, "summary"), _optional_int(item, "exit_code"),
                ) for item in _objects(raw, "important_actions")),
                tuple(CompactFailure(
                    _text(item, "failure_id"), _optional_text(item, "action_id"),
                    _text(item, "category"), _text(item, "summary"),
                ) for item in _objects(raw, "unresolved_failures")),
                tuple(CompactVerification(
                    _text(item, "verification_id"), _text(item, "kind"),
                    _bool(item, "executed"), _text(item, "outcome"),
                    _optional_text(item, "action_id"), _optional_text(item, "summary"),
                    str(item.get("category", "unknown")),
                    str(item.get("coverage", "unknown")),
                    int(item.get("mutation_generation", 0)),
                    bool(item.get("project_bench", False)),
                ) for item in _objects(raw, "verifications")),
                _text_tuple(raw, "files_read"),
                _text_tuple(raw, "files_modified"),
                _text_tuple(raw, "files_created"),
                _text_tuple(raw, "files_deleted"),
                _text_tuple(raw, "open_work"),
                tuple(EvidenceReference(
                    _text(item, "reference_id"), _text(item, "kind"),
                    _bool(item, "available"),
                ) for item in _objects(raw, "evidence")),
                _text(raw, "terminal_status"),
            )
        except (KeyError, TypeError) as exc:
            raise CompactionError(f"malformed structured_state: {exc}") from exc


def _text(raw: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = raw.get(key)

    if not isinstance(value, str) or (not allow_empty and not value):
        raise CompactionError(f"{key} must be text")

    return value


def _optional_text(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)

    if value is not None and not isinstance(value, str):
        raise CompactionError(f"{key} must be text or null")

    return value


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    value = raw.get(key)

    # bool is an int subclass, so it has to be excluded explicitly.

    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise CompactionError(f"{key} must be an integer or null")

    return value


def _bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)

    if not isinstance(value, bool):
        raise CompactionError(f"{key} must be boolean")

    return value


def _objects(raw: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = raw.get(key)

    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise CompactionError(f"{key} must be an array of objects")

    return tuple(value)


def _text_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)

    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise CompactionError(f"{key} must be an array of text")

    return tuple(value)


@dataclass(frozen=True)
class ConversationSummary:
    generation: int
    content: str
    token_estimate: int
    summary_derived: bool = True


@dataclass(frozen=True)
class CompactionArtifact:
    structured_state: StructuredCompactionState
    conversation_summary: ConversationSummary
    compacted_item_ids: frozenset[str]
    messages_compacted: int
    tool_evidence_compacted: int
    tokens_before: int
    tokens_after: int
    trigger_reason: str
    retries: int

    def summary_context_item(self) -> ContextItem:
        return ContextItem(
            item_id=f"compaction-summary:{self.structured_state.task_id}",
            layer=ContextLayer.CONVERSATION_SUMMARY,
            source="semantic_compaction",
            content="\n\n## Conversation continuity\n\n" + self.conversation_summary.content,
            priority=80,
            freshness=Freshness.RECENT,
            protected=True,
            token_estimate=self.conversation_summary.token_estimate,
            inclusion_reason="validated compacted conversational continuity",
        )


@dataclass(frozen=True)
class CompactionResult:
    attempted: bool
    succeeded: bool
    artifact: CompactionArtifact | None = None
    error_category: str | None = None
    error_summary: str | None = None
    retries: int = 0


@dataclass(frozen=True)
class CompactionPolicy:
    mode: CompactionMode = CompactionMode.AUTOMATIC
    pressure_threshold: float = 0.80
    recent_tail_groups: int = 3
    minimum_compactable_tokens: int = 128

    # The summary's bound, in CHARACTERS.
    #
    # It used to be tokens, and the ask and the check then spoke different
    # units: the model was told "under 512 approximate tokens" and complied
    # in real ones -- roughly four characters each, so about 2000 characters
    # -- while the check measured with the harness's estimator at three
    # characters per token, called that 680, and rejected it. Every summary
    # written to the letter of the instruction failed the test of it, on
    # every attempt, for the whole turn.
    #
    # Characters are what the model can be asked for and what can be counted
    # exactly, with no estimator in between. 2048 is the old 512-token
    # intent at the ordinary prose ratio.
    max_summary_chars: int = 2048

    # Two, because the first retry is the first one that asks for anything
    # different: the attempt before it overran, and the ask is lowered by the
    # ratio it overran by. One retry left a summarizer that writes long
    # failing every compaction of a turn -- eighteen of them in a row.
    max_retries: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.pressure_threshold <= 1:
            raise CompactionError("pressure threshold must be in (0, 1]")

        for name in ("recent_tail_groups", "minimum_compactable_tokens",
                     "max_summary_chars", "max_retries"):
            value = getattr(self, name)

            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CompactionError(f"{name} must be non-negative")

        if self.max_summary_chars < 1:
            raise CompactionError("max_summary_chars must be positive")

    def triggered(self, raw_tokens: int, available_tokens: int, eligible_tokens: int) -> bool:
        """Whether to compact now: on request in MANUAL, under pressure otherwise."""

        if self.mode == CompactionMode.MANUAL:
            return eligible_tokens > 0

        # No budget at all means the pressure ratio is undefined, so the size
        # floor alone decides.

        if available_tokens <= 0:
            return eligible_tokens >= self.minimum_compactable_tokens

        return (raw_tokens / available_tokens >= self.pressure_threshold
                and eligible_tokens >= self.minimum_compactable_tokens)


@dataclass(frozen=True)
class CompactionRequest:
    working_state: WorkingState
    context_request: ContextRequest
    existing_artifact: CompactionArtifact | None = None
    trigger_reason: str = "context_pressure"


class ConversationSummarizer(Protocol):
    def summarize(
        self, *, structured_state: StructuredCompactionState,
        existing_summary: str | None, source_text: str,
        max_summary_chars: int,
    ) -> str: ...


class ExtractiveSummarizer:
    """Bounded local fallback requiring no model or external dependency."""

    def summarize(self, *, structured_state, existing_summary, source_text,
                  max_summary_chars):

        # The tail is kept rather than the head: the most recent exchanges are
        # the ones continuity depends on.

        budget = max_summary_chars
        parts = [part for part in (existing_summary, source_text) if part]
        text = "\n".join(parts)

        if len(text) > budget:
            text = text[-budget:]

        return json.dumps({"narrative_summary": text}, ensure_ascii=False)


class ModelBackendSummarizer:
    """Provider-neutral adapter over the existing canonical ModelBackend API."""

    def __init__(
        self, backend: Any, *, trace: TraceEmitter | None = None,
        provider: str | None = None, model: str | None = None,
        budget_manager=None,
    ):
        self.backend = backend
        self.trace = trace
        self.provider = provider or type(backend).__name__
        self.model = model or getattr(backend, "model", None)
        self.budget_manager = budget_manager

    def summarize(self, *, structured_state, existing_summary, source_text,
                  max_summary_chars):
        from model_backend import ConversationMessage, StopReason, TextBlock

        # The authoritative facts go in the prompt so the summary can be
        # checked against them afterwards; the model may narrate, not amend.

        facts = json.dumps(
            safe_value(structured_state.to_dict()), ensure_ascii=False, sort_keys=True,
        )

        prompt = (
            "Return JSON only: {\"narrative_summary\": \"...\"}. Summarize "
            "conversation continuity without changing authoritative facts. Do not "
            "claim files changed or verification passed unless the supplied facts say so. "
            f"Keep it under {max_summary_chars} characters.\n"
            f"AUTHORITATIVE_FACTS:\n{facts}\n"
            f"PREVIOUS_SUMMARY:\n{existing_summary or ''}\n"
            f"OLDER_CONTEXT:\n{source_text}"
        )

        span = (self.trace.start_span(
            EventType.MODEL_CALL_STARTED,
            EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED,
            structured_state.task_id,
            provider=self.provider,
            model=self.model,
            metadata={"purpose": "context_compaction", "use_tools": False},
        ) if self.trace is not None else None)

        try:
            # Compaction is auxiliary work, budgeted apart from the agent's own
            # model calls so it cannot consume the task's turns.

            if self.budget_manager is not None:
                from budgets import BudgetKind

                self.budget_manager.consume(BudgetKind.AUXILIARY_MODEL_CALLS)

            turn = self.backend.complete(
                system=("You compact agent context. Never emit hidden reasoning "
                        "or chain-of-thought."),
                conversation=(ConversationMessage("user", (TextBlock(prompt),)),),
                tools=(), use_tools=False, on_token=None,
            )
        except Exception as exc:
            if span is not None:
                span.fail(exc, provider=self.provider, model=self.model)

            raise

        usage = turn.usage or {}

        if self.budget_manager is not None:
            from budgets import BudgetKind

            input_tokens = _usage_token(usage, "input_tokens", "prompt_tokens")
            output_tokens = _usage_token(usage, "output_tokens", "completion_tokens")
            self.budget_manager.consume(
                BudgetKind.INPUT_TOKENS,
                input_tokens if input_tokens is not None else len(prompt) // 4 + 1,
                approximate=input_tokens is None,
            )
            self.budget_manager.consume(
                BudgetKind.OUTPUT_TOKENS,
                output_tokens if output_tokens is not None else len(turn.text) // 4 + 1,
                approximate=output_tokens is None,
            )

        fields = {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": _usage_token(usage, "input_tokens", "prompt_tokens"),
            "output_tokens": _usage_token(usage, "output_tokens", "completion_tokens"),
            "metadata": {
                "purpose": "context_compaction",
                "stop_reason": str(turn.stop_reason),
                "response_chars": len(turn.text),
            },
        }

        if turn.stop_reason == StopReason.ERROR:
            if span is not None:
                span.fail(turn.error or "compaction model failed",
                          error_category="model_error", **fields)

            raise CompactionError(turn.error or "compaction model failed")

        if span is not None:
            span.finish(**fields)

        return turn.text


def _usage_token(usage: Mapping[str, int], *names: str) -> int | None:
    """The first of ``names`` the provider reported, since each spells it its own way."""

    for name in names:
        value = usage.get(name)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None


@dataclass(frozen=True)
class CompactionValidationResult:
    valid: bool
    errors: tuple[str, ...]


def validate_compaction(
    state: WorkingState, candidate: StructuredCompactionState,
    narrative: str, expected_projection: StructuredCompactionState | None = None,
) -> CompactionValidationResult:
    expected = expected_projection or StructuredCompactionState.from_working_state(state)
    errors = []

    # Every authoritative field must come back byte-identical. Evidence is the
    # sole exception: compaction legitimately adds references of its own.

    for field in expected.__dataclass_fields__:
        if field != "evidence" and getattr(candidate, field) != getattr(expected, field):
            errors.append(f"authoritative field changed: {field}")

    expected_evidence = set(expected.evidence)
    candidate_evidence = set(candidate.evidence)

    if not expected_evidence.issubset(candidate_evidence):
        errors.append("evidence references disappeared")

    # An added reference is only legitimate if it points at compacted context
    # and does not claim to be available; anything else was invented.

    action_ids = {action.action_id for action in state.actions}

    for reference in candidate.evidence:
        if reference.available and reference.reference_id not in action_ids:
            errors.append(f"invalid evidence reference: {reference.reference_id}")

        if (reference not in expected_evidence and
                (reference.kind != "compacted_context" or reference.available)):
            errors.append(f"invented evidence reference: {reference.reference_id}")

    # The narrative is prose the model wrote, so the two claims that matter
    # most -- that something passed -- are checked against the working state.

    lowered = narrative.casefold()

    passed_claim = re.search(r"\b(?:tests?|verification|bench)\b.{0,32}\bpassed\b", lowered)
    any_passed = any(
        item.executed and item.outcome.value == "passed"
        and item.mutation_generation == state.mutation_generation
        and item.coverage != "none"
        for item in state.verifications
    )

    if passed_claim and not any_passed:
        errors.append("narrative invents passed verification")

    if (state.verifications and state.verifications[-1].outcome.value == "failed"
            and re.search(r"\b(?:all tests|verification|bench)\b.{0,24}\bpassed\b", lowered)):
        errors.append("narrative contradicts failed verification")

    return CompactionValidationResult(not errors, tuple(errors))


class CompactionService:
    def __init__(
        self, *, engine: ContextEngine | None = None,
        summarizer: ConversationSummarizer | None = None,
        policy: CompactionPolicy | None = None,
        trace: TraceEmitter | None = None,
    ):
        self.engine = engine or ContextEngine()
        self.summarizer = summarizer or ExtractiveSummarizer()
        self.policy = policy or CompactionPolicy()
        self.trace = trace

    def _trigger_inputs(self, request: CompactionRequest):
        """(prior_ids, candidates, eligible, raw, available): the trigger's inputs."""

        # Already-compacted items are never reconsidered, so successive passes
        # summarize new context rather than re-summarizing their own summary.

        prior_ids = (request.existing_artifact.compacted_item_ids
                     if request.existing_artifact else frozenset())

        candidates = self._eligible_items(request.context_request.items, prior_ids)
        eligible_tokens = sum(self._item_tokens(item) for item in candidates)
        raw_tokens = self.engine.estimate_request_tokens(request.context_request)

        available = max(0, request.context_request.context_limit
                        - request.context_request.output_reserve
                        - request.context_request.safety_margin)

        return prior_ids, candidates, eligible_tokens, raw_tokens, available

    def will_compact(self, request: CompactionRequest) -> bool:
        """Whether :meth:`compact` would do anything at all.

        The decision itself, not a second opinion about it: `compact` asks
        the same question of the same numbers, and answers `(False, False)`
        when it comes back no. A caller that needs to know BEFORE calling --
        to spend a spinner, say -- asks here rather than guessing, so the
        two can never disagree.
        """

        if not isinstance(request.working_state, WorkingState):
            return False

        *_, eligible_tokens, raw_tokens, available = self._trigger_inputs(request)

        return self.policy.triggered(raw_tokens, available, eligible_tokens)

    def compact(self, request: CompactionRequest) -> CompactionResult:
        state = request.working_state

        if not isinstance(state, WorkingState):
            raise CompactionError("compaction request requires WorkingState")

        (prior_ids, candidates, eligible_tokens, raw_tokens,
         available) = self._trigger_inputs(request)

        if not self.policy.triggered(raw_tokens, available, eligible_tokens):
            return CompactionResult(False, False)

        started_ns = time.monotonic_ns()
        self._emit(EventType.COMPACTION_STARTED, state.task_id,
                   status=EventStatus.STARTED, metadata={
                       "trigger_reason": request.trigger_reason,
                       "context_tokens_before": raw_tokens,
                       "eligible_item_count": len(candidates),
                       "raw_content_recorded": False,
                   })
        structured = StructuredCompactionState.from_working_state(state)

        # What is about to be compacted away is itself recorded as evidence, so
        # the projection names the context it replaced, this pass and previous ones.

        prior_context_evidence = (
            tuple(reference for reference in request.existing_artifact.structured_state.evidence
                  if reference.kind == "compacted_context")
            if request.existing_artifact else ()
        )
        compacted_context_evidence = tuple(EvidenceReference(
            item.item_id, "compacted_context", False,
        ) for item in candidates)

        structured = StructuredCompactionState(
            **{**structured.__dict__, "evidence": tuple(dict.fromkeys(
                structured.evidence + prior_context_evidence + compacted_context_evidence
            ))}
        )

        source_text = self._source_text(candidates)
        existing_summary = (request.existing_artifact.conversation_summary.content
                            if request.existing_artifact else None)

        # Nothing is committed until an attempt validates: a summarizer that
        # invents or drops a fact costs a retry, never the context.

        last_error = None

        # What the summarizer is ASKED for, which a retry lowers. The bound it
        # is checked against never moves. Re-asking with the identical prompt
        # is not a retry: a summarizer that writes long wrote long again, and
        # a trace of one writing turn held twenty-four consecutive rejections
        # for "narrative summary exceeds configured bound" -- compaction never
        # succeeded once, so context pressure was never relieved and every
        # later round paid for it.

        asked_chars = self.policy.max_summary_chars

        for attempt in range(self.policy.max_retries + 1):
            try:
                raw = self.summarizer.summarize(
                    structured_state=structured,
                    existing_summary=existing_summary,
                    source_text=source_text,
                    max_summary_chars=asked_chars,
                )
                narrative, claimed = self._parse(raw)

                # A summary that overruns its own bound has not compacted
                # anything, so it is rejected before it is validated.

                # Counted, not estimated. The estimator still sizes the
                # summary's cost in the context below, which is a different
                # question from whether the model obeyed its instruction.
                written = len(narrative)

                if written > self.policy.max_summary_chars:
                    # Ask the next attempt for as much less as this one
                    # overran by, so the retry has a reason to come out
                    # shorter than the attempt it is replacing.
                    asked_chars = max(
                        200, asked_chars * self.policy.max_summary_chars
                        // max(written, 1))

                    raise CompactionError("narrative summary exceeds configured bound")

                # The model may restate the projection; if it does, its version
                # is what gets validated against the working state.

                candidate = (StructuredCompactionState.from_dict(claimed)
                             if claimed is not None else structured)

                validation = validate_compaction(
                    state, candidate, narrative, expected_projection=structured,
                )

                if not validation.valid:
                    raise CompactionError("; ".join(validation.errors))

                compacted_ids = prior_ids | frozenset(item.item_id for item in candidates)
                summary_tokens = self.engine.estimator.estimate(
                    "\n\n## Conversation continuity\n\n" + narrative
                )

                generation = (request.existing_artifact.conversation_summary.generation + 1
                              if request.existing_artifact else 1)
                remaining = tuple(item for item in request.context_request.items
                                  if item.item_id not in compacted_ids)

                artifact = CompactionArtifact(
                    candidate,
                    ConversationSummary(generation, narrative, summary_tokens),
                    compacted_ids,
                    sum(item.layer == ContextLayer.RECENT_CONVERSATION
                        for item in candidates),
                    sum(item.layer == ContextLayer.TOOL_EVIDENCE
                        for item in candidates),
                    raw_tokens,
                    self.engine.estimate_request_tokens(ContextRequest(
                        items=remaining,
                        context_limit=request.context_request.context_limit,
                        output_reserve=request.context_request.output_reserve,
                        safety_margin=request.context_request.safety_margin,
                        fixed_input_tokens=request.context_request.fixed_input_tokens,
                    )) + summary_tokens,
                    request.trigger_reason,
                    attempt,
                )

                # Last gate: a compaction that still does not fit has not
                # solved the problem it was run for.

                if self.engine.compose(
                    self.apply_artifact(request.context_request, artifact)
                ).overflow_tokens:
                    raise CompactionError("compacted context cannot fit effective input budget")

                self._emit(EventType.COMPACTION_FINISHED, state.task_id,
                           status=EventStatus.OK,
                           duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                           metadata={
                               "trigger_reason": request.trigger_reason,
                               "context_tokens_before": artifact.tokens_before,
                               "context_tokens_after": artifact.tokens_after,
                               "messages_compacted": artifact.messages_compacted,
                               "tool_evidence_compacted": artifact.tool_evidence_compacted,
                               "summary_token_estimate": summary_tokens,
                               "validation_result": "passed",
                               "retries": attempt,
                               "raw_content_recorded": False,
                           })

                return CompactionResult(True, True, artifact, retries=attempt)
            except Exception as exc:
                last_error = exc

        # Every attempt failed. The context is returned untouched: a failed
        # compaction costs tokens, never fidelity.

        category = type(last_error).__name__ if last_error else "CompactionError"
        summary = str(last_error or "compaction failed")[:240]

        self._emit(EventType.COMPACTION_FAILED, state.task_id,
                   status=EventStatus.FAILED,
                   duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                   error_category=category, error_summary=summary,
                   metadata={
                       "trigger_reason": request.trigger_reason,
                       "context_tokens_before": raw_tokens,
                       "validation_result": "failed",
                       "retries": self.policy.max_retries,
                       "raw_content_recorded": False,
                   })

        return CompactionResult(
            True, False, error_category=category, error_summary=summary,
            retries=self.policy.max_retries,
        )

    def apply_artifact(
        self, context_request: ContextRequest, artifact: CompactionArtifact,
    ) -> ContextRequest:
        """Replace compacted raw items; caller commits the artifact transaction."""

        # The previous summary is dropped along with the compacted items: the
        # new artifact's summary already subsumes it.

        items = tuple(item for item in context_request.items
                      if item.item_id not in artifact.compacted_item_ids
                      and item.layer != ContextLayer.CONVERSATION_SUMMARY)

        return ContextRequest(
            items=items + (artifact.summary_context_item(),),
            context_limit=context_request.context_limit,
            output_reserve=context_request.output_reserve,
            safety_margin=context_request.safety_margin,
            fixed_input_tokens=context_request.fixed_input_tokens,
        )

    def _eligible_items(
        self, items: tuple[ContextItem, ...], prior_ids: frozenset[str],
    ) -> tuple[ContextItem, ...]:
        # Grouped in encounter order: a tool call and its results form one
        # group, and compacting half of one would break the exchange.

        groups: list[tuple[str, list[ContextItem]]] = []
        by_group: dict[str, list[ContextItem]] = {}

        for index, item in enumerate(items):
            if (item.item_id in prior_ids or item.layer not in {
                    ContextLayer.RECENT_CONVERSATION, ContextLayer.TOOL_EVIDENCE}):
                continue

            key = item.eviction_group or f"item:{index}"

            if key not in by_group:
                by_group[key] = []
                groups.append((key, by_group[key]))

            by_group[key].append(item)

        # The most recent groups are left verbatim, and a group holding
        # anything protected is kept whole.

        cutoff = max(0, len(groups) - self.policy.recent_tail_groups)
        result = []

        for _, members in groups[:cutoff]:
            if not any(item.protected for item in members):
                result.extend(members)

        return tuple(result)

    def _item_tokens(self, item: ContextItem) -> int:
        return (item.token_estimate if item.token_estimate is not None
                else self.engine.estimator.estimate(item.content) + item.token_overhead)

    @staticmethod
    def _source_text(items: tuple[ContextItem, ...]) -> str:
        parts = []

        for item in items:
            # Reuse tracing's bounded obvious-secret redaction before model use.

            content = safe_value(item.content)
            parts.append(f"[{item.layer.value}:{item.item_id}]\n{content}")

        return "\n\n".join(parts)

    @staticmethod
    def _parse(raw: str) -> tuple[str, Mapping[str, Any] | None]:
        if not isinstance(raw, str):
            raise CompactionError("summarizer output must be text")

        # Models fence JSON even when told not to, and the fence is the one
        # deviation tolerated here.

        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                          flags=re.IGNORECASE)

        # strict=False accepts the literal newlines models put inside a JSON
        # string. It is the same deviation as the fence above: a summary
        # rejected for an unescaped line break has not saved anyone from
        # anything, and four compactions of one turn died on exactly that.

        try:
            value = json.loads(text, strict=False)
        except json.JSONDecodeError as exc:
            raise CompactionError(f"malformed summarizer JSON: {exc}") from exc

        if not isinstance(value, Mapping):
            raise CompactionError("summarizer output must be a JSON object")

        narrative = value.get("narrative_summary")

        if not isinstance(narrative, str) or not narrative.strip():
            raise CompactionError("narrative_summary must be non-empty text")

        # Restating the projection is optional; claiming it in the wrong shape
        # is not, since the validator has to compare it field by field.

        claimed = value.get("structured_state")

        if claimed is not None and not isinstance(claimed, Mapping):
            raise CompactionError("structured_state claim must be an object")

        return narrative.strip(), claimed

    def _emit(self, event_type: EventType, task_id: str, **fields: Any) -> None:
        if self.trace is not None:
            self.trace.emit(event_type, task_id, **fields)
