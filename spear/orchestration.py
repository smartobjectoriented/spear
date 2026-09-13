"""Constrained synchronous orchestration for trusted Explorer/Reviewer roles.

The generic contract wraps, rather than erases, their typed deliverables. It
does not support arbitrary roles, recursion, parallelism, or child transcripts.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from agent_roles import AgentRole, AgentRoleSpec, explorer_role, reviewer_role
from agent_runtime import AgentContext, AgentResult, AgentRuntime, RuntimeTerminalReason
from cancellation import CancellationSource, linked_cancellation
from compaction import CompactionPolicy
from context_engine import ContextItem, ContextLayer, Freshness
from model_backend import ConversationMessage, ModelBackend, TextBlock, ToolDefinition
from progress_monitor import ProgressMonitor
from tool_exposure import READ_ONLY_RULE_ID
from session_store import SessionEventType, SessionHandle
from tracing import EventStatus, EventType
from working_state import StateEventType, StateSource, TerminalStatus, WorkingState
from budgets import BudgetExceeded, BudgetKind, BudgetLimit, BudgetManager


class DelegationStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class DelegationRequest:
    parent_task_id: str
    role: AgentRole
    objective: str
    scope: tuple[str, ...]
    context_manifest: tuple[str, ...]
    expected_deliverable_type: str
    tool_policy: str = "trusted_role_default"
    execution_mode: str = "safe"
    model_override: str | None = None
    context_budget: int = 32_768
    model_turn_budget: int = 8
    tool_call_budget: int = 16
    wall_time_ms: int | None = None
    cancellation_relationship: str = "parent_to_child"
    recursion_allowed: bool = False
    delegation_id: str = field(default_factory=lambda: "delegate_" + uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.role not in {AgentRole.EXPLORER, AgentRole.REVIEWER}:
            raise ValueError("delegation role is not trusted/supported")

        if self.execution_mode != "safe" or self.recursion_allowed:
            raise ValueError("delegated roles must be safe and non-recursive")

        if min(self.context_budget, self.model_turn_budget, self.tool_call_budget) < 1:
            raise ValueError("delegation budgets must be positive")


@dataclass(frozen=True)
class DelegationUsage:
    model_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class DelegationResult:
    delegation_id: str
    role: AgentRole
    status: DelegationStatus
    deliverable: object | None
    evidence_references: tuple[str, ...]
    usage: DelegationUsage
    terminal_reason: str
    uncertainties: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def persistence_record(self) -> dict[str, object]:
        # Only a reference to the deliverable is persisted, never the
        # deliverable itself: a child's transcript never reaches the parent.

        deliverable = self.deliverable
        reference = None

        if deliverable is not None:
            reference = getattr(deliverable, "exploration_id", None) or getattr(
                deliverable, "review_id", None
            )

        return {
            "delegation_id": self.delegation_id, "role": self.role.value,
            "status": self.status.value, "deliverable_reference": reference,
            "evidence_references": self.evidence_references,
            "usage": asdict(self.usage), "terminal_reason": self.terminal_reason,
            "uncertainty_count": len(self.uncertainties),
            "error_count": len(self.errors),
        }


class TrustedRoleRegistry:
    """Trusted code owns role capabilities; requests name roles only."""

    def __init__(self) -> None:
        self._roles = {
            AgentRole.EXPLORER: explorer_role,
            AgentRole.REVIEWER: reviewer_role,
        }

    def resolve(self, role: AgentRole) -> AgentRoleSpec:
        try:
            return self._roles[role]()
        except KeyError as exc:
            raise ValueError("untrusted delegation role") from exc


DelegationHandler = Callable[[DelegationRequest, BudgetManager], DelegationResult]


class DelegationManager:
    """One synchronous trusted child at a time; never returns a transcript."""

    def __init__(self, roles: TrustedRoleRegistry | None = None) -> None:
        self.roles = roles or TrustedRoleRegistry()
        self._handlers: dict[AgentRole, DelegationHandler] = {}

    def register(self, role: AgentRole, handler: DelegationHandler) -> None:
        # Resolving first refuses an untrusted role before it can be wired to
        # a handler at all.

        self.roles.resolve(role)

        if role in self._handlers:
            raise ValueError("delegation handler already registered")

        self._handlers[role] = handler

    def delegate(self, request: DelegationRequest, parent: AgentContext) -> DelegationResult:
        if request.parent_task_id != parent.task_id:
            raise ValueError("delegation parent task does not match runtime context")

        # The request may only ever narrow what the trusted role already allows.
        # Every check below refuses a request trying to widen it instead.

        role = self.roles.resolve(request.role)

        if role.execution_mode != request.execution_mode or role.recursive_delegation:
            raise ValueError("delegation cannot override trusted role policy")

        if request.tool_policy != "trusted_role_default":
            raise ValueError("delegation cannot override trusted tool policy")

        if request.cancellation_relationship != "parent_to_child":
            raise ValueError("delegation cannot override cancellation relationship")

        if (request.model_turn_budget > role.max_model_turns
                or request.tool_call_budget > role.max_tool_calls
                or request.context_budget > role.context_limit):
            raise ValueError("delegation allocation exceeds trusted role limits")

        if request.model_override and request.model_override != role.model_override:
            raise ValueError("delegation cannot select an unconfigured model")

        try:
            handler = self._handlers[request.role]
        except KeyError as exc:
            raise ValueError("no trusted handler for delegation role") from exc

        # The child's budget is carved out of the parent's, so a delegation can
        # never spend more than the parent still has.

        parent_budget = parent.budget_manager

        if parent_budget is None:
            parent_budget = BudgetManager.from_legacy(
                component="main", model_turns=parent.max_model_rounds,
                tool_calls=parent.max_tool_actions,
            )
            parent.budget_manager = parent_budget

        limits = {
            BudgetKind.MODEL_TURNS: BudgetLimit(request.model_turn_budget),
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(request.model_turn_budget),
            BudgetKind.TOOL_CALLS: BudgetLimit(request.tool_call_budget),
            BudgetKind.INPUT_TOKENS: BudgetLimit(request.context_budget, True),
        }

        if request.wall_time_ms is not None:
            limits[BudgetKind.WALL_TIME_MS] = BudgetLimit(request.wall_time_ms)

        parent.trace.emit(
            EventType.BUDGET_ALLOCATED, parent.task_id,
            session_id=parent.session_id, status=EventStatus.OK,
            metadata={"delegation_id": request.delegation_id,
                      "component": request.role.value,
                      "model_turns": request.model_turn_budget,
                      "tool_calls": request.tool_call_budget,
                      "context_tokens": request.context_budget,
                      "context_tokens_approximate": True,
                      "raw_content_recorded": False},
        )

        # A parent with nothing left to allocate is a completed delegation with
        # a BUDGET_EXHAUSTED status, not an exception the parent has to handle.

        try:
            child_budget = parent_budget.allocate_child(
                request.delegation_id, component=request.role.value, limits=limits,
            )
        except BudgetExceeded as exc:
            result = DelegationResult(
                request.delegation_id, request.role,
                DelegationStatus.BUDGET_EXHAUSTED, None, (), DelegationUsage(),
                exc.kind.value, errors=(str(exc),),
            )

            parent.delegation_results.append(result.persistence_record())

            AgentRuntime.persist_session_event(
                parent, SessionEventType.DELEGATION_COMPLETED,
                result.persistence_record(),
            )
            parent.trace.emit(
                EventType.BUDGET_EXHAUSTED, parent.task_id,
                session_id=parent.session_id, status=EventStatus.EXHAUSTED,
                metadata={"delegation_id": request.delegation_id,
                          "role": request.role.value, "reason": exc.kind.value,
                          "raw_content_recorded": False},
            )
            parent.trace.emit(
                EventType.DELEGATION_FINISHED, parent.task_id,
                session_id=parent.session_id, status=EventStatus.FAILED,
                metadata={"delegation_id": request.delegation_id,
                          "role": request.role.value,
                          "status": result.status.value,
                          "raw_content_recorded": False},
            )

            return result

        parent.trace.emit(
            EventType.DELEGATION_STARTED, parent.task_id,
            session_id=parent.session_id, status=EventStatus.STARTED,
            metadata={"delegation_id": request.delegation_id,
                      "role": request.role.value, "recursion": False,
                      "raw_content_recorded": False},
        )

        # However the child ends, the parent gets a result rather than an
        # exception: a failed delegation is an outcome, not a parent failure.

        try:
            result = handler(request, child_budget)
        except BudgetExceeded as exc:
            result = DelegationResult(
                request.delegation_id, request.role,
                DelegationStatus.BUDGET_EXHAUSTED, None, (),
                DelegationUsage(
                    child_budget.consumed.get(BudgetKind.MODEL_TURNS, 0),
                    child_budget.consumed.get(BudgetKind.TOOL_CALLS, 0),
                    child_budget.consumed.get(BudgetKind.INPUT_TOKENS, 0),
                    child_budget.consumed.get(BudgetKind.OUTPUT_TOKENS, 0),
                ), exc.kind.value, errors=(str(exc),),
            )
        except Exception as exc:
            result = DelegationResult(
                request.delegation_id, request.role, DelegationStatus.FAILED,
                None, (), DelegationUsage(), type(exc).__name__,
                errors=(str(exc)[:240],),
            )

        parent.delegation_results.append(result.persistence_record())

        AgentRuntime.persist_session_event(
            parent, SessionEventType.DELEGATION_COMPLETED,
            result.persistence_record(),
        )
        parent.trace.emit(
            EventType.DELEGATION_FINISHED, parent.task_id,
            session_id=parent.session_id,
            status=(EventStatus.OK if result.status == DelegationStatus.COMPLETED
                    else EventStatus.FAILED),
            metadata={"delegation_id": request.delegation_id,
                      "role": request.role.value, "status": result.status.value,
                      "model_turns": result.usage.model_turns,
                      "tool_calls": result.usage.tool_calls,
                      "raw_content_recorded": False},
        )

        return result


class ExplorationStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"
    BUDGET_EXHAUSTED = "budget_exhausted"


def new_exploration_id() -> str:
    return "explore_" + uuid.uuid4().hex


@dataclass(frozen=True)
class ExplorationRequest:
    parent_task_id: str
    objective: str
    scope: tuple[str, ...] = ()
    known_files: tuple[str, ...] = ()
    project_rules: tuple[str, ...] = ()
    known_facts: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    expected_deliverable: str = "repository architecture and relevant file map"
    exploration_id: str = field(default_factory=new_exploration_id)
    deduplication_key: str = "initial_repository_exploration"

    def __post_init__(self) -> None:
        if not self.parent_task_id or not self.objective.strip():
            raise ValueError("exploration requires parent task and objective")


@dataclass(frozen=True)
class ExplorationReport:
    exploration_id: str
    parent_task_id: str
    child_task_id: str
    child_session_id: str | None
    deduplication_key: str
    status: ExplorationStatus
    summary: str
    key_findings: tuple[str, ...] = ()
    relevant_files: tuple[str, ...] = ()
    entry_points: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    call_chains: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    architecture: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()
    evidence_references: tuple[str, ...] = ()
    files_inspected: tuple[str, ...] = ()
    model_calls: int = 0
    tool_calls: int = 0
    terminal_reason: str = "completed"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["status"] = self.status.value

        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ExplorationReport":
        values = dict(raw)
        values["status"] = ExplorationStatus(str(values["status"]))

        # JSON has no tuples, so every sequence field comes back as a list and
        # is restored to the frozen shape the dataclass declares.

        for key in (
            "key_findings", "relevant_files", "entry_points", "symbols",
            "call_chains", "tests", "architecture", "uncertainties",
            "unanswered_questions", "evidence_references", "files_inspected",
        ):
            values[key] = tuple(str(item) for item in values.get(key, ()))

        return cls(**values)

    def context_item(self) -> ContextItem:
        payload = {
            "status": self.status.value, "summary": self.summary,
            "key_findings": self.key_findings,
            "relevant_files": self.relevant_files,
            "entry_points": self.entry_points, "symbols": self.symbols,
            "call_chains": self.call_chains, "tests": self.tests,
            "architecture": self.architecture,
            "uncertainties": self.uncertainties,
            "unanswered_questions": self.unanswered_questions,
            "evidence_references": self.evidence_references,
            "files_inspected": self.files_inspected,
        }

        # Labelled as evidence rather than as truth: the parent decides what to
        # do with a child's findings, and the payload is capped for the window.

        return ContextItem(
            f"exploration:{self.exploration_id}", ContextLayer.RETRIEVED_CONTEXT,
            f"explorer:{self.exploration_id}",
            "Explorer report (structured evidence, not task truth):\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)[:12_000],
            priority=84, freshness=Freshness.CURRENT, protected=False,
            inclusion_reason="isolated repository exploration report",
        )


@dataclass(frozen=True)
class ExplorationDecision:
    should_explore: bool
    reason: str


class ExplorationPolicy:
    """Conservative deterministic gate; it adds no model-selection call."""

    _REPOSITORY_INTENT = re.compile(
        r"\b(architecture|codebase|repository|call[ -]?chain|entry point|"
        r"across (?:multiple|several) files|where (?:is|are).*(?:implemented|defined))\b",
        re.IGNORECASE | re.DOTALL,
    )
    _KNOWN_FILE = re.compile(r"(?:^|\s)[\w./-]+\.[A-Za-z0-9]{1,8}(?=\s|$|[:,])")

    def decide(
        self, objective: str, *, known_files: Sequence[str] = (),
        prior_reports: Sequence[Mapping[str, object]] = (),
    ) -> ExplorationDecision:
        if prior_reports:
            return ExplorationDecision(False, "completed exploration already available")

        # Naming a file is naming the answer: there is nothing to discover, so
        # a whole child agent would be pure cost.

        if known_files or self._KNOWN_FILE.search(objective):
            return ExplorationDecision(False, "target file already known")

        if self._REPOSITORY_INTENT.search(objective):
            return ExplorationDecision(True, "repository-scale discovery requested")

        return ExplorationDecision(False, "task does not require repository-scale discovery")


ChildSessionFactory = Callable[[AgentContext, WorkingState], SessionHandle]
ModelOverrideResolver = Callable[[str], ModelBackend]


class ExplorationService:
    def __init__(
        self, runtime: AgentRuntime, *, role: AgentRoleSpec | None = None,
        child_session_factory: ChildSessionFactory | None = None,
        model_override_resolver: ModelOverrideResolver | None = None,
    ) -> None:
        self.runtime = runtime
        self.role = role or explorer_role()
        self.child_session_factory = child_session_factory
        self.model_override_resolver = model_override_resolver
        self._active_cancellation: CancellationSource | None = None

    def cancel_active(self, reason: str = "Explorer cancelled") -> bool:
        """Cancel only the current child; never signal the parent token."""
        return (self._active_cancellation.cancel(reason)
                if self._active_cancellation is not None else False)

    def explore(
        self, request: ExplorationRequest, parent: AgentContext, *,
        tools: tuple[ToolDefinition, ...], tool_executor,
        budget_manager: BudgetManager | None = None,
    ) -> ExplorationReport:
        # The same exploration is never run twice for one task: the parent
        # keeps the report, and re-exploring would only spend budget again.

        existing = self._existing_report(parent, request.deduplication_key)

        if existing is not None:
            return existing

        child_state = WorkingState.start(
            "task_" + uuid.uuid4().hex, request.objective,
            max_model_rounds=self.role.max_model_turns,
            max_tool_actions=self.role.max_tool_calls,
        )

        # Linked, not shared: cancelling the parent cancels the child, while
        # cancel_active() stops the child alone.

        local_source, child_token = linked_cancellation(parent.cancellation)
        self._active_cancellation = local_source

        items = self._context_items(request, parent)
        backend = parent.backend

        if self.role.model_override:
            if self.model_override_resolver is None:
                raise ValueError("Explorer model override requires a resolver")

            backend = self.model_override_resolver(self.role.model_override)

        # The child gets no checkpoint: it is read-only by role, so there is
        # nothing of its own to roll back.

        child = AgentContext(
            working_state=child_state, backend=backend,
            context_engine=parent.context_engine, trace=parent.trace,
            system_prompt="".join(item.content for item in items),
            context_items=items,
            conversation=[ConversationMessage("user", (TextBlock(
                self._request_text(request),
            ),))],
            tools=tools, tool_executor=tool_executor,
            max_model_rounds=self.role.max_model_turns,
            max_tool_actions=self.role.max_tool_calls,
            context_limit=min(parent.context_limit, self.role.context_limit),
            output_reserve=parent.output_reserve,
            compaction_policy=CompactionPolicy(),
            provider=parent.provider, model=self.role.model_override or parent.model,
            cancellation=child_token, progress_monitor=ProgressMonitor(),
            role=self.role.role.value,
            budget_manager=budget_manager,
            checkpoint_manager=None, checkpoint=None,
            task_trace_metadata={
                "role": self.role.role.value,
                "parent_task_id": parent.task_id,
                "exploration_id": request.exploration_id,
                "raw_content_recorded": False,
            },
        )

        if self.child_session_factory is not None:
            child.session = self.child_session_factory(parent, child_state)

        started = time.monotonic()

        parent.trace.emit(
            EventType.EXPLORATION_STARTED, parent.task_id,
            session_id=parent.session_id, status=EventStatus.STARTED,
            metadata={"exploration_id": request.exploration_id,
                      "child_task_id": child.task_id,
                      "child_session_id": child.session_id,
                      "role": "explorer", "raw_content_recorded": False},
        )

        # A child that fails outright still yields a report: the parent is
        # given a status it can act on, never the child's exception.

        try:
            result = self.runtime.run(child)
            report = self._report(request, child, result)
        except Exception as exc:
            report = ExplorationReport(
                request.exploration_id, parent.task_id, child.task_id,
                child.session_id, request.deduplication_key,
                ExplorationStatus.CANCELLED if child_token.is_cancelled
                else ExplorationStatus.FAILED,
                "Explorer did not produce a complete report.",
                uncertainties=(str(exc)[:240],), terminal_reason=type(exc).__name__,
            )
        finally:
            self._active_cancellation = None

        # A partial report is still a usable one, so it is traced as finished
        # rather than failed.

        status_event = (EventType.EXPLORATION_FINISHED
                        if report.status in {ExplorationStatus.COMPLETED,
                                             ExplorationStatus.PARTIAL}
                        else EventType.EXPLORATION_FAILED)

        parent.trace.emit(
            status_event, parent.task_id, session_id=parent.session_id,
            status=(EventStatus.OK if status_event == EventType.EXPLORATION_FINISHED
                    else EventStatus.CANCELLED if report.status == ExplorationStatus.CANCELLED
                    else EventStatus.FAILED),
            duration_ms=(time.monotonic() - started) * 1000,
            metadata={"exploration_id": report.exploration_id,
                      "child_task_id": report.child_task_id,
                      "child_session_id": report.child_session_id,
                      "files_inspected": len(report.files_inspected),
                      "model_calls": report.model_calls,
                      "tool_calls": report.tool_calls,
                      "terminal_reason": report.terminal_reason,
                      "report_chars": len(report.summary),
                      "raw_content_recorded": False},
        )

        self.ingest(parent, report)

        return report

    @staticmethod
    def ingest(parent: AgentContext, report: ExplorationReport) -> None:
        """Fold a report into the parent as evidence, exactly once."""

        if any(item.get("deduplication_key") == report.deduplication_key
               for item in parent.exploration_reports):
            return

        parent.exploration_reports.append(report.to_dict())
        parent.context_items = (*parent.context_items, report.context_item())

        # Files the child actually read become grounded discoveries on the
        # parent, capped so a wide exploration cannot flood its working state.

        for path in report.files_inspected[:32]:
            parent.apply_state_event(
                StateEventType.DISCOVERY_RECORDED, source=StateSource.RETRIEVAL,
                summary="Explorer inspected relevant repository evidence",
                file_path=path, action_id=report.exploration_id,
            )

        AgentRuntime.persist_session_event(
            parent, SessionEventType.EXPLORATION_COMPLETED,
            {"exploration_id": report.exploration_id,
             "child_task_id": report.child_task_id,
             "child_session_id": report.child_session_id,
             "deduplication_key": report.deduplication_key,
             "status": report.status.value},
        )

    @staticmethod
    def _existing_report(
        parent: AgentContext, key: str,
    ) -> ExplorationReport | None:
        for raw in parent.exploration_reports:
            if raw.get("deduplication_key") == key:
                return ExplorationReport.from_dict(raw)

        return None

    def _context_items(
        self, request: ExplorationRequest, parent: AgentContext,
    ) -> tuple[ContextItem, ...]:
        role_item = ContextItem(
            f"explorer-role:{request.exploration_id}", ContextLayer.SYSTEM_RULES,
            "agent_role:explorer", self.role.prompt_addition,
            priority=100, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="enforce isolated read-only explorer role",
            truncatable=False,
        )

        # The child inherits the project's rules but none of the parent's
        # conversation: isolation is the point of delegating at all.

        project_items = tuple(item for item in parent.context_items
                              if item.layer == ContextLayer.PROJECT_RULES)

        # And the one task-scoped rule that is not the parent's to keep: a
        # request that forbade changes forbids them here too. The explorer
        # cannot write in any case, but it can spend the turn's budget
        # investigating how to, and it is asked to report, not to repair.

        scope_items = tuple(item for item in parent.context_items
                            if item.item_id == READ_ONLY_RULE_ID)

        explicit = tuple(ContextItem(
            f"explorer-project-rule:{request.exploration_id}:{index}",
            ContextLayer.PROJECT_RULES, "exploration_request", rule,
            priority=95, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="project rule relevant to exploration",
            truncatable=False,
        ) for index, rule in enumerate(request.project_rules))

        facts = (() if not request.known_facts else (ContextItem(
            f"explorer-known-facts:{request.exploration_id}",
            ContextLayer.RETRIEVED_CONTEXT, "parent_selected_facts",
            "\n".join(f"- {fact}" for fact in request.known_facts),
            priority=75, freshness=Freshness.CURRENT,
            inclusion_reason="parent-selected exploration facts",
        ),))

        return (role_item, *scope_items, *project_items, *explicit, *facts)

    @staticmethod
    def _request_text(request: ExplorationRequest) -> str:
        fields = {
            "objective": request.objective, "scope": request.scope,
            "known_files": request.known_files, "exclusions": request.exclusions,
            "questions": request.questions,
            "expected_deliverable": request.expected_deliverable,
        }

        return "Exploration request:\n" + json.dumps(fields, ensure_ascii=False)

    @staticmethod
    def _report(
        request: ExplorationRequest, child: AgentContext, result: AgentResult,
    ) -> ExplorationReport:
        parsed = _parse_report(result.final_response)
        status = _report_status(result)

        # The child's own claims are ranked first but the files it actually
        # read are always present, so a report cannot omit its own evidence.

        grounded_files = tuple(sorted(result.working_state.files_read))
        reported_files = _strings(parsed.get("relevant_files"))
        ranked = tuple(dict.fromkeys((*reported_files, *grounded_files)))

        # Evidence references, by contrast, are intersected: a child may only
        # cite actions and stored results that really happened.

        valid_evidence = ({action.action_id for action in result.working_state.actions}
                          | set(child.result_references))
        reported_evidence = set(_strings(parsed.get("evidence_references")))

        return ExplorationReport(
            request.exploration_id, request.parent_task_id, result.task_id,
            child.session_id, request.deduplication_key, status,
            str(parsed.get("summary") or result.final_response or
                "Explorer returned no narrative summary.")[:4000],
            _strings(parsed.get("key_findings")), ranked,
            _strings(parsed.get("entry_points")), _strings(parsed.get("symbols")),
            _strings(parsed.get("call_chains")), _strings(parsed.get("tests")),
            _strings(parsed.get("architecture")), _strings(parsed.get("uncertainties")),
            _strings(parsed.get("unanswered_questions")),
            tuple(sorted((reported_evidence & valid_evidence)
                         | set(child.result_references))),
            grounded_files, result.model_calls, result.tool_actions,
            result.terminal_reason.value,
        )


def _strings(value: object) -> tuple[str, ...]:
    """A model-supplied list, coerced to bounded strings; anything else is empty."""

    if not isinstance(value, (list, tuple)):
        return ()

    return tuple(str(item)[:1000] for item in value if str(item).strip())


def _parse_report(text: str) -> Mapping[str, object]:
    """The child's JSON deliverable, fenced or bare; prose degrades to a summary."""

    value = (text or "").strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.DOTALL)
    candidate = fenced.group(1) if fenced else value

    try:
        parsed = json.loads(candidate)

        return parsed if isinstance(parsed, Mapping) else {}
    except (json.JSONDecodeError, TypeError):
        # An unstructured answer is kept as the summary and flagged, rather
        # than discarded: the prose is usually still worth reading.

        return {"summary": value[:4000],
                "uncertainties": ["Explorer response was not structured JSON."]}


def _report_status(result: AgentResult) -> ExplorationStatus:
    """How the child ended, from most specific reason to plain success or failure."""

    if result.terminal_reason == RuntimeTerminalReason.INTERRUPTED:
        return ExplorationStatus.CANCELLED

    if result.terminal_reason == RuntimeTerminalReason.STALLED:
        return ExplorationStatus.STALLED

    if result.terminal_reason in {
        RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
        RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
        RuntimeTerminalReason.BUDGET_EXHAUSTED,
    }:
        return ExplorationStatus.BUDGET_EXHAUSTED

    if result.terminal_status == TerminalStatus.COMPLETED:
        return ExplorationStatus.COMPLETED

    # Something was said but the task never completed, which is still of use
    # to the parent.

    if result.final_response:
        return ExplorationStatus.PARTIAL

    return ExplorationStatus.FAILED
