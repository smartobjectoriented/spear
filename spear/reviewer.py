"""Independent, context-isolated read-only code review orchestration."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from agent_roles import AgentRoleSpec, reviewer_role
from agent_runtime import AgentContext, AgentResult, AgentRuntime, RuntimeTerminalReason
from cancellation import CancellationSource, linked_cancellation
from compaction import CompactionPolicy
from context_engine import ContextItem, ContextLayer, Freshness
from diff_evidence import DiffEvidence
from model_backend import ConversationMessage, ModelBackend, TextBlock, ToolDefinition
from progress_monitor import ProgressMonitor
from session_store import SessionEventType, SessionHandle
from tracing import EventStatus, EventType
from verification import CompletionEvaluation, CompletionVerificationStatus
from working_state import StateEventType, StateSource, TerminalStatus, WorkingState
from budgets import BudgetManager


class ReviewVerdict(StrEnum):
    ACCEPT = "accept"
    REPAIR_REQUIRED = "repair_required"
    BLOCKED = "blocked"


class ReviewExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingCategory(StrEnum):
    REQUIREMENT_MISSING = "requirement_missing"
    FUNCTIONAL_REGRESSION = "functional_regression"
    VERIFICATION_GAP = "verification_gap"
    TEST_GAP = "test_gap"
    UNSAFE_CHANGE = "unsafe_change"
    UNRELATED_CHANGE = "unrelated_change"
    PROJECT_RULE_VIOLATION = "project_rule_violation"
    INCOMPLETE_ERROR_HANDLING = "incomplete_error_handling"
    COMPATIBILITY_ISSUE = "compatibility_issue"
    SECURITY_ISSUE = "security_issue"
    MAINTAINABILITY_ISSUE = "maintainability_issue"


@dataclass(frozen=True)
class ReviewFinding:
    severity: FindingSeverity
    category: FindingCategory
    description: str
    affected_path: str | None = None
    symbol: str | None = None
    requirement_reference: str | None = None
    evidence_reference: str | None = None
    confidence: float = 0.5
    blocking: bool = False
    suggested_remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("review finding description cannot be empty")

        if not 0 <= self.confidence <= 1:
            raise ValueError("review finding confidence must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["severity"] = self.severity.value
        result["category"] = self.category.value

        return result


def new_review_id() -> str:
    return "review_" + uuid.uuid4().hex


@dataclass(frozen=True)
class ReviewRequest:
    parent_task_id: str
    original_objective: str
    mutation_generation: int
    diff: DiffEvidence
    user_constraints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    project_rules: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()
    created_files: tuple[str, ...] = ()
    deleted_files: tuple[str, ...] = ()
    verification_status: str = CompletionVerificationStatus.UNVERIFIED.value
    verification_evidence: tuple[Mapping[str, object], ...] = ()
    project_bench_evidence: tuple[Mapping[str, object], ...] = ()
    unresolved_failures: tuple[Mapping[str, object], ...] = ()
    checkpoint_id: str | None = None
    checkpoint_status: str | None = None
    result_references: tuple[str, ...] = ()
    exploration_findings: tuple[Mapping[str, object], ...] = ()
    max_model_turns: int = 8
    max_tool_calls: int = 16
    context_limit: int = 32_768
    review_id: str = field(default_factory=new_review_id)
    deduplication_key: str | None = None

    def __post_init__(self) -> None:
        if not self.parent_task_id or not self.original_objective.strip():
            raise ValueError("review requires parent task and objective")

        if self.mutation_generation < 0:
            raise ValueError("review mutation generation cannot be negative")

        if min(self.max_model_turns, self.max_tool_calls, self.context_limit) < 1:
            raise ValueError("review budgets must be positive")

        # One review per mutation generation by default: reviewing the same
        # unchanged code twice would spend a child agent to learn nothing.

        if self.deduplication_key is None:
            object.__setattr__(
                self, "deduplication_key",
                f"review_generation_{self.mutation_generation}",
            )

    @property
    def changed_files(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            (*self.modified_files, *self.created_files, *self.deleted_files)
        ))

    def context_payload(self) -> dict[str, object]:
        """Everything the reviewer is allowed to see: requirements and evidence.

        The parent's conversation is deliberately absent. The reviewer judges
        the diff against the stated requirements, not the story of how it came
        about, which is what makes the review independent.
        """

        return {
            "review_id": self.review_id,
            "objective": self.original_objective,
            "constraints": self.user_constraints,
            "acceptance_criteria": self.acceptance_criteria,
            "mutation_generation": self.mutation_generation,
            "changed_files": self.changed_files,
            "created_files": self.created_files,
            "deleted_files": self.deleted_files,
            "diff": self.diff.to_dict(),
            "verification_status": self.verification_status,
            "verification_evidence": self.verification_evidence,
            "project_bench_evidence": self.project_bench_evidence,
            "unresolved_failures": self.unresolved_failures,
            "checkpoint": {"id": self.checkpoint_id, "status": self.checkpoint_status},
            "result_references": self.result_references,
            "exploration_findings": self.exploration_findings,
            "review_budget": {
                "model_turns": self.max_model_turns,
                "tool_calls": self.max_tool_calls,
                "context_limit": self.context_limit,
            },
        }


@dataclass(frozen=True)
class ReviewResult:
    review_id: str
    parent_task_id: str
    child_task_id: str
    child_session_id: str | None
    deduplication_key: str
    mutation_generation: int
    status: ReviewExecutionStatus
    verdict: ReviewVerdict
    confidence: float
    blocking_findings: tuple[ReviewFinding, ...] = ()
    non_blocking_findings: tuple[ReviewFinding, ...] = ()
    requirement_coverage: tuple[str, ...] = ()
    verification_assessment: str = ""
    suspected_regressions: tuple[str, ...] = ()
    unrelated_changes: tuple[str, ...] = ()
    test_gaps: tuple[str, ...] = ()
    security_concerns: tuple[str, ...] = ()
    recommended_repairs: tuple[str, ...] = ()
    evidence_references: tuple[str, ...] = ()
    files_inspected: tuple[str, ...] = ()
    model_calls: int = 0
    tool_calls: int = 0
    terminal_reason: str = "completed"
    diff_reference: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id, "parent_task_id": self.parent_task_id,
            "child_task_id": self.child_task_id,
            "child_session_id": self.child_session_id,
            "deduplication_key": self.deduplication_key,
            "mutation_generation": self.mutation_generation,
            "status": self.status.value, "verdict": self.verdict.value,
            "confidence": self.confidence,
            "blocking_findings": [item.to_dict() for item in self.blocking_findings],
            "non_blocking_findings": [item.to_dict() for item in self.non_blocking_findings],
            "requirement_coverage": list(self.requirement_coverage),
            "verification_assessment": self.verification_assessment,
            "suspected_regressions": list(self.suspected_regressions),
            "unrelated_changes": list(self.unrelated_changes),
            "test_gaps": list(self.test_gaps),
            "security_concerns": list(self.security_concerns),
            "recommended_repairs": list(self.recommended_repairs),
            "evidence_references": list(self.evidence_references),
            "files_inspected": list(self.files_inspected),
            "model_calls": self.model_calls, "tool_calls": self.tool_calls,
            "terminal_reason": self.terminal_reason,
            "diff_reference": self.diff_reference,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ReviewResult":
        values = dict(raw)
        values["status"] = ReviewExecutionStatus(str(values["status"]))
        values["verdict"] = ReviewVerdict(str(values["verdict"]))
        values["blocking_findings"] = tuple(
            _finding(item, forced_blocking=True)
            for item in values.get("blocking_findings", ())
        )
        values["non_blocking_findings"] = tuple(
            _finding(item, forced_blocking=False)
            for item in values.get("non_blocking_findings", ())
        )

        for key in (
            "requirement_coverage", "suspected_regressions", "unrelated_changes",
            "test_gaps", "security_concerns", "recommended_repairs",
            "evidence_references", "files_inspected",
        ):
            values[key] = _strings(values.get(key))

        return cls(**values)

    def repair_context_item(self) -> ContextItem:
        """The review as the parent sees it: findings and repairs, no transcript."""

        feedback = {
            "review_id": self.review_id,
            "mutation_generation": self.mutation_generation,
            "verdict": self.verdict.value,
            "blocking_findings": [item.to_dict() for item in self.blocking_findings],
            "recommended_repairs": self.recommended_repairs,
            "evidence_references": self.evidence_references,
        }

        return ContextItem(
            f"review-feedback:{self.review_id}", ContextLayer.RETRIEVED_CONTEXT,
            f"reviewer:{self.review_id}",
            "Independent review feedback (structured; no reviewer transcript):\n"
            + json.dumps(feedback, ensure_ascii=False, sort_keys=True)[:12_000],
            priority=92, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="current-generation independent review feedback",
        )


@dataclass(frozen=True)
class ReviewDecision:
    should_review: bool
    reason: str


class ReviewPolicy:
    _CODE_SUFFIXES = (
        ".py", ".c", ".h", ".cc", ".cpp", ".rs", ".go", ".java", ".js",
        ".ts", ".tsx", ".jsx", ".rb", ".php", ".sh", ".toml", ".yaml", ".yml",
    )

    def decide(self, state: WorkingState) -> ReviewDecision:
        """Whether this task's changes warrant an independent review."""

        if state.mutation_generation == 0:
            return ReviewDecision(False, "no grounded mutation")

        changed = state.modified_files | state.created_files | state.deleted_files

        # Any one of these makes the change consequential enough to review:
        # it spans files, it was given criteria to meet, it touched code, or a
        # project bench ran against it.

        if (len(changed) > 1 or state.acceptance_criteria
                or any(path.lower().endswith(self._CODE_SUFFIXES) for path in changed)
                or any(item.project_bench for item in state.verifications)):
            return ReviewDecision(True, "mutating coding/risk-bearing task")

        return ReviewDecision(False, "trivial non-code single-file mutation")

    def completion_allowed(
        self, state: WorkingState, verification: CompletionEvaluation,
    ) -> bool:
        """A reviewable task completes only once verified AND accepted."""

        decision = self.decide(state)

        if not decision.should_review:
            return True

        return (verification.status == CompletionVerificationStatus.VERIFIED
                and state.current_review_accepted)


@dataclass(frozen=True)
class ReviewRepairPolicy:
    maximum_cycles: int = 1

    def __post_init__(self) -> None:
        if self.maximum_cycles < 0:
            raise ValueError("maximum repair cycles cannot be negative")

    def allows(self, completed_cycles: int, result: ReviewResult) -> bool:
        return (result.verdict == ReviewVerdict.REPAIR_REQUIRED
                and completed_cycles < self.maximum_cycles)


ChildSessionFactory = Callable[[AgentContext, WorkingState], SessionHandle]
ModelOverrideResolver = Callable[[str], ModelBackend]


class ReviewService:
    def __init__(
        self, runtime: AgentRuntime, *, role: AgentRoleSpec | None = None,
        child_session_factory: ChildSessionFactory | None = None,
        model_override_resolver: ModelOverrideResolver | None = None,
    ) -> None:
        self.runtime = runtime
        self.role = role or reviewer_role()
        self.child_session_factory = child_session_factory
        self.model_override_resolver = model_override_resolver
        self._active_cancellation: CancellationSource | None = None

    def cancel_active(self, reason: str = "Reviewer cancelled") -> bool:
        return (self._active_cancellation.cancel(reason)
                if self._active_cancellation is not None else False)

    def review(
        self, request: ReviewRequest, parent: AgentContext, *,
        tools: tuple[ToolDefinition, ...], tool_executor,
        budget_manager: BudgetManager | None = None,
    ) -> ReviewResult:
        # A completed review of this generation already answers the question.

        existing = self._existing_result(parent, request.deduplication_key or "")

        if existing is not None:
            return existing

        # Both the request and the role cap the child, and the smaller wins:
        # a caller may ask for less than the role allows, never for more.

        state = WorkingState.start(
            "task_" + uuid.uuid4().hex,
            f"Independently review parent task {request.parent_task_id}",
            max_model_rounds=min(request.max_model_turns, self.role.max_model_turns),
            max_tool_actions=min(request.max_tool_calls, self.role.max_tool_calls),
        )

        # Linked, not shared: cancelling the parent cancels the review, while
        # cancel_active() stops the review alone.

        local_source, child_token = linked_cancellation(parent.cancellation)
        self._active_cancellation = local_source

        items = self._context_items(request)
        backend = parent.backend

        if self.role.model_override:
            if self.model_override_resolver is None:
                raise ValueError("Reviewer model override requires a resolver")

            backend = self.model_override_resolver(self.role.model_override)

        child = AgentContext(
            working_state=state, backend=backend,
            context_engine=parent.context_engine, trace=parent.trace,
            system_prompt="".join(item.content for item in items),
            context_items=items,
            conversation=[ConversationMessage("user", (TextBlock(
                "Perform the independent review and return only the requested JSON result."
            ),))],
            tools=tools, tool_executor=tool_executor,
            max_model_rounds=min(request.max_model_turns, self.role.max_model_turns),
            max_tool_actions=min(request.max_tool_calls, self.role.max_tool_calls),
            context_limit=min(parent.context_limit, request.context_limit,
                              self.role.context_limit),
            output_reserve=parent.output_reserve,
            compaction_policy=CompactionPolicy(),
            provider=parent.provider, model=self.role.model_override or parent.model,
            cancellation=child_token, progress_monitor=ProgressMonitor(),
            role=self.role.role.value, checkpoint_manager=None, checkpoint=None,
            budget_manager=budget_manager,
            task_trace_metadata={
                "role": "reviewer", "parent_task_id": parent.task_id,
                "review_id": request.review_id, "mutation_generation": request.mutation_generation,
                "raw_content_recorded": False,
            },
        )

        if self.child_session_factory:
            child.session = self.child_session_factory(parent, state)

        started = time.monotonic()
        parent.trace.emit(
            EventType.REVIEW_STARTED, parent.task_id, session_id=parent.session_id,
            status=EventStatus.STARTED,
            metadata={"review_id": request.review_id, "child_task_id": child.task_id,
                      "child_session_id": child.session_id,
                      "mutation_generation": request.mutation_generation,
                      "diff_bytes": request.diff.byte_count,
                      "raw_content_recorded": False},
        )

        # A review that fails outright still yields a result, and a failed
        # review is BLOCKED rather than silently absent.

        try:
            runtime_result = self.runtime.run(child)
            result = self._result(request, child, runtime_result)
        except Exception as exc:
            result = self._failed_result(request, child, exc)
        finally:
            self._active_cancellation = None

        event = (EventType.REVIEW_FINISHED
                 if result.status == ReviewExecutionStatus.COMPLETED
                 else EventType.REVIEW_FAILED)
        parent.trace.emit(
            event, parent.task_id, session_id=parent.session_id,
            status=(EventStatus.OK if event == EventType.REVIEW_FINISHED
                    else EventStatus.CANCELLED
                    if result.status == ReviewExecutionStatus.CANCELLED
                    else EventStatus.FAILED),
            duration_ms=(time.monotonic() - started) * 1000,
            metadata={"review_id": result.review_id,
                      "child_task_id": result.child_task_id,
                      "child_session_id": result.child_session_id,
                      "mutation_generation": result.mutation_generation,
                      "verdict": result.verdict.value,
                      "blocking_count": len(result.blocking_findings),
                      "non_blocking_count": len(result.non_blocking_findings),
                      "files_inspected": len(result.files_inspected),
                      "model_calls": result.model_calls,
                      "tool_calls": result.tool_calls,
                      "terminal_reason": result.terminal_reason,
                      "raw_content_recorded": False},
        )
        self.ingest(parent, result)

        return result

    @staticmethod
    def ingest(parent: AgentContext, result: ReviewResult) -> None:
        """Record the review on the parent as a state event, exactly once."""

        if any(item.get("review_id") == result.review_id
               for item in parent.review_results):
            return

        parent.review_results.append(result.to_dict())
        parent.context_items = (*parent.context_items, result.repair_context_item())
        parent.apply_state_event(
            StateEventType.REVIEW_RECORDED, source=StateSource.HARNESS,
            review_id=result.review_id,
            mutation_generation=result.mutation_generation,
            verdict=result.verdict.value,
            blocking_count=len(result.blocking_findings),
            evidence_reference=result.diff_reference,
        )
        AgentRuntime.persist_session_event(
            parent, SessionEventType.REVIEW_COMPLETED,
            {"review_id": result.review_id,
             "child_task_id": result.child_task_id,
             "child_session_id": result.child_session_id,
             "mutation_generation": result.mutation_generation,
             "deduplication_key": result.deduplication_key,
             "status": result.status.value, "verdict": result.verdict.value,
             "blocking_count": len(result.blocking_findings)},
        )
        trace_type = {
            ReviewVerdict.ACCEPT: EventType.REVIEW_ACCEPTED,
            ReviewVerdict.REPAIR_REQUIRED: EventType.REVIEW_REPAIR_REQUESTED,
            ReviewVerdict.BLOCKED: EventType.REVIEW_BLOCKED,
        }[result.verdict]
        parent.trace.emit(
            trace_type, parent.task_id, session_id=parent.session_id,
            status=(EventStatus.OK if result.verdict == ReviewVerdict.ACCEPT
                    else EventStatus.DETECTED),
            metadata={"review_id": result.review_id,
                      "mutation_generation": result.mutation_generation,
                      "blocking_count": len(result.blocking_findings),
                      "raw_content_recorded": False},
        )

    @staticmethod
    def _existing_result(parent: AgentContext, key: str) -> ReviewResult | None:
        # Only a COMPLETED review is reusable: a failed or cancelled one never
        # examined the code, so it must not stand in for a review of it.

        for raw in parent.review_results:
            if (raw.get("deduplication_key") == key
                    and raw.get("status") == ReviewExecutionStatus.COMPLETED.value):
                return ReviewResult.from_dict(raw)

        return None

    def _context_items(self, request: ReviewRequest) -> tuple[ContextItem, ...]:
        role = ContextItem(
            f"reviewer-role:{request.review_id}", ContextLayer.SYSTEM_RULES,
            "agent_role:reviewer", self.role.prompt_addition,
            priority=100, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="enforce independent read-only reviewer role",
            truncatable=False,
        )
        rules = tuple(ContextItem(
            f"review-project-rule:{request.review_id}:{index}",
            ContextLayer.PROJECT_RULES, "review_request", value,
            priority=96, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="project rule relevant to independent review",
            truncatable=False,
        ) for index, value in enumerate(request.project_rules))
        evidence = ContextItem(
            f"review-evidence:{request.review_id}", ContextLayer.TOOL_EVIDENCE,
            "grounded_review_request",
            json.dumps(request.context_payload(), ensure_ascii=False, sort_keys=True),
            priority=95, freshness=Freshness.CURRENT, protected=True,
            inclusion_reason="authoritative requirements, diff, and verification evidence",
            truncatable=True,
        )

        return (role, *rules, evidence)

    def _result(
        self, request: ReviewRequest, child: AgentContext,
        runtime: AgentResult,
    ) -> ReviewResult:
        """Turn the child's answer into a validated result.

        Every failure path below lands on BLOCKED. A review that did not run,
        or whose output could not be read, must never be mistaken for one that
        found nothing wrong.
        """

        status = _execution_status(runtime)

        if status != ReviewExecutionStatus.COMPLETED:
            verdict = ReviewVerdict.BLOCKED
            return ReviewResult(
                request.review_id, request.parent_task_id, runtime.task_id,
                child.session_id, request.deduplication_key or "",
                request.mutation_generation, status, verdict, 0.0,
                terminal_reason=runtime.terminal_reason.value,
                files_inspected=tuple(sorted(runtime.working_state.files_read)),
                model_calls=runtime.model_calls, tool_calls=runtime.tool_actions,
                diff_reference=request.diff.result_reference,
            )

        parsed = _parse_json(runtime.final_response)

        if parsed is None:
            return ReviewResult(
                request.review_id, request.parent_task_id, runtime.task_id,
                child.session_id, request.deduplication_key or "",
                request.mutation_generation, ReviewExecutionStatus.FAILED,
                ReviewVerdict.BLOCKED, 0.0,
                blocking_findings=(ReviewFinding(
                    FindingSeverity.HIGH, FindingCategory.VERIFICATION_GAP,
                    "Reviewer returned malformed structured output.",
                    blocking=True,
                ),),
                terminal_reason="malformed_review_output",
                files_inspected=tuple(sorted(runtime.working_state.files_read)),
                model_calls=runtime.model_calls, tool_calls=runtime.tool_actions,
                diff_reference=request.diff.result_reference,
            )

        try:
            verdict = ReviewVerdict(str(parsed.get("verdict", "blocked")).lower())
            confidence = float(parsed.get("confidence", 0.5))
            raw_findings = parsed.get("findings", ())

            if not isinstance(raw_findings, (list, tuple)):
                raise ValueError("findings must be a list")

            findings = tuple(_finding(item) for item in raw_findings)
        except (TypeError, ValueError, KeyError):
            return self._result_from_malformed(request, child, runtime)

        findings, verdict = self._validate(request, child, findings, verdict)

        blocking = tuple(item for item in findings if item.blocking)
        nonblocking = tuple(item for item in findings if not item.blocking)

        return ReviewResult(
            request.review_id, request.parent_task_id, runtime.task_id,
            child.session_id, request.deduplication_key or "",
            request.mutation_generation, ReviewExecutionStatus.COMPLETED,
            verdict, max(0.0, min(1.0, confidence)), blocking, nonblocking,
            _strings(parsed.get("requirement_coverage")),
            str(parsed.get("verification_assessment", ""))[:2000],
            _strings(parsed.get("suspected_regressions")),
            _strings(parsed.get("unrelated_changes")),
            _strings(parsed.get("test_gaps")),
            _strings(parsed.get("security_concerns")),
            _strings(parsed.get("recommended_repairs")),
            tuple(sorted(set(_strings(parsed.get("evidence_references")))
                         & self._valid_evidence(request, child))),
            tuple(sorted(runtime.working_state.files_read)),
            runtime.model_calls, runtime.tool_actions,
            runtime.terminal_reason.value, request.diff.result_reference,
        )

    def _result_from_malformed(
        self, request: ReviewRequest, child: AgentContext, runtime: AgentResult,
    ) -> ReviewResult:
        return ReviewResult(
            request.review_id, request.parent_task_id, runtime.task_id,
            child.session_id, request.deduplication_key or "",
            request.mutation_generation, ReviewExecutionStatus.FAILED,
            ReviewVerdict.BLOCKED, 0.0,
            blocking_findings=(ReviewFinding(
                FindingSeverity.HIGH, FindingCategory.VERIFICATION_GAP,
                "Reviewer output could not be validated.", blocking=True,
            ),), terminal_reason="malformed_review_output",
            files_inspected=tuple(sorted(runtime.working_state.files_read)),
            model_calls=runtime.model_calls, tool_calls=runtime.tool_actions,
            diff_reference=request.diff.result_reference,
        )

    def _validate(
        self, request: ReviewRequest, child: AgentContext,
        findings: tuple[ReviewFinding, ...], verdict: ReviewVerdict,
    ) -> tuple[tuple[ReviewFinding, ...], ReviewVerdict]:
        """Hold the reviewer's findings to the same evidence rule as the agent.

        The reviewer may say anything, but only a finding it can tie to a
        changed file, real evidence, or a stated requirement is allowed to
        block. The two findings appended afterwards are the harness's own, and
        are grounded by construction.
        """

        valid_evidence = self._valid_evidence(request, child)
        changed = set(request.changed_files)
        requirements = {request.original_objective, *request.user_constraints,
                        *request.acceptance_criteria, *request.project_rules}
        validated = []

        for finding in findings:
            # A verification gap is grounded by the recorded status rather than
            # by a citation: the absence of evidence is the finding.

            grounded = (
                finding.affected_path in changed
                or finding.evidence_reference in valid_evidence
                or finding.requirement_reference in requirements
                or (finding.category == FindingCategory.VERIFICATION_GAP
                    and request.verification_status
                    != CompletionVerificationStatus.VERIFIED.value)
            )
            blocking = finding.blocking and grounded

            # Taste never blocks: a maintainability remark only counts if the
            # project actually asked for what it is asking for.

            if (finding.category == FindingCategory.MAINTAINABILITY_ISSUE
                    and finding.requirement_reference not in requirements):
                blocking = False

            # An unrecognised evidence reference is dropped rather than kept,
            # so a finding never cites something that does not exist.

            validated.append(ReviewFinding(
                finding.severity, finding.category, finding.description,
                finding.affected_path, finding.symbol,
                finding.requirement_reference,
                finding.evidence_reference if finding.evidence_reference in valid_evidence else None,
                finding.confidence, blocking, finding.suggested_remediation,
            ))

        # The workspace moved outside the task's own mutations, so the diff
        # under review is not the whole change. Nothing else can be trusted.

        if request.diff.external_change_paths:
            validated.append(ReviewFinding(
                FindingSeverity.CRITICAL, FindingCategory.UNSAFE_CHANGE,
                "Workspace content differs from the last grounded task-owned mutation.",
                affected_path=request.diff.external_change_paths[0], confidence=1.0,
                blocking=True, suggested_remediation="Inspect external changes before proceeding.",
            ))
            verdict = ReviewVerdict.BLOCKED

        # Unverified code cannot be accepted however good it looks, so this
        # finding is added by the harness regardless of what the reviewer said.

        if (request.mutation_generation > 0
                and request.verification_status
                != CompletionVerificationStatus.VERIFIED.value):
            validated.append(ReviewFinding(
                FindingSeverity.HIGH, FindingCategory.VERIFICATION_GAP,
                "Current mutation generation lacks successful full verification.",
                evidence_reference=next(iter(sorted(valid_evidence)), None),
                confidence=1.0, blocking=True,
                suggested_remediation="Run meaningful current-generation verification.",
            ))

            if verdict == ReviewVerdict.ACCEPT:
                verdict = ReviewVerdict.REPAIR_REQUIRED

        # The verdict is reconciled with the findings in both directions: it
        # cannot accept while something blocks, nor demand repairs for nothing.

        has_blocking = any(item.blocking for item in validated)

        if has_blocking and verdict == ReviewVerdict.ACCEPT:
            verdict = ReviewVerdict.REPAIR_REQUIRED

        if not has_blocking and verdict == ReviewVerdict.REPAIR_REQUIRED:
            verdict = ReviewVerdict.ACCEPT

        return tuple(validated), verdict

    @staticmethod
    def _valid_evidence(request: ReviewRequest, child: AgentContext) -> set[str]:
        """Every reference a finding may legitimately cite.

        Both sides count: what the parent handed over as evidence, and what the
        reviewer itself observed while looking.
        """

        result = set(request.result_references)

        if request.diff.result_reference:
            result.add(request.diff.result_reference)

        for item in (*request.verification_evidence, *request.project_bench_evidence):
            for key in ("result_reference", "action_id", "verification_id"):
                if item.get(key):
                    result.add(str(item[key]))

        result.update(action.action_id for action in child.working_state.actions)
        result.update(child.result_references)

        return result

    @staticmethod
    def _failed_result(
        request: ReviewRequest, child: AgentContext, exc: Exception,
    ) -> ReviewResult:
        return ReviewResult(
            request.review_id, request.parent_task_id, child.task_id,
            child.session_id, request.deduplication_key or "",
            request.mutation_generation,
            ReviewExecutionStatus.CANCELLED if child.cancellation.is_cancelled
            else ReviewExecutionStatus.FAILED,
            ReviewVerdict.BLOCKED, 0.0,
            terminal_reason=type(exc).__name__, diff_reference=request.diff.result_reference,
        )


def review_request_from_parent(
    parent: AgentContext, diff: DiffEvidence,
    evaluation: CompletionEvaluation, *, project_rules: Sequence[str] = (),
) -> ReviewRequest:
    """Assemble the review request from what the parent task actually recorded."""

    state = parent.working_state
    evidence = tuple(_verification_dict(item) for item in state.verifications)
    bench = tuple(item for item in evidence if item.get("project_bench"))
    exploration = tuple({
        "exploration_id": item.get("exploration_id"),
        "summary": item.get("summary"),
        "relevant_files": item.get("relevant_files", ()),
        "evidence_references": item.get("evidence_references", ()),
    } for item in parent.exploration_reports)

    return ReviewRequest(
        parent.task_id, state.objective, state.mutation_generation, diff,
        tuple(state.user_constraints), tuple(state.acceptance_criteria),
        tuple(project_rules), tuple(sorted(state.modified_files)),
        tuple(sorted(state.created_files)), tuple(sorted(state.deleted_files)),
        evaluation.status.value, evidence, bench,
        tuple({"failure_id": failure.failure_id, "category": failure.category,
               "summary": failure.summary, "action_id": failure.action_id}
              for failure in state.unresolved_failures),
        state.checkpoint_id, state.checkpoint_status,
        tuple(sorted(parent.result_references)), exploration,
    )


def _verification_dict(item) -> dict[str, object]:
    return {
        "verification_id": item.verification_id, "category": item.category,
        "coverage": item.coverage, "executed": item.executed,
        "outcome": item.outcome.value, "mutation_generation": item.mutation_generation,
        "action_id": item.action_id, "result_reference": item.result_reference,
        "project_bench": item.project_bench, "summary": item.summary,
    }


def _finding(raw: object, forced_blocking: bool | None = None) -> ReviewFinding:
    """Build a finding from model output, or from a stored result.

    ``forced_blocking`` is how deserialization restores the blocking split the
    validator decided: the two lists already say which a finding belongs to.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("review finding must be an object")

    return ReviewFinding(
        FindingSeverity(str(raw.get("severity", "medium")).lower()),
        FindingCategory(str(raw["category"]).lower()),
        str(raw["description"]),
        str(raw["affected_path"]) if raw.get("affected_path") else None,
        str(raw["symbol"]) if raw.get("symbol") else None,
        str(raw["requirement_reference"]) if raw.get("requirement_reference") else None,
        str(raw["evidence_reference"]) if raw.get("evidence_reference") else None,
        float(raw.get("confidence", 0.5)),
        bool(raw.get("blocking", False)) if forced_blocking is None else forced_blocking,
        str(raw["suggested_remediation"]) if raw.get("suggested_remediation") else None,
    )


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    return tuple(str(item)[:1000] for item in value if str(item).strip())


def _parse_json(text: str) -> Mapping[str, object] | None:
    """The reviewer's JSON verdict, fenced or bare; None if it is neither."""

    value = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.DOTALL)

    try:
        parsed = json.loads(fenced.group(1) if fenced else value)
        return parsed if isinstance(parsed, Mapping) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _execution_status(result: AgentResult) -> ReviewExecutionStatus:
    """How the reviewer ended, from most specific reason to plain success."""

    if result.terminal_reason == RuntimeTerminalReason.INTERRUPTED:
        return ReviewExecutionStatus.CANCELLED

    if result.terminal_reason == RuntimeTerminalReason.STALLED:
        return ReviewExecutionStatus.STALLED

    if result.terminal_reason in {
        RuntimeTerminalReason.ROUND_BUDGET_EXHAUSTED,
        RuntimeTerminalReason.TOOL_BUDGET_EXHAUSTED,
        RuntimeTerminalReason.BUDGET_EXHAUSTED,
    }:
        return ReviewExecutionStatus.BUDGET_EXHAUSTED

    if result.terminal_status == TerminalStatus.COMPLETED:
        return ReviewExecutionStatus.COMPLETED

    return ReviewExecutionStatus.FAILED
