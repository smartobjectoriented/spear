"""UI-free orchestration for one SPEAR agent task.

The controller coordinates existing policies and runtimes.  It does not build
providers, read terminal input, render output, or execute tools directly.
Application-specific compatibility and persistence projections are injected as
narrow callbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Callable, Mapping, Sequence

from agent_roles import AgentRole, explorer_role, reviewer_role
from agent_runtime import AgentContext, AgentResult, AgentRuntime, RuntimeTerminalReason
from context_engine import ContextItem, ContextLayer, Freshness
from diff_evidence import DiffEvidence
from model_backend import ConversationMessage, TextBlock, ToolDefinition
from orchestration import (
    DelegationManager, DelegationRequest, DelegationResult, DelegationStatus,
    DelegationUsage, ExplorationPolicy, ExplorationRequest, ExplorationReport,
    ExplorationService, ExplorationStatus,
)
from planning import PlanningPolicy, PlanningService
from reviewer import (
    ReviewPolicy, ReviewRepairPolicy, ReviewResult, ReviewService,
    review_request_from_parent,
)
from session_store import SessionEventType, SessionHandle
from tool_exposure import (
    READ_ONLY_RULE, READ_ONLY_RULE_ID, ToolExposurePolicy, ToolView,
)
from tool_registry import ToolMutability, ToolRegistry
from tracing import EventStatus, EventType
from verification import CompletionEvaluation, CompletionVerificationStatus
from working_state import PlanStepStatus, StateEventType, WorkingState
from training_data import TrainingRecorder, discover_git_commit


ToolExecutor = Callable[..., object]
ChildSessionFactory = Callable[[AgentRole, AgentContext, WorkingState], SessionHandle]
VerificationRunner = Callable[[AgentContext], bool | None]
BenchObserver = Callable[[AgentContext, AgentResult, bool, str], None]
DiffProvider = Callable[[AgentContext], DiffEvidence]
CompatibilityMutation = Callable[[AgentContext, AgentResult], AgentResult]


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class TaskRequest:
    objective: str
    context: AgentContext
    project_scope: tuple[str, ...]
    project_rules: tuple[str, ...] = ()
    web_enabled: bool = True
    memory_write_enabled: bool = True

    # Child agents are experimental and opt-in; the main task path remains
    # synchronous and zero-child by default.

    enable_explorer: bool = False
    enable_reviewer: bool = False

    # Explicit policy switches used by benchmark/ablation callers.  Defaults
    # preserve the normal production flow; they are not global feature flags.

    enable_planning: bool = True
    enable_compaction: bool = True
    selected_memory: bool = True
    role_aware_tools: bool = True
    enable_review_repair: bool = False
    standard_binding: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("task objective must not be empty")

        if self.context.task_id != self.context.working_state.task_id:
            raise ValueError("task context identity mismatch")

        if self.standard_binding is not None:
            if (self.context.standard_binding is not None
                    and dict(self.context.standard_binding) != dict(self.standard_binding)):
                raise ValueError("task request standard binding mismatch")

            self.context.standard_binding = dict(self.standard_binding)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    session_id: str | None
    status: TaskStatus
    final_response: str
    working_state: WorkingState
    agent_result: AgentResult
    verification: CompletionEvaluation
    review: ReviewResult | None
    exploration: ExplorationReport | None
    checkpoint_status: str | None
    budget: Mapping[str, object]
    terminal_reason: str
    warnings: tuple[str, ...] = ()


class TaskController:
    """Synchronous bounded orchestration for Main, Explorer, and Reviewer."""

    def __init__(
        self, runtime: AgentRuntime, registry: ToolRegistry, *,
        tool_executor: ToolExecutor,
        child_session_factory: ChildSessionFactory | None = None,
        verification_runner: VerificationRunner | None = None,
        bench_observer: BenchObserver | None = None,
        diff_provider: DiffProvider | None = None,
        compatibility_mutation: CompatibilityMutation | None = None,
        planning_policy: PlanningPolicy | None = None,
        exploration_policy: ExplorationPolicy | None = None,
        review_policy: ReviewPolicy | None = None,
        tool_exposure_policy: ToolExposurePolicy | None = None,
        review_repair_policy=None,
        training_store=None,
        standard_store=None,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.tool_executor = tool_executor
        self.child_session_factory = child_session_factory
        self.verification_runner = verification_runner
        self.bench_observer = bench_observer
        self.diff_provider = diff_provider
        self.compatibility_mutation = compatibility_mutation
        self.planning_policy = planning_policy or PlanningPolicy()
        self.exploration_policy = exploration_policy or ExplorationPolicy()
        self.review_policy = review_policy or ReviewPolicy()
        self.tool_exposure_policy = tool_exposure_policy or ToolExposurePolicy()
        self.review_repair_policy = review_repair_policy
        self.training_store = training_store
        self.standard_store = standard_store

    def run(self, request: TaskRequest) -> TaskResult:
        context = request.context

        if context.standard_binding is not None:
            from standard_schema import StandardBinding
            persisted = StandardBinding.from_dict(context.standard_binding)

            if self.standard_store is not None:
                current = self.standard_store.binding(
                    persisted.standard_id, persisted.revision,
                    bound_at=persisted.bound_at,
                )

                if (persisted.pdf_sha256 != current.pdf_sha256
                        or persisted.corpus_manifest_sha256 != current.corpus_manifest_sha256):
                    raise ValueError("canonical normative source changed; task blocked")

                if ((persisted.retrieval_fingerprint or persisted.index_fingerprint)
                        != (current.retrieval_fingerprint or current.index_fingerprint)):
                    context.standard_binding = current.to_dict()

                    if context.session is not None:
                        AgentRuntime.persist_session_event(
                            context, SessionEventType.STANDARD_RETRIEVAL_CHANGED,
                            {"old_retrieval_fingerprint": (
                                 persisted.retrieval_fingerprint
                                 or persisted.index_fingerprint),
                             "new_retrieval_fingerprint": (
                                 current.retrieval_fingerprint
                                 or current.index_fingerprint),
                             "canonical_source_unchanged": True,
                             "cached_retrieval_invalidated": True},
                        )

                        if persisted.index_fingerprint != current.index_fingerprint:
                            AgentRuntime.persist_session_event(
                                context, SessionEventType.STANDARD_INDEX_CHANGED,
                                {"old_index_fingerprint": persisted.index_fingerprint,
                                 "new_index_fingerprint": current.index_fingerprint,
                                 "canonical_source_unchanged": True,
                                 "cached_retrieval_invalidated": True},
                            )

                    persisted = current

            identity = "\n".join((
                "\nBOUND STANDARD",
                persisted.standard_id,
                f"revision {persisted.revision}",
                f"PDF sha256 {persisted.pdf_sha256}",
                f"corpus sha256 {persisted.corpus_manifest_sha256}",
                "Normative claims must use retrieved canonical sources and deterministic citations.",
                "Distinguish requirements, recommendations, informative text, and examples.",
                "Do not invent values or silently switch revisions; report insufficient or conflicting sources.",
            )) + "\n"

            # What outranks what, and what may speak for the standard. Both
            # are attached to the BINDING, which is the one place a rule
            # reaches every bound turn whatever prompt the project carries.
            #
            # The evidence half used to live in a project's system prompt
            # instead, and only one corpus KIND was given that prompt -- so an
            # ad-hoc session could bind a standard and never be told what a
            # value group is, which is exactly the question the tools exist to
            # answer. Whether a standard is bound is a property of the turn;
            # which corpus it was bound from is not.

            import normative_precedence

            identity += normative_precedence.PRECEDENCE_RULE + "\n"
            identity += normative_precedence.EVIDENCE_RULE + "\n"

            # A conformance question gets the per-clause form asked for
            # once, here. The assessment record the policy appends does not
            # depend on the model honouring it.

            import conformance_mode
            from agent_runtime import (
                _asked, carried_obligations, is_write_request)

            asked = _asked(context.conversation)

            if conformance_mode.is_conformance_turn(asked):
                identity += conformance_mode.FORMAT_RULE + "\n"

            # The clauses the previous turn retrieved, named for this one.
            # Two prompts is the normal shape -- "how should X work?", then
            # "change the code" -- and the reasoning was done in the first.
            # Without this the second starts over and lands on the wrong
            # sections: one read six clauses of glossary and context fields,
            # then justified a change by the acknowledgement rule it had not
            # opened.

            # The same question the completion gate asks, asked once here:
            # clauses carried from an earlier turn are this turn's
            # obligations only when this turn was asked to change the code.
            # Two answers to that would mean a gate demanding file-and-line
            # for clauses the prompt never presented as work.

            prior = carried_obligations(context, asked)
            answered = str(getattr(context, "prior_answer", "") or "")

            # The requirements this change must meet were established one
            # prompt ago, by the model itself, from the bound standard. A
            # turn that has to rediscover them reads whatever a fresh search
            # surfaces -- glossary sections, context fields -- and changes
            # the code against nothing in particular.

            if answered and is_write_request(asked):
                identity += "\n".join((
                    "\nWHAT THIS SESSION ALREADY ESTABLISHED",
                    answered.strip(),
                    "That is the specification for the change you are now "
                    "asked to make. Every rule stated above must hold in the "
                    "code when you are done, or you must say which does not "
                    "and why.",
                )) + "\n"

            if prior:
                identity += "\n".join((
                    "\nCLAUSES ALREADY ESTABLISHED THIS SESSION",
                    ", ".join(f"§{item}" for item in prior[:16]),
                    "These are what the change must satisfy. Before editing, "
                    "state for each one what the code does today, with file "
                    "and line. Retrieve any clause you intend to cite that is "
                    "not in this list; do not cite one you have not read.",
                )) + "\n"

            context.context_items = tuple(
                item for item in context.context_items
                if item.item_id != "standard:binding"
            ) + (ContextItem(
                "standard:binding", ContextLayer.SYSTEM_RULES,
                "standard_binding", identity, 100, Freshness.CURRENT, True,
                inclusion_reason="operator-pinned normative source identity",
                truncatable=False,
            ),)

            if context.session is not None and not any(
                event.event_type == SessionEventType.STANDARD_BINDING_SET
                for event in context.session.store.events(context.session.session_id)
            ):
                AgentRuntime.persist_session_event(
                    context, SessionEventType.STANDARD_BINDING_SET,
                    {"standard_id": persisted.standard_id,
                     "revision": persisted.revision,
                     "pdf_sha256": persisted.pdf_sha256,
                     "corpus_manifest_sha256": persisted.corpus_manifest_sha256,
                     "index_fingerprint": persisted.index_fingerprint},
                )

        if self.training_store is not None and context.training_recorder is None:
            backend = context.backend
            generation = {
                key: getattr(backend, key) for key in ("temperature", "max_tokens")
                if getattr(backend, key, None) is not None
            }
            context.training_recorder = TrainingRecorder(
                self.training_store, task_id=context.task_id,
                session_id=context.session_id, objective=request.objective,
                workspace=request.project_scope[0],
                project=str(context.task_trace_metadata.get("project") or "") or None,
                data_origin=str(context.task_trace_metadata.get("data_origin") or
                                "NORMAL_USAGE").upper(),
                provider=context.provider, model=context.model,
                generation_config=generation,
                harness_commit=discover_git_commit(os.path.dirname(__file__)),
                standard_binding=context.standard_binding,
            )

        if not request.enable_compaction:
            # Keep the production policy untouched; this context-local
            # override disables automatic compaction for an ablation run.

            context.compaction_policy = replace(
                context.compaction_policy,
                minimum_compactable_tokens=10**9,
            )

        if not request.selected_memory:
            context.context_items = tuple(
                item for item in context.context_items
                if item.layer != ContextLayer.DURABLE_MEMORY
            )

        planning = PlanningService()
        exploration_decision = self.exploration_policy.decide(request.objective)
        planning_decision = self.planning_policy.decide(
            request.objective,
            acceptance_criteria=context.working_state.acceptance_criteria,
            exploration_required=(request.enable_explorer
                                  and exploration_decision.should_explore),
        )

        if request.role_aware_tools:
            view = self.tool_exposure_policy.select(
                self.registry, AgentRole.MAIN, objective=request.objective,
                web_enabled=request.web_enabled,
                memory_write_enabled=request.memory_write_enabled,
                standard_bound=context.standard_binding is not None,
            )
        else:
            # Controlled ablation: expose the same registry definitions in
            # deterministic order, while retaining router authorization.

            # The ablation is role-aware exposure versus the whole registry.
            # An explicit "do not change anything" is not part of that
            # comparison: it is an instruction, and it holds in both arms.

            read_only = self.tool_exposure_policy.read_only_intent(request.objective)
            definitions = tuple(
                ToolDefinition(spec.name, spec.description, dict(spec.input_schema))
                for spec in self.registry.list_specs(model_visible=True)
                if (context.standard_binding is not None
                    or not spec.name.startswith("standard."))
                if not (read_only and spec.mutability == ToolMutability.MUTATING)
            )
            chars = sum(
                len(item.name) + len(item.description) + len(str(item.input_schema))
                for item in definitions
            )
            view = ToolView(
                AgentRole.MAIN, tuple(item.name for item in definitions),
                definitions, chars // 4 + (1 if chars else 0), True,
                "full registry ablation", read_only,
            )

        context.tools = view.definitions
        context.read_only = view.read_only

        # The boundary is enforced, and the model still has to understand its
        # mission. Told to name the file that defines `add`, a run reached for
        # `sed -i` at step four to fix a bug nobody had asked about, then
        # spent thirty-five more steps working around the refusal. Derived
        # from the view, so there is one decision about what read-only means
        # and every role sees the same one.

        context.context_items = tuple(
            item for item in context.context_items
            if item.item_id != READ_ONLY_RULE_ID
        ) + ((ContextItem(
            READ_ONLY_RULE_ID, ContextLayer.SYSTEM_RULES, "tool_exposure",
            READ_ONLY_RULE, 100, Freshness.CURRENT, True,
            inclusion_reason="the request forbade changing anything",
            truncatable=False,
        ),) if view.read_only else ())
        context.tool_exposure = {
            "policy_version": 1, "role": view.role.value,
            "names": view.names, "schema_token_estimate": view.schema_token_estimate,
            "approximate": view.approximate, "read_only": view.read_only,
        }
        context.trace.emit(
            EventType.TOOL_VIEW_SELECTED, context.task_id,
            session_id=context.session_id, status=EventStatus.OK,
            # The NAMES, not only how many. A count cannot tell "the tool was
            # never offered" from "the model ignored the tool it had", and
            # that was the whole question after a run tried fetch_url to
            # write a local file. Tool names are the registry's own
            # vocabulary; nothing about them is task content.
            metadata={"role": "main", "tool_count": len(view.names),
                      "selected_tool_names": list(view.names),
                      "read_only": view.read_only,
                      "schema_token_estimate": view.schema_token_estimate,
                      "approximate": view.approximate,
                      "raw_content_recorded": False},
        )

        if request.enable_planning and planning_decision.should_plan:
            created = planning.create(context)

            if created.created:
                planning.start_next(context)

        exploration = None
        exploration_decision = self.exploration_policy.decide(
            request.objective, prior_reports=context.exploration_reports,
        )

        if request.enable_explorer and exploration_decision.should_explore:
            exploration = self._explore(request)

            if (exploration is not None
                    and any(item.action_id == exploration.exploration_id
                            for item in context.working_state.discoveries)):
                planning.advance(context, "exploration", (exploration.exploration_id,))

        agent_result = self.runtime.run(context, defer_completion=True)

        if agent_result.terminal_reason == RuntimeTerminalReason.INTERRUPTED:
            return self._result(request, TaskStatus.INTERRUPTED, agent_result,
                                exploration=exploration)

        if agent_result.budget_exhausted:
            return self._result(request, TaskStatus.BUDGET_EXHAUSTED, agent_result,
                                exploration=exploration)

        if agent_result.terminal_status == "failed":
            return self._result(request, TaskStatus.FAILED, agent_result,
                                exploration=exploration)

        if self.compatibility_mutation is not None and not agent_result.did_modify:
            agent_result = self.compatibility_mutation(context, agent_result)

        if agent_result.did_modify and context.working_state.mutation_action_ids:
            planning.advance(
                context, "mutation",
                tuple(sorted(context.working_state.mutation_action_ids)),
            )

        if (agent_result.final_response
                and context.working_state.mutation_generation
                and self.verification_runner is not None):
            verdict = self._verify(context, agent_result, "bench")

            if (verdict is False
                    and context.working_state.current_round < context.max_model_rounds
                    and not context.cancellation.is_cancelled):
                context.apply_state_event(
                    StateEventType.RETRY_RECORDED, reason="project_bench_failed",
                )
                context.conversation.append(ConversationMessage(
                    "user", (TextBlock(
                        "The harness acceptance bench failed. Inspect the grounded "
                        "verification failure in current task state, make one bounded "
                        "repair attempt, and verify again."
                    ),), authored_by="harness"))
                repair = self.runtime.run(context, defer_completion=True)

                if repair.terminal_status != "failed" and repair.final_response:
                    agent_result = self._merge(agent_result, repair)
                    self._verify(context, agent_result, "bench_retry")

        verification = context.verification_policy.evaluate_completion(
            context.working_state,
        )
        self._advance_verification(planning, context, verification)
        review = None

        if request.enable_reviewer:
            decision = self.review_policy.decide(context.working_state)
            context.review_required = decision.should_review

            if decision.should_review:
                review, agent_result, verification = self._review(
                    request, planning, agent_result, verification,
                )

        warnings: list[str] = []

        if context.review_required and not self.review_policy.completion_allowed(
            context.working_state, verification,
        ):
            verdict = review.verdict.value if review is not None else "not_run"
            warnings.append(f"review:{verdict}")
            agent_result.final_response = (
                f"⚠ REVIEW {verdict.upper()} — current-generation verification/"
                "review gates are not satisfied. The task checkpoint remains "
                "active.\n\n" + agent_result.final_response
            )

        if (agent_result.final_response and verification.status in {
            CompletionVerificationStatus.UNVERIFIED,
            CompletionVerificationStatus.PARTIALLY_VERIFIED,
            CompletionVerificationStatus.FAILED,
        }):
            changed = sorted(
                context.working_state.modified_files
                | context.working_state.created_files
                | context.working_state.deleted_files
            )
            warnings.append(f"verification:{verification.status.value}")
            agent_result.final_response = (
                "⚠ UNVERIFIED — " + ", ".join(changed)
                + " changed without sufficient current verification.\n\n"
                + agent_result.final_response
            )

        agent_result = self.runtime.complete(context, agent_result)
        status = (TaskStatus.COMPLETED if not agent_result.completion_deferred
                  else TaskStatus.BLOCKED)

        return self._result(
            request, status, agent_result, verification=verification,
            review=review, exploration=exploration, warnings=tuple(warnings),
        )

    def _explore(self, request: TaskRequest) -> ExplorationReport | None:
        context = request.context
        role = explorer_role(
            max_model_turns=max(2, min(8, context.max_model_rounds)),
            max_tool_calls=max(4, min(16, context.max_tool_actions)),
            # DelegationManager's trusted Explorer role is capped at its
            # provider-neutral 32K child context. A larger parent context must
            # not turn into an invalid child allocation.
            context_limit=min(context.context_limit, 32_768),
        )
        role.validate_registry(self.registry)
        service = ExplorationService(
            self.runtime, role=role,
            child_session_factory=(
                (lambda parent, state: self.child_session_factory(
                    AgentRole.EXPLORER, parent, state,
                )) if self.child_session_factory else None
            ),
        )
        typed = ExplorationRequest(
            context.task_id, request.objective, scope=request.project_scope,
            expected_deliverable="Locate relevant implementation, call chains, and tests",
        )
        tools = self.registry.definitions_for_model(
            role="explorer", names=role.allowed_tool_names,
        )
        generic = DelegationRequest(
            context.task_id, AgentRole.EXPLORER, request.objective,
            request.project_scope,
            ("objective", "project_rules", "selected_known_facts"),
            "ExplorationReport", context_budget=role.context_limit,
            model_turn_budget=role.max_model_turns,
            tool_call_budget=role.max_tool_calls,
        )

        def handler(child_request, child_budget):
            report = service.explore(
                typed, context, tools=tools, tool_executor=self.tool_executor,
                budget_manager=child_budget,
            )
            status = {
                ExplorationStatus.COMPLETED: DelegationStatus.COMPLETED,
                ExplorationStatus.PARTIAL: DelegationStatus.PARTIAL,
                ExplorationStatus.BUDGET_EXHAUSTED: DelegationStatus.BUDGET_EXHAUSTED,
                ExplorationStatus.CANCELLED: DelegationStatus.CANCELLED,
            }.get(report.status, DelegationStatus.FAILED)

            return DelegationResult(
                child_request.delegation_id, AgentRole.EXPLORER, status,
                report, report.evidence_references,
                DelegationUsage(report.model_calls, report.tool_calls),
                report.terminal_reason, report.uncertainties,
            )

        manager = DelegationManager()
        manager.register(AgentRole.EXPLORER, handler)

        return manager.delegate(generic, context).deliverable

    def _review(
        self, request: TaskRequest, planning: PlanningService,
        agent_result: AgentResult, verification: CompletionEvaluation,
    ) -> tuple[ReviewResult | None, AgentResult, CompletionEvaluation]:
        context = request.context
        role = reviewer_role(
            max_model_turns=max(2, min(8, context.max_model_rounds)),
            max_tool_calls=max(4, min(16, context.max_tool_actions)),
            context_limit=min(context.context_limit, 32_768),
        )
        role.validate_registry(self.registry)
        service = ReviewService(
            self.runtime, role=role,
            child_session_factory=(
                (lambda parent, state: self.child_session_factory(
                    AgentRole.REVIEWER, parent, state,
                )) if self.child_session_factory else None
            ),
        )
        tools = self.registry.definitions_for_model(
            role="reviewer", names=role.allowed_tool_names,
        )

        def delegate(current_verification):
            diff = (self.diff_provider(context) if self.diff_provider is not None
                    else DiffEvidence(
                        "unavailable", tuple(sorted(context.working_state.relevant_files)),
                        "", 0, False, limitation="no diff provider configured",
                    ))
            typed = review_request_from_parent(
                context, diff, current_verification,
                project_rules=request.project_rules,
            )
            generic = DelegationRequest(
                context.task_id, AgentRole.REVIEWER,
                "Independently review current-generation implementation",
                tuple(typed.changed_files),
                ("requirements", "grounded_diff", "verification_evidence"),
                "ReviewResult", context_budget=typed.context_limit,
                model_turn_budget=typed.max_model_turns,
                tool_call_budget=typed.max_tool_calls,
            )

            def handler(child_request, child_budget):
                result = service.review(
                    typed, context, tools=tools, tool_executor=self.tool_executor,
                    budget_manager=child_budget,
                )
                status = (DelegationStatus.COMPLETED
                          if result.status.value == "completed"
                          else DelegationStatus.BUDGET_EXHAUSTED
                          if result.status.value == "budget_exhausted"
                          else DelegationStatus.CANCELLED
                          if result.status.value == "cancelled"
                          else DelegationStatus.BLOCKED
                          if result.verdict.value == "blocked"
                          else DelegationStatus.FAILED)

                return DelegationResult(
                    child_request.delegation_id, AgentRole.REVIEWER, status,
                    result, result.evidence_references,
                    DelegationUsage(result.model_calls, result.tool_calls),
                    result.terminal_reason,
                    errors=(() if status == DelegationStatus.COMPLETED
                            else (result.terminal_reason,)),
                )

            manager = DelegationManager()
            manager.register(AgentRole.REVIEWER, handler)

            return manager.delegate(generic, context).deliverable

        review = delegate(verification)
        repair_policy = (self.review_repair_policy or ReviewRepairPolicy(
            maximum_cycles=1 if request.enable_review_repair else 0
        ))

        if (review is not None and repair_policy.allows(0, review)
                and not context.cancellation.is_cancelled
                and context.working_state.current_round < context.max_model_rounds):
            active = context.working_state.current_step

            if active is not None:
                planning.revise_step(
                    context, active.step_id, "Independent review after bounded repair",
                    "independent reviewer requested current-generation repair",
                    status=active.status if active.status != PlanStepStatus.COMPLETED
                    else PlanStepStatus.ACTIVE,
                )

            generation = context.working_state.mutation_generation
            context.conversation.append(ConversationMessage(
                "user", (TextBlock(
                    "Address the blocking findings in the current structured review "
                    "feedback. Make one bounded repair, then run meaningful verification. "
                    "Do not address non-blocking style preferences."
                ),), authored_by="harness"))
            repair = self.runtime.run(context, defer_completion=True)

            if repair.terminal_status != "failed" and repair.final_response:
                agent_result = self._merge(agent_result, repair)

            if context.working_state.mutation_generation > generation:
                planning.advance(
                    context, "mutation",
                    tuple(sorted(context.working_state.mutation_action_ids)),
                )

                if self.verification_runner is not None:
                    self._verify(context, agent_result, "review_repair")

                verification = context.verification_policy.evaluate_completion(
                    context.working_state,
                )
                self._advance_verification(planning, context, verification)
                review = delegate(verification)

        verification = context.verification_policy.evaluate_completion(
            context.working_state,
        )

        if (review is not None
                and self.review_policy.completion_allowed(
                    context.working_state, verification,
                )):
            planning.advance(context, "review", (review.review_id,))

        return review, agent_result, verification

    def _verify(
        self, context: AgentContext, result: AgentResult, source: str,
    ) -> bool | None:
        verdict = self.verification_runner(context) if self.verification_runner else None

        if verdict is not None and self.bench_observer is not None:
            self.bench_observer(context, result, verdict, source)

        return verdict

    @staticmethod
    def _advance_verification(
        planning: PlanningService, context: AgentContext,
        verification: CompletionEvaluation,
    ) -> None:
        if verification.status != CompletionVerificationStatus.VERIFIED:
            return

        evidence = tuple(
            item.verification_id for item in context.working_state.verifications
            if item.mutation_generation == context.working_state.mutation_generation
            and item.outcome.value == "passed"
        )

        if evidence:
            planning.advance(context, "verification", evidence)

    @staticmethod
    def _merge(original: AgentResult, newer: AgentResult) -> AgentResult:
        newer.transcript = (*original.transcript, *newer.transcript)
        newer.did_modify = original.did_modify or newer.did_modify

        return newer

    @staticmethod
    def _result(
        request: TaskRequest, status: TaskStatus, result: AgentResult, *,
        verification: CompletionEvaluation | None = None,
        review: ReviewResult | None = None,
        exploration: ExplorationReport | None = None,
        warnings: Sequence[str] = (),
    ) -> TaskResult:
        context = request.context
        verification = verification or context.verification_policy.evaluate_completion(
            context.working_state,
        )
        task_result = TaskResult(
            context.task_id, context.session_id, status, result.final_response,
            context.working_state, result, verification, review, exploration,
            context.checkpoint.status.value if context.checkpoint else None,
            context.budget_manager.to_dict() if context.budget_manager else {},
            result.terminal_reason.value, tuple(warnings),
        )

        if context.training_recorder is not None:
            episode = None

            try:
                review_status = None

                if review is not None:
                    review_status = getattr(getattr(review, "verdict", None), "value", None)

                episode = context.training_recorder.finalize(
                    state=context.working_state, task_status=status.value,
                    terminal_reason=result.terminal_reason.value,
                    verification_status=verification.status.value,
                    checkpoint_status=(context.checkpoint.status.value
                                       if context.checkpoint else None),
                    final_response=result.final_response,
                    review_status=review_status,
                    budget=(context.budget_manager.to_dict()
                            if context.budget_manager else {}),
                    standard_source_ids=tuple(sorted(
                        context.standard_source_ids_used)),
                )
            except Exception as exc:
                context.observer.notice("training_capture_error", {
                    "stage": "finalize", "error_category": type(exc).__name__,
                })

            if episode is not None:
                export_prohibited = bool(
                    episode.training_metadata.get("standard_export_prohibited"))

                try:
                    # Candidate artifacts are small and immutable.  Trainer
                    # JSONL materialization remains an explicit separate job.

                    if not export_prohibited:
                        from sft_dataset import SFTDatasetBuilder
                        SFTDatasetBuilder(context.training_recorder.store).persist_episode_candidates(
                            episode,
                        )
                except Exception as exc:
                    context.observer.notice("training_capture_error", {
                        "stage": "sft_derivation",
                        "error_category": type(exc).__name__,
                    })

                try:
                    # Preference pairing is deferred; this cheap update only
                    # indexes the newly observed decision contexts/outcomes.

                    if not export_prohibited:
                        from preference_dataset import PreferenceDatasetBuilder
                        PreferenceDatasetBuilder(
                            context.training_recorder.store,
                        ).index_episode(episode)
                except Exception as exc:
                    context.observer.notice("training_capture_error", {
                        "stage": "preference_derivation",
                        "error_category": type(exc).__name__,
                    })

                try:
                    from training_readiness import TrainingReadinessIndex
                    TrainingReadinessIndex(
                        context.training_recorder.store.root,
                    ).update_episode(episode)
                except Exception as exc:
                    context.observer.notice("training_capture_error", {
                        "stage": "readiness_update",
                        "error_category": type(exc).__name__,
                    })

        return task_result
