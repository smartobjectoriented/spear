import json
import unittest

from agent_roles import AgentRole, AgentRoleSpec, explorer_role
from agent_runtime import AgentRuntime
from budgets import BudgetKind, BudgetLimit, BudgetManager
from model_backend import ModelTurn, StopReason
from orchestration import (
    DelegationManager, DelegationRequest, DelegationResult, DelegationStatus,
    DelegationUsage, ExplorationRequest, ExplorationService,
)
from reviewer import ReviewService
from tests.test_agent_runtime import ScriptedBackend, make_context
from tests.test_orchestration import ReadExecutor
from tests.test_reviewer import (
    ReadExecutor as ReviewReadExecutor, add_verification, parent_context,
    request as review_request, review_json,
)


def delegated(role=AgentRole.EXPLORER, **changes):
    values = dict(
        parent_task_id="task_runtime", role=role, objective="inspect",
        scope=(".",), context_manifest=("objective",),
        expected_deliverable_type=("ExplorationReport" if role == AgentRole.EXPLORER
                                   else "ReviewResult"),
        context_budget=4096, model_turn_budget=4, tool_call_budget=4,
    )
    values.update(changes)
    return DelegationRequest(**values)


class DelegationTests(unittest.TestCase):
    def test_trusted_roles_only_and_recursion_denied(self):
        with self.assertRaises(ValueError):
            delegated(role="implementation")
        with self.assertRaises(ValueError):
            delegated(recursion_allowed=True)

    def test_request_cannot_override_execution_or_model_policy(self):
        with self.assertRaises(ValueError):
            delegated(execution_mode="auto")
        parent = make_context(ScriptedBackend([]))
        manager = DelegationManager()
        manager.register(AgentRole.EXPLORER, lambda request, budget: None)
        with self.assertRaises(ValueError):
            manager.delegate(delegated(model_override="untrusted"), parent)
        with self.assertRaises(ValueError):
            manager.delegate(delegated(model_turn_budget=999), parent)
        with self.assertRaises(ValueError):
            manager.delegate(delegated(tool_policy="model_defined"), parent)

    def test_generic_result_has_deliverable_not_transcript_and_is_persisted(self):
        parent = make_context(ScriptedBackend([]))
        parent.budget_manager = BudgetManager("task", {
            BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(1),
            BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(4),
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(4),
            BudgetKind.TOOL_CALLS: BudgetLimit(4),
        })
        manager = DelegationManager()
        report = {"summary": "bounded"}
        manager.register(AgentRole.EXPLORER, lambda request, budget: DelegationResult(
            request.delegation_id, request.role, DelegationStatus.COMPLETED,
            report, ("evidence",), DelegationUsage(1, 1), "completed",
        ))
        result = manager.delegate(delegated(), parent)
        self.assertIs(result.deliverable, report)
        self.assertFalse(hasattr(result, "transcript"))
        self.assertEqual(parent.delegation_results[0]["role"], "explorer")

    def test_child_budget_exhaustion_is_typed_partial_boundary(self):
        parent = make_context(ScriptedBackend([]))
        parent.budget_manager = BudgetManager("task", {
            BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(1),
            BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(2),
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(2),
            BudgetKind.TOOL_CALLS: BudgetLimit(2),
        })
        manager = DelegationManager()
        def exhaust(request, budget):
            budget.consume(BudgetKind.TOOL_CALLS, 3)
        manager.register(AgentRole.EXPLORER, exhaust)
        result = manager.delegate(delegated(tool_call_budget=2), parent)
        self.assertEqual(result.status, DelegationStatus.BUDGET_EXHAUSTED)
        self.assertIsNone(result.deliverable)

    def test_explorer_runs_through_generic_delegation(self):
        payload = json.dumps({"summary": "found", "relevant_files": ["main.py"]})
        parent = make_context(ScriptedBackend([ModelTurn(payload, (), StopReason.END_TURN)]))
        service = ExplorationService(AgentRuntime(), role=explorer_role())
        typed = ExplorationRequest(parent.task_id, "locate entry point")
        manager = DelegationManager()
        def handler(generic, budget):
            report = service.explore(
                typed, parent, tools=(), tool_executor=ReadExecutor(),
                budget_manager=budget,
            )
            return DelegationResult(
                generic.delegation_id, generic.role, DelegationStatus.COMPLETED,
                report, report.evidence_references,
                DelegationUsage(report.model_calls, report.tool_calls),
                report.terminal_reason,
            )
        manager.register(AgentRole.EXPLORER, handler)
        result = manager.delegate(delegated(), parent)
        self.assertEqual(result.deliverable.summary, "found")
        self.assertIsNot(result.deliverable.child_task_id, parent.task_id)

    def test_reviewer_runs_through_generic_delegation(self):
        parent = parent_context(ScriptedBackend([
            ModelTurn(review_json(), (), StopReason.END_TURN),
        ]))
        add_verification(parent.working_state)
        typed = review_request(parent)
        service = ReviewService(AgentRuntime())
        manager = DelegationManager()
        def handler(generic, budget):
            result = service.review(
                typed, parent, tools=(), tool_executor=ReviewReadExecutor(),
                budget_manager=budget,
            )
            return DelegationResult(
                generic.delegation_id, generic.role, DelegationStatus.COMPLETED,
                result, result.evidence_references,
                DelegationUsage(result.model_calls, result.tool_calls),
                result.terminal_reason,
            )
        manager.register(AgentRole.REVIEWER, handler)
        result = manager.delegate(delegated(
            role=AgentRole.REVIEWER, parent_task_id=parent.task_id,
        ), parent)
        self.assertEqual(result.deliverable.verdict.value, "accept")
        self.assertFalse(hasattr(result, "transcript"))
        self.assertGreater(parent.budget_manager.consumed[BudgetKind.PRIMARY_MODEL_CALLS], 0)
        self.assertGreater(parent.budget_manager.consumed[BudgetKind.CHILD_MODEL_TURNS], 0)


if __name__ == "__main__":
    unittest.main()
