import json
import tempfile
import unittest
from pathlib import Path

from agent_runtime import AgentContext, AgentRuntime
from budgets import BudgetKind, BudgetLimit, BudgetManager
from checkpoint import CheckpointManager, CheckpointStatus, MutationType
from context_engine import ContextEngine
from diff_evidence import DiffEvidenceService
from model_backend import ModelTurn, StopReason
from orchestration import (
    ExplorationReport, ExplorationRequest, ExplorationService, ExplorationStatus,
)
from planning import PlanStepDefinition, PlanningService
from reviewer import ReviewService, review_request_from_parent
from session_store import (
    FileSessionStore, SessionConfiguration, SessionHandle, SessionSnapshot,
)
from tests.test_agent_runtime import (
    GroundedToolExecutor, ScriptedBackend, make_context, text_turn, tool_turn,
)
from tests.test_reviewer import ReadExecutor as ReviewExecutor, finding, review_json
from tracing import TraceEmitter
from verification import CompletionVerificationStatus, VerificationPolicy
from working_state import PlanStepStatus, StateEventType, StateSource


def task_budget():
    return BudgetManager("task", {
        BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(30),
        BudgetKind.AUXILIARY_MODEL_CALLS: BudgetLimit(3),
        BudgetKind.MODEL_TURNS: BudgetLimit(12),
        BudgetKind.TOOL_CALLS: BudgetLimit(20),
        BudgetKind.COMMAND_EXECUTIONS: BudgetLimit(10),
        BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(3),
        BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(12),
        BudgetKind.INPUT_TOKENS: BudgetLimit(100_000, True),
        BudgetKind.OUTPUT_TOKENS: BudgetLimit(50_000, True),
    })


class FullOrchestrationFlowTests(unittest.TestCase):
    def test_plan_explore_implement_verify_repair_review_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            target = root / "service.py"
            target.write_text("value = 1\n")
            checkpoint_manager = CheckpointManager(Path(directory) / "checkpoints", root)
            backend = ScriptedBackend([
                tool_turn("edit1", "edit_file", path="service.py",
                          old_text="1", new_text="2"),
                tool_turn("test1", "bash", command="pytest"),
                text_turn("implemented"),
            ])
            executor = GroundedToolExecutor(["OK: edited", "1 passed"])
            context = make_context(
                backend, rounds=12, actions=20,
                executor=executor,
            )
            context.budget_manager = task_budget()
            checkpoint = checkpoint_manager.begin_checkpoint(
                context.task_id, "session_flow1234",
            )
            checkpoint_manager.capture(checkpoint, "service.py")
            context.checkpoint_manager = checkpoint_manager
            context.checkpoint = checkpoint
            context.apply_state_event(
                StateEventType.CHECKPOINT_RECORDED,
                checkpoint_id=checkpoint.checkpoint_id,
                status=checkpoint.status.value,
            )
            planning = PlanningService()
            planning.apply(context, (
                PlanStepDefinition("explore", "Locate implementation"),
                PlanStepDefinition("implement", "Implement change", ("explore",)),
                PlanStepDefinition("verify", "Run tests", ("implement",)),
                PlanStepDefinition("review", "Independent review", ("verify",)),
            ))
            planning.start_next(context)
            context.apply_state_event(
                StateEventType.DISCOVERY_RECORDED, source=StateSource.RETRIEVAL,
                summary="located",
                file_path="service.py", action_id="explore_grounded",
            )
            planning.advance(context, "exploration", ("explore_grounded",))

            runtime = AgentRuntime()
            first = runtime.run(context, defer_completion=True)
            target.write_text("value = 2\n")
            checkpoint_manager.record_mutation(
                checkpoint, "service.py", action_id="tool_1_task_runtime",
                mutation_type=MutationType.MODIFIED,
            )
            planning.advance(context, "mutation",
                             tuple(context.working_state.mutation_action_ids))
            evaluation = VerificationPolicy().evaluate_completion(context.working_state)
            self.assertEqual(evaluation.status, CompletionVerificationStatus.VERIFIED)
            planning.advance(context, "verification", tuple(
                item.verification_id for item in context.working_state.verifications
                if item.mutation_generation == context.working_state.mutation_generation
            ))

            context.backend = ScriptedBackend([ModelTurn(review_json(
                "repair_required", (finding(
                    "functional_regression", "Explicit requirement is not met.",
                    path="service.py", requirement=None,
                ),)), (), StopReason.END_TURN)])
            reviewer = ReviewService(runtime)
            diff_service = DiffEvidenceService()
            request = review_request_from_parent(
                context, diff_service.from_checkpoint(checkpoint_manager, checkpoint),
                evaluation,
            )
            rejected = reviewer.review(
                request, context, tools=(), tool_executor=ReviewExecutor(),
            )
            self.assertEqual(rejected.verdict.value, "repair_required")
            active = context.working_state.current_step
            planning.revise_step(
                context, active.step_id, "Independent review after bounded repair",
                "reviewer requested repair", status=PlanStepStatus.ACTIVE,
            )

            context.backend = ScriptedBackend([
                tool_turn("edit2", "edit_file", path="service.py",
                          old_text="2", new_text="3"),
                tool_turn("test2", "bash", command="pytest"),
                text_turn("repaired"),
            ])
            executor.outputs.extend(["OK: repaired", "1 passed"])
            context.tool_executor = executor
            repaired = runtime.run(context, defer_completion=True)
            target.write_text("value = 3\n")
            checkpoint_manager.record_mutation(
                checkpoint, "service.py", action_id="tool_3_task_runtime",
                mutation_type=MutationType.MODIFIED,
            )
            evaluation = VerificationPolicy().evaluate_completion(context.working_state)
            self.assertEqual(evaluation.status, CompletionVerificationStatus.VERIFIED)
            context.backend = ScriptedBackend([
                ModelTurn(review_json("accept"), (), StopReason.END_TURN),
            ])
            accepted = reviewer.review(
                review_request_from_parent(
                    context,
                    diff_service.from_checkpoint(checkpoint_manager, checkpoint),
                    evaluation,
                ), context, tools=(), tool_executor=ReviewExecutor(),
            )
            self.assertEqual(accepted.verdict.value, "accept")
            planning.advance(context, "review", (accepted.review_id,))
            context.review_required = True
            runtime.complete(context, repaired)
            self.assertEqual(checkpoint.status, CheckpointStatus.FINALIZED)
            self.assertTrue(all(step.status == PlanStepStatus.COMPLETED
                                for step in context.working_state.plan_steps.values()))
            self.assertEqual(context.working_state.mutation_generation, 2)
            self.assertTrue(context.working_state.current_review_accepted)
            self.assertFalse(hasattr(accepted, "transcript"))

    def test_resume_restores_plan_budget_delegation_checkpoint_and_child_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root / "a.py").write_text("a\n")
            checkpoint_manager = CheckpointManager(Path(directory) / "checkpoints", root)
            context = make_context(ScriptedBackend([]), task_id="task_resumeflow")
            PlanningService.apply(context, (PlanStepDefinition(
                "explore", "Locate code", requires_evidence=True,
            ),))
            checkpoint = checkpoint_manager.begin_checkpoint(
                context.task_id, "session_resumeflow",
            )
            context.checkpoint = checkpoint
            context.checkpoint_manager = checkpoint_manager
            context.budget_manager = task_budget()
            context.budget_manager.consume(BudgetKind.MODEL_TURNS, 2)
            report = ExplorationReport(
                "explore_resume", context.task_id, "task_childresume", None,
                "initial_repository_exploration", ExplorationStatus.COMPLETED,
                "found", relevant_files=("a.py",), files_inspected=("a.py",),
            )
            context.exploration_reports.append(report.to_dict())
            context.delegation_results.append({
                "delegation_id": "delegate_resume", "role": "explorer",
                "status": "completed",
            })
            store = FileSessionStore(Path(directory) / "sessions")
            config = SessionConfiguration(str(root), "project", "safe")
            handle = SessionHandle(store, checkpoint.session_id, config)
            snapshot = SessionSnapshot(
                handle.session_id, context.task_id, 0, config,
                context.working_state, tuple(context.conversation),
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_status=checkpoint.status.value,
                exploration_reports=tuple(context.exploration_reports),
                budget_state=context.budget_manager.to_dict(),
                delegation_results=tuple(context.delegation_results),
            )
            store.save_snapshot(snapshot)
            loaded = store.load_snapshot(handle.session_id)
            resumed = AgentContext.from_session_snapshot(
                loaded, session=handle, backend=ScriptedBackend([]),
                context_engine=ContextEngine(), trace=TraceEmitter(), tools=(),
                tool_executor=GroundedToolExecutor(), context_limit=4096,
                checkpoint_manager=checkpoint_manager,
            )
            existing = ExplorationService(AgentRuntime()).explore(
                ExplorationRequest(context.task_id, "locate"), resumed,
                tools=(), tool_executor=GroundedToolExecutor(),
            )
            self.assertEqual(existing.exploration_id, report.exploration_id)
            self.assertEqual(resumed.budget_manager.consumed[BudgetKind.MODEL_TURNS], 2)
            self.assertIn("explore", resumed.working_state.plan_steps)
            self.assertEqual(resumed.checkpoint.checkpoint_id, checkpoint.checkpoint_id)
            self.assertEqual(len(resumed.delegation_results), 1)


if __name__ == "__main__":
    unittest.main()
