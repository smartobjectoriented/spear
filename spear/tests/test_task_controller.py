import json
import tempfile
import unittest
from pathlib import Path

from agent_runtime import AgentRuntime
from budgets import BudgetKind, BudgetLimit, BudgetManager
from cancellation import CancellationSource
from checkpoint import CheckpointManager, CheckpointStatus
from diff_evidence import DiffEvidence
from model_backend import ModelTurn, StopReason
from task_controller import TaskController, TaskRequest, TaskStatus
from tests.test_agent_runtime import (
    GroundedToolExecutor, ScriptedBackend, make_context, text_turn, tool_turn,
)
from tests.test_planning import plan_turn
from tests.test_reviewer import finding, review_json
from tests.test_orchestration import final_report
from tool_registry import ToolCategory, ToolRegistry, native_tool_specs
from verification import CompletionVerificationStatus
from working_state import StateEventType


def registry():
    result = ToolRegistry()
    for spec in native_tool_specs():
        result.register(spec, None if spec.category == ToolCategory.COMMAND
                        else lambda *_: None)
    return result


def controller(context, **kwargs):
    return TaskController(
        AgentRuntime(), registry(), tool_executor=context.tool_executor, **kwargs,
    )


class TaskControllerTests(unittest.TestCase):
    def test_default_profile_is_main_agent_only(self):
        context = make_context(ScriptedBackend([text_turn("answer")]))
        request = TaskRequest("answer", context, ("/workspace",))
        self.assertFalse(request.enable_explorer)
        self.assertFalse(request.enable_reviewer)
        self.assertFalse(request.enable_review_repair)

    def test_simple_task_runs_without_terminal_ui(self):
        context = make_context(ScriptedBackend([text_turn("answer")]))
        result = controller(context).run(TaskRequest(
            "answer a factual question", context, ("/workspace",),
            enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_response, "answer")
        self.assertFalse(result.working_state.plan_steps)

    def test_complex_task_uses_non_mutating_structured_plan(self):
        context = make_context(ScriptedBackend([plan_turn(), text_turn("done")]), rounds=8)
        result = controller(context).run(TaskRequest(
            "architectural migration across multiple modules", context,
            ("/workspace",), enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(tuple(result.working_state.plan_steps),
                         ("explore", "implement", "verify"))
        self.assertEqual(context.backend.calls[0]["tools"], ())

    def test_compatibility_mutation_boundary_is_explicit(self):
        context = make_context(ScriptedBackend([text_turn("legacy block")]))
        calls = []

        def compatibility(ctx, result):
            calls.append(ctx.task_id)
            return result

        controller(context, compatibility_mutation=compatibility).run(TaskRequest(
            "fix known.py", context, ("/workspace",),
            enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(calls, [context.task_id])

    def test_cancellation_and_budget_exhaustion_are_typed(self):
        cancelled = make_context(ScriptedBackend([text_turn()]))
        source = CancellationSource()
        source.cancel("stop")
        cancelled.cancellation = source.token
        result = controller(cancelled).run(TaskRequest(
            "answer", cancelled, ("/workspace",),
            enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(result.status, TaskStatus.INTERRUPTED)

        exhausted = make_context(ScriptedBackend([text_turn()]))
        exhausted.budget_manager = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(1),
            BudgetKind.MODEL_TURNS: BudgetLimit(2),
        })
        exhausted.budget_manager.consume(BudgetKind.PRIMARY_MODEL_CALLS)
        result = controller(exhausted).run(TaskRequest(
            "answer", exhausted, ("/workspace",),
            enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(result.status, TaskStatus.BUDGET_EXHAUSTED)

    def test_mutation_verification_reviewer_and_checkpoint_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            (workspace / "service.py").write_text("value = 1\n")
            backend = ScriptedBackend([
                tool_turn("edit", "edit_file", path="service.py",
                          old_text="1", new_text="2"),
                tool_turn("test", "bash", command="pytest"),
                text_turn("implemented"),
                ModelTurn(review_json("accept"), (), StopReason.END_TURN),
            ])
            executor = GroundedToolExecutor(["OK: edited", "1 passed"])
            context = make_context(backend, executor=executor, rounds=10, actions=12)
            context.budget_manager = BudgetManager("task", {
                BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(20),
                BudgetKind.MODEL_TURNS: BudgetLimit(20),
                BudgetKind.TOOL_CALLS: BudgetLimit(20),
                BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(2),
                BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(10),
            })
            manager = CheckpointManager(Path(directory) / "checkpoints", workspace)
            checkpoint = manager.begin_checkpoint(context.task_id, "session_controller")
            context.checkpoint_manager = manager
            context.checkpoint = checkpoint
            context.apply_state_event(
                StateEventType.CHECKPOINT_RECORDED,
                checkpoint_id=checkpoint.checkpoint_id,
                status=checkpoint.status.value,
            )

            def bench(ctx):
                evidence = ctx.verification_policy.project_bench_evidence(
                    ctx.working_state, executed=True, passed=True, action_id="bench",
                )
                AgentRuntime._record_verification(ctx, evidence)
                return True

            ctl = controller(
                context, verification_runner=bench,
                diff_provider=lambda ctx: DiffEvidence(
                    "test", ("service.py",), "diff", 4, False,
                ),
            )
            result = ctl.run(TaskRequest(
                "fix service.py behavior and test it", context, (str(workspace),),
                enable_explorer=False, enable_reviewer=True,
                enable_review_repair=True,
            ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(result.verification.status,
                             CompletionVerificationStatus.VERIFIED)
            self.assertEqual(result.review.verdict.value, "accept")
            self.assertEqual(checkpoint.status, CheckpointStatus.FINALIZED)

    def test_explorer_path_uses_isolated_child_and_returns_report(self):
        backend = ScriptedBackend([
            plan_turn(),
            tool_turn("search", "bash", command="rg service"),
            ModelTurn(final_report(), (), StopReason.END_TURN),
            text_turn("parent continued"),
        ])
        executor = GroundedToolExecutor(["service.py: def run(): pass"])
        context = make_context(backend, executor=executor, rounds=10, actions=12)
        # A parent may use a 65K context while trusted children remain capped
        # at their role limit; delegation must clamp rather than reject it.
        context.context_limit = 65_536
        context.budget_manager = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(20),
            BudgetKind.MODEL_TURNS: BudgetLimit(20),
            BudgetKind.TOOL_CALLS: BudgetLimit(20),
            BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(2),
            BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(10),
        })
        result = controller(context).run(TaskRequest(
            "map repository architecture and locate the service entry point",
            context, ("/workspace",), enable_explorer=True, enable_reviewer=False,
        ))
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(result.exploration)
        self.assertNotEqual(result.exploration.child_task_id, context.task_id)
        self.assertFalse(hasattr(result.exploration, "transcript"))

    def test_reviewer_requests_one_bounded_repair_then_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            (workspace / "service.py").write_text("value = 1\n")
            backend = ScriptedBackend([
                tool_turn("edit1", "edit_file", path="service.py",
                          old_text="1", new_text="2"),
                tool_turn("test1", "bash", command="pytest"),
                text_turn("implemented"),
                ModelTurn(review_json("repair_required", (finding(
                    "functional_regression", "requirement mismatch",
                    path="service.py",
                ),)), (), StopReason.END_TURN),
                tool_turn("edit2", "edit_file", path="service.py",
                          old_text="2", new_text="3"),
                tool_turn("test2", "bash", command="pytest"),
                text_turn("repaired"),
                ModelTurn(review_json("accept"), (), StopReason.END_TURN),
            ])
            executor = GroundedToolExecutor([
                "OK: edit one", "1 passed", "OK: edit two", "1 passed",
            ])
            context = make_context(backend, executor=executor, rounds=14, actions=20)
            context.budget_manager = BudgetManager("task", {
                BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(30),
                BudgetKind.MODEL_TURNS: BudgetLimit(30),
                BudgetKind.TOOL_CALLS: BudgetLimit(30),
                BudgetKind.CHILD_AGENT_CALLS: BudgetLimit(3),
                BudgetKind.CHILD_MODEL_TURNS: BudgetLimit(15),
            })
            manager = CheckpointManager(Path(directory) / "checkpoints", workspace)
            checkpoint = manager.begin_checkpoint(context.task_id, "session_repair")
            context.checkpoint_manager = manager
            context.checkpoint = checkpoint
            context.apply_state_event(
                StateEventType.CHECKPOINT_RECORDED,
                checkpoint_id=checkpoint.checkpoint_id,
                status=checkpoint.status.value,
            )

            def bench(ctx):
                evidence = ctx.verification_policy.project_bench_evidence(
                    ctx.working_state, executed=True, passed=True,
                    action_id=f"bench_{ctx.working_state.mutation_generation}",
                )
                AgentRuntime._record_verification(ctx, evidence)
                return True

            result = controller(
                context, verification_runner=bench,
                diff_provider=lambda ctx: DiffEvidence(
                    "test", ("service.py",), "diff", 4, False,
                ),
            ).run(TaskRequest(
                "fix service.py behavior and test it", context, (str(workspace),),
                enable_explorer=False, enable_reviewer=True,
                enable_review_repair=True,
            ))
            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(result.review.verdict.value, "accept")
            self.assertEqual(context.working_state.mutation_generation, 2)
            self.assertEqual(len(context.review_results), 2)

    def test_resumed_context_keeps_identity_and_budget(self):
        context = make_context(ScriptedBackend([text_turn("continued")]),
                               task_id="task_controller_resume")
        context.budget_manager = BudgetManager("task", {
            BudgetKind.PRIMARY_MODEL_CALLS: BudgetLimit(5),
            BudgetKind.MODEL_TURNS: BudgetLimit(5),
        })
        context.budget_manager.consume(BudgetKind.PRIMARY_MODEL_CALLS, 2)
        result = controller(context).run(TaskRequest(
            "continue known.py", context, ("/workspace",),
            enable_explorer=False, enable_reviewer=False,
        ))
        self.assertEqual(result.task_id, "task_controller_resume")
        consumed = result.budget["consumed"]
        self.assertEqual(consumed[BudgetKind.PRIMARY_MODEL_CALLS.value], 3)


class ReadOnlyScopeTests(unittest.TestCase):
    """The boundary is enforced; the model still has to know its mission.

    Told to name the file that defines `add`, a run reached for `sed -i` at
    step four to fix a bug nobody asked about, then spent thirty-five more
    steps looking for a syntax that would get through.
    """

    def test_the_rule_is_stated_and_derived_from_the_view(self):
        from tool_exposure import READ_ONLY_RULE, READ_ONLY_RULE_ID, ToolExposurePolicy

        self.assertTrue(ToolExposurePolicy.read_only_intent(
            "Answer which file defines add without editing files."))
        self.assertIn("READ-ONLY TASK", READ_ONLY_RULE)
        self.assertIn("Do not edit, repair, compile, or run tests", READ_ONLY_RULE)
        self.assertIn("answer immediately", READ_ONLY_RULE)
        self.assertEqual(READ_ONLY_RULE_ID, "task:read_only")

        # One definition, used by the controller and inherited by the child
        # roles -- never re-inferred from the wording a second time.

        controller = Path(__file__).resolve().parents[1] / "task_controller.py"
        orchestration = Path(__file__).resolve().parents[1] / "orchestration.py"
        self.assertIn("READ_ONLY_RULE_ID", controller.read_text())
        self.assertIn("view.read_only", controller.read_text())
        self.assertIn("READ_ONLY_RULE_ID", orchestration.read_text())

    def test_the_explorer_inherits_the_scope_rule(self):
        from agent_runtime import AgentContext
        from context_engine import ContextItem, ContextLayer, Freshness
        from orchestration import ExplorationRequest, ExplorationService
        from tool_exposure import READ_ONLY_RULE, READ_ONLY_RULE_ID

        rule = ContextItem(READ_ONLY_RULE_ID, ContextLayer.SYSTEM_RULES,
                           "tool_exposure", READ_ONLY_RULE, 100,
                           Freshness.CURRENT, True)
        other = ContextItem("noise", ContextLayer.RECENT_CONVERSATION,
                            "conversation", "chatter")

        class Parent:
            context_items = (rule, other)

        items = ExplorationService.__dict__["_context_items"](
            type("S", (), {"role": type("R", (), {"prompt_addition": "explore"})()})(),
            ExplorationRequest("task", "look around"), Parent())
        identifiers = [item.item_id for item in items]

        self.assertIn(READ_ONLY_RULE_ID, identifiers)
        self.assertNotIn("noise", identifiers)


if __name__ == "__main__":
    unittest.main()
