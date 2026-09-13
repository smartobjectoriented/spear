import json
import tempfile
import unittest

from compaction import StructuredCompactionState
from model_backend import ModelTurn, StopReason
from planning import PlanStepDefinition, PlanningPolicy, PlanningService
from session_store import FileSessionStore, SessionConfiguration, SessionSnapshot
from tests.test_agent_runtime import ScriptedBackend, make_context
from working_state import (
    ActionKind, PlanStepStatus, StateEvent, StateEventType, StateSource,
    StateTransitionError, WorkingState,
)


def plan_turn():
    return ModelTurn(json.dumps({"steps": [
        {"step_id": "explore", "objective": "Locate implementation and tests",
         "dependencies": [], "completion_criteria": ["relevant paths found"],
         "requires_evidence": True},
        {"step_id": "implement", "objective": "Implement the multi-file change",
         "dependencies": ["explore"], "completion_criteria": ["files changed"],
         "requires_evidence": True},
        {"step_id": "verify", "objective": "Run verification and review",
         "dependencies": ["implement"], "completion_criteria": ["tests and review pass"],
         "requires_evidence": True},
    ]}), (), StopReason.END_TURN, usage={"input_tokens": 20, "output_tokens": 30})


class PlanningTests(unittest.TestCase):
    def test_policy_skips_simple_and_plans_complex(self):
        policy = PlanningPolicy()
        self.assertFalse(policy.decide("fix src/a.py typo").should_plan)
        self.assertTrue(policy.decide("architectural multi-file migration").should_plan)
        self.assertTrue(policy.decide("change behavior", acceptance_criteria=("a", "b")).should_plan)
        self.assertTrue(policy.decide("locate it", exploration_required=True).should_plan)

    def test_planning_call_is_non_mutating_and_creates_working_state_plan(self):
        context = make_context(ScriptedBackend([plan_turn()]))
        result = PlanningService().create(context)
        self.assertTrue(result.created)
        self.assertEqual(tuple(context.working_state.plan_steps),
                         ("explore", "implement", "verify"))
        self.assertFalse(context.working_state.modified_files)
        call = context.backend.calls[0]
        self.assertFalse(call["use_tools"])
        self.assertEqual(call["tools"], ())

    def test_grounded_completion_and_model_prose_rejection(self):
        context = make_context(ScriptedBackend([]))
        PlanningService.apply(context, (PlanStepDefinition(
            "change", "Implement change", completion_criteria=("edit",),
        ),))
        PlanningService.start_step(context, "change")
        with self.assertRaises(StateTransitionError):
            context.apply_state_event(
                StateEventType.PLAN_STEP_UPDATED, source=StateSource.MODEL,
                step_id="change", status="completed", evidence_action_ids=("claimed",),
            )
        context.apply_state_event(
            StateEventType.ACTION_SUCCEEDED, source=StateSource.TOOL_RUNTIME,
            action_id="edit_grounded", kind=ActionKind.TOOL.value,
            name="edit_file", observed_status="ok", round_number=1,
        )
        PlanningService.complete_step(context, "change", ("edit_grounded",))
        self.assertEqual(context.working_state.plan_steps["change"].status,
                         PlanStepStatus.COMPLETED)

    def test_failed_action_cannot_complete_plan_step(self):
        context = make_context(ScriptedBackend([]))
        PlanningService.apply(context, (PlanStepDefinition("change", "Implement"),))
        PlanningService.start_step(context, "change")
        context.apply_state_event(
            StateEventType.ACTION_FAILED, source=StateSource.TOOL_RUNTIME,
            action_id="failed_change", kind=ActionKind.TOOL.value,
            name="edit_file", observed_status="failed", category="tool_error",
            summary="failed", round_number=1,
        )
        with self.assertRaises(StateTransitionError):
            PlanningService.complete_step(context, "change", ("failed_change",))

    def test_dependencies_block_early_start_and_blocked_step(self):
        context = make_context(ScriptedBackend([]))
        PlanningService.apply(context, (
            PlanStepDefinition("one", "First", requires_evidence=False),
            PlanStepDefinition("two", "Second", dependencies=("one",)),
        ))
        with self.assertRaises(StateTransitionError):
            PlanningService.start_step(context, "two")
        PlanningService.start_step(context, "one")
        context.apply_state_event(StateEventType.PLAN_STEP_UPDATED,
                                  step_id="one", status="blocked")
        self.assertEqual(context.working_state.blocked_steps[0].step_id, "one")

    def test_revision_preserves_history_and_failed_approach(self):
        context = make_context(ScriptedBackend([]))
        PlanningService.apply(context, (PlanStepDefinition(
            "change", "Original approach", requires_evidence=False,
        ),))
        PlanningService.start_step(context, "change")
        context.apply_state_event(
            StateEventType.ACTION_FAILED, source=StateSource.TOOL_RUNTIME,
            action_id="failed_edit", kind=ActionKind.TOOL.value,
            name="edit_file", observed_status="failed", category="tool_error",
            summary="approach failed", round_number=1,
        )
        PlanningService.revise_step(
            context, "change", "Use grounded alternative", "new evidence",
            status=PlanStepStatus.ACTIVE,
        )
        self.assertEqual(context.working_state.plan_revisions[0].prior_description,
                         "Original approach")
        self.assertEqual(len(context.working_state.unresolved_failures), 1)

    def test_explorer_evidence_and_reviewer_repair_support_plan_revision(self):
        context = make_context(ScriptedBackend([]))
        PlanningService.apply(context, (
            PlanStepDefinition("explore", "Locate implementation"),
            PlanStepDefinition("review", "Review final change", dependencies=("explore",)),
        ))
        PlanningService.start_step(context, "explore")
        context.apply_state_event(
            StateEventType.DISCOVERY_RECORDED, source=StateSource.RETRIEVAL,
            summary="found", file_path="src.py", action_id="explore_report",
        )
        self.assertTrue(PlanningService.advance(
            context, "exploration", ("explore_report",)))
        PlanningService.revise_step(
            context, "review", "Repair grounded blocking review findings",
            "reviewer requested repair", status=PlanStepStatus.ACTIVE,
        )
        self.assertEqual(context.working_state.current_step.step_id, "review")

    def test_plan_survives_serialization_compaction_and_session_snapshot(self):
        state = WorkingState.start("task_planpersist", "complex migration")
        state.apply(StateEvent.create(
            StateEventType.PLAN_STEP_ADDED, state.task_id, StateSource.HARNESS,
            step_id="one", description="Inspect", dependencies=(),
            completion_criteria=("found",), requires_evidence=True,
        ))
        restored = WorkingState.from_json(state.to_json())
        self.assertEqual(restored.plan_steps["one"].completion_criteria, ("found",))
        compact = StructuredCompactionState.from_working_state(restored)
        self.assertEqual(compact.plan[0].step_id, "one")
        with tempfile.TemporaryDirectory() as directory:
            store = FileSessionStore(directory)
            config = SessionConfiguration("/workspace", "project", "safe")
            snapshot = SessionSnapshot(
                "session_planpersist", state.task_id, 0, config, restored, (),
            )
            store.save_snapshot(snapshot)
            loaded = store.load_snapshot(snapshot.session_id)
            self.assertIn("one", loaded.working_state.plan_steps)


if __name__ == "__main__":
    unittest.main()
