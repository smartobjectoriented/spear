import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracing import EventType, TraceEmitter
from agent_runtime import AgentContext, AgentRuntime
from compaction import CompactionPolicy
from context_engine import ContextEngine, ContextItem, ContextLayer
from model_backend import ConversationMessage, TextBlock

from working_state import (
    ActionKind,
    ActionStatus,
    DuplicateEventError,
    PlanStepStatus,
    SerializationError,
    StateEvent,
    StateEventType,
    StateSource,
    StateTransitionError,
    TerminalStatus,
    VerificationOutcome,
    WorkingState,
    replay_state,
)


def event(event_type, task_id="task_test", source=StateSource.HARNESS, **data):
    return StateEvent.create(event_type, task_id, source, **data)


def started(task_id="task_test", objective="Implement the requested change"):
    return WorkingState.start(task_id, objective, max_model_rounds=30, max_tool_actions=60)


def succeeded_action(action_id="action_1", kind=ActionKind.TOOL, name="read"):
    return event(
        StateEventType.ACTION_SUCCEEDED,
        source=StateSource.TOOL_RUNTIME if kind != ActionKind.MODEL else StateSource.HARNESS,
        action_id=action_id,
        kind=kind.value,
        name=name,
        observed_status="ok",
        output_chars=10,
        round_number=1,
    )


class WorkingStateCoreTests(unittest.TestCase):
    def test_empty_and_started_state(self):
        empty = WorkingState("task_empty")
        self.assertEqual(empty.terminal_status, TerminalStatus.RUNNING)
        self.assertFalse(empty.started)
        self.assertEqual(empty.actions, [])

        state = started()
        self.assertTrue(state.started)
        self.assertEqual(state.objective, "Implement the requested change")
        self.assertEqual(state.max_model_rounds, 30)
        self.assertEqual(state.max_tool_actions, 60)

    def test_working_state_owns_and_enforces_task_id(self):
        state = started()
        self.assertEqual(state.task_id, "task_test")
        with self.assertRaises(StateTransitionError):
            state.apply(event(StateEventType.RETRY_RECORDED, task_id="another"))

    def test_event_payload_is_detached_from_mutable_caller_data(self):
        evidence = ["action_1"]
        transition = event(
            StateEventType.PLAN_STEP_ADDED,
            step_id="inspect", description="Inspect", evidence=evidence,
        )
        evidence.append("action_2")
        self.assertEqual(transition.data["evidence"], ["action_1"])

    def test_rejected_transition_does_not_partially_mutate_state(self):
        state = WorkingState("task_test")
        invalid = event(
            StateEventType.TASK_STARTED,
            source=StateSource.USER,
            objective="objective",
            max_model_rounds=0,
            max_tool_actions=60,
        )
        with self.assertRaises(StateTransitionError):
            state.apply(invalid)
        self.assertFalse(state.started)
        self.assertEqual(state.objective, "")
        self.assertEqual(state.events, [])

    def test_objective_constraints_and_acceptance_criteria(self):
        state = started()
        state.apply(event(
            StateEventType.OBJECTIVE_RECORDED, source=StateSource.USER,
            objective="Refine the objective",
        ))
        state.apply(event(
            StateEventType.CONSTRAINT_RECORDED, source=StateSource.USER,
            constraint="Remain provider-neutral",
        ))
        state.apply(event(
            StateEventType.ACCEPTANCE_CRITERION_RECORDED, source=StateSource.USER,
            criterion="All tests pass",
        ))
        self.assertEqual(state.objective, "Refine the objective")
        self.assertEqual(state.user_constraints, ["Remain provider-neutral"])
        self.assertEqual(state.acceptance_criteria, ["All tests pass"])

    def test_grounded_file_read_transition(self):
        state = started()
        state.apply(succeeded_action())
        state.apply(event(
            StateEventType.FILE_READ, source=StateSource.TOOL_RUNTIME, path="src/main.py",
            action_id="action_1",
        ))
        self.assertEqual(state.files_read, {"src/main.py"})
        self.assertIn("src/main.py", state.relevant_files)

    def test_grounded_file_modification_and_creation_transitions(self):
        state = started()
        state.apply(succeeded_action("edit_1", name="edit_file"))
        state.apply(event(
            StateEventType.FILE_MODIFIED, source=StateSource.TOOL_RUNTIME, path="src/main.py",
            action_id="edit_1",
        ))
        state.apply(succeeded_action("write_1", name="write_file"))
        state.apply(event(
            StateEventType.FILE_CREATED, source=StateSource.TOOL_RUNTIME, path="src/new.py",
            action_id="write_1",
        ))
        self.assertEqual(state.modified_files, {"src/main.py"})
        self.assertEqual(state.created_files, {"src/new.py"})

    def test_assistant_prose_cannot_create_grounded_file_state(self):
        state = started()
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.FILE_MODIFIED,
                source=StateSource.MODEL,
                path="claimed.py",
                action_id="model_claim",
                assistant_text="I modified claimed.py",
            ))
        self.assertEqual(state.modified_files, set())
        self.assertEqual(len(state.events), 1)

    def test_file_mutation_requires_a_successful_recorded_tool_action(self):
        state = started()
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.FILE_MODIFIED, source=StateSource.TOOL_RUNTIME,
                path="unobserved.py", action_id="missing",
            ))
        state.apply(event(
            StateEventType.ACTION_FAILED, source=StateSource.TOOL_RUNTIME,
            action_id="failed_edit", kind=ActionKind.TOOL.value, name="edit_file",
            observed_status="failed", summary="no match",
        ))
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.FILE_MODIFIED, source=StateSource.TOOL_RUNTIME,
                path="still_unmodified.py", action_id="failed_edit",
            ))
        self.assertEqual(state.modified_files, set())

    def test_successful_action_updates_grounded_counters(self):
        state = started()
        state.apply(succeeded_action())
        self.assertEqual(state.actions[-1].status, ActionStatus.SUCCEEDED)
        self.assertEqual(state.tool_actions, 1)
        self.assertEqual(state.tool_budget_used, 1)
        self.assertEqual(state.unresolved_failures, ())

    def test_failed_action_cannot_be_recorded_as_successful(self):
        state = started()
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.ACTION_SUCCEEDED,
                source=StateSource.TOOL_RUNTIME,
                action_id="command_1", kind=ActionKind.COMMAND.value, name="make",
                observed_status="failed",
            ))
        self.assertEqual(state.actions, [])

        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.ACTION_SUCCEEDED,
                source=StateSource.TOOL_RUNTIME,
                action_id="command_nonzero", kind=ActionKind.COMMAND.value, name="make",
                observed_status="ok", exit_code=2,
            ))
        self.assertEqual(state.actions, [])

        state.apply(event(
            StateEventType.ACTION_FAILED,
            source=StateSource.TOOL_RUNTIME,
            action_id="command_1", kind=ActionKind.COMMAND.value, name="make",
            observed_status="failed", category="command_failed", summary="exit 2",
            exit_code=2,
        ))
        self.assertEqual(state.actions[-1].status, ActionStatus.FAILED)
        self.assertEqual(state.actions[-1].exit_code, 2)
        self.assertEqual(state.command_actions, 1)
        self.assertEqual(len(state.unresolved_failures), 1)

    def test_failure_remains_unresolved_until_explicit_resolution(self):
        state = started()
        state.apply(event(
            StateEventType.ACTION_FAILED, source=StateSource.TOOL_RUNTIME,
            action_id="tool_1", kind=ActionKind.TOOL.value, name="edit_file",
            observed_status="failed", summary="exact match absent",
        ))
        failure_id = "failure:tool_1"
        self.assertFalse(state.failures[failure_id].resolved)
        state.apply(event(StateEventType.RETRY_RECORDED, reason="adjusted input"))
        self.assertFalse(state.failures[failure_id].resolved)
        state.apply(event(StateEventType.FAILURE_RESOLVED, failure_id=failure_id))
        self.assertTrue(state.failures[failure_id].resolved)


class VerificationAndPlanTests(unittest.TestCase):
    def test_verification_not_run_is_truthful(self):
        state = started()
        state.apply(event(
            StateEventType.VERIFICATION_RECORDED,
            verification_id="verify_1", kind="existing_turn_rule",
            executed=False, outcome=VerificationOutcome.NOT_RUN.value,
        ))
        self.assertEqual(state.verification_outcome, VerificationOutcome.NOT_RUN)

    def test_failed_verification_remains_visible(self):
        state = started()
        state.apply(event(
            StateEventType.VERIFICATION_RECORDED,
            verification_id="bench_1", kind="project_bench",
            executed=True, outcome=VerificationOutcome.FAILED.value,
            summary="tests failed",
        ))
        self.assertEqual(state.verification_outcome, VerificationOutcome.FAILED)
        self.assertIn("verification:bench_1", state.failures)
        self.assertFalse(state.failures["verification:bench_1"].resolved)

    def test_verification_pass_requires_execution_and_grounded_source(self):
        state = started()
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.VERIFICATION_RECORDED,
                verification_id="verify_false", kind="claim", executed=False,
                outcome=VerificationOutcome.PASSED.value,
            ))
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.VERIFICATION_RECORDED,
                source=StateSource.MODEL,
                verification_id="verify_claim", kind="claim", executed=True,
                outcome=VerificationOutcome.PASSED.value,
            ))
        state.apply(event(
            StateEventType.VERIFICATION_RECORDED,
            verification_id="verify_real", kind="project_bench", executed=True,
            outcome=VerificationOutcome.PASSED.value,
        ))
        self.assertEqual(state.verification_outcome, VerificationOutcome.PASSED)
        self.assertEqual(len(state.verifications), 1)

    def test_failed_verification_cannot_silently_disappear_after_later_pass(self):
        state = started()
        state.apply(event(
            StateEventType.VERIFICATION_RECORDED,
            verification_id="bench_fail", kind="project_bench", executed=True,
            outcome=VerificationOutcome.FAILED.value,
        ))
        state.apply(event(
            StateEventType.VERIFICATION_RECORDED,
            verification_id="bench_retry", kind="project_bench", executed=True,
            outcome=VerificationOutcome.PASSED.value,
        ))
        self.assertEqual(state.verification_outcome, VerificationOutcome.PASSED)
        self.assertEqual([item.outcome for item in state.verifications], [
            VerificationOutcome.FAILED, VerificationOutcome.PASSED,
        ])
        self.assertFalse(state.failures["verification:bench_fail"].resolved)

    def test_plan_step_transitions(self):
        state = started()
        state.apply(event(
            StateEventType.PLAN_STEP_ADDED,
            step_id="inspect", description="Inspect the implementation",
        ))
        state.apply(event(
            StateEventType.PLAN_STEP_UPDATED,
            step_id="inspect", status=PlanStepStatus.ACTIVE.value,
        ))
        self.assertEqual(state.current_step.step_id, "inspect")
        state.apply(event(
            StateEventType.PLAN_STEP_ADDED,
            step_id="implement", description="Implement the change",
        ))
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.PLAN_STEP_UPDATED,
                step_id="implement", status=PlanStepStatus.ACTIVE.value,
            ))
        state.apply(event(
            StateEventType.PLAN_STEP_UPDATED,
            step_id="inspect", status=PlanStepStatus.COMPLETED.value,
            evidence_action_ids=["action_1"],
        ))
        self.assertEqual(state.current_step, None)
        self.assertEqual(state.completed_steps[0].evidence_action_ids, ("action_1",))
        with self.assertRaises(StateTransitionError):
            state.apply(event(
                StateEventType.PLAN_STEP_UPDATED,
                step_id="inspect", status=PlanStepStatus.ACTIVE.value,
            ))


class TerminalAndControlTests(unittest.TestCase):
    def test_completed_task_has_one_terminal_status(self):
        state = started()
        state.apply(event(StateEventType.TASK_COMPLETED, summary="done"))
        self.assertEqual(state.terminal_status, TerminalStatus.COMPLETED)
        with self.assertRaises(StateTransitionError):
            state.apply(event(StateEventType.TASK_INTERRUPTED, summary="late interrupt"))
        self.assertEqual(state.terminal_status, TerminalStatus.COMPLETED)

    def test_interruption(self):
        state = started()
        state.apply(event(StateEventType.TASK_INTERRUPTED, summary="user interrupt"))
        self.assertEqual(state.terminal_status, TerminalStatus.INTERRUPTED)

    def test_budget_exhaustion_and_control_counters(self):
        state = started()
        state.apply(event(StateEventType.ROUND_STARTED, round_number=1))
        state.apply(event(StateEventType.RETRY_RECORDED, reason="malformed call"))
        state.apply(event(StateEventType.REPEATED_ACTION_RECORDED, action_id="a"))
        state.apply(event(StateEventType.VERIFICATION_REPROMPT_RECORDED))
        state.apply(event(StateEventType.BUDGET_EXHAUSTED, summary="round budget"))
        self.assertEqual(state.current_round, 1)
        self.assertEqual(state.retry_count, 1)
        self.assertEqual(state.repeated_action_count, 1)
        self.assertEqual(state.verification_reprompt_count, 1)
        self.assertEqual(state.terminal_status, TerminalStatus.BUDGET_EXHAUSTED)

    def test_reaching_budget_can_precede_forced_final_completion(self):
        state = started()
        state.apply(event(StateEventType.BUDGET_REACHED, reason="round_budget"))
        self.assertTrue(state.budget_reached)
        self.assertEqual(state.terminal_status, TerminalStatus.RUNNING)
        state.apply(event(StateEventType.TASK_COMPLETED, summary="forced synthesis returned"))
        self.assertEqual(state.terminal_status, TerminalStatus.COMPLETED)


class SerializationAndReplayTests(unittest.TestCase):
    def populated(self):
        state = started()
        state.apply(succeeded_action())
        state.apply(event(
            StateEventType.FILE_READ, source=StateSource.TOOL_RUNTIME, path="README.md",
            action_id="action_1",
        ))
        state.apply(event(
            StateEventType.ACTION_FAILED, source=StateSource.TOOL_RUNTIME,
            action_id="build_1", kind=ActionKind.COMMAND.value, name="make",
            observed_status="failed", summary="exit 2",
        ))
        return state

    def test_serialization_is_json_compatible_and_versioned(self):
        state = self.populated()
        payload = state.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["task_id"], "task_test")
        json.dumps(payload)

    def test_restoration_preserves_unresolved_failures_and_round_trips(self):
        state = self.populated()
        restored = WorkingState.from_json(state.to_json())
        self.assertEqual(restored.to_dict(), state.to_dict())
        self.assertEqual(restored.files_read, {"README.md"})
        self.assertFalse(restored.failures["failure:build_1"].resolved)
        second = WorkingState.from_json(restored.to_json())
        self.assertEqual(second.to_dict(), state.to_dict())

    def test_malformed_serialized_state_is_rejected(self):
        for value in (
            "not-json",
            json.dumps({"schema_version": 99, "task_id": "x", "events": []}),
            json.dumps({"schema_version": 1, "task_id": "x", "events": {}}),
            json.dumps({
                "schema_version": 1, "task_id": "x", "events": [{
                    "event_id": "e", "event_type": "future_unknown", "task_id": "x",
                    "source": "harness", "timestamp": "now", "monotonic_ns": 1,
                    "data": {},
                }],
            }),
        ):
            with self.subTest(value=value):
                with self.assertRaises(SerializationError):
                    WorkingState.from_json(value)

    def test_unknown_additive_envelope_fields_are_ignored(self):
        payload = self.populated().to_dict()
        payload["future_metadata"] = {"new": True}
        restored = WorkingState.from_dict(payload)
        self.assertEqual(restored.task_id, "task_test")

    def test_duplicate_event_is_rejected_without_double_counting(self):
        state = started()
        action = succeeded_action()
        state.apply(action)
        with self.assertRaises(DuplicateEventError):
            state.apply(action)
        self.assertEqual(state.tool_actions, 1)

    def test_deterministic_event_replay(self):
        original = self.populated()
        replayed = replay_state(original.task_id, original.events)
        self.assertEqual(replayed.to_dict(), original.to_dict())
        self.assertEqual(replayed.actions, original.actions)
        self.assertEqual(replayed.failures, original.failures)


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, trace_event):
        self.events.append(trace_event)


class WorkingStateRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def test_scripted_task_uses_grounded_runtime_facts_without_prose_parsing(self):
        from model_backend import ModelTurn, StopReason
        from tool_runtime import AuditLogger, ExecutionMode, Workspace

        class FakeBackend:
            model = "scripted-local"

            def complete(self, **kwargs):
                # Deliberately false prose: it must not create file or
                # verification facts in WorkingState.
                return ModelTurn(
                    "I changed claimed.py and all tests passed.", (), StopReason.END_TURN
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_values = {
                "WORKSPACE": self.rag_chat.WORKSPACE,
                "PROJECT_ROOT": self.rag_chat.PROJECT_ROOT,
                "EXECUTION_MODE": self.rag_chat.EXECUTION_MODE,
                "AUDIT_LOGGER": self.rag_chat.AUDIT_LOGGER,
                "TRAJECTORY_FILE": self.rag_chat.TRAJECTORY_FILE,
            }
            recorder = MemoryRecorder()
            state = WorkingState.start("task_runtime", "Create a grounded file")
            self.rag_chat.WORKSPACE = Workspace.from_path(root)
            self.rag_chat.PROJECT_ROOT = str(root)
            self.rag_chat.EXECUTION_MODE = ExecutionMode.AUTO
            self.rag_chat.AUDIT_LOGGER = AuditLogger(root / "audit.jsonl")
            self.rag_chat.TRAJECTORY_FILE = str(root / "trajectory.jsonl")
            context = AgentContext(
                state, FakeBackend(), ContextEngine(), TraceEmitter(recorder),
                "system", (ContextItem(
                    "system", ContextLayer.SYSTEM_RULES, "test", "system",
                    protected=True, inclusion_reason="test",
                ),), [ConversationMessage("user", (TextBlock("create"),))], (),
                lambda *_: "OK", 3, 3, 4096, output_reserve=64,
                compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
            )
            try:
                context.apply_state_event(
                    StateEventType.ROUND_STARTED, round_number=1
                )
                AgentRuntime().complete_model_turn(context, use_tools=False)
                with patch.object(self.rag_chat, "tool_use"), patch.object(
                    self.rag_chat, "tool_result"
                ):
                    result = self.rag_chat.execute_tool(
                        "write_file", {"path": "grounded.txt", "content": "value"}, {},
                        agent_context=context,
                    )
                self.assertTrue(result.startswith("OK"))
                context.apply_state_event(
                    StateEventType.VERIFICATION_RECORDED,
                    verification_id="scripted_check",
                    kind="scripted_check",
                    executed=True,
                    outcome=VerificationOutcome.PASSED.value,
                    summary="scripted check passed",
                )
                context.apply_state_event(
                    StateEventType.TASK_COMPLETED, summary="done"
                )
                self.rag_chat.save_trajectory(
                    "question", [], "answer", "pass", "bench", state.task_id
                )
            finally:
                for name, value in old_values.items():
                    setattr(self.rag_chat, name, value)

            sample = json.loads((root / "trajectory.jsonl").read_text())
            self.assertEqual(sample["task_id"], "task_runtime")
            self.assertEqual(state.created_files, {"grounded.txt"})
            self.assertNotIn("claimed.py", state.relevant_files)
            self.assertEqual(state.model_rounds, 1)
            self.assertEqual(state.tool_actions, 1)
            self.assertEqual(state.verification_outcome, VerificationOutcome.PASSED)
            self.assertEqual(state.terminal_status, TerminalStatus.COMPLETED)
            transition_events = [item for item in recorder.events
                                 if item.event_type == EventType.WORKING_STATE_UPDATED]
            self.assertGreaterEqual(len(transition_events), 5)
            encoded = json.dumps([item.to_dict() for item in transition_events])
            self.assertNotIn("Create a grounded file", encoded)
            self.assertNotIn("I changed claimed.py", encoded)

    def test_nonzero_command_result_is_a_grounded_failed_action(self):
        from tool_runtime import ToolResult

        state = WorkingState.start("task_command", "Run a command")
        context = AgentContext(
            state, object(), ContextEngine(), TraceEmitter(), "system",
            (ContextItem("system", ContextLayer.SYSTEM_RULES, "test", "system",
                         protected=True, inclusion_reason="test"),),
            [], (), lambda *_: "OK", 3, 3, 4096, output_reserve=64,
            compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        )
        with patch.object(
            self.rag_chat, "run_cmd_result",
            return_value=ToolResult(
                "failed", "command failed", stdout="compiler output", exit_code=2,
            ),
        ):
            result = self.rag_chat.execute_tool(
                "bash", {"command": "make"}, {}, agent_context=context,
            )
        self.assertIn("exit 2", result)
        self.assertEqual(state.actions[-1].status, ActionStatus.FAILED)
        self.assertEqual(state.actions[-1].exit_code, 2)
        self.assertEqual(state.command_actions, 1)
        self.assertEqual(len(state.unresolved_failures), 1)


if __name__ == "__main__":
    unittest.main()
