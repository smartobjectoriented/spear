import json
import unittest
from unittest.mock import patch

from compaction import (
    CompactionMode,
    CompactionPolicy,
    CompactionRequest,
    CompactionService,
    ExtractiveSummarizer,
    ModelBackendSummarizer,
    StructuredCompactionState,
    validate_compaction,
)
from agent_runtime import AgentContext, AgentRuntime
from context_engine import ContextEngine, ContextItem, ContextLayer, ContextRequest, Freshness
from model_backend import ToolDefinition
from tracing import EventType, TraceEmitter
from working_state import (
    ActionKind,
    PlanStepStatus,
    StateEvent,
    StateEventType,
    StateSource,
    VerificationOutcome,
    WorkingState,
)


def apply(state, event_type, source=StateSource.HARNESS, **data):
    state.apply(StateEvent.create(event_type, state.task_id, source, **data))


def populated_state(task_id="task_compact"):
    state = WorkingState.start(task_id, "Implement safe compaction")
    for value in ("Remain local", "Preserve security", "No provider coupling"):
        apply(state, StateEventType.CONSTRAINT_RECORDED, StateSource.USER,
              constraint=value)
    for value in ("Tests pass", "Facts remain", "Summary bounded", "No regressions"):
        apply(state, StateEventType.ACCEPTANCE_CRITERION_RECORDED, StateSource.USER,
              criterion=value)
    for step_id, description in (("one", "Inspect"), ("two", "Implement"),
                                 ("three", "Verify")):
        apply(state, StateEventType.PLAN_STEP_ADDED, step_id=step_id,
              description=description)
    apply(state, StateEventType.PLAN_STEP_UPDATED, step_id="one",
          status=PlanStepStatus.ACTIVE.value)
    apply(state, StateEventType.PLAN_STEP_UPDATED, step_id="one",
          status=PlanStepStatus.COMPLETED.value, evidence_action_ids=[])
    apply(state, StateEventType.PLAN_STEP_UPDATED, step_id="two",
          status=PlanStepStatus.ACTIVE.value)
    apply(state, StateEventType.PLAN_STEP_UPDATED, step_id="two",
          status=PlanStepStatus.BLOCKED.value)
    for number in range(2):
        apply(state, StateEventType.ACTION_FAILED, StateSource.TOOL_RUNTIME,
              action_id=f"failed_{number}", kind=ActionKind.COMMAND.value,
              name="build", observed_status="failed", category="command_nonzero",
              summary=f"failed approach {number}", exit_code=2)
    apply(state, StateEventType.ACTION_SUCCEEDED, StateSource.TOOL_RUNTIME,
          action_id="edit_1", kind=ActionKind.TOOL.value, name="edit_file",
          observed_status="ok", summary="edited")
    apply(state, StateEventType.FILE_MODIFIED, StateSource.TOOL_RUNTIME,
          action_id="edit_1", path="a.py")
    apply(state, StateEventType.FILE_CREATED, StateSource.TOOL_RUNTIME,
          action_id="edit_1", path="new.py")
    apply(state, StateEventType.ACTION_SUCCEEDED, StateSource.TOOL_RUNTIME,
          action_id="read_1", kind=ActionKind.TOOL.value, name="read_file",
          observed_status="ok", summary="read")
    for path in ("a.py", "b.py", "c.py", "d.py", "e.py"):
        apply(state, StateEventType.FILE_READ, StateSource.TOOL_RUNTIME,
              action_id="read_1", path=path)
    apply(state, StateEventType.DISCOVERY_RECORDED,
          summary="Entry point found", file_path="a.py", symbol="main",
          action_id="read_1")
    apply(state, StateEventType.DECISION_RECORDED,
          summary="Keep API small", rationale="Lower risk", action_id="edit_1")
    apply(state, StateEventType.VERIFICATION_RECORDED,
          verification_id="pass", kind="unit", executed=True,
          outcome=VerificationOutcome.PASSED.value, summary="unit passed")
    apply(state, StateEventType.VERIFICATION_RECORDED,
          verification_id="fail", kind="build", executed=True,
          outcome=VerificationOutcome.FAILED.value, summary="build failed")
    apply(state, StateEventType.VERIFICATION_RECORDED,
          verification_id="never", kind="integration", executed=False,
          outcome=VerificationOutcome.NOT_RUN.value, summary="not run")
    return state


def context_item(number, *, layer=ContextLayer.RECENT_CONVERSATION,
                 protected=False, group=None, content=None):
    return ContextItem(
        item_id=f"item:{number}", layer=layer, source="test_history",
        content=content or (f"old conversational fact {number} " * 20),
        priority=20 + number, freshness=Freshness.STALE,
        protected=protected, token_overhead=4,
        inclusion_reason="test", eviction_group=group or f"group:{number}",
        truncatable=layer != ContextLayer.TOOL_EVIDENCE,
    )


def context_request(items, *, limit=500, reserve=0):
    return ContextRequest(tuple(items), context_limit=limit,
                          output_reserve=reserve, safety_margin=0)


class ScriptedSummarizer:
    def __init__(self, outputs=None, error=None):
        self.outputs = list(outputs or [])
        self.error = error
        self.calls = 0

    def summarize(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        if self.outputs:
            output = self.outputs.pop(0)
            if callable(output):
                return output(kwargs)
            return output
        return json.dumps({"narrative_summary": "User clarified the implementation order."})


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class CompactionTests(unittest.TestCase):
    def service(self, summarizer=None, *, mode=CompactionMode.MANUAL,
                threshold=.8, tail=1, retries=0, max_chars=512, trace=None):
        return CompactionService(
            engine=ContextEngine(), summarizer=summarizer or ScriptedSummarizer(),
            policy=CompactionPolicy(
                mode=mode, pressure_threshold=threshold, recent_tail_groups=tail,
                minimum_compactable_tokens=1, max_summary_chars=max_chars,
                max_retries=retries,
            ), trace=trace,
        )

    def request(self, state=None, items=None, artifact=None, limit=500):
        state = state or populated_state()
        items = items or tuple(context_item(index) for index in range(5))
        return CompactionRequest(state, context_request(items, limit=limit), artifact)

    def test_successful_manual_compaction(self):
        result = self.service().compact(self.request())
        self.assertTrue(result.succeeded)
        self.assertGreater(result.artifact.messages_compacted, 0)
        self.assertEqual(result.artifact.structured_state.objective,
                         "Implement safe compaction")

    def test_threshold_triggered_automatic_compaction(self):
        result = self.service(mode=CompactionMode.AUTOMATIC, threshold=.2).compact(
            self.request(limit=200)
        )
        self.assertTrue(result.succeeded)

    def test_no_compaction_below_threshold(self):
        summarizer = ScriptedSummarizer()
        result = self.service(summarizer, mode=CompactionMode.AUTOMATIC,
                              threshold=.9).compact(self.request(limit=10000))
        self.assertFalse(result.attempted)
        self.assertEqual(summarizer.calls, 0)

    def test_authoritative_task_definition_preserved(self):
        state = populated_state()
        compact = self.service().compact(self.request(state=state)).artifact.structured_state
        self.assertEqual(compact.objective, state.objective)
        self.assertEqual(compact.user_constraints, tuple(state.user_constraints))
        self.assertEqual(compact.acceptance_criteria, tuple(state.acceptance_criteria))

    def test_unresolved_failures_and_failed_approaches_preserved(self):
        state = populated_state()
        compact = self.service().compact(self.request(state=state)).artifact.structured_state
        self.assertEqual({item.failure_id for item in compact.unresolved_failures},
                         {item.failure_id for item in state.unresolved_failures})
        self.assertIn("failed", {item.status for item in compact.important_actions})

    def test_all_verification_outcomes_preserved(self):
        compact = self.service().compact(self.request()).artifact.structured_state
        self.assertEqual([item.outcome for item in compact.verifications],
                         ["passed", "failed", "not_run"])

    def test_files_and_plan_states_preserved(self):
        compact = self.service().compact(self.request()).artifact.structured_state
        self.assertEqual(compact.files_modified, ("a.py",))
        self.assertEqual(compact.files_created, ("new.py",))
        self.assertEqual([item.status for item in compact.plan],
                         ["completed", "blocked", "pending"])

    def test_terminal_state_preserved(self):
        state = populated_state()
        apply(state, StateEventType.TASK_INTERRUPTED, summary="cancelled")
        compact = StructuredCompactionState.from_working_state(state)
        self.assertEqual(compact.terminal_status, "interrupted")

    def test_the_ask_and_the_check_speak_the_same_unit(self):
        """A summary written to the letter of the instruction must pass it.

        The bound used to be tokens: the model was told "under 512
        approximate tokens", complied at roughly four characters each, and
        the check measured with an estimator at three -- so it called two
        thousand characters six hundred and eighty tokens and rejected every
        summary, on every attempt, for the whole turn.
        """
        asked = []

        def obedient(kwargs):
            limit = kwargs["max_summary_chars"]
            asked.append(limit)

            # Exactly what it was told, to the character.
            return json.dumps({"narrative_summary": "x" * limit})

        result = self.service(ScriptedSummarizer([obedient]),
                              max_chars=512).compact(self.request())

        self.assertEqual(asked, [512])
        self.assertTrue(result.succeeded,
                        "obeying the instruction must satisfy the check")

    def test_a_summary_over_the_bound_is_still_refused(self):
        def verbose(kwargs):
            return json.dumps({
                "narrative_summary": "x" * (kwargs["max_summary_chars"] + 1)})

        result = self.service(ScriptedSummarizer([verbose]),
                              max_chars=512).compact(self.request())

        self.assertFalse(result.succeeded)

    def test_a_retry_asks_for_a_shorter_summary_than_the_one_that_overran(self):
        """Re-asking with the same prompt is not a retry.

        A live trace held twenty-four consecutive rejections for "narrative
        summary exceeds configured bound" in one turn: the summarizer wrote
        long, was asked again in exactly the same words, and wrote long
        again. Compaction never succeeded, so the pressure that triggered it
        was never relieved.
        """
        asked = []

        def long_then_short(kwargs):
            asked.append(kwargs["max_summary_chars"])

            return json.dumps({"narrative_summary":
                               "word " * (400 if len(asked) == 1 else 4)})

        summarizer = ScriptedSummarizer([long_then_short, long_then_short])
        result = self.service(summarizer, retries=1, max_chars=512).compact(
            self.request())

        self.assertEqual(len(asked), 2)
        self.assertLess(asked[1], asked[0],
                        "the second attempt must be asked for less")
        self.assertTrue(result.succeeded)

    def test_terminal_state_cannot_be_changed_by_summarizer(self):
        state = populated_state()
        claimed = StructuredCompactionState.from_working_state(state).to_dict()
        claimed["terminal_status"] = "completed"
        raw = json.dumps({"narrative_summary": "Continuing.",
                          "structured_state": claimed})
        self.assertFalse(self.service(ScriptedSummarizer([raw])).compact(
            self.request(state=state)).succeeded)

    def test_claimed_objective_change_is_rejected(self):
        state = populated_state()
        claimed = StructuredCompactionState.from_working_state(state).to_dict()
        claimed["objective"] = "Different task"
        summary = json.dumps({"narrative_summary": "Continuing work.",
                              "structured_state": claimed})
        result = self.service(ScriptedSummarizer([summary])).compact(
            self.request(state=state)
        )
        self.assertFalse(result.succeeded)
        self.assertIn("objective", result.error_summary)

    def test_claimed_constraint_or_acceptance_loss_is_rejected(self):
        state = populated_state()
        for field in ("user_constraints", "acceptance_criteria"):
            claimed = StructuredCompactionState.from_working_state(state).to_dict()
            claimed[field] = []
            raw = json.dumps({"narrative_summary": "Continuing.",
                              "structured_state": claimed})
            result = self.service(ScriptedSummarizer([raw])).compact(
                self.request(state=state)
            )
            self.assertFalse(result.succeeded)

    def test_failed_verification_cannot_become_passed(self):
        state = populated_state()
        claimed = StructuredCompactionState.from_working_state(state).to_dict()
        claimed["verifications"][1]["outcome"] = "passed"
        raw = json.dumps({"narrative_summary": "Continuing.",
                          "structured_state": claimed})
        self.assertFalse(self.service(ScriptedSummarizer([raw])).compact(
            self.request(state=state)).succeeded)

    def test_not_run_verification_cannot_become_passed(self):
        state = populated_state()
        claimed = StructuredCompactionState.from_working_state(state).to_dict()
        claimed["verifications"][2]["outcome"] = "passed"
        raw = json.dumps({"narrative_summary": "Continuing.",
                          "structured_state": claimed})
        self.assertFalse(self.service(ScriptedSummarizer([raw])).compact(
            self.request(state=state)).succeeded)

    def test_no_modified_file_loss_or_invention(self):
        state = populated_state()
        for files in ([], ["a.py", "invented.py"]):
            claimed = StructuredCompactionState.from_working_state(state).to_dict()
            claimed["files_modified"] = files
            raw = json.dumps({"narrative_summary": "Continuing.",
                              "structured_state": claimed})
            self.assertFalse(self.service(ScriptedSummarizer([raw])).compact(
                self.request(state=state)).succeeded)

    def test_plan_state_cannot_change_through_summary(self):
        state = populated_state()
        claimed = StructuredCompactionState.from_working_state(state).to_dict()
        claimed["plan"][0]["status"] = "pending"
        claimed["plan"][1]["status"] = "completed"
        raw = json.dumps({"narrative_summary": "Continuing.",
                          "structured_state": claimed})
        self.assertFalse(self.service(ScriptedSummarizer([raw])).compact(
            self.request(state=state)).succeeded)

    def test_false_narrative_success_is_rejected(self):
        state = WorkingState.start("truth", "Build")
        apply(state, StateEventType.VERIFICATION_RECORDED,
              verification_id="build", kind="build", executed=True,
              outcome="failed", summary="failed")
        raw = json.dumps({"narrative_summary": "All tests passed."})
        result = self.service(ScriptedSummarizer([raw])).compact(
            self.request(state=state)
        )
        self.assertFalse(result.succeeded)

    def test_malformed_output_and_model_failure_are_non_destructive(self):
        state = populated_state()
        before = state.to_json()
        malformed = self.service(ScriptedSummarizer(["not json"])).compact(
            self.request(state=state)
        )
        failed = self.service(ScriptedSummarizer(error=RuntimeError("offline"))).compact(
            self.request(state=state)
        )
        self.assertFalse(malformed.succeeded)
        self.assertFalse(failed.succeeded)
        self.assertEqual(state.to_json(), before)
        self.assertIsNone(malformed.artifact)

    def test_bounded_retry(self):
        summarizer = ScriptedSummarizer(["bad", "also bad", "unused"])
        result = self.service(summarizer, retries=1).compact(self.request())
        self.assertFalse(result.succeeded)
        self.assertEqual(summarizer.calls, 2)
        self.assertEqual(result.retries, 1)

    def test_impossible_budget_does_not_commit_compaction(self):
        state = populated_state()
        request = ContextRequest(
            tuple(context_item(index) for index in range(4)),
            context_limit=100, output_reserve=0, safety_margin=0,
            fixed_input_tokens=120,
        )
        result = self.service().compact(CompactionRequest(state, request))
        self.assertTrue(result.attempted)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.artifact)

    def test_repeated_compaction_replaces_summary_without_recursive_growth(self):
        state = populated_state()
        service = self.service(tail=1, max_chars=128)
        first = service.compact(self.request(state=state)).artifact
        new_items = tuple(context_item(index) for index in range(8))
        effective = service.apply_artifact(context_request(new_items), first)
        second = service.compact(CompactionRequest(state, effective, first)).artifact
        self.assertEqual(second.conversation_summary.generation, 2)
        self.assertGreater(len(second.compacted_item_ids), len(first.compacted_item_ids))
        self.assertLessEqual(second.conversation_summary.token_estimate, 40)

    def test_three_successive_compactions_retain_seeded_facts(self):
        state = populated_state("stress")
        service = self.service(tail=1, max_chars=256)
        artifact = None
        irrelevant_counts = []
        for generation in range(3):
            items = tuple(context_item(index) for index in range(5 + generation * 3))
            request = context_request(items, limit=600)
            if artifact:
                request = service.apply_artifact(request, artifact)
            result = service.compact(CompactionRequest(state, request, artifact))
            self.assertTrue(result.succeeded)
            artifact = result.artifact
            compact = artifact.structured_state
            self.assertEqual(compact.user_constraints, tuple(state.user_constraints))
            self.assertEqual(compact.acceptance_criteria, tuple(state.acceptance_criteria))
            self.assertEqual(compact.files_modified, ("a.py",))
            self.assertEqual([item.outcome for item in compact.verifications],
                             ["passed", "failed", "not_run"])
            self.assertEqual(len(compact.unresolved_failures), 3)
            self.assertLessEqual(compact.conversation_summary.token_estimate
                                 if hasattr(compact, "conversation_summary") else
                                 artifact.conversation_summary.token_estimate, 72)
            remaining = service.apply_artifact(context_request(items), artifact)
            irrelevant_counts.append(sum(
                item.layer == ContextLayer.RECENT_CONVERSATION
                for item in remaining.items
            ))
        self.assertTrue(all(value <= 1 for value in irrelevant_counts))

    def test_recent_tail_is_preserved(self):
        service = self.service(tail=2)
        artifact = service.compact(self.request()).artifact
        self.assertNotIn("item:3", artifact.compacted_item_ids)
        self.assertNotIn("item:4", artifact.compacted_item_ids)

    def test_tool_call_result_group_is_atomic(self):
        items = (
            context_item(0, layer=ContextLayer.TOOL_EVIDENCE, group="call:1"),
            context_item(1, layer=ContextLayer.TOOL_EVIDENCE, group="call:1"),
            context_item(2), context_item(3),
        )
        artifact = self.service(tail=1).compact(self.request(items=items)).artifact
        self.assertEqual({"item:0", "item:1"} & artifact.compacted_item_ids,
                         {"item:0", "item:1"})

    def test_existing_summary_is_replaced(self):
        service = self.service(tail=1)
        first = service.compact(self.request()).artifact
        applied = service.apply_artifact(
            context_request(tuple(context_item(i) for i in range(7))), first
        )
        second = service.compact(CompactionRequest(
            populated_state(), applied, first
        )).artifact
        rendered = service.apply_artifact(
            context_request(tuple(context_item(i) for i in range(7))), second
        )
        summaries = [item for item in rendered.items
                     if item.layer == ContextLayer.CONVERSATION_SUMMARY]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(second.conversation_summary.generation, 2)

    def test_trace_metadata_is_safe_and_content_free(self):
        recorder = MemoryRecorder()
        trace = TraceEmitter(recorder)
        item = context_item(0, content="authorization=super-secret " * 40)
        items = (item, context_item(1), context_item(2))
        result = self.service(trace=trace).compact(self.request(items=items))
        self.assertTrue(result.succeeded)
        self.assertEqual([event.event_type for event in recorder.events],
                         [EventType.COMPACTION_STARTED, EventType.COMPACTION_FINISHED])
        encoded = json.dumps([event.to_dict() for event in recorder.events])
        self.assertNotIn("super-secret", encoded)
        self.assertNotIn("narrative_summary", encoded)

    def test_context_engine_integration_reduces_raw_context(self):
        service = self.service(tail=1)
        request = self.request().context_request
        artifact = service.compact(self.request()).artifact
        compacted = service.apply_artifact(request, artifact)
        snapshot = service.engine.compose(compacted)
        self.assertTrue(any(item.layer == ContextLayer.CONVERSATION_SUMMARY
                            for item in snapshot.selected_items))
        self.assertLess(len(compacted.items), len(request.items))

    def test_structured_state_round_trip_and_validation(self):
        state = populated_state()
        original = StructuredCompactionState.from_working_state(state)
        restored = StructuredCompactionState.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )
        self.assertEqual(restored, original)
        self.assertTrue(validate_compaction(state, restored, "Continuing.").valid)

    def test_compacted_large_results_have_explicit_missing_references(self):
        items = (
            context_item(0, layer=ContextLayer.TOOL_EVIDENCE,
                         content="large result " * 100),
            context_item(1), context_item(2),
        )
        artifact = self.service(tail=1).compact(self.request(items=items)).artifact
        reference = next(item for item in artifact.structured_state.evidence
                         if item.reference_id == "item:0")
        self.assertEqual(reference.kind, "compacted_context")
        self.assertFalse(reference.available)

    def test_provider_independent_core_has_no_provider_names(self):
        from pathlib import Path
        source = Path(__file__).resolve().parents[1].joinpath("compaction.py").read_text()
        self.assertNotIn("OpenAI", source)
        self.assertNotIn("Anthropic", source)

    def test_model_summarizer_emits_model_timing_pair(self):
        from model_backend import ModelTurn, StopReason

        class FakeBackend:
            model = "local-test"

            def complete(self, **kwargs):
                return ModelTurn(
                    json.dumps({"narrative_summary": "Continuing."}), (),
                    StopReason.END_TURN,
                    usage={"input_tokens": 10, "output_tokens": 3},
                )

        recorder = MemoryRecorder()
        summarizer = ModelBackendSummarizer(
            FakeBackend(), trace=TraceEmitter(recorder), provider="fake",
        )
        result = self.service(summarizer).compact(self.request())
        self.assertTrue(result.succeeded)
        model_events = [event for event in recorder.events
                        if event.event_type in {
                            EventType.MODEL_CALL_STARTED,
                            EventType.MODEL_CALL_FINISHED,
                        }]
        self.assertEqual([event.event_type for event in model_events], [
            EventType.MODEL_CALL_STARTED, EventType.MODEL_CALL_FINISHED,
        ])
        self.assertEqual(model_events[-1].input_tokens, 10)
        self.assertEqual(model_events[-1].metadata["purpose"], "context_compaction")


class ProductionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def test_long_task_compacts_then_uses_existing_context_engine_path(self):
        from model_backend import ConversationMessage, ModelTurn, StopReason, TextBlock

        class FakeBackend:
            model = "scripted-local"
            max_tokens = 32

            def __init__(self):
                self.calls = []

            def complete(self, **kwargs):
                self.calls.append(kwargs)
                if "compact agent context" in kwargs["system"]:
                    return ModelTurn(
                        json.dumps({"narrative_summary": "Earlier discussion retained."}),
                        (), StopReason.END_TURN,
                    )
                return ModelTurn("done", (), StopReason.END_TURN)

        state = WorkingState.start("production_compaction", "Finish safely")
        conversation = tuple(
            ConversationMessage("user" if index % 2 == 0 else "assistant",
                                (TextBlock((f"old {index} ") * 80),))
            for index in range(9)
        )
        backend = FakeBackend()
        items = (ContextItem(
            "system", ContextLayer.SYSTEM_RULES, "test", "SYSTEM",
            priority=100, protected=True, inclusion_reason="test",
        ),)
        context = AgentContext(
            state, backend, ContextEngine(), TraceEmitter(), "SYSTEM", items,
            list(conversation), (ToolDefinition("test", "test", {}),),
            lambda *_: "OK", 3, 3, 1800, output_reserve=32,
            compaction_policy=CompactionPolicy(
                pressure_threshold=.2, recent_tail_groups=1,
                minimum_compactable_tokens=1, max_retries=0,
            ),
        )
        turn = AgentRuntime().complete_model_turn(context, use_tools=False)
        artifact = context.compaction_artifact
        self.assertEqual(turn.text, "done")
        self.assertIsNotNone(artifact)
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("Conversation continuity", backend.calls[-1]["system"])
        self.assertEqual(artifact.structured_state.objective, "Finish safely")

    def test_invalid_configuration_falls_back_without_breaking_composition(self):
        with patch.dict(
            "os.environ", {"SPEAR_COMPACTION_THRESHOLD": "invalid"}, clear=False,
        ):
            policy = self.rag_chat.compaction_policy_from_environment()
        self.assertEqual(policy, CompactionPolicy())


if __name__ == "__main__":
    unittest.main()
