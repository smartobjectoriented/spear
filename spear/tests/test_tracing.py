import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracing import (
    EventStatus,
    EventType,
    JsonlTraceRecorder,
    NullTraceRecorder,
    TraceEmitter,
    TraceEvent,
    create_trace_emitter,
    new_task_id,
)
from agent_runtime import AgentContext, AgentRuntime
from compaction import CompactionPolicy
from context_engine import ContextEngine, ContextItem, ContextLayer
from model_backend import ConversationMessage, TextBlock
from working_state import WorkingState


class MemoryRecorder:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class FailingRecorder:
    def record(self, event):
        raise OSError("disk unavailable")


class HostileValue:
    def __repr__(self):
        raise RuntimeError("do not render me")


class TraceEventTests(unittest.TestCase):
    def test_event_creation_and_serialization(self):
        event = TraceEvent(
            EventType.MODEL_CALL_FINISHED,
            "task_one",
            status=EventStatus.OK,
            provider="local",
            input_tokens=12,
            metadata={"stop_reason": "end_turn"},
        )
        encoded = event.to_dict()
        self.assertEqual(encoded["event_type"], "model_call_finished")
        self.assertEqual(encoded["task_id"], "task_one")
        self.assertEqual(encoded["status"], "ok")
        self.assertEqual(encoded["input_tokens"], 12)
        self.assertNotIn("tool_name", encoded)

    def test_task_id_is_stable_when_reused_and_distinct_across_tasks(self):
        task_id = new_task_id()
        events = [TraceEvent(EventType.TASK_STARTED, task_id),
                  TraceEvent(EventType.TASK_FINISHED, task_id)]
        self.assertEqual(events[0].task_id, events[1].task_id)
        self.assertNotEqual(task_id, new_task_id())
        self.assertTrue(task_id.startswith("task_"))

    def test_secret_fields_and_secret_looking_strings_are_redacted(self):
        event = TraceEvent(EventType.TOOL_CALL_STARTED, "task", metadata={
            "api_key": "sk-secret",
            "nested": {"Authorization": "Bearer abc.def", "safe": "token=xyz command"},
            "command_summary": "curl -H Authorization:BearerSecret endpoint",
        })
        serialized = json.dumps(event.to_dict())
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("abc.def", serialized)
        self.assertNotIn("xyz", serialized)
        self.assertNotIn("BearerSecret", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_malformed_metadata_is_json_safe(self):
        cyclic = {}
        cyclic["self"] = cyclic
        event = TraceEvent(EventType.CONTEXT_COMPOSED, "task", metadata={
            "cycle": cyclic, "set": {3, 1}, "hostile": HostileValue(), "blob": b"secret",
        })
        encoded = json.dumps(event.to_dict())
        self.assertIn("[CYCLE]", encoded)
        self.assertIn("[HostileValue]", encoded)
        self.assertIn("[bytes:6]", encoded)


class RecorderTests(unittest.TestCase):
    def test_jsonl_preserves_event_order_and_each_line_is_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "trace.jsonl"
            recorder = JsonlTraceRecorder(path)
            task_id = new_task_id()
            recorder.record(TraceEvent(EventType.TASK_STARTED, task_id))
            recorder.record(TraceEvent(EventType.TASK_FINISHED, task_id))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event_type"] for row in rows],
                             ["task_started", "task_finished"])
            self.assertEqual({row["task_id"] for row in rows}, {task_id})

    def test_null_and_disabled_recorder_write_nothing(self):
        NullTraceRecorder().record(TraceEvent(EventType.TASK_STARTED, "task"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            emitter = create_trace_emitter(path, {})
            emitter.emit(EventType.TASK_STARTED, "task")
            self.assertFalse(path.exists())

    def test_enabled_configuration_honours_explicit_trace_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chosen.jsonl"
            emitter = create_trace_emitter("unused.jsonl", {
                "SPEAR_TRACE": "1", "SPEAR_TRACE_FILE": str(path),
            })
            emitter.emit(EventType.TASK_STARTED, "task")
            self.assertEqual(json.loads(path.read_text())["event_type"], "task_started")

    def test_jsonl_write_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "not-a-file"
            directory.mkdir()
            recorder = JsonlTraceRecorder(directory)
            recorder.record(TraceEvent(EventType.TASK_STARTED, "task"))
            self.assertEqual(recorder.failure_count, 1)
            self.assertIn("IsADirectoryError", recorder.last_error)

    def test_custom_recorder_failure_is_isolated_by_emitter(self):
        emitter = TraceEmitter(FailingRecorder())
        event = emitter.emit(EventType.TASK_STARTED, "task")
        self.assertEqual(event.task_id, "task")


class TimingTests(unittest.TestCase):
    def test_model_call_timing_pair(self):
        recorder = MemoryRecorder()
        emitter = TraceEmitter(recorder)
        span = emitter.start_span(
            EventType.MODEL_CALL_STARTED, EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED, "task", provider="fake", model="scripted",
        )
        span.finish(input_tokens=10, output_tokens=2)
        self.assertEqual([event.event_type for event in recorder.events], [
            EventType.MODEL_CALL_STARTED, EventType.MODEL_CALL_FINISHED,
        ])
        self.assertEqual(recorder.events[1].parent_event_id, recorder.events[0].event_id)
        self.assertGreaterEqual(recorder.events[1].duration_ms, 0)

    def test_tool_call_timing_pair(self):
        recorder = MemoryRecorder()
        emitter = TraceEmitter(recorder)
        span = emitter.start_span(
            EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED,
            EventType.TOOL_CALL_FAILED, "task", tool_name="read_file",
        )
        span.finish(metadata={"output_chars": 20})
        self.assertEqual([event.event_type.value for event in recorder.events],
                         ["tool_call_started", "tool_call_finished"])
        self.assertEqual(recorder.events[0].action_id, recorder.events[1].action_id)

    def test_error_event_generation(self):
        recorder = MemoryRecorder()
        emitter = TraceEmitter(recorder)
        span = emitter.start_span(
            EventType.MODEL_CALL_STARTED, EventType.MODEL_CALL_FINISHED,
            EventType.MODEL_CALL_FAILED, "task",
        )
        span.fail(ValueError("bad response"), error_category="malformed_output")
        failure = recorder.events[-1]
        self.assertEqual(failure.event_type, EventType.MODEL_CALL_FAILED)
        self.assertEqual(failure.error_category, "malformed_output")
        self.assertEqual(failure.status, EventStatus.FAILED)


class RagChatTracingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_chat
        cls.rag_chat = rag_chat

    def setUp(self):
        self.recorder = MemoryRecorder()
        self.emitter = TraceEmitter(self.recorder)
        self.state = WorkingState.start(
            "task_integration", "scripted integration task"
        )

    def context(self, backend, trace=None):
        return AgentContext(
            self.state, backend, ContextEngine(), trace or self.emitter, "system",
            (ContextItem(
                "system", ContextLayer.SYSTEM_RULES, "test", "system",
                protected=True, inclusion_reason="test",
            ),),
            [ConversationMessage("user", (TextBlock("test"),))], (),
            lambda *_: "OK", 3, 3, 4096, output_reserve=256,
            compaction_policy=CompactionPolicy(minimum_compactable_tokens=10000),
        )

    def test_model_boundary_emits_context_and_timing_pair(self):
        from model_backend import ModelTurn, StopReason

        class FakeBackend:
            model = "scripted-local"
            max_tokens = 256

            def complete(self, **kwargs):
                return ModelTurn("done", (), StopReason.END_TURN,
                                 usage={"input_tokens": 8, "output_tokens": 1})

        AgentRuntime().complete_model_turn(self.context(FakeBackend()), use_tools=False)
        types = [event.event_type for event in self.recorder.events]
        self.assertEqual(types, [
            EventType.CONTEXT_COMPOSED,
            EventType.MODEL_CALL_STARTED,
            EventType.MODEL_CALL_FINISHED,
            EventType.WORKING_STATE_UPDATED,
        ])
        finished = self.recorder.events[-2]
        self.assertEqual(finished.input_tokens, 8)
        self.assertEqual(finished.output_tokens, 1)
        self.assertEqual(finished.metadata["tool_call_count"], 0)

    def test_tool_boundary_emits_timing_pair_without_recording_values(self):
        arguments = {"query": "api_key=do-not-record"}
        with patch.object(self.rag_chat, "search_corpus", return_value="OK: done"):
            self.rag_chat.execute_tool(
                "search_corpus", arguments, {},
                agent_context=self.context(object()),
            )
        tool_events = [event for event in self.recorder.events
                       if event.event_type in (EventType.TOOL_CALL_STARTED,
                                               EventType.TOOL_CALL_FINISHED)]
        self.assertEqual([event.event_type for event in tool_events], [
            EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_FINISHED,
        ])
        encoded = json.dumps([event.to_dict() for event in tool_events])
        self.assertNotIn("do-not-record", encoded)
        self.assertIn("argument_keys", encoded)

    def test_model_exception_raises_and_tool_exception_becomes_structured_failure(self):
        class BrokenBackend:
            def complete(self, **kwargs):
                raise RuntimeError("provider unavailable")

        with self.assertRaises(RuntimeError):
            AgentRuntime().complete_model_turn(
                self.context(BrokenBackend()), use_tools=False,
            )
        with patch.object(
            self.rag_chat, "search_corpus", side_effect=ValueError("tool bug")
        ):
            result = self.rag_chat.execute_tool(
                "search_corpus", {"query": "x"}, {},
                agent_context=self.context(object()),
            )
        self.assertTrue(result.startswith("ERROR:"))
        types = [event.event_type for event in self.recorder.events]
        self.assertIn(EventType.MODEL_CALL_FAILED, types)
        self.assertIn(EventType.TOOL_CALL_FAILED, types)

    def test_broken_recorder_cannot_change_model_or_tool_results(self):
        from model_backend import ModelTurn, StopReason

        class FakeBackend:
            def complete(self, **kwargs):
                return ModelTurn("unchanged", (), StopReason.END_TURN)

        failing = TraceEmitter(FailingRecorder())
        turn = AgentRuntime().complete_model_turn(
            self.context(FakeBackend(), trace=failing), use_tools=False,
        )
        with patch.object(self.rag_chat, "search_corpus", return_value="OK: unchanged"):
            result = self.rag_chat.execute_tool(
                "search_corpus", {"query": "x"}, {},
                agent_context=self.context(object(), trace=failing),
            )
        self.assertEqual(turn.text, "unchanged")
        self.assertEqual(result, "OK: unchanged")

    def test_task_id_is_correlated_with_existing_trajectory_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_path = self.rag_chat.TRAJECTORY_FILE
            self.rag_chat.TRAJECTORY_FILE = str(Path(temporary) / "trajectory.jsonl")
            try:
                self.rag_chat.save_trajectory(
                    "question", [], "answer", "pass", "bench", "task_integration"
                )
                sample = json.loads(Path(self.rag_chat.TRAJECTORY_FILE).read_text())
            finally:
                self.rag_chat.TRAJECTORY_FILE = old_path
        self.assertEqual(sample["task_id"], "task_integration")

    def test_scripted_task_has_expected_end_to_end_event_sequence(self):
        from model_backend import ModelTurn, StopReason

        class FakeBackend:
            model = "scripted-local"

            def complete(self, **kwargs):
                return ModelTurn("done", (), StopReason.END_TURN)

        self.emitter.emit(EventType.TASK_STARTED, "task_integration",
                          status=EventStatus.STARTED)
        context = self.context(FakeBackend())
        AgentRuntime().complete_model_turn(context, use_tools=False)
        with patch.object(self.rag_chat, "search_corpus", return_value="OK: result"):
            self.rag_chat.execute_tool(
                "search_corpus", {"query": "identifier"}, {},
                agent_context=context,
            )
        self.emitter.emit(EventType.TASK_FINISHED, "task_integration",
                          status=EventStatus.OK)
        sequence = [event.event_type for event in self.recorder.events]
        self.assertEqual(sequence, [
            EventType.TASK_STARTED,
            EventType.CONTEXT_COMPOSED,
            EventType.MODEL_CALL_STARTED,
            EventType.MODEL_CALL_FINISHED,
            EventType.WORKING_STATE_UPDATED,
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_FINISHED,
            EventType.WORKING_STATE_UPDATED,
            EventType.TASK_FINISHED,
        ])


if __name__ == "__main__":
    unittest.main()
