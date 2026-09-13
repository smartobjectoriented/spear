import tempfile
import unittest

from agent_runtime import AgentContext, AgentRuntime, RuntimeTerminalReason
from cancellation import CancellationSource
from context_engine import ContextEngine
from failure_policy import FailureKind, RetryPolicy
from session_store import (
    FileSessionStore, SessionConfiguration, SessionHandle, new_session_id,
    restore_session,
)
from tests.test_agent_runtime import (
    GroundedToolExecutor, ScriptedBackend, make_context, text_turn, tool_turn,
)
from tracing import TraceEmitter
from working_state import TerminalStatus


class RuntimeResumptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = FileSessionStore(self.temp.name)
        self.configuration = SessionConfiguration(
            "/workspace", "project", "safe", provider="scripted", model="one",
        )

    def attach_session(self, context):
        context.session = SessionHandle(
            self.store, new_session_id(), self.configuration,
        )
        return context

    def test_crash_resume_keeps_ids_state_results_and_does_not_repeat_mutation(self):
        source = CancellationSource()
        delegate = GroundedToolExecutor(["OK: changed"])

        def cancelling_executor(context, call_id, name, arguments, cache):
            result = delegate(context, call_id, name, arguments, cache)
            source.cancel("simulated process interruption")
            return result

        first = self.attach_session(make_context(
            ScriptedBackend([tool_turn("call_write", "write_file", path="a.py")]),
            executor=cancelling_executor,
        ))
        first.cancellation = source.token
        interrupted = AgentRuntime().run(first)
        self.assertEqual(interrupted.terminal_reason, RuntimeTerminalReason.INTERRUPTED)
        sid, task_id = first.session_id, first.task_id

        handle, snapshot = restore_session(
            self.store, sid,
            SessionConfiguration("/workspace", "project", "safe",
                                 provider="other-local", model="two"),
        )
        called = []
        resumed = AgentContext.from_session_snapshot(
            snapshot, session=handle,
            backend=ScriptedBackend([text_turn("finished"), text_turn("finished")]),
            context_engine=ContextEngine(), trace=TraceEmitter(), tools=first.tools,
            tool_executor=lambda *_: called.append(True), context_limit=4096,
            output_reserve=64,
        )
        result = AgentRuntime().run(resumed)
        # A resumed turn that wrote and verified nothing is told so, like
        # any other; what this test pins is that the write is not repeated.
        self.assertTrue(result.final_response.startswith("finished"))
        self.assertIn("a.py changed", result.final_response)
        self.assertEqual(result.task_id, task_id)
        self.assertEqual(resumed.session_id, sid)
        self.assertEqual(result.working_state.modified_files, {"a.py"})
        self.assertFalse(called, "grounded mutation must not be replayed")
        self.assertEqual(result.terminal_status, TerminalStatus.COMPLETED)

    def test_cancel_before_start_is_persisted_and_cannot_complete(self):
        source = CancellationSource(); source.cancel("before start")
        context = self.attach_session(make_context(ScriptedBackend([text_turn()])))
        context.cancellation = source.token
        result = AgentRuntime().run(context)
        snapshot = self.store.load_snapshot(context.session_id)
        self.assertEqual(result.failure_kind, FailureKind.INTERRUPTED)
        self.assertEqual(snapshot.working_state.terminal_status,
                         TerminalStatus.INTERRUPTED)

    def test_explicit_bounded_model_retry_is_accounted(self):
        backend = ScriptedBackend([RuntimeError("temporary"), text_turn("ok")])
        context = make_context(backend)
        context.retry_policy = RetryPolicy({FailureKind.MODEL_ERROR: 1})
        result = AgentRuntime().run(context)
        self.assertEqual(result.final_response, "ok")
        self.assertEqual(result.working_state.retry_count, 1)
        self.assertEqual(result.working_state.model_rounds, 2)

    def test_repeated_failed_approach_reaches_structured_stall(self):
        turns = [tool_turn(f"read_{n}", "read_file", path="same.py")
                 for n in range(4)] + [text_turn("stopped")]
        context = make_context(ScriptedBackend(turns), rounds=8)
        result = AgentRuntime().run(context)
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        self.assertEqual(result.failure_kind, FailureKind.STALLED)
        self.assertEqual(result.terminal_status, TerminalStatus.FAILED)

    def test_session_write_failure_is_surfaced_as_runtime_failure(self):
        class BrokenStore:
            def append(self, _event):
                raise OSError("disk unavailable")
            def save_snapshot(self, _snapshot):
                raise OSError("disk unavailable")
        context = make_context(ScriptedBackend([text_turn("must not run")]))
        context.session = SessionHandle(
            BrokenStore(), new_session_id(), self.configuration,
        )
        result = AgentRuntime().run(context)
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.RUNTIME_FAILURE)
        self.assertEqual(result.terminal_status, TerminalStatus.FAILED)
        self.assertIn("session event", result.error_summary)
