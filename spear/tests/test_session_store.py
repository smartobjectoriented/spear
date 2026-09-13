import json
import tempfile
import unittest
from pathlib import Path

from compaction import (
    CompactionArtifact, ConversationSummary, StructuredCompactionState,
)
from context_engine import ContextItem, ContextLayer
from model_backend import ConversationMessage, TextBlock, ToolUseBlock
from session_store import (
    FileSessionStore, InFlightOperation, SessionCompatibilityError,
    SessionConfiguration, SessionError, SessionEvent, SessionEventType,
    SessionHandle, SessionSnapshot, new_session_id, restore_session,
)
from working_state import (
    StateEvent, StateEventType, StateSource, TerminalStatus, WorkingState,
)


def config(workspace="/workspace", provider="local"):
    return SessionConfiguration(workspace, "project", "safe",
                                provider=provider, model="model")


def state(task="task_session"):
    return WorkingState.start(task, "finish task", max_model_rounds=8,
                              max_tool_actions=12)


def artifact(working):
    return CompactionArtifact(
        StructuredCompactionState.from_working_state(working),
        ConversationSummary(1, "Earlier discussion.", 5), frozenset({"old"}),
        2, 1, 100, 40, "test", 0,
    )


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = FileSessionStore(self.temp.name)
        self.sid = new_session_id()
        self.ws = state()

    def snapshot(self, sequence=1, **kwargs):
        values = dict(
            session_id=self.sid, task_id=self.ws.task_id, sequence=sequence,
            configuration=config(), working_state=self.ws,
            conversation=(ConversationMessage("user", (TextBlock("continue"),)),),
            context_items=(ContextItem("system", ContextLayer.SYSTEM_RULES,
                                       "test", "rules", protected=True),),
        )
        values.update(kwargs)
        return SessionSnapshot(**values)

    def test_create_append_and_deterministic_order(self):
        handle = SessionHandle(self.store, self.sid, config())
        first = handle.append(SessionEventType.SESSION_STARTED, self.ws.task_id)
        second = handle.append(SessionEventType.TASK_STARTED, self.ws.task_id)
        self.assertEqual([e.sequence for e in self.store.events(self.sid)], [1, 2])
        self.assertEqual((first.event_id, second.event_id),
                         tuple(e.event_id for e in self.store.events(self.sid)))

    def test_atomic_snapshot_load_and_private_layout(self):
        self.store.save_snapshot(self.snapshot())
        loaded = self.store.load_snapshot(self.sid)
        self.assertEqual(loaded.working_state.objective, "finish task")
        path = Path(self.temp.name) / self.sid / "snapshot.json"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_active_checkpoint_reference_survives_snapshot_and_resume(self):
        snap = self.snapshot(
            checkpoint_id="checkpoint_12345678", checkpoint_status="active",
            checkpoint_paths=("a.py", "new.bin"),
        )
        self.store.save_snapshot(snap)
        loaded = self.store.load_snapshot(self.sid)
        self.assertEqual(loaded.checkpoint_id, "checkpoint_12345678")
        self.assertEqual(loaded.checkpoint_status, "active")
        self.assertEqual(loaded.checkpoint_paths, ("a.py", "new.bin"))
        serialized = json.dumps(loaded.to_dict())
        self.assertNotIn("original file contents", serialized)

    def test_snapshot_restores_working_state_failure_conversation_and_artifact(self):
        self.ws.apply(StateEvent.create(
            StateEventType.ACTION_FAILED, self.ws.task_id, StateSource.HARNESS,
            action_id="failed_1", kind="command", name="build",
            observed_status="failed", category="command_failed", summary="failed",
            round_number=0, exit_code=2,
        ))
        snap = self.snapshot(
            compaction_artifact=artifact(self.ws), result_references=("result_abc",),
        )
        self.store.save_snapshot(snap)
        loaded = self.store.load_snapshot(self.sid)
        self.assertEqual(len(loaded.working_state.unresolved_failures), 1)
        self.assertEqual(loaded.compaction_artifact.structured_state.task_id,
                         self.ws.task_id)
        self.assertEqual(loaded.result_references, ("result_abc",))
        self.assertEqual(loaded.conversation[0].content[0].text, "continue")

    def test_result_reference_does_not_duplicate_large_payload(self):
        self.store.save_snapshot(self.snapshot(result_references=("result_big",)))
        raw = (Path(self.temp.name) / self.sid / "snapshot.json").read_text()
        self.assertIn("result_big", raw)
        self.assertNotIn("x" * 1000, raw)

    def test_obvious_secrets_are_redacted_from_snapshot_and_events(self):
        secret_state = WorkingState.start("task_secret", "use api_key=very-secret")
        sid = new_session_id()
        snapshot = SessionSnapshot(
            sid, secret_state.task_id, 1, config(), secret_state,
            (ConversationMessage("user", (TextBlock("token=hidden-value"),)),),
        )
        self.store.save_snapshot(snapshot)
        self.store.append(SessionEvent(
            SessionEventType.TASK_STARTED, sid, secret_state.task_id, 1,
            {"authorization": "Bearer should-not-persist"},
        ))
        directory = Path(self.temp.name) / sid
        persisted = ((directory / "snapshot.json").read_text()
                     + (directory / "events.jsonl").read_text())
        self.assertNotIn("very-secret", persisted)
        self.assertNotIn("hidden-value", persisted)
        self.assertNotIn("should-not-persist", persisted)

    def test_schema_and_malformed_snapshot_rejected(self):
        directory = Path(self.temp.name) / self.sid
        directory.mkdir()
        (directory / "snapshot.json").write_text('{"schema_version": 99}')
        with self.assertRaises(SessionError):
            self.store.load_snapshot(self.sid)

    def test_malformed_compaction_artifact_falls_back_without_losing_state(self):
        self.store.save_snapshot(self.snapshot(compaction_artifact=artifact(self.ws)))
        path = Path(self.temp.name) / self.sid / "snapshot.json"
        raw = json.loads(path.read_text())
        raw["compaction_artifact"]["structured_state"]["task_id"] = "task_other"
        path.write_text(json.dumps(raw))
        loaded = self.store.load_snapshot(self.sid)
        self.assertIsNone(loaded.compaction_artifact)
        self.assertEqual(loaded.working_state.objective, "finish task")

    def test_truncated_final_jsonl_is_recovered_but_middle_corruption_is_not(self):
        event = SessionEvent(SessionEventType.SESSION_STARTED, self.sid,
                             self.ws.task_id, 1)
        self.store.append(event)
        path = Path(self.temp.name) / self.sid / "events.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"truncated"')
        self.assertEqual(len(self.store.events(self.sid)), 1)
        self.store.append(SessionEvent(
            SessionEventType.TASK_STARTED, self.sid, self.ws.task_id, 2,
        ))
        self.assertEqual([item.sequence for item in self.store.events(self.sid)], [1, 2])
        path.write_text('{bad}\n' + json.dumps(event.to_dict()) + '\n')
        with self.assertRaises(SessionError):
            self.store.events(self.sid)

    def test_path_traversal_and_malformed_ids_rejected(self):
        with self.assertRaises(SessionError):
            self.store.events("../outside")

    def test_configuration_allows_provider_change_but_not_workspace_change(self):
        self.assertTrue(config(provider="a").compatible_with(config(provider="b")))
        self.assertFalse(config().compatible_with(config("/other")))
        self.store.save_snapshot(self.snapshot())
        with self.assertRaises(SessionCompatibilityError):
            restore_session(self.store, self.sid, config("/other"))

    def test_two_sessions_do_not_leak(self):
        other = new_session_id()
        self.store.save_snapshot(self.snapshot())
        other_state = state("task_other")
        self.store.save_snapshot(SessionSnapshot(
            other, other_state.task_id, 0, config(), other_state, (),
        ))
        self.assertNotEqual(self.store.load_snapshot(self.sid).task_id,
                            self.store.load_snapshot(other).task_id)

    def test_resume_interrupted_task_preserves_identity(self):
        self.ws.apply(StateEvent.create(StateEventType.TASK_INTERRUPTED,
                                        self.ws.task_id, StateSource.HARNESS,
                                        summary="cancelled"))
        self.store.save_snapshot(self.snapshot())
        handle, restored = restore_session(self.store, self.sid, config(provider="new"))
        self.assertEqual(handle.session_id, self.sid)
        self.assertEqual(restored.task_id, self.ws.task_id)
        self.assertEqual(restored.working_state.terminal_status, TerminalStatus.RUNNING)

    def test_inflight_mutation_is_cancelled_and_never_claimed_successful(self):
        conversation = (ConversationMessage("assistant", (
            ToolUseBlock("call_1", "write_file", {"path": "a.py"}),
        )),)
        self.store.save_snapshot(self.snapshot(
            conversation=conversation,
            in_flight=InFlightOperation("tool", "call_1", "write_file", True),
        ))
        _, restored = restore_session(self.store, self.sid, config())
        self.assertFalse(restored.working_state.modified_files)
        self.assertEqual(restored.working_state.actions[-1].status.value, "cancelled")
        self.assertTrue(restored.conversation[-1].content[0].is_error)

    def test_grounded_mutation_after_completion_survives_without_inflight_replay(self):
        self.ws.apply(StateEvent.create(
            StateEventType.ACTION_SUCCEEDED, self.ws.task_id, StateSource.TOOL_RUNTIME,
            action_id="write_1", kind="tool", name="write_file",
            observed_status="ok", round_number=1,
        ))
        self.ws.apply(StateEvent.create(
            StateEventType.FILE_MODIFIED, self.ws.task_id, StateSource.TOOL_RUNTIME,
            action_id="write_1", path="a.py",
        ))
        self.store.save_snapshot(self.snapshot())
        _, restored = restore_session(self.store, self.sid, config())
        self.assertEqual(restored.working_state.modified_files, {"a.py"})
        self.assertEqual(len(restored.working_state.actions), 1)

    def test_legacy_history_is_not_repuposed_or_modified(self):
        history = Path(self.temp.name) / "history.json"
        history.write_text('[{"role":"user","content":"old"}]')
        before = history.read_bytes()
        self.store.save_snapshot(self.snapshot())
        self.assertEqual(history.read_bytes(), before)
