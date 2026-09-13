import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from checkpoint import (
    CheckpointError, CheckpointManager, CheckpointStatus, MutationType,
    PathRollbackStatus, RollbackResult, RollbackStatus,
)
from cancellation import CancellationSource, NEVER_CANCELLED
from tracing import TraceEmitter


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.manager = CheckpointManager(root / "audit" / "checkpoints",
                                         self.workspace)
        self.checkpoint = self.manager.begin_checkpoint(
            "task_12345678", "session_12345678")

    def tearDown(self):
        self.temp.cleanup()

    def mutate(self, relative, data, action="edit_12345678"):
        path = self.workspace / relative
        self.manager.capture(self.checkpoint, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manager.record_mutation(self.checkpoint, path, action)
        return path

    def test_creation_and_private_storage(self):
        self.assertEqual(self.checkpoint.status, CheckpointStatus.ACTIVE)
        directory = (self.manager.storage_root / self.checkpoint.session_id /
                     self.checkpoint.checkpoint_id)
        self.assertTrue((directory / "metadata.json").is_file())
        self.assertEqual(directory.stat().st_mode & 0o077, 0)

    def test_capture_text_binary_and_only_once(self):
        text = self.workspace / "a.txt"
        binary = self.workspace / "a.bin"
        text.write_text("before", encoding="utf-8")
        binary.write_bytes(b"\x00\xffbefore")
        first = self.manager.capture(self.checkpoint, text)
        text.write_text("changed", encoding="utf-8")
        second = self.manager.capture(self.checkpoint, text)
        self.manager.capture(self.checkpoint, binary)
        self.assertIs(first, second)
        self.assertEqual(len(self.checkpoint.paths), 2)

    def test_restore_existing_byte_for_byte(self):
        path = self.workspace / "a.bin"
        original = b"\x00\xfforiginal"
        path.write_bytes(original)
        self.mutate("a.bin", b"changed")
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.SUCCESS)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(result.paths[0].status, PathRollbackStatus.RESTORED)

    def test_created_file_is_removed_but_empty_existing_is_restored(self):
        existing = self.workspace / "empty"
        existing.write_bytes(b"")
        self.mutate("empty", b"not empty", "edit_empty")
        created = self.mutate("new.txt", b"new", "create_new")
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.SUCCESS)
        self.assertEqual(existing.read_bytes(), b"")
        self.assertFalse(created.exists())
        self.assertIn(PathRollbackStatus.REMOVED_CREATED_FILE,
                      {item.status for item in result.paths})

    def test_repeated_edits_track_latest_task_owned_hash(self):
        path = self.workspace / "a.txt"
        path.write_text("original")
        self.mutate("a.txt", b"one", "edit_one")
        self.mutate("a.txt", b"two", "edit_two")
        self.assertEqual(len(self.checkpoint.paths), 1)
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.SUCCESS)
        self.assertEqual(path.read_text(), "original")

    def test_deleted_file_is_restored(self):
        path = self.workspace / "gone.txt"
        path.write_text("original")
        self.manager.capture(self.checkpoint, path)
        path.unlink()
        self.manager.record_mutation(
            self.checkpoint, path, "delete_12345678", MutationType.DELETED)
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.SUCCESS)
        self.assertEqual(path.read_text(), "original")

    def test_external_change_conflict_is_all_or_nothing(self):
        a = self.workspace / "a.txt"
        b = self.workspace / "b.txt"
        a.write_text("A")
        b.write_text("B")
        self.mutate("a.txt", b"task A", "edit_a")
        self.mutate("b.txt", b"task B", "edit_b")
        a.write_text("external")
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.CONFLICT)
        self.assertEqual(a.read_text(), "external")
        self.assertEqual(b.read_text(), "task B")

    def test_external_deletion_conflicts(self):
        path = self.workspace / "a.txt"
        path.write_text("A")
        self.mutate("a.txt", b"task")
        path.unlink()
        result = self.manager.rollback(self.checkpoint)
        self.assertEqual(result.status, RollbackStatus.CONFLICT)
        self.assertEqual(result.paths[0].status, PathRollbackStatus.MISSING)

    def test_failed_mutation_has_no_post_state(self):
        path = self.workspace / "a.txt"
        path.write_text("A")
        self.manager.capture(self.checkpoint, path)
        self.assertIsNone(self.checkpoint.paths["a.txt"].post_mutation_hash)

    def test_finalize_is_not_git_commit_and_prevents_rollback(self):
        self.manager.finalize(self.checkpoint)
        self.assertEqual(self.checkpoint.status, CheckpointStatus.FINALIZED)
        with self.assertRaises(CheckpointError):
            self.manager.rollback(self.checkpoint)

    def test_traversal_symlink_and_wrong_identity_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.write_text("outside")
        with self.assertRaises(CheckpointError):
            self.manager.capture(self.checkpoint, "../outside")
        link = self.workspace / "link"
        link.symlink_to(outside)
        with self.assertRaises(CheckpointError):
            self.manager.capture(self.checkpoint, link)
        with self.assertRaises(CheckpointError):
            self.manager.load(self.checkpoint.checkpoint_id,
                              task_id="task_wrong123")

    def test_malformed_metadata_rejected(self):
        metadata = (self.manager.storage_root / self.checkpoint.session_id /
                    self.checkpoint.checkpoint_id / "metadata.json")
        metadata.write_text("{bad", encoding="utf-8")
        with self.assertRaises(CheckpointError):
            self.manager.load(self.checkpoint.checkpoint_id)

    def test_resume_active_checkpoint(self):
        path = self.mutate("a.txt", b"task")
        loaded = self.manager.load(
            self.checkpoint.checkpoint_id, task_id=self.checkpoint.task_id,
            session_id=self.checkpoint.session_id)
        self.assertEqual(loaded.status, CheckpointStatus.ACTIVE)
        self.assertEqual(loaded.paths["a.txt"].post_mutation_hash,
                         self.checkpoint.paths["a.txt"].post_mutation_hash)
        self.assertEqual(path.read_bytes(), b"task")

    def test_rollback_result_serialization(self):
        result = RollbackResult(
            self.checkpoint.checkpoint_id, RollbackStatus.SUCCESS, ())
        self.assertEqual(json.loads(json.dumps(result.to_dict()))["status"], "success")


class ProductionMutationCheckpointTests(unittest.TestCase):
    def setUp(self):
        import rag_chat
        from tool_runtime import Workspace
        self.rag_chat = rag_chat
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()
        self.old_workspace, self.old_root = rag_chat.WORKSPACE, rag_chat.PROJECT_ROOT
        self.addCleanup(setattr, rag_chat, "WORKSPACE", self.old_workspace)
        self.addCleanup(setattr, rag_chat, "PROJECT_ROOT", self.old_root)
        rag_chat.WORKSPACE = Workspace.from_path(self.workspace)
        rag_chat.PROJECT_ROOT = str(self.workspace)
        self.manager = CheckpointManager(Path(self.temp.name) / "checkpoints",
                                         self.workspace)
        self.checkpoint = self.manager.begin_checkpoint(
            "task_abcdefgh", "session_abcdefgh")
        self.context = SimpleNamespace(
            task_id="task_abcdefgh", trace=TraceEmitter(),
            cancellation=NEVER_CANCELLED, checkpoint_manager=self.manager,
            checkpoint=self.checkpoint,
        )

    def route(self, name, arguments, cache=None):
        # These exercise the production mutation handlers, so the session has
        # to be one that permits mutation: the process default is safe mode,
        # where the handlers refuse before they reach the checkpoint at all.
        # Patching only the approval left that premise implicit and, once a
        # categorical mode check was added ahead of it, untrue.

        from tool_runtime import ExecutionMode

        with patch.object(self.rag_chat, "EXECUTION_MODE", ExecutionMode.AUTO), \
                patch.object(self.rag_chat, "authorize_mutation", return_value=None):
            return self.rag_chat.route_tool_envelope(
                name, arguments, cache or {}, task_id=self.context.task_id,
                trace=self.context.trace, cancellation=self.context.cancellation,
                agent_context=self.context,
            )

    def test_write_edit_append_and_code_block_facade_capture_before_mutation(self):
        created = self.route("write_file", {"path": "new.txt", "content": "one"})
        self.assertTrue(created.success)
        cache = {}
        self.rag_chat._note_files_read(cache, "cat new.txt")
        edited = self.route("edit_file", {
            "path": "new.txt", "old_text": "one", "new_text": "two",
        }, cache)
        appended = self.route("append_file", {
            "path": "new.txt", "content": "three",
        }, cache)
        self.assertTrue(edited.success)
        self.assertTrue(appended.success)
        item = self.checkpoint.paths["new.txt"]
        self.assertFalse(item.original_exists)
        self.assertIsNotNone(item.post_mutation_hash)
        rollback = self.manager.rollback(self.checkpoint)
        self.assertEqual(rollback.status, RollbackStatus.SUCCESS)
        self.assertFalse((self.workspace / "new.txt").exists())

    def test_unified_diff_path_is_checkpointed(self):
        target = self.workspace / "a.txt"
        target.write_text("old\n", encoding="utf-8")
        diff = "@@ -1 +1 @@\n-old\n+new\n"
        result = self.route("write_file", {"path": "a.txt", "content": diff})
        self.assertTrue(result.success)
        self.assertIn("a.txt", self.checkpoint.paths)
        self.manager.rollback(self.checkpoint)
        self.assertEqual(target.read_text(), "old\n")

    def test_cancel_before_mutation_captures_nothing(self):
        source = CancellationSource()
        source.cancel("stop")
        self.context.cancellation = source.token
        result = self.route("write_file", {"path": "cancelled.txt", "content": "x"})
        self.assertFalse(result.success)
        self.assertEqual(self.checkpoint.paths, {})
        self.assertFalse((self.workspace / "cancelled.txt").exists())

    def test_cancel_after_grounded_mutation_keeps_recovery_data(self):
        result = self.route("write_file", {"path": "made.txt", "content": "x"})
        self.assertTrue(result.success)
        source = CancellationSource()
        source.cancel("after")
        self.assertIn("made.txt", self.checkpoint.paths)
        self.assertEqual(self.checkpoint.status, CheckpointStatus.ACTIVE)


if __name__ == "__main__":
    unittest.main()


class SecondaryRootCheckpointTests(unittest.TestCase):
    """A declared corpus is writable, so a mutation there must checkpoint.

    Without this, every edit_file in a secondary root died on "checkpoint
    target escapes workspace" -- and the model, told by the banner that the
    tree was writable, kept abandoning edit_file for shell splices that
    mangled the file.
    """
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.workspace = root / "workspace"
        self.corpus = root / "corpus"
        for directory in (self.workspace, self.corpus):
            directory.mkdir()
        self.target = self.corpus / "user_space.rst"
        self.target.write_text("original\n")
        self.manager = CheckpointManager(root / "store", self.workspace,
                                         extra_roots=(self.corpus,))

    def test_a_file_in_a_declared_corpus_can_be_captured_and_rolled_back(self):
        checkpoint = self.manager.begin_checkpoint("task_12345678", "session_12345678")
        self.manager.capture(checkpoint, self.target)
        self.target.write_text("mangled\n")
        self.manager.record_mutation(checkpoint, self.target, "edit_12345678")
        self.manager.rollback(checkpoint)
        self.assertEqual("original\n", self.target.read_text())

    def test_a_path_in_no_declared_root_is_still_refused(self):
        outside = Path(self.tmp.name) / "elsewhere.txt"
        outside.write_text("x\n")
        checkpoint = self.manager.begin_checkpoint("task_12345678", "session_12345678")
        with self.assertRaises(CheckpointError):
            self.manager.capture(checkpoint, outside)

    def test_the_primary_workspace_keeps_its_relative_keys(self):
        inside = self.workspace / "main.c"
        inside.write_text("int main(void) { return 0; }\n")
        checkpoint = self.manager.begin_checkpoint("task_12345678", "session_12345678")
        self.manager.capture(checkpoint, inside)
        self.assertIn("main.c", checkpoint.paths)
