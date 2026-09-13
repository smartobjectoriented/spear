import tempfile
import unittest
from pathlib import Path

from result_store import ResultStore, ResultStoreError
from tool_registry import ToolCategory, ToolRegistry, ToolResultPolicy, ToolSpec
from tool_router import ToolExecutionContext, ToolRouter
from tracing import TraceEmitter


class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = ResultStore(self.temporary.name)

    def test_put_get_metadata_and_existence(self):
        stored = self.store.put("complete output", task_id="task", metadata={"tool": "x"})
        self.assertTrue(self.store.exists(stored.reference))
        self.assertEqual(self.store.get_text(stored.reference, task_id="task"),
                         "complete output")
        self.assertEqual(self.store.metadata(stored.reference)["byte_count"], 15)

    def test_reference_is_opaque_content_derived_and_collision_safe(self):
        first = self.store.put("same", task_id="one")
        repeated = self.store.put("same", task_id="one")
        other_task = self.store.put("same", task_id="two")
        self.assertRegex(first.reference, r"^result_[0-9a-f]{64}$")
        self.assertEqual(first.reference, repeated.reference)
        self.assertNotEqual(first.reference, other_task.reference)

    def test_task_association_is_enforced(self):
        stored = self.store.put("private", task_id="one")
        with self.assertRaises(ResultStoreError):
            self.store.get(stored.reference, task_id="two")

    def test_path_traversal_and_malformed_references_are_rejected(self):
        for reference in ("../secret", "/tmp/result", "result_bad", ""):
            with self.subTest(reference=reference):
                with self.assertRaises(ResultStoreError):
                    self.store.get(reference)
                self.assertFalse(self.store.exists(reference))

    def test_missing_well_formed_reference(self):
        missing = "result_" + "0" * 64
        with self.assertRaises(KeyError):
            self.store.get(missing)

    def test_binary_non_utf8_is_preserved_and_text_is_safe(self):
        payload = b"prefix\xffsuffix"
        stored = self.store.put(payload)
        self.assertEqual(self.store.get(stored.reference), payload)
        self.assertIn("\ufffd", self.store.get_text(stored.reference))

    def test_metadata_cannot_choose_a_filename(self):
        stored = self.store.put("x", metadata={"filename": "../../escape"})
        self.assertTrue(self.store.exists(stored.reference))
        self.assertFalse(Path(self.temporary.name).parent.joinpath("escape").exists())


class ResultStoreRouterTests(unittest.TestCase):
    def router(self, root, output, *, cap=40, preview=12):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            "large", "Return output", {"type": "object", "properties": {}},
            ToolCategory.OTHER,
            result_policy=ToolResultPolicy(cap, True, preview),
        ), lambda *_: output)
        store = ResultStore(root)
        context = ToolExecutionContext("task", TraceEmitter(), {}, store)
        return ToolRouter(registry), context, store

    def test_small_result_remains_inline_and_unstored(self):
        router, context, _ = self.router(self.temporary_dir(), "small")
        result = router.execute(context, "call", "large", {})
        self.assertEqual(result.model_content, "small")
        self.assertFalse(result.truncated_for_model)
        self.assertIsNone(result.result_reference)

    def test_large_result_is_bounded_stored_and_retrievable(self):
        root = self.temporary_dir()
        full = "build-line\n" * 100
        router, context, store = self.router(root, full)
        result = router.execute(context, "call", "large", {})
        self.assertTrue(result.truncated_for_model)
        self.assertLess(len(result.model_content), len(full))
        self.assertTrue(result.model_content.startswith(full[:12]))
        self.assertIn("full result reference", result.model_content)
        self.assertEqual(store.get_text(result.result_reference, task_id="task"), full)
        self.assertEqual(result.text, full)
        self.assertEqual(result.output_bytes, len(full.encode()))

    def test_large_binary_result_is_safe_and_not_fully_reinjected(self):
        root = self.temporary_dir()
        payload = b"x" * 100 + b"\xff"
        router, context, store = self.router(root, payload)
        result = router.execute(context, "call", "large", {})
        self.assertNotEqual(result.model_content.encode("utf-8"), payload)
        self.assertEqual(store.get(result.result_reference), payload)

    def temporary_dir(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary.name


if __name__ == "__main__":
    unittest.main()
