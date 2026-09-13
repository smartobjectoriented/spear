import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from agent_roles import explorer_role, reviewer_role
from memory_store import (
    MarkdownMemoryStore, MemoryRecord, MemoryScope, MemorySource,
    MemoryStoreError,
)
from tool_registry import ToolRegistry, native_tool_specs


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "memories.md"
        self.store = MarkdownMemoryStore(self.path)
        self.now = datetime(2026, 8, 24, 12, 30, tzinfo=UTC)

    def test_existing_markdown_import_is_human_readable_and_structured(self):
        original = "# Notes\n\n- [2026-01-02] Build with make test\n- Plain preference\n"
        self.path.write_text(original, encoding="utf-8")
        records = self.store.records()
        self.assertEqual([item.content for item in records],
                         ["Build with make test", "Plain preference"])
        self.assertTrue(all(item.source == MemorySource.IMPORTED for item in records))
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)
        self.assertFalse(self.store.metadata_path.exists())

    def test_creation_serialization_and_scope_filtering(self):
        record = self.store.add(
            "Use the local test runner", scope=MemoryScope.PROJECT,
            source=MemorySource.AUTHORIZED_TOOL, provenance="confirmed tool call",
            tags=("tests",), now=self.now,
        )
        restored = MemoryRecord.from_dict(record.to_dict())
        self.assertEqual(restored, record)
        selected = self.store.select("test runner", scopes=(MemoryScope.PROJECT,))
        self.assertEqual(selected.records[0].content, record.content)
        self.assertFalse(self.store.select(
            "test runner", scopes=(MemoryScope.USER,),
        ).records)
        self.assertIn("- [2026-08-24] Use the local test runner",
                      self.path.read_text(encoding="utf-8"))

    def test_relevance_filters_broad_memory_injection(self):
        self.store.add("Kernel build uses make kernel", now=self.now)
        self.store.add("Frontend formatting uses prettier", now=self.now)
        selection = self.store.select("fix kernel build")
        self.assertEqual([item.content for item in selection.records],
                         ["Kernel build uses make kernel"])
        self.assertNotIn("Frontend", selection.render())
        self.assertEqual(selection.excluded_count, 1)

    def test_supersession_and_inactive_memory_exclusion(self):
        old = self.store.add("Use make old", now=self.now)
        new = self.store.add(
            "Use make new", supersedes=(old.memory_id,),
            now=datetime(2026, 8, 25, tzinfo=UTC),
        )
        records = {item.memory_id: item for item in self.store.records()}
        self.assertFalse(records[old.memory_id].active)
        self.assertEqual(records[old.memory_id].superseded_by, new.memory_id)
        self.assertEqual(self.store.select("make").records, (records[new.memory_id],))

    def test_malformed_metadata_is_rejected(self):
        self.path.write_text("- [2026-01-02] Valid\n", encoding="utf-8")
        memory_id = self.store.records()[0].memory_id
        self.store.metadata_path.write_text('{"schema_version": 99}', encoding="utf-8")
        with self.assertRaises(MemoryStoreError):
            self.store.records()
        self.store.metadata_path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(MemoryStoreError):
            self.store.records()
        self.store.metadata_path.write_text(json.dumps({
            "schema_version": 1,
            "records": {memory_id: {"scope": "invalid"}},
        }), encoding="utf-8")
        with self.assertRaises(MemoryStoreError):
            self.store.records()

    def test_metadata_is_json_and_does_not_duplicate_content(self):
        self.store.add("A unique durable fact", now=self.now)
        raw = json.loads(self.store.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema_version"], 1)
        self.assertNotIn("A unique durable fact", self.store.metadata_path.read_text())

    def test_explorer_and_reviewer_structurally_cannot_write_memory(self):
        registry = ToolRegistry()
        for spec in native_tool_specs():
            registry.register(spec, lambda *_: None)
        for role in (explorer_role(), reviewer_role()):
            role.validate_registry(registry)
            self.assertFalse(role.can_write_memory)
            self.assertNotIn("remember", role.allowed_tool_names)

    def test_confirmed_legacy_write_boundary_uses_store_and_selected_context(self):
        import rag_chat
        with patch.object(rag_chat, "MEMORIES_FILE", str(self.path)), \
                patch.object(rag_chat, "authorize_mutation", return_value=None) as authorize, \
                patch.object(rag_chat, "tool_use"), patch.object(rag_chat, "tool_result"):
            result = rag_chat._registered_remember(None, {"note": "Kernel uses ninja"})
            authorize.assert_called_once()
            self.assertTrue(result.text.startswith("OK:"))
            rag_chat.save_memory("Frontend uses prettier")
            selected = rag_chat.load_memories("kernel build")
            self.assertIn("Kernel uses ninja", selected)
            self.assertNotIn("Frontend uses prettier", selected)


if __name__ == "__main__":
    unittest.main()
