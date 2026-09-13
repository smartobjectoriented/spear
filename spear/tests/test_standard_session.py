import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from model_backend import ConversationMessage, TextBlock
from session_store import (
    FileSessionStore, SessionCompatibilityError, SessionConfiguration,
    SessionEventType, SessionSnapshot, new_session_id, restore_session,
)
from standard_ingest import ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore
from standard_vector_index import rebuild_vector_index
from tests.standard_fixture import synthetic_pdf_bytes
from working_state import WorkingState


class StandardSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        pdf = root / "fixture.pdf"; pdf.write_bytes(synthetic_pdf_bytes())
        self.standards = StandardStore(root / "standards")
        ingest_pdf(self.standards, pdf, standard_id="TEST-STD", revision="TEST-1",
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.standards, "TEST-STD", "TEST-1",
                              created_at="first")
        self.binding = self.standards.binding("TEST-STD", "TEST-1")
        self.sessions = FileSessionStore(root / "sessions")
        self.sid = new_session_id(); self.task_id = "task_standard_123"
        self.configuration = SessionConfiguration("/workspace", "project", "safe")

    def tearDown(self): self.temp.cleanup()

    def save(self):
        snapshot = SessionSnapshot(
            self.sid, self.task_id, 0, self.configuration,
            WorkingState.start(self.task_id, "normative question"),
            (ConversationMessage("user", (TextBlock("question"),)),),
            standard_binding=self.binding.to_dict(),
            standard_source_ids_used=("std-" + "a" * 32,),
            standard_retrieval_cache_fingerprint=self.binding.index_fingerprint,
        )
        self.sessions.save_snapshot(snapshot)

    def test_binding_and_source_provenance_round_trip_without_normative_text(self):
        self.save(); loaded = self.sessions.load_snapshot(self.sid)
        self.assertEqual(loaded.standard_binding, self.binding.to_dict())
        self.assertEqual(loaded.standard_source_ids_used, ("std-" + "a" * 32,))
        serialized = str(loaded.to_dict())
        self.assertNotIn("reserved value", serialized)

    def test_same_corpus_resume_succeeds(self):
        self.save()
        _, snapshot = restore_session(
            self.sessions, self.sid, self.configuration,
            standard_store=self.standards)
        self.assertEqual(snapshot.standard_binding["corpus_manifest_sha256"],
                         self.binding.corpus_manifest_sha256)

    def test_index_only_change_allows_resume_records_event_and_invalidates_cache(self):
        self.save()
        changed = rebuild_lexical_index(
            self.standards, "TEST-STD", "TEST-1", indexer_version="bm25-v2",
            created_at="second")
        _, snapshot = restore_session(
            self.sessions, self.sid, self.configuration,
            standard_store=self.standards)
        self.assertEqual(snapshot.standard_binding["index_fingerprint"],
                         changed.index_fingerprint)
        events = self.sessions.events(self.sid)
        event = next(item for item in events
                     if item.event_type == SessionEventType.STANDARD_INDEX_CHANGED)
        self.assertTrue(event.payload["canonical_source_unchanged"])
        self.assertTrue(event.payload["cached_retrieval_invalidated"])

    def test_vector_model_change_allows_resume_and_records_retrieval_change(self):
        self.save()

        class Embedder:
            model_id = "TEST_FIXTURE/local"
            model_revision = "v1"
            def embed_documents(self, texts):
                return [[1.0, float(bool(text.strip()))] for text in texts]
            def embed_query(self, _text): return [1.0, 1.0]

        rebuild_vector_index(self.standards, "TEST-STD", "TEST-1", Embedder(),
                             created_at="fixed")
        _, snapshot = restore_session(
            self.sessions, self.sid, self.configuration,
            standard_store=self.standards)
        self.assertNotEqual(snapshot.standard_retrieval_cache_fingerprint,
                            self.binding.index_fingerprint)
        event = next(item for item in self.sessions.events(self.sid)
                     if item.event_type == SessionEventType.STANDARD_RETRIEVAL_CHANGED)
        self.assertTrue(event.payload["canonical_source_unchanged"])
        self.assertTrue(event.payload["cached_retrieval_invalidated"])

    def test_pdf_or_corpus_change_blocks_with_operator_diagnostic(self):
        self.save()
        current = self.binding

        class ChangedSource:
            def binding(self, *_args, **_kwargs):
                return replace(current, corpus_manifest_sha256="b" * 64)

        with self.assertRaisesRegex(
            SessionCompatibilityError, "canonical normative source changed"):
            restore_session(self.sessions, self.sid, self.configuration,
                            standard_store=ChangedSource())

    def test_two_sessions_do_not_share_binding(self):
        self.save()
        other = new_session_id()
        self.sessions.save_snapshot(SessionSnapshot(
            other, "task_other_123", 0, self.configuration,
            WorkingState.start("task_other_123", "ordinary task"), (),
        ))
        self.assertIsNone(self.sessions.load_snapshot(other).standard_binding)
