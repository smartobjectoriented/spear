"""A document is checked in full once, then trusted until a file moves.

The full check -- every unit rehashed, every index's fingerprint recomputed
-- ran on every binding check and every search. At 298 709 units and a 5.8 GB
vector index that was five minutes, at startup and per query.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import standard_store
from standard_commands import StandardOperator, handle_standard_command
from standard_crossrefs import rebuild_cross_reference_index
from standard_ingest import ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore, StandardStoreError, VERIFICATION_FILE
from standard_vector_index import rebuild_vector_index
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_standard_hybrid_retrieval import HYBRID_PAGES, FixtureSemanticEmbedder

SID, REV = "SYNTH-STD", "R1"


class VerificationStampTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "hybrid.pdf"
        pdf.write_bytes(synthetic_pdf_bytes(HYBRID_PAGES))
        self.root = root / "standards"
        store = StandardStore(self.root)
        ingest_pdf(store, pdf, standard_id=SID, revision=REV,
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(store, SID, REV, created_at="fixed")
        rebuild_cross_reference_index(store, SID, REV, created_at="fixed")
        rebuild_vector_index(store, SID, REV, FixtureSemanticEmbedder(),
                             created_at="fixed")

    def tearDown(self):
        self.temp.cleanup()

    def fresh(self):
        """A new store object: what a new spear-chat process sees."""
        return StandardStore(self.root)

    def counting(self):
        """Patch the two expensive checks and count how often they run."""
        calls = {"units": 0, "fingerprint": 0}
        read_units = StandardStore._read_units
        fingerprint = standard_store.corpus_fingerprint

        def units(store, *args):
            calls["units"] += 1
            return read_units(store, *args)

        def corpus(units_):
            calls["fingerprint"] += 1
            return fingerprint(units_)

        return calls, [mock.patch.object(StandardStore, "_read_units", units),
                       mock.patch.object(standard_store, "corpus_fingerprint", corpus)]

    def test_what_the_store_wrote_needs_no_second_check(self):
        record = json.loads((self.fresh().revision_dir(SID, REV)
                             / VERIFICATION_FILE).read_text())
        self.assertEqual(set(record), {"corpus", "lexical", "crossrefs", "vector"})

    def test_a_new_process_binds_without_rehashing(self):
        calls, patches = self.counting()

        for patch in patches:
            patch.start()

        try:
            binding = self.fresh().binding(SID, REV)
        finally:
            for patch in patches:
                patch.stop()

        self.assertIsNotNone(binding.vector_index_fingerprint)
        self.assertIsNotNone(binding.cross_reference_index_fingerprint)
        self.assertEqual(calls, {"units": 0, "fingerprint": 0})

    def test_an_edited_unit_is_checked_again_and_caught(self):
        store = self.fresh()
        corpus = store.revision_dir(SID, REV) / "corpus"
        victim = sorted(corpus.iterdir())[0]
        payload = json.loads(victim.read_text())
        payload["text"] += " tampered"
        victim.write_text(json.dumps(payload))

        # Caught by the unit's own content hash, before the corpus-wide one.
        with self.assertRaisesRegex(Exception, "hash mismatch|integrity"):
            self.fresh().verify_corpus(SID, REV)

    def test_a_rewritten_index_is_checked_again_and_caught(self):
        directory = self.fresh().revision_dir(SID, REV) / "indexes" / "lexical"
        index = json.loads((directory / "index.json").read_text())
        index["document_count"] += 1
        (directory / "index.json").write_text(json.dumps(index))

        with self.assertRaisesRegex(StandardStoreError, "fingerprint mismatch"):
            self.fresh().binding(SID, REV)

    def test_a_removed_record_only_costs_a_full_check(self):
        store = self.fresh()
        (store.revision_dir(SID, REV) / VERIFICATION_FILE).unlink()
        calls, patches = self.counting()

        for patch in patches:
            patch.start()

        try:
            self.fresh().binding(SID, REV)
        finally:
            for patch in patches:
                patch.stop()

        self.assertGreater(calls["fingerprint"], 0)
        self.assertTrue((store.revision_dir(SID, REV) / VERIFICATION_FILE).exists())

    def test_within_a_process_units_are_read_once(self):
        store = self.fresh()
        calls, patches = self.counting()

        for patch in patches:
            patch.start()

        try:
            for _ in range(3):
                store.load_units(SID, REV)
        finally:
            for patch in patches:
                patch.stop()

        self.assertEqual(calls["units"], 1)

    def test_verify_checks_everything_in_full_whatever_was_recorded(self):
        calls, patches = self.counting()
        operator = StandardOperator(self.fresh())

        for patch in patches:
            patch.start()

        try:
            output = handle_standard_command(f"/standard verify {SID} {REV}", operator)
        finally:
            for patch in patches:
                patch.stop()

        self.assertGreater(calls["fingerprint"], 0)

        for name in ("corpus", "lexical", "vector", "crossrefs"):
            self.assertRegex(output, rf"{name}\s*: ok")

    def test_verify_reports_a_broken_index_instead_of_stopping(self):
        store = self.fresh()
        directory = store.revision_dir(SID, REV) / "indexes" / "crossrefs"
        index = json.loads((directory / "index.json").read_text())
        index["counts"]["resolved"] += 1
        (directory / "index.json").write_text(json.dumps(index))

        results = self.fresh().verify_everything(SID, REV)
        self.assertEqual(results["corpus"], "ok")
        self.assertTrue(results["crossrefs"].startswith("FAILED"))


if __name__ == "__main__":
    unittest.main()
