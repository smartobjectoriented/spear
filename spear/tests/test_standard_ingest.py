import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from standard_ingest import StandardIngestionError, ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_schema import StandardContentType, StandardModality
from standard_store import StandardCollisionError, StandardStore
from tests.standard_fixture import PAGES, synthetic_pdf_bytes


class StandardIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / "synthetic.pdf"
        self.pdf.write_bytes(synthetic_pdf_bytes())
        self.store = StandardStore(self.root / "private")

    def tearDown(self): self.temp.cleanup()

    def ingest(self):
        return ingest_pdf(self.store, self.pdf, standard_id="TEST-STD",
                          revision="TEST-1", source_origin="TEST_FIXTURE",
                          ingestion_timestamp="2026-01-01T00:00:00+00:00")

    def test_pdf_hash_semantic_counts_page_section_modality_table_and_cross_reference(self):
        manifest = self.ingest()
        self.assertEqual(manifest.source_pdf_sha256,
                         hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        self.assertEqual(manifest.page_count, 2)
        units = self.store.load_units("TEST-STD", "TEST-1")
        shall = next(unit for unit in units if unit.modality == StandardModality.SHALL)
        should = next(unit for unit in units if unit.modality == StandardModality.SHOULD)
        definition = next(unit for unit in units
                          if unit.content_type == StandardContentType.DEFINITION)
        table = next(unit for unit in units if unit.content_type == StandardContentType.TABLE)
        self.assertEqual((shall.page, shall.section), (1, "3.1"))
        self.assertIn("3.1 Reserved Values", shall.heading_path)
        self.assertEqual(should.content_type, StandardContentType.RECOMMENDATION)
        self.assertIn("3.2", shall.cross_references)
        self.assertEqual(definition.section, "3.1")
        self.assertTrue(table.needs_structured_review)
        self.assertTrue(table.needs_review)

    def test_identical_reingestion_is_idempotent_with_same_ids_and_corpus_hash(self):
        first = self.ingest()
        ids = [item.source_id for item in self.store.load_units("TEST-STD", "TEST-1")]
        second = self.ingest()
        self.assertEqual(first.corpus_manifest_sha256, second.corpus_manifest_sha256)
        self.assertEqual(ids, [item.source_id for item in
                              self.store.load_units("TEST-STD", "TEST-1")])

    def test_revision_collision_with_different_pdf_fails_closed(self):
        self.ingest()
        other = self.root / "other.pdf"
        changed = list(PAGES)
        changed[0] = tuple((*changed[0], (570, "Materially different canonical text.")))
        other.write_bytes(synthetic_pdf_bytes(tuple(changed)))
        with self.assertRaisesRegex(StandardCollisionError, "different source PDF"):
            ingest_pdf(self.store, other, standard_id="TEST-STD", revision="TEST-1")

    def test_malformed_pdf_fails_and_blank_page_is_warned_without_ocr(self):
        bad = self.root / "bad.pdf"; bad.write_bytes(b"not a PDF")
        with self.assertRaises(StandardIngestionError):
            ingest_pdf(self.store, bad, standard_id="BAD", revision="1")
        partial = self.root / "partial.pdf"
        partial.write_bytes(synthetic_pdf_bytes((PAGES[0], ())))
        manifest = ingest_pdf(self.store, partial, standard_id="PARTIAL", revision="1",
                              source_origin="TEST_FIXTURE")
        self.assertTrue(any("no extractable text" in item for item in manifest.warnings))

    def test_raw_pdf_is_not_retained_by_default_but_corpus_survives_source_removal(self):
        self.ingest(); self.pdf.unlink()
        directory = self.store.revision_dir("TEST-STD", "TEST-1")
        self.assertFalse((directory / "source" / "original.pdf").exists())
        self.assertTrue(self.store.verify_corpus("TEST-STD", "TEST-1"))
