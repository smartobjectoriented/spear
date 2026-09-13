import tempfile
import unittest
import json
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_retrieval import StandardRetrieval, rebuild_lexical_index
from standard_store import StandardStore
from standard_store import StandardStoreError
from tests.standard_fixture import synthetic_pdf_bytes


class StandardRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        pdf = root / "fixture.pdf"; pdf.write_bytes(synthetic_pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(
            self.store, pdf, standard_id="TEST-STD", revision="TEST-1",
            source_origin="TEST_FIXTURE")
        self.index = rebuild_lexical_index(
            self.store, "TEST-STD", "TEST-1", created_at="2026-01-01T00:00:00Z")
        self.retrieval = StandardRetrieval(self.store)

    def tearDown(self): self.temp.cleanup()

    def test_exact_terms_acronym_hex_section_and_ranking(self):
        reserved = self.retrieval.search(
            "TEST-STD", "TEST-1", "shall reject reserved value")
        self.assertTrue(reserved)
        self.assertIn("shall reject reserved value", reserved[0].snippet)
        self.assertGreaterEqual(reserved[0].score, reserved[-1].score)
        self.assertTrue(self.retrieval.search("TEST-STD", "TEST-1", "0xF"))
        fixture = self.retrieval.search("TEST-STD", "TEST-1", "TEST_FIXTURE")
        self.assertTrue(fixture)
        section = self.retrieval.search("TEST-STD", "TEST-1", "3.1")
        self.assertEqual(section[0].section, "3.1")
        self.assertGreater(section[0].score, 999)

    def test_section_filter_exact_fetch_parent_and_citation(self):
        results = self.retrieval.search(
            "TEST-STD", "TEST-1", "shall reject", section="3.1", limit=5)
        result = next(item for item in results if item.modality == "SHALL")
        fetched = self.retrieval.fetch("TEST-STD", "TEST-1", result.source_id)
        self.assertEqual(fetched["unit"]["source_id"], result.source_id)
        self.assertIsNotNone(fetched["parent"])
        self.assertIn("§3.1", fetched["citation"]["rendered"])

    def test_every_unit_a_fetch_returns_carries_its_own_citation(self):
        """The citation travels with the text, so quoting costs no round trip.

        A single turn made eighteen standard.cite calls, seventeen of them
        for units a fetch had just returned -- eighteen model round trips to
        obtain a string whose every component was already on screen.
        """

        hit = self.retrieval.search("TEST-STD", "TEST-1", "3.1")[0]
        fetched = self.retrieval.fetch("TEST-STD", "TEST-1", hit.source_id)
        units = ([fetched["unit"]] + fetched["neighbors"]
                 + ([fetched["parent"]] if fetched["parent"] else []))

        self.assertGreater(len(units), 1, "the fixture must offer neighbours")

        for unit in units:
            with self.subTest(source_id=unit["source_id"]):
                self.assertIn("citation_rendered", unit)
                self.assertIn(unit["source_id"], unit["citation_rendered"])

                # The rendered string and nothing else: the five fields a
                # citation is made of are already on the unit, and repeating
                # them would buy the citation at the price of the text.

                self.assertNotIn("citation", unit)

    def test_fetch_and_cite_render_a_source_identically(self):
        """One formatter, so the two can never disagree about one source."""

        hit = self.retrieval.search("TEST-STD", "TEST-1", "3.1")[0]
        fetched = self.retrieval.fetch("TEST-STD", "TEST-1", hit.source_id)
        units = ([fetched["unit"]] + fetched["neighbors"]
                 + ([fetched["parent"]] if fetched["parent"] else []))

        for unit in units:
            with self.subTest(source_id=unit["source_id"]):
                cited = self.retrieval.cite("TEST-STD", "TEST-1",
                                            unit["source_id"])
                self.assertEqual(unit["citation_rendered"], cited.render())
                self.assertEqual(unit["citation_rendered"],
                                 cited.to_dict()["rendered"])

        # And the top-level citation still renders the unit it belongs to.

        self.assertEqual(fetched["citation"]["rendered"],
                         fetched["unit"]["citation_rendered"])

    def test_index_manifest_is_source_bound_and_rebuildable(self):
        rebuilt = rebuild_lexical_index(
            self.store, "TEST-STD", "TEST-1", created_at="later")
        self.assertEqual(self.index.index_fingerprint, rebuilt.index_fingerprint)
        changed = rebuild_lexical_index(
            self.store, "TEST-STD", "TEST-1", indexer_version="bm25-v2",
            created_at="later")
        self.assertEqual(changed.source_corpus_sha256,
                         self.manifest.corpus_manifest_sha256)
        self.assertNotEqual(changed.index_fingerprint, self.index.index_fingerprint)

    def test_index_corruption_is_detected_without_harming_canonical_corpus(self):
        directory = self.store.revision_dir(
            "TEST-STD", "TEST-1") / "indexes" / "lexical"
        index = json.loads((directory / "index.json").read_text("utf-8"))
        index["document_count"] += 1
        (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(StandardStoreError, "fingerprint mismatch"):
            self.store.load_index("TEST-STD", "TEST-1")
        self.assertEqual(self.store.verify_corpus(
            "TEST-STD", "TEST-1").corpus_manifest_sha256,
            self.manifest.corpus_manifest_sha256)
