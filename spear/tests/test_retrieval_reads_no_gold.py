"""Retrieval must work with no evaluation directory present at all.

The benchmark consumes retrieval; retrieval never consumes the benchmark. A
retriever that could read its own gold -- expected provisions, case ids,
acceptable evidence sets -- would be tunable to questions it is supposed to
answer blind, and every measurement taken with it would be worthless.

This is cheap to state and easy to lose: the store and the benchmark share a
directory, and `evaluation/` sits beside `corpus/`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from standard_crossrefs import rebuild_cross_reference_index
from standard_ingest import ingest_pdf
from standard_retrieval import StandardRetrieval, rebuild_lexical_index
from standard_store import StandardStore
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_standard_hybrid_retrieval import HYBRID_PAGES


class RetrievalNeverReadsTheBenchmark(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        pdf = root / "synth.pdf"
        pdf.write_bytes(synthetic_pdf_bytes(HYBRID_PAGES))
        self.store = StandardStore(root / "standards")
        ingest_pdf(self.store, pdf, standard_id="SYNTH-STD", revision="R1",
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.store, "SYNTH-STD", "R1")
        rebuild_cross_reference_index(self.store, "SYNTH-STD", "R1")
        self.revision = self.store.revision_dir("SYNTH-STD", "R1")

    def search(self):
        return StandardRetrieval(self.store).search_response(
            "SYNTH-STD", "R1", "reserved identifier", mode="lexical")

    def test_it_works_with_no_evaluation_directory(self):
        shutil.rmtree(self.revision / "evaluation", ignore_errors=True)

        self.assertTrue(self.search().results)

    def test_it_works_when_gold_is_present_but_unreadable(self):
        """Present-and-unreadable is the shape that catches a reader: a
        retriever that never opens the file does not care that it is
        corrupt."""
        evaluation = self.revision / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / "acceptable-evidence-sets.json").write_text("{ not json")
        (evaluation / "normative-matrix-candidate.json").write_text("{ not json")

        self.assertTrue(self.search().results)

    def test_planted_gold_does_not_change_the_ranking(self):
        """The decisive one. If ranking moved when gold appeared, something
        read it."""
        shutil.rmtree(self.revision / "evaluation", ignore_errors=True)
        without = [item.source_id for item in self.search().results]

        evaluation = self.revision / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / "acceptable-evidence-sets.json").write_text(json.dumps(
            {"cases": [{"case_id": "X1",
                        "acceptable_evidence_sets": [[without[-1]]]}]}))
        with_gold = [item.source_id for item in self.search().results]

        self.assertEqual(without, with_gold)


if __name__ == "__main__":
    unittest.main()
