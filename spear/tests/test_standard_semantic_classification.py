"""STD1F: semantic content-type classification. Every fixture here is synthetic.

No licensed normative text appears in this file.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_ingest import (
    canonical_units, extract_pdf_pages, ingest_pdf, verify_against_contents,
)
from standard_review import (
    UNREVIEWED, StandardReview, build_review_sample, write_review_sample,
)
from standard_schema import StandardContentType, StandardLayoutKind, StandardModality
from standard_store import StandardStore
from tests.standard_extraction_fixture import semantic_pdf_bytes


class SemanticClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        pdf = Path(cls.temp.name) / "semantic.pdf"
        pdf.write_bytes(semantic_pdf_bytes())
        pages, _ = extract_pdf_pages(pdf)
        cls.units = canonical_units(pages, standard_id="SEM", revision="R1",
                                    pdf_sha256="a" * 64)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def unit_for(self, phrase):
        matches = [unit for unit in self.units if phrase in unit.text]
        self.assertEqual(len(matches), 1, f"expected one unit for {phrase!r}")
        return matches[0]

    # -- modality is evidence, not a verdict ------------------------------
    def test_front_matter_keeps_its_modality_but_is_not_a_requirement(self):
        boilerplate = self.unit_for("No person shall have authority")
        self.assertTrue(boilerplate.is_front_matter)
        self.assertEqual(boilerplate.content_type, StandardContentType.FRONT_MATTER)
        # The words are still there; erasing them would lose real evidence.
        self.assertEqual(boilerplate.modality, StandardModality.SHALL)

    def test_a_labelled_paragraph_keeps_its_modality_too(self):
        observation = self.unit_for("Rule 4-1")
        self.assertEqual(observation.modality, StandardModality.SHALL)

    # -- genuine normative body text ---------------------------------------
    def test_body_normative_statements_stay_normative(self):
        for phrase, modality in (("shall reject reserved values", StandardModality.SHALL),
                                 ("must reject malformed packets", StandardModality.MUST)):
            unit = self.unit_for(phrase)
            self.assertEqual(unit.content_type, StandardContentType.REQUIREMENT, phrase)
            self.assertEqual(unit.modality, modality, phrase)

    def test_should_and_may_keep_their_modality(self):
        recommendation = self.unit_for("should retain the identifier")
        self.assertEqual(recommendation.content_type, StandardContentType.RECOMMENDATION)
        self.assertEqual(recommendation.modality, StandardModality.SHOULD)
        permission = self.unit_for("may omit the optional field")
        self.assertEqual(permission.modality, StandardModality.MAY)
        self.assertNotEqual(permission.content_type, StandardContentType.REQUIREMENT)

    # -- definitions --------------------------------------------------------
    def test_a_term_followed_by_means_is_a_definition(self):
        self.assertEqual(self.unit_for("Transport Identifier means").content_type,
                         StandardContentType.DEFINITION)

    def test_means_that_and_by_means_of_are_not_definitions(self):
        for phrase in ("This approach means that", "carried by means of"):
            self.assertNotEqual(self.unit_for(phrase).content_type,
                                StandardContentType.DEFINITION, phrase)

    # -- coarse units --------------------------------------------------------
    def test_a_unit_merging_two_labelled_paragraphs_is_flagged_but_stays_normative(self):
        coarse = self.unit_for("Rule 4-1")
        self.assertEqual(coarse.content_type, StandardContentType.REQUIREMENT)
        self.assertIn("unit merges more than one labelled paragraph", coarse.warnings)

    def test_a_single_labelled_paragraph_is_not_flagged_as_coarse(self):
        single = self.unit_for("shall reject reserved values")
        self.assertNotIn("unit merges more than one labelled paragraph", single.warnings)


if __name__ == "__main__":
    unittest.main()


class ReviewSampleCoverageTests(unittest.TestCase):
    """The sample must review what its stratum names."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "semantic.pdf"
        pdf.write_bytes(semantic_pdf_bytes())
        self.store = StandardStore(root / "standards")
        ingest_pdf(self.store, pdf, standard_id="SEM", revision="R1",
                   source_origin="TEST_FIXTURE")
        self.units = {unit.source_id: unit
                      for unit in self.store.load_units("SEM", "R1")}

    def tearDown(self):
        self.temp.cleanup()

    def rows_for(self, document, stratum):
        return [row for row in document["rows"] if row["reason"] == stratum]

    def test_the_annex_stratum_picks_a_body_opener_not_a_contents_entry(self):
        document = build_review_sample(self.store, "SEM", "R1")
        rows = self.rows_for(document, "annex_heading")
        self.assertTrue(rows, "annex stratum selected nothing")
        for row in rows:
            unit = self.units[row["candidate_source_id"]]
            self.assertFalse(unit.is_front_matter)
            self.assertFalse(unit.is_toc_entry)
            self.assertIn("annex or appendix heading resets clause numbering",
                          unit.warnings)
        # The contents page does list the same annex; it must not be the sample.
        listed = [unit for unit in self.units.values()
                  if unit.is_toc_entry and "Annex" in unit.text]
        self.assertTrue(listed, "fixture should list the annex in its contents")
        self.assertFalse({unit.source_id for unit in listed}
                         & {row["candidate_source_id"] for row in rows})

    def test_every_sampled_row_resolves_and_loads_as_a_review(self):
        target, archived = write_review_sample(self.store, "SEM", "R1")
        self.assertIsNone(archived)
        review = StandardReview.load(self.store, "SEM", "R1", reviewer="tester")
        self.assertTrue(review.items())
        self.assertEqual(review.summary()[UNREVIEWED], len(review.items()))
        for item in review.items():
            self.assertIn(item.source_id, self.units)

    def test_rebuilding_a_sample_keeps_the_previous_review_as_history(self):
        write_review_sample(self.store, "SEM", "R1")
        review = StandardReview.load(self.store, "SEM", "R1", reviewer="tester")
        review.record(0, "PASS")
        review.save()
        target, archived = write_review_sample(self.store, "SEM", "R1")
        self.assertIsNotNone(archived)
        self.assertTrue(archived.is_file())
        kept = json.loads(archived.read_text())
        self.assertEqual(kept["rows"][0]["human_verdict"], "PASS")
        fresh = json.loads(target.read_text())
        self.assertIsNone(fresh["rows"][0]["human_verdict"])

    def test_review_and_sampling_need_no_vector_index(self):
        indexes = self.store.revision_dir("SEM", "R1") / "indexes"
        self.assertFalse((indexes / "vector").exists())
        write_review_sample(self.store, "SEM", "R1")
        StandardReview.load(self.store, "SEM", "R1", reviewer="tester")


class StructuralGateTests(unittest.TestCase):
    """Classification must not be able to move the structural verification."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        pdf = Path(self.temp.name) / "semantic.pdf"
        pdf.write_bytes(semantic_pdf_bytes())
        pages, _ = extract_pdf_pages(pdf)
        self.units = canonical_units(pages, standard_id="SEM", revision="R1",
                                     pdf_sha256="a" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_verify_ignores_content_type_and_modality_entirely(self):
        before = verify_against_contents(self.units)
        scrambled = tuple(
            replace(unit, content_type=StandardContentType.UNKNOWN,
                    modality=StandardModality.NONE)
            for unit in self.units)
        self.assertEqual(verify_against_contents(scrambled), before)
        self.assertEqual(before["page_agreement"], before["clauses_in_both"])
        self.assertGreater(before["clauses_in_both"], 0)

    def test_classification_registers_no_tool_of_its_own(self):
        from standard_tools import STANDARD_TOOL_NAMES
        self.assertEqual(set(STANDARD_TOOL_NAMES),
                         {"standard.search", "standard.fetch", "standard.cite",
                          "standard.get_structure"})
        self.assertNotIn("standard.validate", STANDARD_TOOL_NAMES)
