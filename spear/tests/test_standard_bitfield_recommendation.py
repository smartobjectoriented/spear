"""STD2B-R and STD2C: the advice shown to an operator before they approve.

STD2C changed one rule here deliberately. A label that states a bit range the
geometry contradicts used to be a reason to reject the candidate; it is now a
reason to believe the label, because the measurement is centred text.

All fixtures are synthetic. No licensed normative text appears in this file.
"""

import json
import tempfile
import unittest
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_semantic import SpanRole
from standard_store import StandardStore
from standard_structure import extract_structures
from standard_structure_review import StandardStructureReview, recommend
from standard_structure_store import StandardStructureStore, build_manifest
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes

SID, REV = "SEM", "R1"


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "bitfield.pdf"
        pdf.write_bytes(semantic_bitfield_pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                                   source_origin="TEST_FIXTURE")
        artifact = extract_layout(pdf, pdf_sha256=self.manifest.source_pdf_sha256)
        (self.store.revision_dir(SID, REV) / "layout.json").write_text(
            json.dumps(artifact))
        structures = extract_structures(
            self.store.load_units(SID, REV), artifact, standard_id=SID,
            revision=REV, corpus_fingerprint=self.manifest.corpus_manifest_sha256)
        StandardStructureStore(self.store).save(build_manifest(
            structures, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version), structures)
        self.review = StandardStructureReview(self.store, SID, REV,
                                              reviewer="tester")
        self.table_id = next(iter(self.review.bitfields))

    def tearDown(self):
        self.temp.cleanup()

    def advise(self, mutate=None):
        if mutate is not None:
            mutate(self.review.bitfields[self.table_id])
        return recommend(self.review, self.table_id)

    def relabel(self, index, text):
        """Rename a span and the cell behind it, so provenance is kept."""
        span = self.review.bitfields[self.table_id]["spans"][index]
        for row in self.review.table(self.table_id)["rows"]:
            for cell in row["cells"]:
                if cell["bbox"] == span["bbox"] and cell["text"] == span["text"]:
                    cell["text"] = text
        span["text"] = text

    def test_a_tiled_ruler_recommends_its_spans_as_fields(self):
        advice = self.advise()
        roles = [span["recommended_role"] for span in advice["spans"]]
        self.assertEqual(roles, [SpanRole.FIELD.value, SpanRole.FIELD.value])
        self.assertIn(advice["candidate"], ("RECOMMEND_PASS",
                                            "RECOMMEND_ACCEPTABLE_WARNING"))
        self.assertEqual(advice["bit_order"], "MSB_TO_LSB")

    def test_one_centred_label_is_not_treated_as_a_field(self):
        def drop_second(bitfield):
            bitfield["spans"] = bitfield["spans"][:1]
        advice = self.advise(drop_second)
        self.assertEqual(advice["candidate"], "RECOMMEND_FOLLOWUP")
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.ROW_LABEL.value)
        self.assertTrue(any("tiles the ruler" in span["reason"]
                            for span in advice["spans"]))

    def test_a_stated_range_outranks_a_measurement_that_disagrees(self):
        self.relabel(0, "SUBFIELD_LENGTH (23-12)")
        advice = self.advise()
        span = advice["spans"][0]
        self.assertEqual(span["recommended_role"], SpanRole.FIELD.value)
        self.assertEqual(span["position_source"], "STATED_RANGE")
        self.assertEqual(span["stated"], "23..12")
        self.assertNotEqual(span["geometry"], "23..12")
        self.assertNotEqual(advice["candidate"], "RECOMMEND_REJECT")

    def test_a_stated_range_is_not_downgraded_by_the_tiling_rule(self):
        # One centred label alone never tiles a ruler, but it still states its
        # own bits, so the rule about centred labels must not touch it.
        self.relabel(0, "LONE_FIELD (31-24)")
        def drop_second(bitfield):
            bitfield["spans"] = bitfield["spans"][:1]
        advice = self.advise(drop_second)
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.FIELD.value)
        self.assertEqual(advice["spans"][0]["position_source"], "STATED_RANGE")

    def test_a_label_whose_range_is_refused_is_shown_as_refused(self):
        self.relabel(0, "SCALE_LEVEL (0-9)")
        advice = self.advise()
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.UNKNOWN.value)
        self.assertIn("(0-9)", advice["spans"][0]["refused"])
        self.assertEqual(advice["candidate"], "RECOMMEND_FOLLOWUP")

    def test_the_panel_shows_both_authorities_and_the_status_between_them(self):
        import io

        from standard_structure_review import _render_bitfield
        self.relabel(0, "SUBFIELD_LENGTH (23-12)")
        out = io.StringIO()
        _render_bitfield(self.review, self.table_id, out)
        text = out.getvalue()
        self.assertIn("stated   : 23..12", text)
        self.assertIn("geometry :", text)
        self.assertIn("authority: STATED_RANGE", text)
        self.assertNotIn("claude", text.lower())

    def test_a_label_stating_a_matching_bit_range_stays_a_field(self):
        def restate(bitfield):
            bitfield["spans"][0]["text"] = "PREFIX_SIZE (31..28)"
        advice = self.advise(restate)
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.FIELD.value)
        self.assertNotEqual(advice["candidate"], "RECOMMEND_REJECT")

    def test_a_label_naming_a_whole_word_is_structural(self):
        def restate(bitfield):
            bitfield["spans"][0]["text"] = "Packet Header (1 Word, Mandatory)"
        advice = self.advise(restate)
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.STRUCTURAL_LABEL.value)
        self.assertIn("whole packet word", advice["spans"][0]["reason"])

    def test_a_short_ruler_is_treated_as_a_fragment(self):
        def shorten(bitfield):
            bitfield["bit_labels"] = bitfield["bit_labels"][:5]
        advice = self.advise(shorten)
        self.assertEqual(advice["candidate"], "RECOMMEND_FOLLOWUP")
        self.assertTrue(any("fragment of a longer ruler" in reason
                            for reason in advice["reasons"]))

    def test_a_broken_label_range_is_never_recommended_as_a_field(self):
        def puncture(bitfield):
            bitfield["spans"][0]["covered_labels"] = [31, 29]
        advice = self.advise(puncture)
        self.assertEqual(advice["spans"][0]["recommended_role"],
                         SpanRole.UNKNOWN.value)
        self.assertEqual(advice["candidate"], "RECOMMEND_FOLLOWUP")

    def test_a_recommendation_never_records_a_verdict(self):
        before = dict(self.review.verdicts)
        self.advise()
        self.assertEqual(self.review.verdicts, before)
        self.assertEqual(self.review.summary()["UNREVIEWED"],
                         self.review.summary()["TOTAL"])

    def test_the_suggested_command_names_the_candidate_not_a_reviewer(self):
        import io

        from standard_structure_review import _render_bitfield
        out = io.StringIO()
        _render_bitfield(self.review, self.table_id, out)
        text = out.getvalue()
        self.assertIn("approve-bitfield", text)
        self.assertIn("records your identity", text)
        self.assertNotIn("claude", text.lower())
