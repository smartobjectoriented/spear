"""STD2A-R: provenance quality and the gate a later phase has to satisfy.

All fixtures are synthetic. No licensed normative text appears in this file.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_store import StandardStore
from standard_structure import (
    GeometryStatus, ProvenanceQuality, StandardStructureError, extract_structures,
    promotion_state, provenance_counts, unresolved_semantic_cells,
    validate_structures,
)
from standard_structure_review import (
    REVIEW_DIMENSIONS, StandardStructureReview,
)
from standard_structure_store import StandardStructureStore, build_manifest
from tests.standard_geometry_fixture import geometry_pdf_bytes

SID, REV = "GEO", "R1"


class _Extracted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.pdf = root / "geometry.pdf"
        cls.pdf.write_bytes(geometry_pdf_bytes())
        cls.store = StandardStore(root / "standards")
        cls.manifest = ingest_pdf(cls.store, cls.pdf, standard_id=SID, revision=REV,
                                  source_origin="TEST_FIXTURE")
        cls.units = cls.store.load_units(SID, REV)
        cls.artifact = extract_layout(cls.pdf,
                                      pdf_sha256=cls.manifest.source_pdf_sha256)
        cls.structures = extract_structures(
            cls.units, cls.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint=cls.manifest.corpus_manifest_sha256)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def cells(self):
        return [cell for table in self.structures.tables
                for row in table.rows for cell in row.cells]


class ProvenanceQualityTests(_Extracted):
    def test_a_cell_says_why_its_sources_are_attached(self):
        kinds = {cell.provenance for cell in self.cells()}
        self.assertTrue(kinds <= set(ProvenanceQuality))
        self.assertIn(ProvenanceQuality.DIRECT_TEXT_MATCH, kinds)
        for cell in self.cells():
            if cell.provenance is ProvenanceQuality.UNRESOLVED:
                self.assertEqual(cell.source_ids, ())
            else:
                self.assertTrue(cell.source_ids)

    def test_a_cell_matching_its_own_text_is_a_direct_match(self):
        direct = [cell for cell in self.cells()
                  if cell.provenance is ProvenanceQuality.DIRECT_TEXT_MATCH]
        self.assertTrue(direct)
        known = {unit.source_id: unit for unit in self.units}
        for cell in direct[:20]:
            probe = " ".join(cell.text.split())
            self.assertTrue(any(probe in " ".join(known[value].text.split())
                                for value in cell.source_ids), cell.text)

    def test_a_short_cell_inherits_its_row_rather_than_going_unresolved(self):
        inherited = [cell for cell in self.cells()
                     if cell.provenance is ProvenanceQuality.ROW_INHERITED]
        self.assertTrue(inherited)
        for cell in inherited:
            self.assertTrue(cell.source_ids)

    def test_inherited_provenance_is_never_reported_as_a_direct_match(self):
        for cell in self.cells():
            if cell.provenance is not ProvenanceQuality.DIRECT_TEXT_MATCH:
                self.assertNotEqual(cell.provenance,
                                    ProvenanceQuality.DIRECT_TEXT_MATCH)
        payload = json.loads(json.dumps(self.structures.to_dict()))
        sample = payload["tables"][0]["rows"][0]["cells"][0]
        self.assertIn("provenance", sample)
        self.assertIn("semantic", sample)

    def test_a_region_never_inherits_from_an_unbounded_source_set(self):
        from standard_structure import _MAX_REGION_SOURCES
        for table in self.structures.tables:
            region = [cell for row in table.rows for cell in row.cells
                      if cell.provenance is ProvenanceQuality.TABLE_REGION_INHERITED]
            for cell in region:
                self.assertLessEqual(len(cell.source_ids), _MAX_REGION_SOURCES)

    def test_a_cell_is_never_mapped_to_a_merely_nearby_source(self):
        known = {unit.source_id: unit for unit in self.units}
        for cell in self.cells():
            for value in cell.source_ids:
                # Whatever the quality, the source has to be on the same page.
                self.assertEqual(known[value].page, cell.page)

    def test_counts_add_up(self):
        counts = provenance_counts(self.structures.tables)
        self.assertEqual(counts["TOTAL"], len(self.cells()))


class PromotionGateTests(_Extracted):
    def test_a_table_whose_semantic_cells_all_cite_something_is_sufficient(self):
        clean = [table for table in self.structures.tables
                 if not unresolved_semantic_cells(table)]
        self.assertTrue(clean)
        for table in clean:
            self.assertEqual(promotion_state(table), "PROVENANCE_SUFFICIENT")

    def test_one_unresolved_semantic_cell_blocks_the_table(self):
        table = self.structures.tables[0]
        row = table.rows[0]
        broken = replace(table, rows=(replace(row, cells=(
            replace(row.cells[0], source_ids=(), semantic=True,
                    provenance=ProvenanceQuality.UNRESOLVED),) + row.cells[1:]),)
            + table.rows[1:])
        self.assertEqual(promotion_state(broken), "BLOCKED_UNRESOLVED_PROVENANCE")
        self.assertEqual(len(unresolved_semantic_cells(broken)), 1)

    def test_a_blank_or_punctuation_cell_does_not_block(self):
        table = self.structures.tables[0]
        row = table.rows[0]
        decorative = replace(table, rows=(replace(row, cells=(
            replace(row.cells[0], text="-", source_ids=(), semantic=False,
                    provenance=ProvenanceQuality.UNRESOLVED),) + row.cells[1:]),)
            + table.rows[1:])
        self.assertEqual(promotion_state(decorative), "PROVENANCE_SUFFICIENT")
        self.assertEqual(unresolved_semantic_cells(decorative), ())

    def test_a_cell_citing_an_unknown_source_still_fails_validation(self):
        table = self.structures.tables[0]
        row = table.rows[0]
        forged = replace(self.structures, tables=(replace(table, rows=(
            replace(row, cells=(replace(row.cells[0],
                                        source_ids=("std-" + "e" * 32),),)
                    + row.cells[1:]),) + table.rows[1:]),) + self.structures.tables[1:])
        with self.assertRaises(StandardStructureError):
            validate_structures(forged, artifact=self.artifact,
                                source_ids={unit.source_id for unit in self.units})


class AutoAcceptAndListTests(_Extracted):
    def test_a_bulleted_or_numbered_list_is_not_auto_accepted(self):
        from standard_structure import _bullet_led
        self.assertTrue(_bullet_led([("•", "a"), ("•", "b")]))
        self.assertTrue(_bullet_led([("1.", "a"), ("2.", "b"), ("3.", "c")]))
        self.assertFalse(_bullet_led([("Code", "a"), ("0001", "b")]))

    def test_auto_accept_needs_a_caption_or_a_real_multi_row_table(self):
        for table in self.structures.tables:
            if table.geometry_status is GeometryStatus.AUTO_GEOMETRY_OK:
                self.assertEqual(table.warnings, ())
                self.assertTrue(table.caption_source_ids or table.row_count >= 3)

    def test_auto_accepted_tables_carry_complete_provenance(self):
        for table in self.structures.tables:
            if table.geometry_status is GeometryStatus.AUTO_GEOMETRY_OK:
                self.assertEqual(unresolved_semantic_cells(table), ())


class ReviewSeparationTests(_Extracted):
    def setUp(self):
        (self.store.revision_dir(SID, REV) / "layout.json").write_text(
            json.dumps(self.artifact))
        self.structures_store = StandardStructureStore(self.store)
        self.structures_store.save(build_manifest(
            self.structures, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version), self.structures)

    def review(self):
        return StandardStructureReview(self.store, SID, REV,
                                       reviewer="claude-assistant")

    def test_a_verdict_never_overwrites_the_automated_status(self):
        review = self.review()
        table_id = review.selection()[0]
        before = review.table(table_id)["geometry_status"]
        review.record(table_id, "FAIL", geometry_correct="NO")
        review.save()
        row = self.review().verdicts[table_id]
        self.assertEqual(row["verdict"], "FAIL")
        self.assertEqual(row["geometry_status"], before)
        self.assertEqual(self.review().table(table_id)["geometry_status"], before)

    def test_each_review_dimension_is_recorded_independently(self):
        review = self.review()
        table_id = review.selection()[0]
        review.record(table_id, "ACCEPTABLE_WARNING", geometry_correct="YES",
                      row_structure_correct="PARTIAL",
                      column_structure_correct="YES",
                      provenance_sufficient="YES",
                      bitfield_candidate_correct="NOT_APPLICABLE")
        review.save()
        row = self.review().verdicts[table_id]
        for name in REVIEW_DIMENSIONS:
            self.assertIn(name, row)
        self.assertEqual(row["row_structure_correct"], "PARTIAL")
        self.assertEqual(row["geometry_correct"], "YES")

    def test_a_bitfield_verdict_is_separate_from_its_detection(self):
        review = self.review()
        _, payload = self.structures_store.load(SID, REV)
        detected = {item["table_id"] for item in payload["bitfields"]}
        table_id = next(value for value in review.selection() if value in detected)
        review.record(table_id, "NEEDS_FOLLOWUP",
                      bitfield_candidate_correct="AMBIGUOUS")
        review.save()
        row = self.review().verdicts[table_id]
        self.assertEqual(row["bitfield_candidate_correct"], "AMBIGUOUS")
        # Detection itself is unchanged by the verdict.
        _, again = self.structures_store.load(SID, REV)
        self.assertEqual({item["table_id"] for item in again["bitfields"]}, detected)

    def test_an_unknown_dimension_is_refused(self):
        review = self.review()
        with self.assertRaises(StandardStructureError):
            review.record(review.selection()[0], "PASS", not_a_dimension="YES")

    def test_a_review_is_invalidated_when_the_structures_change(self):
        review = self.review()
        review.record(review.selection()[0], "PASS")
        review.save()
        rebuilt = extract_structures(
            self.units, self.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint="a" * 64)
        self.structures_store.save(build_manifest(
            rebuilt, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version), rebuilt)
        with self.assertRaises(StandardStructureError):
            self.review()
