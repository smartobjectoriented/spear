"""STD2A: deterministic table and bitfield geometry. All fixtures are synthetic.

No licensed normative text appears in this file.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_schema import StandardContentType
from standard_store import StandardStore
from standard_structure import (
    GeometryStatus, HeaderCandidate, StandardStructureError, extract_structures,
    structure_fingerprint, validate_structures,
)
from standard_structure_store import (
    StandardStructureStore, build_manifest,
)
from standard_structure_review import (
    StandardStructureReview, UNREVIEWED, run_review, sample_tables,
)
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from tests.standard_geometry_fixture import geometry_pdf_bytes
from tool_registry import ToolRegistry

SID, REV = "GEO", "R1"


class _Geometry(unittest.TestCase):
    """Extracts the geometry fixture once per test class."""

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

    def on_page(self, page):
        return [table for table in self.structures.tables if table.page_start == page]

    def one_on(self, page):
        tables = self.on_page(page)
        self.assertEqual(len(tables), 1, f"expected one table on page {page}")
        return tables[0]


class TableGeometryTests(_Geometry):
    def test_a_simple_table_is_found_with_its_rows_and_columns(self):
        table = self.one_on(1)
        self.assertEqual((table.row_count, table.column_count), (4, 3))
        self.assertEqual(table.grid()[0], ("Code", "Meaning", "Size"))
        self.assertEqual(table.geometry_status, GeometryStatus.AUTO_GEOMETRY_OK)

    def test_aligned_prose_and_a_numbered_list_are_not_tables(self):
        self.assertEqual(self.on_page(4), [])

    def test_a_wrapped_cell_stays_in_one_row(self):
        table = self.one_on(2)
        self.assertEqual(table.row_count, 4)
        wrapped = table.grid()[1][2]
        self.assertIn("Turns the channel on", wrapped)
        self.assertIn("when the flag is set", wrapped)

    def test_a_blank_cell_leaves_a_gap_rather_than_shifting_the_row(self):
        table = self.one_on(2)
        self.assertEqual(table.grid()[2], ("6", "Reserved", ""))

    def test_a_header_row_is_recognised_and_a_bit_ruler_is_not(self):
        self.assertEqual(self.one_on(1).rows[0].header_candidate,
                         HeaderCandidate.TRUE)
        self.assertNotEqual(self.one_on(3).rows[0].header_candidate,
                            HeaderCandidate.TRUE)

    def test_a_caption_is_attached_only_when_the_classifier_agrees(self):
        self.assertTrue(self.one_on(1).caption_source_ids)
        captions = self.one_on(1).caption_source_ids
        unit = self.store.resolve_source(SID, REV, captions[0])
        self.assertEqual(unit.content_type, StandardContentType.TABLE)

    def test_prose_about_a_figure_or_table_never_becomes_a_caption(self):
        table = self.one_on(5)
        self.assertEqual(len(table.caption_source_ids), 1)
        caption = self.store.resolve_source(SID, REV, table.caption_source_ids[0])
        self.assertTrue(caption.text.strip().startswith("Table 4:"))

    def test_a_table_without_a_caption_is_still_a_table(self):
        tables = self.on_page(6)
        self.assertEqual(len(tables), 2)
        self.assertTrue(all(not table.caption_source_ids for table in tables))

    def test_two_tables_on_one_page_stay_separate(self):
        left, right = sorted(self.on_page(6), key=lambda table: table.bbox[1])
        self.assertEqual(left.column_count, 2)
        self.assertEqual(right.column_count, 3)
        self.assertNotEqual(left.table_id, right.table_id)

    def test_a_heading_and_a_caption_are_not_read_as_table_rows(self):
        table = self.one_on(10)
        self.assertEqual(table.grid()[0], ("Group", "Min", "Max"))
        self.assertEqual(table.row_count, 4)

    def test_a_sub_header_and_a_sparse_row_survive(self):
        grid = self.one_on(10).grid()
        self.assertEqual(grid[1], ("Timing", "", ""))
        self.assertEqual(grid[3], ("Stop", "", "31"))

    def test_rows_and_columns_are_ordered(self):
        for table in self.structures.tables:
            self.assertEqual([row.row_index for row in table.rows],
                             list(range(table.row_count)))
            for row in table.rows:
                indexes = [cell.column_index for cell in row.cells]
                self.assertEqual(indexes, sorted(indexes))


class ContinuationTests(_Geometry):
    def test_a_table_crossing_a_page_is_linked_not_merged(self):
        self.assertEqual(len(self.structures.continuations), 1)
        link = self.structures.continuations[0]
        source = self.one_on(7)
        self.assertEqual(link.from_table_id, source.table_id)
        self.assertEqual(source.continuation_ids, (link.to_table_id,))
        # Linked, never merged: the pages stay separate candidates.
        self.assertEqual(source.page_start, source.page_end)

    def test_a_repeated_header_is_recorded_as_evidence(self):
        link = self.structures.continuations[0]
        self.assertTrue(any("header row repeated" in item for item in link.evidence))
        self.assertEqual(link.warnings, ())

    def test_an_incompatible_next_page_is_not_a_continuation(self):
        linked = {link.to_table_id for link in self.structures.continuations}
        for table in self.on_page(9):
            self.assertNotIn(table.table_id, linked)


class BitfieldTests(_Geometry):
    def test_a_bit_ruler_and_the_labels_a_field_spans_are_recorded(self):
        self.assertEqual(len(self.structures.bitfields), 1)
        bitfield = self.structures.bitfields[0]
        self.assertEqual([label.value for label in bitfield.bit_labels],
                         [31, 30, 29, 28, 27, 26, 25, 24])
        spans = {span.text: span.covered_labels for span in bitfield.spans}
        self.assertIn("Stream Class", spans)
        self.assertEqual(spans["Stream Class"], (31, 30))

    def test_a_bitfield_candidate_asserts_no_packet_semantics(self):
        payload = self.structures.bitfields[0].to_dict()
        for forbidden in ("msb", "lsb", "width", "mask", "shift", "type"):
            self.assertNotIn(forbidden, json.dumps(payload))
        self.assertEqual(self.structures.bitfields[0].review_status, UNREVIEWED)

    def test_an_ordinary_numeric_table_is_not_a_bitfield(self):
        numeric = self.one_on(11)
        self.assertEqual(numeric.grid()[0], ("Sample", "Low", "High", "Mean"))
        self.assertNotIn(numeric.table_id,
                         {item.table_id for item in self.structures.bitfields})


class ProvenanceAndIdentityTests(_Geometry):
    def test_every_cell_traces_back_to_canonical_sources(self):
        known = {unit.source_id for unit in self.units}
        cells = [cell for table in self.structures.tables
                 for row in table.rows for cell in row.cells]
        self.assertTrue(cells)
        for cell in cells:
            self.assertTrue(set(cell.source_ids) <= known)
        mapped = sum(1 for cell in cells if cell.source_ids)
        self.assertEqual(mapped, len(cells))

    def test_identifiers_are_deterministic_for_the_same_input(self):
        again = extract_structures(
            self.units, self.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256)
        self.assertEqual([table.table_id for table in again.tables],
                         [table.table_id for table in self.structures.tables])
        self.assertEqual(
            [cell.cell_id for table in again.tables for row in table.rows
             for cell in row.cells],
            [cell.cell_id for table in self.structures.tables for row in table.rows
             for cell in row.cells])
        self.assertEqual(structure_fingerprint(again),
                         structure_fingerprint(self.structures))

    def test_identifiers_change_when_the_corpus_they_describe_changes(self):
        other = extract_structures(
            self.units, self.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint="b" * 64)
        self.assertNotEqual([table.table_id for table in other.tables],
                            [table.table_id for table in self.structures.tables])

    def test_each_candidate_pins_the_corpus_and_layout_it_came_from(self):
        for table in self.structures.tables:
            self.assertEqual(table.corpus_fingerprint,
                             self.manifest.corpus_manifest_sha256)
            self.assertEqual(table.layout_fingerprint,
                             self.artifact["layout_fingerprint"])


class GeometryValidationTests(_Geometry):
    def source_ids(self):
        return frozenset(unit.source_id for unit in self.units)

    def test_valid_geometry_passes(self):
        validate_structures(self.structures, artifact=self.artifact,
                            source_ids=self.source_ids())

    def test_a_non_finite_or_inverted_box_is_rejected(self):
        from dataclasses import replace
        for box in ((float("nan"), 0.0, 1.0, 1.0), (10.0, 10.0, 1.0, 1.0)):
            broken = replace(self.structures,
                             tables=(replace(self.structures.tables[0], bbox=box),)
                             + self.structures.tables[1:])
            with self.assertRaises(StandardStructureError):
                validate_structures(broken, artifact=self.artifact,
                                    source_ids=self.source_ids())

    def test_a_box_outside_its_page_is_rejected(self):
        from dataclasses import replace
        broken = replace(self.structures,
                         tables=(replace(self.structures.tables[0],
                                         bbox=(0.0, 0.0, 9999.0, 9999.0)),)
                         + self.structures.tables[1:])
        with self.assertRaises(StandardStructureError):
            validate_structures(broken, artifact=self.artifact,
                                source_ids=self.source_ids())

    def test_an_unknown_source_or_page_is_rejected(self):
        from dataclasses import replace
        broken = replace(self.structures,
                         tables=(replace(self.structures.tables[0],
                                         supporting_source_ids=("std-" + "f" * 32,)),)
                         + self.structures.tables[1:])
        with self.assertRaises(StandardStructureError):
            validate_structures(broken, artifact=self.artifact,
                                source_ids=self.source_ids())
        moved = replace(self.structures,
                        tables=(replace(self.structures.tables[0], page_start=999),)
                        + self.structures.tables[1:])
        with self.assertRaises(StandardStructureError):
            validate_structures(moved, artifact=self.artifact,
                                source_ids=self.source_ids())

    def test_a_continuation_to_an_unknown_table_is_rejected(self):
        from dataclasses import replace
        from standard_structure import StandardTableContinuation
        broken = replace(self.structures, continuations=(
            StandardTableContinuation("tbl-0000000000000000",
                                      "tbl-1111111111111111"),))
        with self.assertRaises(StandardStructureError):
            validate_structures(broken, artifact=self.artifact,
                                source_ids=self.source_ids())


class ToolSurfaceTests(_Geometry):
    def test_structures_are_not_reachable_by_the_model(self):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        names = {spec.name for spec in registry.list_specs()}
        self.assertEqual(names, set(STANDARD_TOOL_NAMES))
        for forbidden in ("standard.validate",
                          "standard.structures", "standard.tables"):
            self.assertNotIn(forbidden, names)
        import standard_structure, standard_structure_review
        self.assertFalse(hasattr(standard_structure, "register"))
        self.assertFalse(hasattr(standard_structure_review, "register"))

    def test_extraction_needs_no_vector_index(self):
        indexes = self.store.revision_dir(SID, REV) / "indexes"
        self.assertFalse((indexes / "vector").exists())
        self.assertTrue(self.structures.tables)

    def test_the_canonical_corpus_is_untouched_by_extraction(self):
        after = self.store.verify_corpus(SID, REV)
        self.assertEqual(after.corpus_manifest_sha256,
                         self.manifest.corpus_manifest_sha256)
        self.assertEqual({unit.source_id for unit in self.store.load_units(SID, REV)},
                         {unit.source_id for unit in self.units})
