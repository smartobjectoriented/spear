"""A table the parser read survives extraction, storage and reload.

Before this, a parser that knew the grid had to throw it away at the store
boundary: `StandardDocumentUnit` had nowhere to put a table id, a row, its
ordered columns or the spans behind each cell. The runtime then rebuilt the
relationship from what sat next to what -- guessing at something it had
already been told, and getting a cell identity only by accident.

Everything here is synthetic. No column name, row key or table label below
means anything to the code under test; they are only strings the document
prints.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import provision_identity as pi
from standard_schema import (
    StandardContentType, StandardDocumentUnit, StandardIngestionManifest,
    StandardModality, StandardTableCell, StandardTableStructure,
    HumanValidationStatus, source_content_sha256,
)
from standard_store import StandardStore, corpus_fingerprint

SID, REV = "ACME-1", "2030"
PDF = "a" * 64


def cell(index, text, column="", spans=()):
    return StandardTableCell(column_index=index, text=text, column=column,
                             span_ids=tuple(spans))


def row(source, section, table, row_key, cells, *, columns=(), index=0,
        text=None, caption="", page=7):
    body = text if text is not None else " ".join(c.text for c in cells if c.text)
    return StandardDocumentUnit(
        source_id=source, standard_id=SID, revision=REV, section=section,
        page=page, heading_path=(), content_type=StandardContentType.TABLE,
        modality=StandardModality.NONE, text=body or "-",
        source_pdf_sha256=PDF,
        source_content_sha256=source_content_sha256(body or "-"),
        extractor_version="test",
        table_structure=StandardTableStructure(
            table_id=table, row_index=index, columns=tuple(columns),
            cells=tuple(cells), row_key=row_key, caption=caption, page=page))


def legacy(source, section, text):
    """A unit from a store written before the grid could be kept."""
    return StandardDocumentUnit(
        source_id=source, standard_id=SID, revision=REV, section=section,
        page=7, heading_path=(), content_type=StandardContentType.TABLE,
        modality=StandardModality.NONE, text=text, source_pdf_sha256=PDF,
        source_content_sha256=source_content_sha256(text),
        extractor_version="test", schema_version=2)


def store_and_reload(units):
    """Through a real StandardStore, on disk, and back."""
    directory = tempfile.TemporaryDirectory()
    units = tuple(sorted(units, key=lambda u: u.source_id))
    store = StandardStore(Path(directory.name))
    store.save_ingestion(
        StandardIngestionManifest(
            standard_id=SID, revision=REV, source_pdf_sha256=PDF,
            logical_source_filename="acme.pdf",
            ingestion_timestamp="2030-01-01T00:00:00+00:00",
            extractor_version="test", page_count=9,
            canonical_unit_count=len(units), section_count=1,
            requirement_count=0, recommendation_count=0, definition_count=0,
            table_figure_marker_count=len(units), warnings=(),
            extraction_errors=(),
            human_validation_status=HumanValidationStatus.NOT_REVIEWED,
            corpus_manifest_sha256=corpus_fingerprint(units)),
        units)
    reloaded = store.load_units(SID, REV)
    directory.cleanup()

    return reloaded


def evidence_of(units):
    """What the runtime derives, from the payload a fetch would return."""
    return pi.records_for_units([unit.to_dict() for unit in units])


def rows_of(units):
    return [record for record in evidence_of(units)
            if isinstance(record.key, pi.TableRowKey)]


class AGridSurvivesTheStore(unittest.TestCase):
    """extraction -> serialization -> reload -> runtime evidence."""

    COLUMNS = ("Key", "Short", "Meaning")
    CELLS = (cell(0, "20", "Key", ("sp-0001",)),
             cell(1, "Alpha", "Short", ("sp-0002",)),
             cell(2, "the first one", "Meaning", ("sp-0003", "sp-0004")))

    def setUp(self):
        self.reloaded = store_and_reload(
            [row("std-" + "1" * 32, "4.2", "4.2-1", "20", self.CELLS,
                 columns=self.COLUMNS, index=3, caption="Table 4.2-1: Keys")])
        self.record = rows_of(self.reloaded)[0]

    def test_the_row_key_is_the_one_the_document_prints(self):
        self.assertEqual(self.record.key, pi.TableRowKey("4.2", "4.2-1", 20))

    def test_the_cells_keep_their_column_names(self):
        self.assertEqual([str(item["key"]) for item in self.record.cells],
                         ["TableCell 4.2-1:20[Key]",
                          "TableCell 4.2-1:20[Short]",
                          "TableCell 4.2-1:20[Meaning]"])

    def test_the_columns_keep_their_order(self):
        self.assertEqual(self.reloaded[0].table_structure.columns, self.COLUMNS)

    def test_each_cell_keeps_its_exact_text(self):
        self.assertEqual([item["text"] for item in self.record.cells],
                         ["20", "Alpha", "the first one"])

    def test_each_cell_keeps_its_canonical_spans(self):
        self.assertEqual([item["span_ids"] for item in self.record.cells],
                         [("sp-0001",), ("sp-0002",), ("sp-0003", "sp-0004")])

    def test_the_page_provenance_survives(self):
        self.assertEqual(self.reloaded[0].table_structure.page, 7)

    def test_the_identity_is_recorded_as_native(self):
        self.assertEqual(self.record.structure_origin, pi.NATIVE)


class TwoTablesMayNumberTheirRowsAlike(unittest.TestCase):
    """The row number is the document's, not a global one. Two tables in one
    section both having a row 20 is ordinary, and collapsing them onto one
    identity would let an approval of one answer for the other."""

    def setUp(self):
        self.units = store_and_reload([
            row("std-" + "1" * 32, "4.2", "4.2-1", "20",
                (cell(0, "20", "Key"), cell(1, "Alpha", "Short"))),
            row("std-" + "2" * 32, "4.2", "4.2-2", "20",
                (cell(0, "20", "Key"), cell(1, "Beta", "Short"))),
        ])

    def test_the_two_rows_are_two_identities(self):
        self.assertEqual({str(record.key) for record in rows_of(self.units)},
                         {"TableRow 4.2-1:20", "TableRow 4.2-2:20"})

    def test_their_cells_are_two_identities_too(self):
        found = {str(item["key"]) for record in rows_of(self.units)
                 for item in record.cells}

        self.assertIn("TableCell 4.2-1:20[Short]", found)
        self.assertIn("TableCell 4.2-2:20[Short]", found)


class CellsKeepWhatIsInThem(unittest.TestCase):

    def test_a_multiline_cell_keeps_its_words(self):
        units = store_and_reload([
            row("std-" + "3" * 32, "4.2", "4.2-1", "21",
                (cell(0, "21", "Key"),
                 cell(1, "Set to 1: enabled\nSet to 0: disabled", "Meaning")))])
        cells = rows_of(units)[0].cells

        self.assertEqual(cells[1]["text"],
                         "Set to 1: enabled Set to 0: disabled")

    def test_an_empty_cell_keeps_its_column(self):
        """A blank cell is evidence that the document left it blank. Dropping
        it would shift every column after it by one."""
        units = store_and_reload([
            row("std-" + "4" * 32, "4.2", "4.2-1", "22",
                (cell(0, "22", "Key"), cell(1, "", "Short"),
                 cell(2, "something", "Meaning")))])
        cells = rows_of(units)[0].cells

        self.assertEqual([str(item["key"]).split("[")[1] for item in cells],
                         ["Key]", "Short]", "Meaning]"])
        self.assertEqual(cells[1]["text"], "")


class ATableNeedNotBeCaptioned(unittest.TestCase):
    """A table the document never labelled still has an identity, because the
    parser gave it one. What must never happen is the identity coming from
    array position."""

    def setUp(self):
        self.units = store_and_reload([
            row("std-" + "5" * 32, "4.2", "p7-t1", "20",
                (cell(0, "20", ""), cell(1, "Alpha", "")), caption="")])

    def test_it_still_has_a_row_identity(self):
        self.assertEqual(str(rows_of(self.units)[0].key), "TableRow p7-t1:20")

    def test_unnamed_columns_fall_back_to_position(self):
        self.assertEqual([str(item["key"]) for item in rows_of(self.units)[0].cells],
                         ["TableCell p7-t1:20[col0]", "TableCell p7-t1:20[col1]"])


class ALegacyStoreStillLoads(unittest.TestCase):
    """A corpus ingested before the grid could be kept must keep working, and
    keep its fingerprint: it serializes exactly as it was stored."""

    def setUp(self):
        self.units = store_and_reload(
            [legacy("std-" + "6" * 32, "4.2", "20 Alpha the first one")])

    def test_it_carries_no_table_structure(self):
        self.assertIsNone(self.units[0].table_structure)

    def test_its_serialization_does_not_grow_a_new_key(self):
        self.assertNotIn("table_structure", self.units[0].to_dict())

    def test_reconstruction_still_gives_it_a_row_identity(self):
        found = rows_of(self.units)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].key.row, 20)

    def test_that_identity_is_reported_as_reconstructed(self):
        self.assertEqual(rows_of(self.units)[0].structure_origin,
                         pi.RECONSTRUCTED)


class ANativeTableIsNeverReconstructed(unittest.TestCase):
    """Reconstruction is a guess about a relationship the parser recorded.
    Consulting it anyway is how a stored fact loses to an inference."""

    NATIVE = [row("std-" + "7" * 32, "4.2", "4.2-1", "20",
                  (cell(0, "20", "Key"), cell(1, "Alpha", "Short")))]

    def test_adjacency_is_not_consulted(self):
        import evidence_graph

        def refuse(units):
            raise AssertionError("adjacency reconstruction was invoked")

        original = evidence_graph.tables_in
        evidence_graph.tables_in = refuse
        try:
            found = rows_of(store_and_reload(self.NATIVE))
        finally:
            evidence_graph.tables_in = original

        self.assertEqual(str(found[0].key), "TableRow 4.2-1:20")

    def test_a_ledger_built_from_it_asks_for_no_reconstruction(self):
        import evidence_graph

        original = evidence_graph.tables_in
        evidence_graph.tables_in = lambda units: (_ for _ in ()).throw(
            AssertionError("adjacency reconstruction was invoked"))
        try:
            found = pi.ProvisionLedger()
            found.observe({"results": [unit.to_dict()
                                       for unit in store_and_reload(self.NATIVE)]})
        finally:
            evidence_graph.tables_in = original

        self.assertIn(pi.TableRowKey("4.2", "4.2-1", 20), found.by_key)


if __name__ == "__main__":
    unittest.main()


class WhatTheStoreDescribedIsNeverGuessedAt(unittest.TestCase):
    """A table that numbers nothing gives its rows no citable identity. That
    has always been true; what must not happen is the legacy text heuristic
    stepping in behind a recorded grid and inventing one from the words."""

    UNKEYED = [row("std-" + "8" * 32, "4.2", "4.2-1", "",
                   (cell(0, "20 Alpha", "Key"), cell(1, "so on", "Meaning")),
                   text="20 Alpha so on")]

    def test_an_unkeyed_native_row_gets_no_row_identity(self):
        self.assertEqual(rows_of(store_and_reload(self.UNKEYED)), [])

    def test_a_legacy_unit_with_the_same_text_still_does(self):
        """The heuristic is not removed; it is confined to the stores that
        have nothing better."""
        found = rows_of(store_and_reload(
            [legacy("std-" + "9" * 32, "4.2", "20 Alpha so on")]))

        self.assertEqual([record.key.row for record in found], [20])
