"""What a unit is FOUND by, as against what it returns as evidence.

A table row reads "20 ReqV Request Validation Acknowledge packet Set to 1:
..." and nothing in those words says it is a bit of a Control packet's CAM
field. The caption above it says that; the column header "Bit#" says what the
20 is. The old extraction hid the problem by packing a whole table into one
283-word unit, which retrieved well and cited terribly.

Measured on one bound standard: with the row indexed on its own words alone,
four bit-level questions found their row nowhere in the vector top twenty.

The rule this file holds: search may see the structure a unit sits in; it may
never see a neighbouring provision, and the evidence handed back is untouched.
"""

from __future__ import annotations

import unittest

from standard_retrieval import rebuild_lexical_index
from standard_schema import (
    StandardContentType, StandardDocumentUnit, StandardModality,
    StandardTableCell, StandardTableStructure, source_content_sha256,
)
from standard_vector_index import retrieval_text, structural_context

SID, REV, PDF = "ACME-1", "2030", "a" * 64

ROW_TEXT = "20 ReqV Request Validation Acknowledge packet Set to 1: requested"


def unit(*, structure=None, text=ROW_TEXT, section="8.3.1", headings=("8 Commands",)):
    return StandardDocumentUnit(
        source_id="std-" + "1" * 32, standard_id=SID, revision=REV,
        section=section, page=116, heading_path=headings,
        content_type=StandardContentType.TABLE,
        modality=StandardModality.NONE, text=text, source_pdf_sha256=PDF,
        source_content_sha256=source_content_sha256(text),
        extractor_version="test", table_structure=structure)


def grid(caption="Table 8.3.1-1: Control Packet: Control/Acknowledge Mode Field",
         columns=("Bit#", "Designation", "Name", "Function")):
    return StandardTableStructure(
        table_id="8.3.1-1", row_index=11, columns=columns, row_key="20",
        caption=caption,
        cells=(StandardTableCell(column_index=0, text="20", column="Bit#"),
               StandardTableCell(column_index=1, text="ReqV", column="Designation")))


class ARowIsFoundByTheTableItSitsIn(unittest.TestCase):

    def test_the_caption_joins_the_search_document(self):
        self.assertIn("Control/Acknowledge Mode Field",
                      retrieval_text(unit(structure=grid())))

    def test_the_column_headers_join_it(self):
        found = structural_context(unit(structure=grid()))

        self.assertEqual(found[1:], ("Bit#", "Designation", "Name", "Function"))

    def test_the_evidence_text_is_untouched(self):
        """The row still returns exactly what the document printed."""
        self.assertEqual(unit(structure=grid()).text, ROW_TEXT)

    def test_an_uncaptioned_table_contributes_its_headers_only(self):
        self.assertEqual(structural_context(unit(structure=grid(caption=""))),
                         ("Bit#", "Designation", "Name", "Function"))

    def test_a_table_with_no_headers_contributes_its_caption_only(self):
        self.assertEqual(structural_context(unit(structure=grid(columns=()))),
                         (grid().caption,))


class AUnitWithNoGridIsIndexedExactlyAsBefore(unittest.TestCase):
    """A corpus ingested under an earlier schema must keep its index, and its
    fingerprint: enrichment that quietly rewrote every older store would
    force a rebuild of things nobody asked to change."""

    def test_it_contributes_no_structural_context(self):
        self.assertEqual(structural_context(unit(structure=None)), ())

    def test_its_retrieval_text_is_section_headings_and_text(self):
        self.assertEqual(retrieval_text(unit(structure=None)),
                         "Section 8.3.1\n8 Commands\n" + ROW_TEXT)

    def test_a_unit_without_the_field_at_all_is_tolerated(self):
        """Anything unit-shaped may be passed in; only a store knows about
        table structure."""
        class Bare:
            section, heading_path, text = "1.1", (), "words"

        self.assertEqual(structural_context(Bare()), ())


class StructureIsNotANeighbouringProvision(unittest.TestCase):
    """The whole hazard of enriching a search document is that it becomes a
    way to let one provision be found -- and then read -- through another's
    words. A caption and a column header state no obligation."""

    def test_only_caption_and_columns_are_taken(self):
        structure = grid()
        found = structural_context(unit(structure=structure))

        self.assertEqual(set(found),
                         {structure.caption, *structure.columns})

    def test_cell_text_of_the_row_is_not_duplicated_into_context(self):
        self.assertNotIn("ReqV", " ".join(
            structural_context(unit(structure=grid()))[:1]))


if __name__ == "__main__":
    unittest.main()
