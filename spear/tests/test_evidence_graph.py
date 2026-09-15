"""A table arrives in pieces and nothing links them.

The caption is one unit -- 85 of one store's 98 table units are nothing but a
caption -- and each row is another, carrying no label, no content type and no
way back. Asked which bit of a field was reserved, a session read forty-one
units over fourteen rounds and said the evidence did not settle it. The answer
was in a unit it had already retrieved.

The fixtures are invented. What is real is the shape the extractor leaves.
"""

from __future__ import annotations

import unittest

import evidence_graph as eg
import provision_identity as pi


def unit(position, text, *, section="5.2", page=10, content_type="UNKNOWN",
         source_id=None):
    return {"source_id": source_id or f"std-{position:032d}",
            "section": section, "page": page, "unit_position": position,
            "content_type": content_type, "modality": "NONE", "text": text}


def caption(position, title="Table 5.2-1: Gadget Mode Field", **kw):
    return unit(position, title, content_type="TABLE", **kw)


ROWS = ["7 Reserved Reserved for future modes",
        "6 EnA Enable A Set to 1: enabled",
        "5 EnB Enable B Set to 1: enabled"]


class ACaptionFindsItsRows(unittest.TestCase):
    def test_rows_following_a_caption_are_collected(self):
        units = [caption(1)] + [unit(2 + i, row) for i, row in enumerate(ROWS)]
        tables = eg.tables_in(units)

        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0].rows), 3)

    def test_each_row_becomes_a_keyed_structural_record(self):
        """A row is structural evidence keyed by ITS TABLE and its number,
        not a provision with a null-ordinal ProvisionKey."""
        units = [caption(1)] + [unit(2 + i, row) for i, row in enumerate(ROWS)]
        keys = eg.tables_in(units)[0].keys()

        self.assertEqual(keys, [pi.TableRowKey("5.2", "5.2-1", key)
                                for key in (7, 6, 5)])
        self.assertTrue(all(isinstance(key, pi.TableRowKey) for key in keys))

    def test_a_row_can_be_looked_up_by_its_key(self):
        units = [caption(1)] + [unit(2 + i, row) for i, row in enumerate(ROWS)]
        row = eg.tables_in(units)[0].row_for(7)

        self.assertIn("Reserved", row["text"])

    def test_the_link_is_bidirectional(self):
        units = [caption(1, source_id="std-cap")] + [
            unit(2 + i, row) for i, row in enumerate(ROWS)]
        to_caption, to_rows = eg.link(units)

        self.assertEqual(len(to_rows["std-cap"]), 3)
        self.assertEqual(to_caption[units[1]["source_id"]]["source_id"], "std-cap")


class ProseIsNotAbsorbed(unittest.TestCase):
    """Adjacency alone attaches a paragraph to whatever table precedes it."""

    def test_a_sentence_after_the_rows_ends_the_table(self):
        units = ([caption(1)] + [unit(2 + i, row) for i, row in enumerate(ROWS)]
                 + [unit(9, "The Gadget Mode field is described above."),
                    unit(10, "4 EnC Enable C Set to 1: enabled")])
        table = eg.tables_in(units)[0]

        self.assertEqual(len(table.rows), 3)
        self.assertNotIn(4, [key.row for key in table.keys()])

    def test_a_prose_unit_is_not_row_like(self):
        self.assertFalse(eg.is_row_like(
            unit(1, "The field is eight bits wide and carries three flags.")))

    def test_a_unit_of_a_declared_kind_is_not_a_row(self):
        """A REQUIREMENT beside a table is a provision, not a row."""
        self.assertFalse(eg.is_row_like(
            unit(1, "3 Rule text here", content_type="REQUIREMENT")))


class TablesDoNotBleedIntoEachOther(unittest.TestCase):
    def test_two_adjacent_tables_keep_their_own_rows(self):
        units = [caption(1, "Table 5.2-1: First"),
                 unit(2, "7 AaA First row"),
                 caption(3, "Table 5.2-2: Second"),
                 unit(4, "6 BbB Second row"),
                 unit(5, "5 CcC Second row again")]
        tables = eg.tables_in(units)

        self.assertEqual([len(table.rows) for table in tables], [1, 2])
        self.assertEqual(tables[1].title, "Table 5.2-2: Second")

    def test_a_row_in_another_section_is_not_collected(self):
        units = [caption(1), unit(2, ROWS[0]),
                 unit(3, "6 EnB Enable B", section="5.3")]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 1)

    def test_a_table_continues_onto_the_next_page(self):
        units = [caption(1, page=10), unit(2, ROWS[0], page=10),
                 unit(3, ROWS[1], page=11)]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 2)

    def test_a_row_pages_away_is_not_collected(self):
        units = [caption(1, page=10), unit(2, ROWS[0], page=14)]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 0)


class ATableMayHaveNoRecoverableRows(unittest.TestCase):
    def test_a_lone_caption_yields_an_empty_table_not_an_error(self):
        tables = eg.tables_in([caption(1)])

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].rows, [])
        self.assertEqual(tables[0].keys(), [])

    def test_rows_with_no_caption_are_not_invented_into_a_table(self):
        self.assertEqual(eg.tables_in([unit(1, ROWS[0]), unit(2, ROWS[1])]), [])

    def test_a_missing_row_is_simply_absent(self):
        units = [caption(1), unit(2, ROWS[0])]

        self.assertIsNone(eg.tables_in(units)[0].row_for(6))


if __name__ == "__main__":
    unittest.main()


class AColumnHeaderDoesNotEndTheTable(unittest.TestCase):
    """A table is caption, then a column header, then rows.

    The header carries no leading key -- "Bit# Designation Name Function" --
    so treating the first non-row unit as the end of the table lost every row
    of the table this was written for.
    """

    def test_a_header_between_caption_and_rows_is_tolerated(self):
        units = [caption(1), unit(2, "Bit# Designation Name Function")] + [
            unit(3 + i, row) for i, row in enumerate(ROWS)]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 3)

    def test_prose_between_caption_and_rows_is_not_tolerated(self):
        """Tolerance is for headers, not for paragraphs: a sentence ends it."""
        units = [caption(1),
                 unit(2, "The field below is described in the following text."),
                 unit(3, ROWS[0])]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 0)

    def test_a_long_unit_is_not_a_header(self):
        units = [caption(1), unit(2, "x " * 60), unit(3, ROWS[0])]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 0)

    def test_tolerance_is_bounded(self):
        units = [caption(1), unit(2, "Header one"), unit(3, "Header two"),
                 unit(4, "Header three"), unit(5, ROWS[0])]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 0)

    def test_tolerance_does_not_apply_once_rows_have_begun(self):
        """After a row, a short non-row unit still ends the table."""
        units = [caption(1), unit(2, ROWS[0]), unit(3, "Continued"),
                 unit(4, ROWS[1])]

        self.assertEqual(len(eg.tables_in(units)[0].rows), 1)
