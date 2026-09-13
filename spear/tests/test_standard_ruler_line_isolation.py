"""A field's name comes from its own printed line, not the ruler's.

Cells are grouped into columns without regard for which line a word sat on, so
a wide label can arrive carrying a ruler number printed above it. The bits are
unaffected -- they come from the label's own text -- but the name would be
wrong. Every fixture here is invented; no licensed normative text appears.
"""

import unittest
from dataclasses import replace

from standard_semantic import (
    PositionSource, SemanticRole, StandardBitfieldApproval, promote_bitfield,
)
from standard_word_association import (
    RULER_LINE_EXCLUDED, AssociationSource, associate_words, field_candidates,
    isolate_from_ruler_line, page_words, ruler_band,
)
from standard_structure import HeaderCandidate, StandardTableRow
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, REV, SID, SOURCE, bitfield, cell, line_y,
    merged_cell_table, raw, ruler_row, span_of, table, word_row,
)

OPERATOR = "test-operator"


class _Isolation(unittest.TestCase):
    def entries(self, source, words=(), spans=()):
        mark = bitfield(source, spans=spans)
        return mark, field_candidates(mark.to_dict(), source.to_dict(), words=words)

    def labels(self, source, words=(), spans=()):
        return [item.text for item in self.entries(source, words, spans)[1]]

    def promote(self, source, words=(), spans=(), roles=None):
        mark, entries = self.entries(source, words, spans)
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT,
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(entries)))
        return promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT, words=words)


class ExclusionTests(_Isolation):
    def test_a_ruler_number_from_the_line_above_leaves_the_field_name(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        self.assertEqual(self.labels(source), ["18 FIELD_A (31-16)"])
        self.assertEqual(self.labels(source, words), ["FIELD_A (31-16)"])

    def test_every_ruler_number_on_that_line_is_excluded_not_just_the_first(self):
        source, words = merged_cell_table("FIELD_A (31-16)",
                                          prefix=("18", "17", "16"))
        self.assertEqual(self.labels(source), ["18 17 16 FIELD_A (31-16)"])
        self.assertEqual(self.labels(source, words), ["FIELD_A (31-16)"])

    def test_a_number_printed_on_the_field_line_is_part_of_the_field_name(self):
        # The decision is where the word sat, never what it looks like.
        source, words = merged_cell_table("FIELD_A (31-16)", prefix=("7",),
                                          prefix_on_ruler=False)
        self.assertEqual(self.labels(source, words), ["7 FIELD_A (31-16)"])

    def test_the_exclusion_is_recorded_on_the_candidate(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        _, entries = self.entries(source, words)
        self.assertIn(RULER_LINE_EXCLUDED, entries[0].warnings)
        _, plain = self.entries(source)
        self.assertNotIn(RULER_LINE_EXCLUDED, plain[0].warnings)

    def test_without_raw_geometry_nothing_is_removed(self):
        source, _ = merged_cell_table("FIELD_A (31-16)")
        self.assertEqual(self.labels(source), ["18 FIELD_A (31-16)"])

    def test_a_label_wrapping_onto_a_second_line_keeps_both_lines(self):
        # Only lines that overlap the ruler are dropped, so a label that really
        # does wrap onto a second line is never truncated to one of them.
        source = table((ruler_row(),
                        StandardTableRow(
                            row_index=1, page=1,
                            bbox=(60.0, line_y(0)[0], 560.0, line_y(1)[1]),
                            cells=(cell(1, 0, "0", 64.0, 88.0),
                                   cell(1, 1, "WRAPPED LABEL FIELD_A (31-16)",
                                        240.0, 400.0, top=line_y(0)[0],
                                        bottom=line_y(1)[1])),
                            header_candidate=HeaderCandidate.FALSE)))
        words = ([raw(str(value), 100.0 + offset * 14.0, 108.0 + offset * 14.0,
                      on_ruler=True)
                  for offset, value in enumerate(range(31, -1, -1))]
                 + [raw("WRAPPED", 240.0, 290.0, line=0),
                    raw("LABEL", 300.0, 340.0, line=0),
                    raw("FIELD_A (31-16)", 240.0, 400.0, line=1),
                    raw("0", 64.0, 88.0, line=0)])
        label_cell = source.rows[1].cells[1]
        self.assertLessEqual(label_cell.bbox[1], line_y(0)[0])
        self.assertGreaterEqual(label_cell.bbox[3], line_y(1)[1])
        self.assertEqual(self.labels(source, tuple(words)),
                         ["WRAPPED LABEL FIELD_A (31-16)"])

    def test_a_cell_wholly_on_the_ruler_line_is_left_alone(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        band = ruler_band(bitfield(source).to_dict()["bit_labels"])
        cell = next(item for row in source.rows for item in row.cells
                    if item.text.startswith("18 "))
        # A box that never leaves the ruler strip has no other line to keep.
        squashed = (cell.bbox[0], band[0], cell.bbox[2], band[1])
        self.assertIsNone(isolate_from_ruler_line(cell.text, squashed, words, band))

    def test_no_ruler_means_no_exclusion(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        cell = next(item for row in source.rows for item in row.cells
                    if item.text.startswith("18 "))
        self.assertIsNone(
            isolate_from_ruler_line(cell.text, cell.bbox, words, None))


class PreservationTests(_Isolation):
    """Everything except the name must come through untouched."""

    def promoted(self, words=()):
        source, raw_words = merged_cell_table("FIELD_A (31-16)")
        return source, self.promote(source, raw_words if words else ())

    def test_the_stated_range_is_unchanged(self):
        _, (definition, blocked) = self.promoted(words=True)
        self.assertIsNone(blocked)
        self.assertEqual(definition.fields[0].stated_range_text, "(31-16)")

    def test_the_bit_position_is_unchanged(self):
        _, (before, _) = self.promoted()
        _, (after, _) = self.promoted(words=True)
        self.assertEqual((before.fields[0].msb, before.fields[0].lsb,
                          before.fields[0].width), (31, 16, 16))
        self.assertEqual((after.fields[0].msb, after.fields[0].lsb,
                          after.fields[0].width), (31, 16, 16))

    def test_the_position_source_stays_the_stated_range(self):
        _, (definition, _) = self.promoted(words=True)
        self.assertEqual(definition.fields[0].position_source,
                         PositionSource.STATED_RANGE)

    def test_the_word_association_is_unchanged(self):
        _, (before, _) = self.promoted()
        _, (after, _) = self.promoted(words=True)
        for definition in (before, after):
            self.assertEqual(definition.fields[0].word_index, 0)
            self.assertEqual(definition.fields[0].word_association_source,
                             AssociationSource.EXPLICIT_WORD_INDEX)

    def test_the_canonical_sources_are_kept(self):
        _, (definition, _) = self.promoted(words=True)
        field = definition.fields[0]
        self.assertEqual(field.supporting_source_ids, (SOURCE,))
        self.assertTrue(field.source_cell_ids)

    def test_the_provenance_grade_is_neither_weakened_nor_inflated(self):
        source, (before, _) = self.promoted()
        _, (after, _) = self.promoted(words=True)
        self.assertEqual(before.fields[0].provenance_grades,
                         after.fields[0].provenance_grades)

    def test_the_field_keeps_its_role(self):
        _, (definition, _) = self.promoted(words=True)
        self.assertEqual(definition.fields[0].semantic_role, SemanticRole.FIELD)

    def test_the_field_count_does_not_change(self):
        source, raw_words = merged_cell_table("FIELD_A (31-16)")
        _, plain = self.entries(source)
        _, fixed = self.entries(source, raw_words)
        self.assertEqual(len(plain), len(fixed))
        before, _ = self.promote(source)
        after, _ = self.promote(source, raw_words)
        self.assertEqual(len(before.fields), len(after.fields))


class LabelTests(_Isolation):
    def test_the_normative_label_keeps_its_stated_range(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        definition, _ = self.promote(source, words)
        self.assertEqual(definition.fields[0].label, "FIELD_A (31-16)")

    def test_the_display_label_drops_the_range_and_the_ruler_number(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        definition, _ = self.promote(source, words)
        self.assertEqual(definition.fields[0].display_label, "FIELD_A")
        self.assertNotIn("18", definition.fields[0].display_label)

    def test_the_normalized_identifier_follows_the_cleaned_label(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        definition, _ = self.promote(source, words)
        self.assertEqual(definition.fields[0].normalized_identifier, "FIELD_A")


class UntouchedTests(_Isolation):
    """A label that does not state its own bits is never rewritten."""

    def test_a_measured_span_with_no_stated_range_keeps_its_text(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("PLAIN_LABEL",)),))))
        spans = (span_of(source, "PLAIN_LABEL", covered=(31, 30, 29)),)
        words = page_words({"pages": [{"page": 1, "blocks": [{"lines": [
            {"words": [raw("PLAIN_LABEL", 100.0, 240.0, line=0)]}]}]}]}, 1)
        self.assertEqual(self.labels(source, words, spans), ["PLAIN_LABEL"])

    def test_an_ordinary_stated_field_on_one_line_is_left_exactly_as_it_is(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        words = page_words({"pages": [{"page": 1, "blocks": [{"lines": [
            {"words": [raw("FIELD_A (31-16)", 100.0, 240.0, line=0)]}]}]}]}, 1)
        self.assertEqual(self.labels(source, words), ["FIELD_A (31-16)"])
        _, entries = self.entries(source, words)
        self.assertNotIn(RULER_LINE_EXCLUDED, entries[0].warnings)

    def test_isolation_never_writes_to_the_table_it_reads(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        before = source.to_dict()
        self.entries(source, words)
        self.assertEqual(source.to_dict(), before)

    def test_isolation_never_invents_a_canonical_source(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        _, entries = self.entries(source, words)
        for entry in entries:
            self.assertTrue(set(entry.source_ids) <= {SOURCE})

    def test_a_cell_whose_every_word_is_ruler_is_not_emptied(self):
        source, words = merged_cell_table("FIELD_A (31-16)")
        cell = next(item for row in source.rows for item in row.cells
                    if item.text.startswith("18 "))
        only_ruler = tuple(word for word in words
                           if word["text"] in {"18", "17", "16"})
        self.assertIsNone(isolate_from_ruler_line(
            cell.text, cell.bbox, only_ruler,
            ruler_band(bitfield(source).to_dict()["bit_labels"])))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
