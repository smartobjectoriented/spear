"""STD2G: a run of ruler numbers is the ruler, not a field nobody positioned.

Cells are grouped into columns without regard for printed line, so a cell can
reach from the ruler down into a word's row carrying several ruler positions.
Read as a name, it argues the word is incompletely described. Every fixture
here is invented; no licensed text appears.
"""

import unittest

from standard_semantic import (
    StandardBitfieldApproval, StructuralCompleteness, promote_bitfield,
)
from standard_word_association import (
    is_pure_numeric_ruler_cell, field_candidates, ruler_band, unpositioned_labels,
)
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, RULER_TOP, REV, SID, bitfield, line_y, ruler_row,
    ruler_run_table, table, word_row,
)

OPERATOR = "test-operator"
BAND = (RULER_TOP, RULER_TOP + 12.0)
ON_RULER = (240.0, RULER_TOP, 400.0, RULER_TOP + 12.0)
BELOW_RULER = (240.0, line_y(0)[0], 400.0, line_y(0)[1])


class PredicateTests(unittest.TestCase):
    """Narrow by construction: a false positive would hide a real field."""

    def filtered(self, text, box=ON_RULER, band=BAND):
        return is_pure_numeric_ruler_cell(text, box, band)

    def test_a_descending_unit_run_on_the_ruler_line_is_ruler(self):
        self.assertTrue(self.filtered("31 30 29 28"))
        self.assertTrue(self.filtered("26 25"))

    def test_an_ascending_unit_run_on_the_ruler_line_is_ruler(self):
        self.assertTrue(self.filtered("0 1 2 3"))

    def test_a_run_with_a_gap_in_it_is_not_ruler(self):
        self.assertFalse(self.filtered("31 29 27"))

    def test_a_run_with_a_stride_of_two_is_not_ruler(self):
        self.assertFalse(self.filtered("10 12 14"))

    def test_one_word_among_the_digits_keeps_the_cell(self):
        self.assertFalse(self.filtered("31 30 foo 29"))

    def test_a_name_with_a_number_is_not_ruler(self):
        for text in ("Field 31", "Channel 0 1", "Version 1 2"):
            self.assertFalse(self.filtered(text), text)

    def test_a_word_or_byte_count_is_not_ruler(self):
        for text in ("1 Word", "2 Words", "4 bytes"):
            self.assertFalse(self.filtered(text), text)

    def test_a_single_number_is_not_matched_by_this_predicate(self):
        # Single ruler labels were already excluded, and still are elsewhere.
        self.assertFalse(self.filtered("31"))

    def test_values_beyond_a_bit_number_are_not_ruler(self):
        self.assertFalse(self.filtered("64 65 66"))
        self.assertFalse(self.filtered("100 101"))

    def test_a_numeric_run_away_from_the_ruler_is_kept(self):
        self.assertFalse(self.filtered("31 30 29 28", box=BELOW_RULER))

    def test_without_a_ruler_to_sit_on_nothing_is_filtered(self):
        self.assertFalse(is_pure_numeric_ruler_cell("31 30 29 28"))
        self.assertFalse(is_pure_numeric_ruler_cell("31 30 29 28", ON_RULER, None))


class _Structural(unittest.TestCase):
    def promote(self, source, roles=None):
        mark = bitfield(source)
        entries = field_candidates(mark.to_dict(), source.to_dict())
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT,
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(entries)))
        return mark, promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT)


class CompletenessTests(_Structural):
    def test_a_merged_descending_run_does_not_argue_the_word_is_incomplete(self):
        source = ruler_run_table("31 30 29 28")
        definition, blocked = self.promote(source)[1]
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 1)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)
        self.assertFalse([w for w in definition.warnings if "never positions" in w])

    def test_a_merged_ascending_run_does_not_argue_the_word_is_incomplete(self):
        definition, blocked = self.promote(ruler_run_table("0 1 2 3"))[1]
        self.assertIsNone(blocked)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_a_real_label_in_the_same_place_still_argues_incompleteness(self):
        # The page-228 shape: a named label the diagram never positions.
        source = ruler_run_table("Reserved")
        definition, blocked = self.promote(source)[1]
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 1)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)
        self.assertTrue(any("never positions" in w for w in definition.warnings))

    def test_a_non_unit_run_still_argues_incompleteness(self):
        definition, _ = self.promote(ruler_run_table("31 29 27"))[1]
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)

    def test_a_gap_with_nothing_in_it_stays_a_legitimate_gap(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (15-0)",)),))))
        definition, blocked = self.promote(source)[1]
        self.assertIsNone(blocked)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_filtering_changes_no_field_role_or_position(self):
        ruler = self.promote(ruler_run_table("31 30 29 28"))[1][0]
        named = self.promote(ruler_run_table("Reserved"))[1][0]
        for definition in (ruler, named):
            self.assertEqual(len(definition.fields), 1)
            field = definition.fields[0]
            self.assertEqual((field.msb, field.lsb, field.width), (15, 0, 16))
            self.assertEqual(field.semantic_role.value, "FIELD")
            self.assertEqual(field.position_source.value, "STATED_RANGE")
            self.assertEqual(field.word_index, 0)

    def test_a_run_is_never_offered_as_an_unpositioned_label(self):
        source = ruler_run_table("31 30 29 28")
        mark = bitfield(source)
        labels = unpositioned_labels(mark.to_dict(), source.to_dict())
        self.assertEqual([c["text"] for c in labels], [])

    def test_a_name_is_still_offered_as_an_unpositioned_label(self):
        source = ruler_run_table("Reserved")
        mark = bitfield(source)
        labels = unpositioned_labels(mark.to_dict(), source.to_dict())
        self.assertEqual([c["text"] for c in labels], ["Reserved"])

    def test_the_ruler_band_comes_from_the_candidates_own_labels(self):
        source = ruler_run_table("31 30 29 28")
        band = ruler_band(bitfield(source).to_dict()["bit_labels"])
        self.assertIsNotNone(band)
        self.assertLessEqual(band[0], RULER_TOP + 0.001)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
