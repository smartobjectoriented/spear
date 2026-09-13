"""STD3B: a word the diagram names does not vanish for having no fields.

Structural completeness used to look only at words that produced a field, so a
word whose labels nobody could position left no trace and the definition called
itself whole. Every fixture here is invented; no licensed text appears.
"""

import unittest
from dataclasses import replace

from standard_semantic import (
    SemanticRole, StandardBitfieldApproval, StructuralCompleteness,
    promote_bitfield,
)
from standard_word_association import (
    field_candidates, is_pure_numeric_ruler_cell, unpositioned_labels,
)
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, RULER_TOP, REV, SID, SOURCE, bitfield, cell, line_y,
    ruler_row, ruler_run_table, table, two_word_table, word_row,
)

OPERATOR = "test-operator"
BAND = (RULER_TOP, RULER_TOP + 12.0)
ON_RULER = (240.0, RULER_TOP, 400.0, RULER_TOP + 12.0)


class _Words(unittest.TestCase):
    def promote(self, source, roles=None):
        mark = bitfield(source)
        entries = field_candidates(mark.to_dict(), source.to_dict())
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=FINGERPRINT,
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(entries)))
        return promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT)


class ZeroFieldWordTests(_Words):
    def test_a_word_whose_labels_state_nothing_makes_the_whole_thing_incomplete(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 1)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)
        self.assertEqual(len(definition.unresolved_words), 1)

    def test_the_missing_word_is_named_in_the_definition(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        word = definition.unresolved_words[0]
        self.assertEqual(word.word_index, 1)
        self.assertEqual(word.field_count, 0)
        self.assertEqual(word.unpositioned_labels, ("Reserved",))
        self.assertEqual(word.supporting_source_ids, (SOURCE,))

    def test_several_labels_in_a_missing_word_are_all_reported(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Alpha", "Beta", "Gamma"))
        definition, _ = self.promote(source)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)
        self.assertEqual(definition.unresolved_words[0].unpositioned_labels,
                         ("Alpha", "Beta", "Gamma"))

    def test_a_word_with_fields_is_unaffected(self):
        source = two_word_table(("FIELD_A (31-0)",), ("FIELD_B (31-0)",))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 2)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)
        self.assertEqual(definition.unresolved_words, ())

    def test_a_complete_word_plus_a_missing_word_is_still_incomplete(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        covered = {b for f in definition.fields
                   for b in range(f.lsb, f.msb + 1)}
        self.assertEqual(len(covered), 32)   # word 1 is fully described
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)


class NoiseTests(_Words):
    def test_a_word_holding_only_a_ruler_run_is_ignored(self):
        source = ruler_run_table("31 30 29 28")
        definition, _ = self.promote(source)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)
        self.assertEqual(definition.unresolved_words, ())

    def test_a_word_heading_merged_with_its_ruler_is_ignored(self):
        # The column pass can gather the Word heading and the ruler into one
        # cell; that cell is still ruler, not a name nobody positioned.
        self.assertTrue(is_pure_numeric_ruler_cell("Word 31 30 29 28",
                                                   ON_RULER, BAND))
        source = ruler_run_table("Word 31 30 29 28")
        definition, _ = self.promote(source)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_COMPLETE)

    def test_a_heading_with_a_broken_run_is_not_ignored(self):
        self.assertFalse(is_pure_numeric_ruler_cell("Word 31 29 27",
                                                    ON_RULER, BAND))

    def test_a_name_that_merely_starts_with_word_is_not_ignored(self):
        for text in ("Word Count 31 30", "Wordy 31 30"):
            self.assertFalse(is_pure_numeric_ruler_cell(text, ON_RULER, BAND),
                             text)


class NoInventionTests(_Words):
    def test_no_field_is_created_for_the_missing_word(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        self.assertEqual({f.word_index for f in definition.fields}, {0})
        self.assertNotIn(1, {f.word_index for f in definition.fields})

    def test_no_reserved_role_is_synthesized(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        self.assertNotIn(SemanticRole.RESERVED,
                         [f.semantic_role for f in definition.fields])

    def test_the_missing_word_gets_no_range_of_any_kind(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        word = definition.unresolved_words[0]
        for attribute in ("msb", "lsb", "width", "covered_bit_labels"):
            self.assertFalse(hasattr(word, attribute), attribute)

    def test_no_field_carries_a_packet_global_offset(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        for field in definition.fields:
            self.assertLessEqual(field.msb, 31)
            self.assertGreaterEqual(field.lsb, 0)


class RepresentationTests(_Words):
    def test_the_unresolved_word_survives_serialization(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Reserved",))
        definition, _ = self.promote(source)
        value = definition.to_dict()
        self.assertEqual(len(value["unresolved_words"]), 1)
        row = value["unresolved_words"][0]
        self.assertEqual(row["word_index"], 1)
        self.assertEqual(row["field_count"], 0)
        self.assertEqual(row["unpositioned_labels"], ["Reserved"])
        self.assertIsInstance(row["supporting_source_ids"], list)

    def test_the_build_is_deterministic(self):
        source = two_word_table(("FIELD_A (31-0)",), ("Alpha", "Beta"))
        first, _ = self.promote(source)
        second, _ = self.promote(source)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_a_word_with_both_fields_and_an_orphan_is_still_reported(self):
        # The page-228 shape: fields present, and a label nobody positioned.
        source = table((ruler_row(),
                        word_row(1, ((1, ("Reserved", "FIELD_B (15-0)")),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(definition.structural_completeness,
                         StructuralCompleteness.STRUCTURALLY_INCOMPLETE)
        self.assertEqual(len(definition.unresolved_words), 1)
        self.assertEqual(definition.unresolved_words[0].field_count, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
