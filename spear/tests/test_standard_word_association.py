"""STD2D: a bit range belongs to a word, and the document says which one.

Every fixture here is invented. No licensed normative text appears in this file.
"""

import unittest
from dataclasses import replace

from standard_semantic import (
    DefinitionCompleteness, PositionSource, RULER_FRAGMENTED, SemanticRole,
    SpanRole, StandardBitfieldApproval, StandardSemanticError, promote_bitfield,
    semantic_fingerprint, validate_semantics, StandardSemanticSet,
)
from standard_word_association import (
    NO_WORD_INDEX_IN_ROW, WORD_LINES_NOT_SEPARABLE, WORD_LINE_COUNT_MISMATCH,
    AssociationSource, associate_words, field_candidates, is_word_count,
    word_index_cells,
)
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, LINE_HEIGHT, REV, SID, SOURCE, bitfield, cell,
    line_y, ruler_row, span_of, table, word_row,
)

OPERATOR = "test-operator"


class _Words(unittest.TestCase):
    def association(self, source):
        return associate_words(source.to_dict())

    def candidates(self, source, spans=()):
        mark = bitfield(source, spans=spans)
        return mark, field_candidates(mark.to_dict(), source.to_dict())

    def word_of(self, source, text, spans=()):
        associations = self.association(source)
        _, entries = self.candidates(source, spans)
        entry = next(item for item in entries if item.text == text)
        return associations[entry.cell_id]

    def promote(self, source, *, spans=(), roles=None, verdict="PASS",
                reviewer=OPERATOR, approval=None):
        mark = bitfield(source, spans=spans)
        entries = field_candidates(mark.to_dict(), source.to_dict())
        if approval is None:
            approval = StandardBitfieldApproval(
                candidate_id=mark.bitfield_id, verdict=verdict, reviewer=reviewer,
                reviewed_at="2026-01-01T00:00:00+00:00",
                structure_fingerprint=FINGERPRINT,
                span_roles=tuple(roles if roles is not None
                                 else ["FIELD"] * len(entries)))
        return promote_bitfield(
            mark, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT)


class WordIndexTests(_Words):
    def test_an_explicit_word_zero_is_read_from_the_word_column(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertEqual(association.word_index, 0)
        self.assertEqual(association.association_source,
                         AssociationSource.EXPLICIT_WORD_INDEX)

    def test_an_explicit_word_one_is_read_from_the_word_column(self):
        source = table((ruler_row(),
                        word_row(1, ((1, ("FIELD_B (31-16)",)),))))
        self.assertEqual(self.word_of(source, "FIELD_B (31-16)").word_index, 1)

    def test_a_word_named_by_a_label_needs_no_number(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("Word Alpha", "FIELD_A (31-16)")),),
                                 word_column=False)))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertIsNone(association.word_index)
        self.assertEqual(association.word_label, "Alpha")
        self.assertEqual(association.association_source,
                         AssociationSource.EXPLICIT_WORD_LABEL)

    def test_a_marker_naming_its_own_word_speaks_for_its_row(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("Word 4", "FIELD_A (31-16)")),),
                                 word_column=False)))
        self.assertEqual(self.word_of(source, "FIELD_A (31-16)").word_index, 4)

    def test_a_word_count_is_not_a_word_index(self):
        self.assertTrue(is_word_count("HEADER (2 Words, Optional)"))
        self.assertTrue(is_word_count("PAYLOAD (1 Word)"))
        self.assertFalse(is_word_count("Word 2"))
        source = table((ruler_row(),
                        word_row(1, ((0, ("HEADER (2 Words)", "FIELD_A (31-16)")),),
                                 word_column=False)))
        # Neither cell identifies a word, so the row falls back to geometry.
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertEqual(association.association_source,
                         AssociationSource.GEOMETRIC_ROW)

    def test_arbitrary_integers_are_not_word_indices(self):
        # A column of numbers with no heading vouching for it is just numbers.
        source = table((ruler_row(heading=None),
                        word_row(1, ((7, ("FIELD_A (31-16)",)),))))
        heading, cells = word_index_cells(source.to_dict())
        self.assertIsNone(heading)
        self.assertEqual(cells, [])
        self.assertEqual(self.word_of(source, "FIELD_A (31-16)").association_source,
                         AssociationSource.GEOMETRIC_ROW)

    def test_the_word_column_carries_its_own_provenance(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertTrue(association.source_cell_ids)
        self.assertEqual(association.supporting_source_ids, (SOURCE,))


class BandSplitTests(_Words):
    """The defect STD2D exists to fix."""

    def two_words(self):
        return table((ruler_row(),
                      word_row(1, ((0, ("FIELD_A (31-28)", "FIELD_B (27-24)")),
                                   (1, ("FIELD_C (31-16)", "FIELD_D (15-0)"))))))

    def test_an_explicit_word_index_beats_a_merged_y_band(self):
        source = self.two_words()
        self.assertEqual(len({row.row_index for row in source.rows}), 2)
        words = {text: self.word_of(source, text).word_index
                 for text in ("FIELD_A (31-28)", "FIELD_B (27-24)",
                              "FIELD_C (31-16)", "FIELD_D (15-0)")}
        self.assertEqual(words, {"FIELD_A (31-28)": 0, "FIELD_B (27-24)": 0,
                                 "FIELD_C (31-16)": 1, "FIELD_D (15-0)": 1})

    def test_the_merged_band_promotes_into_two_words_without_overlap(self):
        definition, blocked = self.promote(self.two_words())
        self.assertIsNone(blocked)
        by_word = {}
        for field in definition.fields:
            by_word.setdefault(field.word_index, {})[field.label] = (
                field.msb, field.lsb, field.width)
        self.assertEqual(by_word[0], {"FIELD_A (31-28)": (31, 28, 4),
                                      "FIELD_B (27-24)": (27, 24, 4)})
        self.assertEqual(by_word[1], {"FIELD_C (31-16)": (31, 16, 16),
                                      "FIELD_D (15-0)": (15, 0, 16)})

    def test_a_row_whose_lines_do_not_match_its_word_count_is_unresolved(self):
        # Two words named, three printed lines: nothing safe can be said.
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",)),
                                     (2, ("FIELD_C (31-16)",))))))
        source = replace(source, rows=(source.rows[0], replace(
            source.rows[1], cells=tuple(
                replace(item, text="0 1") if item.column_index == 0 else item
                for item in source.rows[1].cells))))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertEqual(association.association_source, AssociationSource.UNKNOWN)
        self.assertIn(WORD_LINE_COUNT_MISMATCH, association.warnings)

    def test_an_unresolvable_word_blocks_promotion_rather_than_guessing(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",)),
                                     (2, ("FIELD_C (31-16)",))))))
        source = replace(source, rows=(source.rows[0], replace(
            source.rows[1], cells=tuple(
                replace(item, text="0 1") if item.column_index == 0 else item
                for item in source.rows[1].cells))))
        definition, blocked = self.promote(source)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "WORD_ASSOCIATION_UNRESOLVED")

    def test_a_label_covering_several_lines_is_not_split_between_words(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",))))))
        tall = replace(source.rows[1], cells=tuple(
            replace(item, bbox=(item.bbox[0], line_y(0)[0], item.bbox[2],
                                line_y(1)[1]))
            if item.column_index == 1 else item
            for item in source.rows[1].cells))
        source = replace(source, rows=(source.rows[0], tall))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertEqual(association.association_source, AssociationSource.UNKNOWN)
        self.assertIn(WORD_LINES_NOT_SEPARABLE, association.warnings)


class OverlapTests(_Words):
    def test_the_same_range_in_two_words_is_not_an_overlap(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (23-12)",)),
                                     (1, ("FIELD_B (23-12)",))))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual({field.word_index for field in definition.fields}, {0, 1})
        for field in definition.fields:
            self.assertEqual((field.msb, field.lsb, field.width), (23, 12, 12))

    def test_two_overlapping_ranges_in_one_word_are_rejected(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (23-12)", "FIELD_B (15-8)")),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "OVERLAPPING_FIELDS")
        self.assertIn("word 0", blocked["detail"])

    def test_several_fields_tile_one_word_without_complaint(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-24)", "FIELD_B (23-12)",
                                          "FIELD_C (11-0)")),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        covered = set()
        for field in definition.fields:
            covered |= set(field.covered_bit_labels)
        self.assertEqual(covered, set(range(32)))

    def test_gaps_within_a_word_are_left_as_the_document_wrote_them(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-28)", "FIELD_B (23-20)")),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        covered = set()
        for field in definition.fields:
            covered |= set(field.covered_bit_labels)
        self.assertNotIn(27, covered)
        self.assertEqual(len(definition.fields), 2)


class LocalNumberingTests(_Words):
    def test_bit_numbers_stay_local_to_their_word(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",))))))
        definition, _ = self.promote(source)
        for field in definition.fields:
            self.assertEqual((field.msb, field.lsb), (31, 16))

    def test_no_field_is_given_a_global_packet_offset(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",))))))
        definition, _ = self.promote(source)
        for field in definition.fields:
            self.assertLessEqual(field.msb, 31)
            self.assertGreaterEqual(field.lsb, 0)
        self.assertNotIn(63, [field.msb for field in definition.fields])

    def test_the_word_and_its_evidence_survive_serialization(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",))))))
        definition, _ = self.promote(source)
        value = definition.to_dict()
        words = {field["word_index"] for field in value["fields"]}
        self.assertEqual(words, {0, 1})
        for field in value["fields"]:
            self.assertEqual(field["word_association_source"],
                             AssociationSource.EXPLICIT_WORD_INDEX.value)


class GeometricFallbackTests(_Words):
    def test_a_table_with_no_word_column_falls_back_to_its_rows(self):
        source = table((ruler_row(heading=None),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),),
                                 word_column=False)))
        association = self.word_of(source, "FIELD_A (31-16)")
        self.assertEqual(association.association_source,
                         AssociationSource.GEOMETRIC_ROW)
        self.assertEqual(association.word_index, 1)

    def test_a_row_with_no_word_number_says_so(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),)),
                        word_row(2, ((0, ("FIELD_B (31-16)",)),), first_line=2,
                                 word_column=False)))
        association = self.word_of(source, "FIELD_B (31-16)")
        self.assertEqual(association.association_source,
                         AssociationSource.GEOMETRIC_ROW)
        self.assertIn(NO_WORD_INDEX_IN_ROW, association.warnings)

    def test_a_multi_word_definition_resting_on_geometry_is_only_partial(self):
        source = table((ruler_row(heading=None),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),),
                                 word_column=False),
                        word_row(2, ((0, ("FIELD_B (31-16)",)),), first_line=2,
                                 word_column=False)))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(definition.completeness, DefinitionCompleteness.PARTIAL)
        self.assertTrue(any("rests on geometry" in item
                            for item in definition.warnings))

    def test_a_single_word_geometric_definition_stays_complete(self):
        source = table((ruler_row(heading=None),
                        word_row(1, ((0, ("FIELD_A (31-16)", "FIELD_B (15-0)")),),
                                 word_column=False)))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(definition.completeness, DefinitionCompleteness.COMPLETE)


class ReachabilityTests(_Words):
    """STD2C left stated ranges unreachable unless a span measured them."""

    def test_a_stated_range_cell_becomes_a_field_with_no_geometric_span(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        mark, entries = self.candidates(source)
        self.assertEqual(mark.spans, ())
        self.assertEqual([item.text for item in entries], ["FIELD_A (31-16)"])
        self.assertFalse(entries[0].from_geometric_span)
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(definition.fields[0].position_source,
                         PositionSource.STATED_RANGE)

    def test_a_measured_span_keeps_its_place_at_the_front_of_the_list(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)", "FIELD_B (15-0)")),))))
        spans = (span_of(source, "FIELD_B (15-0)", covered=(20, 19, 18)),)
        _, entries = self.candidates(source, spans)
        self.assertEqual([item.text for item in entries],
                         ["FIELD_B (15-0)", "FIELD_A (31-16)"])
        self.assertTrue(entries[0].from_geometric_span)
        self.assertFalse(entries[1].from_geometric_span)

    def test_a_cell_is_never_counted_as_both_a_span_and_a_stated_cell(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        spans = (span_of(source, "FIELD_A (31-16)", covered=(20, 19, 18)),)
        _, entries = self.candidates(source, spans)
        self.assertEqual(len(entries), 1)

    def test_a_fragmented_ruler_does_not_block_a_stated_range(self):
        source = table((ruler_row(labels=range(17, -1, -1)),
                        word_row(1, ((0, ("FIELD_A (31-24)",)),))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual((definition.fields[0].msb, definition.fields[0].lsb),
                         (31, 24))
        self.assertIn(RULER_FRAGMENTED, definition.warnings)

    def test_a_whole_ruler_is_not_reported_as_fragmented(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-24)",)),))))
        definition, _ = self.promote(source)
        self.assertNotIn(RULER_FRAGMENTED, definition.warnings)

    def test_a_bare_ruler_label_is_never_a_field_candidate(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),))))
        _, entries = self.candidates(source)
        self.assertTrue(all(not item.text.strip().isdigit() for item in entries))

    def test_the_word_column_is_never_a_field_candidate(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (31-16)",)),
                                     (1, ("FIELD_B (31-16)",))))))
        _, entries = self.candidates(source)
        self.assertEqual({item.text for item in entries},
                         {"FIELD_A (31-16)", "FIELD_B (31-16)"})


class GateTests(_Words):
    def two_words(self):
        return table((ruler_row(),
                      word_row(1, ((0, ("FIELD_A (31-16)",)),
                                   (1, ("FIELD_B (31-16)",))))))

    def test_word_association_does_not_bypass_human_review(self):
        source = self.two_words()
        mark = bitfield(source)
        definition, blocked = promote_bitfield(
            mark, source, None, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=FINGERPRINT)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "NO_HUMAN_REVIEW")

    def test_an_assistant_may_not_approve_a_word_local_definition(self):
        with self.assertRaises(StandardSemanticError):
            StandardBitfieldApproval(
                candidate_id="bit-0123456789abcdef", verdict="PASS",
                reviewer="claude-assistant", reviewed_at="2026-01-01T00:00:00+00:00",
                structure_fingerprint=FINGERPRINT, span_roles=("FIELD", "FIELD"))

    def test_every_field_keeps_the_canonical_source_of_its_cell(self):
        definition, _ = self.promote(self.two_words())
        for field in definition.fields:
            self.assertEqual(field.supporting_source_ids, (SOURCE,))
            self.assertTrue(field.source_cell_ids)

    def test_a_field_without_provenance_blocks_the_definition(self):
        source = self.two_words()
        stripped = replace(source.rows[1], cells=tuple(
            replace(item, source_ids=()) if item.text.startswith("FIELD_A")
            else item for item in source.rows[1].cells))
        source = replace(source, rows=(source.rows[0], stripped))
        definition, blocked = self.promote(source)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "UNRESOLVED_PROVENANCE")

    def test_an_unclassified_candidate_still_blocks(self):
        definition, blocked = self.promote(
            self.two_words(), roles=["FIELD", SpanRole.UNKNOWN.value])
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPAN_UNCLASSIFIED")

    def test_a_role_list_that_misses_a_candidate_is_refused(self):
        definition, blocked = self.promote(self.two_words(), roles=["FIELD"])
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPANS_NOT_CLASSIFIED")

    def test_a_stale_approval_still_blocks(self):
        source = self.two_words()
        mark = bitfield(source)
        approval = StandardBitfieldApproval(
            candidate_id=mark.bitfield_id, verdict="PASS", reviewer=OPERATOR,
            reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint="c" * 64, span_roles=("FIELD", "FIELD"))
        definition, blocked = self.promote(source, approval=approval)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_APPROVAL")

    def test_a_rejected_review_is_never_promoted(self):
        definition, blocked = self.promote(self.two_words(), verdict="FAIL")
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "HUMAN_REVIEW_REJECTED")

    def test_validation_rejects_a_field_with_no_resolvable_word(self):
        definition, _ = self.promote(self.two_words())
        broken = replace(definition, fields=(
            replace(definition.fields[0],
                    word_association_source=AssociationSource.UNKNOWN),
        ) + definition.fields[1:])
        with self.assertRaises(StandardSemanticError):
            validate_semantics(
                StandardSemanticSet((broken,), (), ()),
                structures=_Structures((bitfield(self.two_words()),)),
                source_ids=frozenset({SOURCE}),
                corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
                structure_fingerprint=FINGERPRINT)

    def test_validation_accepts_the_same_range_in_different_words(self):
        source = table((ruler_row(),
                        word_row(1, ((0, ("FIELD_A (23-12)",)),
                                     (1, ("FIELD_B (23-12)",))))))
        definition, _ = self.promote(source)
        validate_semantics(
            StandardSemanticSet((definition,), (), ()),
            structures=_Structures((bitfield(source),)),
            source_ids=frozenset({SOURCE}), corpus_fingerprint=FINGERPRINT,
            layout_fingerprint=LAYOUT, structure_fingerprint=FINGERPRINT)


class _Structures:
    def __init__(self, bitfields):
        self.bitfields = bitfields
        self.tables = ()


class DeterminismTests(_Words):
    def two_words(self):
        return table((ruler_row(),
                      word_row(1, ((0, ("FIELD_A (23-12)",)),
                                   (1, ("FIELD_B (23-12)",))))))

    def test_a_word_local_definition_is_reproducible(self):
        first, _ = self.promote(self.two_words())
        second, _ = self.promote(self.two_words())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.semantic_fingerprint, second.semantic_fingerprint)

    def test_the_word_takes_part_in_a_field_identity(self):
        # The same label, the same bits, two different words: two fields, and
        # two identities. Without the word in the hash they would collapse.
        source = table((ruler_row(),
                        word_row(1, ((0, ("SAME_LABEL (23-12)",)),
                                     (1, ("SAME_LABEL (23-12)",))))))
        definition, blocked = self.promote(source)
        self.assertIsNone(blocked)
        self.assertEqual(len(definition.fields), 2)
        self.assertEqual(len({field.field_id for field in definition.fields}), 2)
        self.assertEqual(sorted(field.word_index for field in definition.fields),
                         [0, 1])

    def test_an_empty_semantic_set_hashes_the_same_way_twice(self):
        empty = StandardSemanticSet((), (), ())
        self.assertEqual(semantic_fingerprint(empty), semantic_fingerprint(empty))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
