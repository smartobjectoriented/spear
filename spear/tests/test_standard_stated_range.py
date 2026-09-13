"""STD2C: recognising the bit ranges a standard states in its own words.

Every fixture here is invented. No licensed normative text appears in this file.
"""

import unittest

from standard_stated_range import (
    AMBIGUOUS_ASCENDING_RANGE, COUNT_DESCRIPTOR, DISQUALIFYING_CONTEXT,
    GEOMETRY_RANGE_CONFLICT, MAX_BIT, NO_BITFIELD_CONTEXT, RANGE_EXCEEDS_WORD_WIDTH,
    PROSE_CONTEXT, RANGE_OUTSIDE_RULER, RANGE_OUT_OF_BOUNDS, UNIT_FOLLOWS_RANGE,
    GeometryAgreement, RangeContext, RangeDomain, RangeStatus, RulerAgreement,
    StatedOrder, SyntaxKind, accepted_ranges, agreement_with_geometry,
    agreement_with_ruler, display_label, is_count_descriptor, parse_stated_ranges,
    reads_as_prose, ruler_is_trusted, stated_ranges_for_cell,
)

WORD = tuple(range(31, -1, -1))
BITFIELD = RangeContext(in_bitfield_region=True, ruler_labels=WORD)


class SyntaxTests(unittest.TestCase):
    """The shapes an explicit bit range is written in."""

    def one(self, text, context=BITFIELD):
        found = accepted_ranges(text, context=context)
        self.assertEqual(len(found), 1, f"{text!r} -> {found}")
        return found[0]

    def test_a_parenthesised_dash_range_is_read_high_to_low(self):
        parsed = self.one("FIELD_A (23-12)")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (23, 12, 12))
        self.assertEqual(parsed.syntax_kind, SyntaxKind.PAREN_DASH)
        self.assertEqual(parsed.stated_order, StatedOrder.HIGH_FIRST)

    def test_a_parenthesised_double_dot_range_is_read(self):
        parsed = self.one("FIELD_B (31..24)")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (31, 24, 8))
        self.assertEqual(parsed.syntax_kind, SyntaxKind.PAREN_DOTDOT)

    def test_a_bracketed_colon_range_is_read(self):
        parsed = self.one("FIELD_C [15:8]")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (15, 8, 8))
        self.assertEqual(parsed.syntax_kind, SyntaxKind.BRACKET_COLON)

    def test_a_bits_keyword_range_is_read(self):
        parsed = self.one("FIELD_D bits 7-4")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (7, 4, 4))
        self.assertEqual(parsed.syntax_kind, SyntaxKind.BITS_DASH)

    def test_a_spelled_out_through_range_is_read(self):
        parsed = self.one("FIELD_E Bits 11 through 0")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (11, 0, 12))
        self.assertEqual(parsed.syntax_kind, SyntaxKind.BITS_THROUGH)

    def test_a_single_bit_is_a_range_of_width_one(self):
        parsed = self.one("FIELD_F bit 3")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (3, 3, 1))
        self.assertEqual(parsed.stated_order, StatedOrder.SINGLE)

    def test_an_angle_bracket_range_is_read(self):
        parsed = self.one("FIELD_G <31:24>")
        self.assertEqual((parsed.high_bit, parsed.low_bit, parsed.width), (31, 24, 8))

    def test_a_keyword_range_written_low_first_is_still_unambiguous(self):
        # "bits" names what the numbers are, so either order can be normalized.
        parsed = self.one("FIELD_H bits 4 to 7")
        self.assertEqual((parsed.high_bit, parsed.low_bit), (7, 4))
        self.assertEqual(parsed.stated_order, StatedOrder.LOW_FIRST)

    def test_every_accepted_range_normalizes_high_at_or_above_low(self):
        for text in ("A (23-12)", "B (31..24)", "C [15:8]", "D bits 7-4",
                     "E Bits 11 through 0", "F bit 3", "G bits 4 to 7"):
            parsed = self.one(text)
            self.assertGreaterEqual(parsed.high_bit, parsed.low_bit, text)
            self.assertEqual(parsed.width, parsed.high_bit - parsed.low_bit + 1)


class RefusalTests(unittest.TestCase):
    """Number pairs that look like ranges and are not."""

    def refused(self, text, context=BITFIELD):
        found = parse_stated_ranges(text, context=context)
        self.assertFalse([item for item in found
                          if item.status is RangeStatus.ACCEPTED],
                         f"{text!r} was accepted")
        return found

    def test_a_bare_number_pair_is_never_a_range(self):
        for text in ("Version 23-12", "Clause 7-4", "Figure 31-24", "Part 15:8",
                     "Date 2026-08", "Frequency 31-24 MHz"):
            self.assertEqual(self.refused(text), (),
                             f"{text!r} produced a range token")

    def test_a_date_is_not_a_range_even_in_parentheses(self):
        self.assertEqual(self.refused("Date (2026-08)"), ())

    def test_a_clause_number_is_refused_by_the_word_in_front_of_it(self):
        found = self.refused("Clause (7-4)")
        self.assertIn(DISQUALIFYING_CONTEXT, found[0].warnings)

    def test_a_figure_number_is_refused_by_the_word_in_front_of_it(self):
        found = self.refused("Figure (31-24)")
        self.assertIn(DISQUALIFYING_CONTEXT, found[0].warnings)

    def test_a_version_number_is_refused_by_the_word_in_front_of_it(self):
        found = self.refused("Version (23-12)")
        self.assertIn(DISQUALIFYING_CONTEXT, found[0].warnings)

    def test_a_bracketed_part_number_is_refused(self):
        found = self.refused("Part [15:8]")
        self.assertIn(DISQUALIFYING_CONTEXT, found[0].warnings)

    def test_a_measurement_is_refused_by_the_unit_after_it(self):
        found = self.refused("Frequency (31-24) MHz")
        self.assertIn(UNIT_FOLLOWS_RANGE, found[0].warnings)

    def test_a_unit_after_a_comma_does_not_refuse_the_range(self):
        # A field may legitimately name the unit of the value it carries.
        parsed = accepted_ranges("SOME_ANGLE (15..0), degrees", context=BITFIELD)
        self.assertEqual((parsed[0].high_bit, parsed[0].low_bit), (15, 0))

    def test_a_word_count_is_not_a_bit_range(self):
        self.assertTrue(is_count_descriptor("HEADER (2 Words, Optional)"))
        self.assertEqual(self.refused("HEADER (2 Words, Optional)"), ())
        self.assertEqual(self.refused("PAYLOAD (1 Word)"), ())

    def test_a_byte_or_octet_count_is_not_a_bit_range(self):
        for text in ("BLOCK (4 bytes)", "BLOCK (8 octets)", "BLOCK (16 bits)"):
            self.assertEqual(self.refused(text), (), text)

    def test_an_ascending_bare_pair_is_too_ambiguous_to_accept(self):
        # A scale written low to high with nothing naming bits reads exactly
        # like a bit range written backwards. Refuse rather than guess.
        found = self.refused("SCALE_LEVEL (0-9)")
        self.assertIn(AMBIGUOUS_ASCENDING_RANGE, found[0].warnings)

    def test_a_range_outside_any_sane_bit_bound_is_refused(self):
        found = self.refused(f"WIDE ({MAX_BIT + 10}-{MAX_BIT + 2})")
        self.assertIn(RANGE_OUT_OF_BOUNDS, found[0].warnings)

    def test_a_malformed_range_is_not_a_range(self):
        for text in ("FIELD (23-)", "FIELD (-12)", "FIELD (..12)", "FIELD [15:]"):
            self.assertEqual(self.refused(text), (), text)

    def test_a_range_needs_a_bitfield_context_unless_it_names_bits(self):
        found = parse_stated_ranges("FIELD_A (23-12)", context=RangeContext())
        self.assertEqual(found[0].status, RangeStatus.REJECTED)
        self.assertIn(NO_BITFIELD_CONTEXT, found[0].warnings)
        # naming bits is its own evidence
        self.assertTrue(accepted_ranges("FIELD_A bits 23-12", context=RangeContext()))

    def test_bit_terminology_in_the_same_text_is_enough_context(self):
        self.assertTrue(accepted_ranges("Bitmapped subfield (23-12)",
                                        context=RangeContext()))

    def test_a_bit_mentioned_in_a_sentence_is_not_that_cell_position(self):
        # A sentence explaining where a radix point sits mentions a bit; the
        # cell it sits in is prose, not a field label.
        found = self.refused(
            "and a fractional part, the binary point falling after bit 22.")
        self.assertIn(PROSE_CONTEXT, found[0].warnings)

    def test_a_long_run_of_text_is_prose_however_it_is_punctuated(self):
        self.assertTrue(reads_as_prose("x" * 120))
        self.assertFalse(reads_as_prose("SOME_FIELD (23-12)"))
        self.assertFalse(reads_as_prose("Angle (15..0), degrees"))

    def test_a_range_wider_than_the_word_is_not_a_word_local_position(self):
        parsed = parse_stated_ranges("WIDE_VALUE (63..32)", context=BITFIELD)[0]
        self.assertEqual(parsed.domain, RangeDomain.EXCEEDS_WORD_WIDTH)
        self.assertIn(RANGE_EXCEEDS_WORD_WIDTH, parsed.warnings)


class RulerTests(unittest.TestCase):
    def test_a_complete_power_of_two_run_from_zero_is_a_whole_ruler(self):
        self.assertTrue(ruler_is_trusted(list(range(31, -1, -1))))
        self.assertTrue(ruler_is_trusted(list(range(0, 16))))

    def test_a_short_or_offset_run_is_a_fragment_not_a_ruler(self):
        self.assertFalse(ruler_is_trusted([3, 2, 1, 0]))
        self.assertFalse(ruler_is_trusted(list(range(31, 23, -1))))
        self.assertFalse(ruler_is_trusted(list(range(19, -1, -1))))
        self.assertFalse(ruler_is_trusted([]))

    def test_a_range_inside_a_whole_ruler_is_accepted(self):
        parsed = accepted_ranges("FIELD (23-12)", context=BITFIELD)[0]
        self.assertEqual(agreement_with_ruler(parsed, WORD),
                         RulerAgreement.INSIDE_RULER)

    def test_a_range_outside_a_whole_ruler_is_marked_and_blocks_promotion(self):
        ruler = tuple(range(15, -1, -1))
        bound = stated_ranges_for_cell(
            "FIELD (23-12)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH",
            context=RangeContext(in_bitfield_region=True, ruler_labels=ruler))[0]
        self.assertEqual(bound.ruler_agreement, RulerAgreement.OUTSIDE_RULER)
        self.assertIn(RANGE_OUTSIDE_RULER, bound.warnings)
        self.assertEqual(bound.status, RangeStatus.REJECTED)
        self.assertFalse(bound.promotable)

    def test_a_range_is_never_clamped_to_the_ruler(self):
        ruler = tuple(range(15, -1, -1))
        bound = stated_ranges_for_cell(
            "FIELD (23-12)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH",
            context=RangeContext(in_bitfield_region=True, ruler_labels=ruler))[0]
        self.assertEqual((bound.high_bit, bound.low_bit), (23, 12))

    def test_a_fragment_ruler_cannot_veto_a_stated_range(self):
        fragment = (17, 16, 15, 14, 13, 12, 11, 10)
        bound = stated_ranges_for_cell(
            "FIELD (31..24)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH",
            context=RangeContext(in_bitfield_region=True, ruler_labels=fragment))[0]
        self.assertEqual(bound.ruler_agreement, RulerAgreement.RULER_UNTRUSTED)
        self.assertTrue(bound.promotable)


class GeometryAgreementTests(unittest.TestCase):
    def bound(self, text, covered, ruler=WORD):
        return stated_ranges_for_cell(
            text, cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH", covered_labels=covered,
            context=RangeContext(in_bitfield_region=True, ruler_labels=ruler))[0]

    def test_a_measurement_matching_the_stated_range_agrees(self):
        bound = self.bound("FIELD (23-20)", (23, 22, 21, 20))
        self.assertEqual(bound.geometry_agreement, GeometryAgreement.AGREES)
        self.assertFalse(bound.geometry_conflict)

    def test_a_measurement_inside_the_stated_range_partly_overlaps(self):
        bound = self.bound("FIELD (23-12)", (17, 16, 15, 14))
        self.assertEqual(bound.geometry_agreement,
                         GeometryAgreement.PARTIAL_OVERLAP)
        self.assertTrue(bound.geometry_conflict)
        self.assertIn(GEOMETRY_RANGE_CONFLICT, bound.warnings)

    def test_a_disjoint_measurement_contradicts(self):
        bound = self.bound("FIELD (23-20)", (11, 10, 9, 8))
        self.assertEqual(bound.geometry_agreement, GeometryAgreement.CONTRADICTS)
        self.assertTrue(bound.geometry_conflict)

    def test_a_contradiction_never_changes_the_stated_range(self):
        bound = self.bound("FIELD (23-12)", (17, 16, 15, 14))
        self.assertEqual((bound.high_bit, bound.low_bit, bound.width), (23, 12, 12))
        self.assertTrue(bound.promotable)

    def test_a_measurement_against_a_fragment_ruler_means_nothing(self):
        bound = self.bound("FIELD (31..24)", (17, 16), ruler=(17, 16, 15, 14))
        self.assertEqual(bound.geometry_agreement,
                         GeometryAgreement.GEOMETRY_NOT_MEANINGFUL)
        self.assertFalse(bound.geometry_conflict)


class LabelTests(unittest.TestCase):
    def test_the_normative_label_is_kept_whole(self):
        bound = stated_ranges_for_cell(
            "Some Field (23-12)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH", context=BITFIELD)[0]
        self.assertEqual(bound.normative_label, "Some Field (23-12)")

    def test_a_display_label_drops_only_the_range_token(self):
        parsed = accepted_ranges("Some Field (23-12)", context=BITFIELD)[0]
        self.assertEqual(display_label("Some Field (23-12)", parsed), "Some Field")
        parsed = accepted_ranges("Angle (15..0), degrees", context=BITFIELD)[0]
        self.assertEqual(display_label("Angle (15..0), degrees", parsed),
                         "Angle, degrees")

    def test_a_label_that_is_only_a_range_keeps_its_text(self):
        parsed = accepted_ranges("(23-12)", context=BITFIELD)[0]
        self.assertEqual(display_label("(23-12)", parsed), "(23-12)")


class ProvenanceTests(unittest.TestCase):
    def test_a_bound_range_carries_the_cell_and_its_canonical_sources(self):
        bound = stated_ranges_for_cell(
            "FIELD (23-12)", cell_id="cel-abc", source_ids=("std-1", "std-2"),
            provenance="DIRECT_TEXT_MATCH", context=BITFIELD)[0]
        self.assertEqual(bound.raw_source_cell_id, "cel-abc")
        self.assertEqual(bound.supporting_source_ids, ("std-1", "std-2"))
        self.assertEqual(bound.provenance_grade, "DIRECT_TEXT_MATCH")
        self.assertTrue(bound.promotable)

    def test_a_range_with_no_canonical_source_is_not_promotable(self):
        bound = stated_ranges_for_cell(
            "FIELD (23-12)", cell_id="cel-abc", source_ids=(),
            provenance="UNRESOLVED", context=BITFIELD)[0]
        self.assertTrue(bound.positional)
        self.assertFalse(bound.promotable)

    def test_a_bound_range_round_trips_through_its_dictionary(self):
        bound = stated_ranges_for_cell(
            "FIELD (23-12)", cell_id="cel-abc", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH", context=BITFIELD)[0]
        value = bound.to_dict()
        self.assertEqual(value["high_bit"], 23)
        self.assertEqual(value["syntax_kind"], SyntaxKind.PAREN_DASH.value)
        self.assertEqual(value["status"], RangeStatus.ACCEPTED.value)
        self.assertIsInstance(value["supporting_source_ids"], list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
