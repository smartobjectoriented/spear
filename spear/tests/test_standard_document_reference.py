"""STD3C: a noun that names a numbered part of a document, versus a field name.

"Figure 3-1" refers; "Noise Figure" names. The guard used to reject whichever
range followed the word, which threw away a real field and — because a genuine
reference puts its number between the noun and the range — let the references
through. Every fixture here is invented; no licensed text appears.
"""

import unittest

from standard_stated_range import (
    DISQUALIFYING_CONTEXT, PROSE_CONTEXT, RangeContext, RangeStatus,
    accepted_ranges, is_document_reference, parse_stated_ranges,
)

BITFIELD = RangeContext(in_bitfield_region=True)


class _Ranges(unittest.TestCase):
    def rejected(self, text):
        found = parse_stated_ranges(text, context=BITFIELD)
        self.assertTrue(found, f"{text!r} produced no range token to judge")
        self.assertFalse([item for item in found
                          if item.status is RangeStatus.ACCEPTED],
                         f"{text!r} was accepted")
        return found[0].warnings

    def accepted(self, text):
        found = accepted_ranges(text, context=BITFIELD)
        self.assertEqual(len(found), 1, f"{text!r} -> {found}")
        return found[0]


class DocumentReferenceTests(_Ranges):
    """A numbered reference must stay rejected, and now actually is."""

    def test_a_numbered_figure_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT, self.rejected("Figure 3-1 (15..0)"))

    def test_a_figure_reference_in_a_sentence_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT,
                      self.rejected("Figure 7 shows bits 15..0"))

    def test_a_see_figure_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT,
                      self.rejected("see Figure 4-2, bits 31..16"))

    def test_an_abbreviated_figure_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT, self.rejected("Fig. 2 (15..0)"))
        self.assertIn(DISQUALIFYING_CONTEXT, self.rejected("Fig 2 (15..0)"))

    def test_a_clause_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT, self.rejected("clause 9.3 (15..0)"))

    def test_a_version_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT, self.rejected("version 2 (15..0)"))

    def test_a_table_or_rule_reference_is_refused(self):
        self.assertIn(DISQUALIFYING_CONTEXT,
                      self.rejected("Table 9.3.2-1 (15..0)"))
        self.assertIn(DISQUALIFYING_CONTEXT,
                      self.rejected("Rule 9.3.2-5 (31..28)"))

    def test_a_bare_noun_with_a_range_hung_off_it_is_refused(self):
        for text in ("Figure (31-24)", "Clause (7-4)", "Version (23-12)",
                     "Part [15:8]", "Section (7..0)"):
            self.assertIn(DISQUALIFYING_CONTEXT, self.rejected(text), text)

    def test_the_predicate_reads_the_text_before_the_range(self):
        self.assertTrue(is_document_reference("Figure 3-1 "))
        self.assertTrue(is_document_reference("see Figure 4-2, "))
        self.assertTrue(is_document_reference("Figure "))
        self.assertFalse(is_document_reference("Noise Figure "))
        self.assertFalse(is_document_reference("Figure of Merit "))
        self.assertFalse(is_document_reference(""))


class FieldNameTests(_Ranges):
    """A noun inside a field's name is not a reference."""

    def test_a_field_name_ending_in_figure_is_accepted(self):
        parsed = self.accepted("Noise Figure (15..0), dB")
        self.assertEqual((parsed.high_bit, parsed.low_bit), (15, 0))

    def test_a_field_name_beginning_with_figure_is_accepted(self):
        parsed = self.accepted("Figure of Merit (7..0)")
        self.assertEqual((parsed.high_bit, parsed.low_bit), (7, 0))

    def test_a_field_name_with_figure_in_the_middle_is_accepted(self):
        self.accepted("Configuration Figure (31..16)")

    def test_a_unit_after_a_comma_still_does_not_refuse_the_name(self):
        parsed = self.accepted("Gain Figure (7..4), dB")
        self.assertEqual((parsed.high_bit, parsed.low_bit), (7, 4))

    def test_an_unrelated_neighbouring_name_is_unaffected(self):
        self.accepted("Signal-to-Noise Ratio (15..0), dB")

    def test_other_nouns_are_free_inside_names_too(self):
        for text in ("Frame Section (7..0)", "Packet Part (3..0)",
                     "Revision Table (15..8)"):
            parsed = self.accepted(text)
            self.assertGreaterEqual(parsed.high_bit, parsed.low_bit)


class UnchangedGuardTests(_Ranges):
    """Everything else STD2C refuses must still be refused."""

    def test_the_prose_guard_is_untouched(self):
        self.assertIn(PROSE_CONTEXT, self.rejected(
            "and a fractional part, the binary point falling after bit 22."))
        self.assertIn(PROSE_CONTEXT, self.rejected(
            "Implementations shall ignore bits 15-8 of the second word."))

    def test_a_unit_after_the_range_still_refuses(self):
        self.assertIn("UNIT_FOLLOWS_RANGE", self.rejected("Frequency (31-24) MHz"))

    def test_an_ascending_bare_pair_still_refuses(self):
        self.assertIn("AMBIGUOUS_ASCENDING_RANGE",
                      self.rejected("SCALE_LEVEL (0-9)"))

    def test_a_word_count_is_still_not_a_range(self):
        self.assertEqual(parse_stated_ranges("HEADER (2 Words, Optional)",
                                             context=BITFIELD), ())

    def test_a_date_is_still_not_a_range(self):
        self.assertEqual(parse_stated_ranges("Date (2026-08)",
                                             context=BITFIELD), ())

    def test_context_is_still_required_for_a_bare_bracketed_pair(self):
        found = parse_stated_ranges("Noise Figure (15..0)", context=RangeContext())
        self.assertEqual(found[0].status, RangeStatus.REJECTED)
        self.assertIn("NO_BITFIELD_CONTEXT", found[0].warnings)

    def test_parsing_is_deterministic(self):
        for text in ("Noise Figure (15..0), dB", "Figure 3-1 (15..0)"):
            first = parse_stated_ranges(text, context=BITFIELD)
            second = parse_stated_ranges(text, context=BITFIELD)
            self.assertEqual([item.to_dict() for item in first],
                             [item.to_dict() for item in second])

    def test_parsing_mutates_nothing_it_is_given(self):
        text = "Noise Figure (15..0), dB"
        before = str(text)
        parse_stated_ranges(text, context=BITFIELD)
        self.assertEqual(text, before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
