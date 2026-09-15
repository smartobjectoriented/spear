"""A fragment is kept and not ranked; structure is ranked at any length.

The rule exists because exhaustive span coverage and lexical ranking pull
against each other: covering every span is what makes a corpus verifiable,
and it is also what fills the corpus with three-word fragments that distort
BM25's length normalisation for everything else.

Nothing here knows any particular document. The examples are invented, and a
change that made this pass only for one standard's column names would be the
hack this file exists to prevent.
"""

from __future__ import annotations

import unittest

import standard_retrieval_policy as policy


class AFragmentIsKeptButNotRanked(unittest.TestCase):

    def test_a_lone_number_is_a_fragment(self):
        self.assertFalse(policy.retrievable("8"))

    def test_a_two_word_bullet_is_a_fragment(self):
        self.assertFalse(policy.retrievable("• Data Payload"))

    def test_an_axis_label_is_a_fragment(self):
        self.assertFalse(policy.retrievable("Window"))

    def test_a_sentence_is_not_a_fragment(self):
        self.assertTrue(policy.retrievable(
            "A Controllee shall return the packet unchanged."))


class StructureIsEvidenceAtAnyLength(unittest.TestCase):
    """Short is how a table cell, a caption and a heading are SUPPOSED to be.
    Judging them by length would delete the document's own structure from
    search while claiming to remove noise."""

    def test_a_row_with_its_own_grid_is_retrievable(self):
        self.assertTrue(policy.retrievable("24 A1 See Table 9", has_structure=True))

    def test_a_short_row_is_retrievable_by_content_type(self):
        self.assertTrue(policy.retrievable("24 A1", content_type="TABLE_ROW"))

    def test_a_caption_is_retrievable(self):
        self.assertTrue(policy.retrievable("Table 9-1: Modes",
                                           content_type="CAPTION"))

    def test_a_heading_is_retrievable(self):
        self.assertTrue(policy.retrievable("9.1 Modes", content_type="HEADING"))

    def test_a_formula_is_retrievable(self):
        self.assertTrue(policy.retrievable("t = n/f", content_type="FORMULA"))

    def test_an_unknown_short_unit_is_not_rescued_by_its_type(self):
        self.assertFalse(policy.retrievable("t = n/f", content_type="UNKNOWN"))


class APrintedNameIsEvidence(unittest.TestCase):
    """A provision the document numbered is citable however tersely it is
    written, and a corpus that hid it from search would make the citation
    unreachable."""

    def test_a_short_labelled_provision_is_retrievable(self):
        self.assertTrue(policy.retrievable("Rule 5.2-1: Reserved."))

    def test_every_numbered_kind_counts(self):
        for label in ("Rule 5.2-1", "Recommendation 5.2-1", "Permission 5.2-1",
                      "Observation 5.2-1", "Definition 5.2-1"):
            with self.subTest(label=label):
                self.assertTrue(policy.retrievable(label + ": x."))

    def test_a_bare_section_number_is_not_a_printed_name(self):
        """"8.3.1" names a section, not a provision, and a fragment that
        happens to contain one is still a fragment."""
        self.assertFalse(policy.carries_printed_identity("8.3.1"))


class TheThresholdIsAPolicyNotAConstant(unittest.TestCase):

    def test_it_can_be_moved_without_touching_the_rule(self):
        self.assertTrue(policy.retrievable("one two three", minimum=3))
        self.assertFalse(policy.retrievable("one two three", minimum=4))

    def test_empty_text_is_never_ranked(self):
        self.assertFalse(policy.retrievable(""))
        self.assertFalse(policy.retrievable(None))


if __name__ == "__main__":
    unittest.main()
