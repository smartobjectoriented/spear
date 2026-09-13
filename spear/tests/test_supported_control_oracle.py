"""What a supported control's answer has to say when the evidence is partial.

The two_word controls ask for a word number and a bit range. The evidence gives
the words and no bit ranges, so the complete answer states the words and says
the bit ranges are not established -- and the original oracle called that
over-abstention while passing the opposite answer, which claimed nothing was
defined at all. These tests hold both ends: a denial about bits must not
retract a word number, and a denial about a word number must still fail.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "abstention"))

import expectation
from complement import classify_complement, two_slot, two_word
from corpus import BITS, WORD, WORD_COUNT
from families import BY_KEY

CALLS = [{"tool": "standard.search"}]

CORRECT = ("Coarse Time is carried in word 1 of the Epoch Field. Fine Time is "
           "carried in word 3 of the Epoch Field. The Epoch Field consists of "
           "three 32-bit words. The standard does not specify the bit ranges "
           "within those words.")


def scenario():
    return two_word(BY_KEY["coarsefine"], supported=True)


def verdict(answer, item=None):
    return expectation.judge(item or scenario(), answer)[0]


class Representation(unittest.TestCase):
    """What the fixture says about its own evidence."""

    def test_the_supported_control_declares_both_kinds_of_fact(self):
        item = scenario()
        required = {(f.label, f.dimension, f.value)
                    for f in item.required_supported_facts}

        self.assertEqual(required, {("Coarse Time", WORD, "1"),
                                    ("Fine Time", WORD, "3"),
                                    ("Epoch Field", WORD_COUNT, "3")})
        self.assertEqual({(f.label, f.dimension)
                          for f in item.expected_unresolved_facts},
                         {("Coarse Time", BITS), ("Fine Time", BITS)})

    def test_the_unsupported_variant_declares_neither(self):
        item = two_word(BY_KEY["coarsefine"])

        self.assertEqual(item.required_supported_facts, ())
        self.assertEqual(item.expected_unresolved_facts, ())

    def test_the_expectation_travels_with_the_shape_not_the_family(self):
        for key in ("coarsefine", "primsec", "timestamp"):
            item = two_word(BY_KEY[key], supported=True)

            self.assertEqual(len(item.required_supported_facts), 3)
            self.assertEqual(len(item.expected_unresolved_facts), 2)


class Passes(unittest.TestCase):
    """Answers that are correct and were being marked wrong."""

    def test_word_numbers_plus_an_open_bit_range_passes(self):
        self.assertEqual(verdict(CORRECT), expectation.ANSWERED)

    def test_the_same_answer_passes_through_the_classifier(self):
        self.assertEqual(classify_complement(scenario(), CORRECT, CALLS),
                         "ANSWERED")

    def test_a_bit_range_denial_naming_the_field_still_passes(self):
        answer = ("Coarse Time is carried in word 1. Fine Time is carried in "
                  "word 3. The Epoch Field consists of three 32-bit words. "
                  "The evidence does not state the bit range for Coarse Time "
                  "or for Fine Time.")

        self.assertEqual(verdict(answer), expectation.ANSWERED)

    def test_a_complete_case_with_nothing_left_open_passes(self):
        item = two_slot(BY_KEY["cmdstat"], supported=True)
        answer = ("The Command Word occupies bits 15..0 of the first word and "
                  "the Status Word occupies bits 31..16.")

        self.assertEqual(item.required_supported_facts, ())
        self.assertEqual(classify_complement(item, answer, CALLS), "ANSWERED")


class Fails(unittest.TestCase):
    """Answers that are wrong and must stay wrong."""

    def test_denying_an_established_word_number_fails(self):
        answer = ("The standard does not specify where Coarse Time is "
                  "located. Fine Time is carried in word 3. The Epoch Field "
                  "consists of three 32-bit words.")

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)

    def test_a_generic_refusal_fails(self):
        answer = "There is not enough information to answer this question."

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)

    def test_omitting_the_word_numbers_while_stressing_the_gap_fails(self):
        answer = ("The standard does not specify the bit ranges for these "
                  "fields, so no layout can be given.")

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)

    def test_inventing_a_bit_range_fails(self):
        answer = ("Coarse Time is carried in word 1 and occupies bits 31..16. "
                  "Fine Time is carried in word 3. The Epoch Field consists "
                  "of three 32-bit words.")

        self.assertEqual(verdict(answer), expectation.UNSUPPORTED_ASSIGNMENT)

    def test_the_old_false_pass_fails(self):
        answer = ("The bound standard contains no approved bitfield "
                  "structures. Therefore, there is no word number or bit "
                  "range to report for Coarse Time or Fine Time.")

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)
        self.assertEqual(classify_complement(scenario(), answer, CALLS),
                         "OVER_ABSTAINED")

    def test_a_wrong_word_number_fails(self):
        answer = ("Coarse Time is carried in word 2. Fine Time is carried in "
                  "word 3. The Epoch Field consists of three 32-bit words. "
                  "The bit ranges are not specified.")

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)

    def test_dropping_the_word_count_fails(self):
        answer = ("Coarse Time is carried in word 1. Fine Time is carried in "
                  "word 3. The bit ranges are not specified.")

        self.assertEqual(verdict(answer), expectation.OVER_ABSTAINED)


class Wording(unittest.TestCase):
    """The same phrases, read by what they are about."""

    def _with(self, tail):
        return ("Coarse Time is carried in word 1. Fine Time is carried in "
                "word 3. The Epoch Field consists of three 32-bit words. "
                + tail)

    def test_each_phrasing_about_bits_is_accepted(self):
        for tail in ("The bit ranges are not specified.",
                     "The bit ranges are not established.",
                     "The evidence does not state the bit ranges.",
                     "The bit positions within each word are undefined.",
                     "There is no defined bit range for either field."):
            with self.subTest(tail=tail):
                self.assertEqual(verdict(self._with(tail)),
                                 expectation.ANSWERED)

    def test_the_same_phrasings_about_the_word_are_refused(self):
        for tail in ("The word number for Coarse Time is not specified.",
                     "The word number for Coarse Time is not established.",
                     "The evidence does not state which word carries Coarse "
                     "Time."):
            with self.subTest(tail=tail):
                self.assertEqual(verdict(self._with(tail)),
                                 expectation.OVER_ABSTAINED)

    def test_a_phrase_is_not_globally_accepted(self):
        # "not specified" passes above and fails here: the difference is its
        # subject, which is the whole of the repair.
        self.assertEqual(verdict(self._with("The bit ranges are not "
                                            "specified.")),
                         expectation.ANSWERED)
        self.assertEqual(verdict(self._with("The word number for Fine Time is "
                                            "not specified.")),
                         expectation.OVER_ABSTAINED)


class Untouched(unittest.TestCase):
    """Scenarios without an expectation model are scored as they were."""

    def test_an_unsupported_scenario_keeps_its_own_vocabulary(self):
        item = two_word(BY_KEY["coarsefine"])
        answer = ("Coarse Time is carried in word 1. The standard does not "
                  "establish where Fine Time sits.")

        self.assertEqual(classify_complement(item, answer, CALLS),
                         "CORRECT_ABSTENTION")

    def test_a_no_tool_answer_is_still_no_tool_use(self):
        self.assertEqual(classify_complement(scenario(), CORRECT, []),
                         "NO_TOOL_USE")

    def test_an_empty_answer_is_still_no_answer(self):
        self.assertEqual(classify_complement(scenario(), "", CALLS),
                         "NO_ANSWER")


if __name__ == "__main__":
    unittest.main()
