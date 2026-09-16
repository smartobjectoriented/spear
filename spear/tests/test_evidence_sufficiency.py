"""Evidence sets, on invented identities.

The rule that matters is the joint one: a conclusion with two normative
clauses is not supported by retrieving one of them, however highly that one
is ranked. A scorer that counted it as a hit would report a benchmark green
while the turn it models could not have reached the conclusion.
"""

from __future__ import annotations

import unittest

import evidence_sufficiency as es

ALONE = [["Rule 1-1"]]
EITHER = [["Rule 1-1"], ["Rule 1-2"]]
BOTH = [["Rule 1-1", "Rule 1-2"]]


class OneSetOneProvision(unittest.TestCase):

    def test_it_is_covered_when_retrieved(self):
        self.assertTrue(es.is_covered(ALONE, ["Rule 1-1", "Rule 9-9"]))

    def test_it_is_not_covered_otherwise(self):
        self.assertFalse(es.is_covered(ALONE, ["Rule 1-2"]))

    def test_its_arrival_is_its_own_rank(self):
        self.assertEqual(es.first_covering_rank(ALONE, {"Rule 1-1": 4}), 4)


class EitherOfTwo(unittest.TestCase):

    def test_the_first_alone_suffices(self):
        self.assertTrue(es.is_covered(EITHER, ["Rule 1-1"]))

    def test_the_second_alone_suffices(self):
        self.assertTrue(es.is_covered(EITHER, ["Rule 1-2"]))

    def test_neither_does_not(self):
        self.assertFalse(es.is_covered(EITHER, ["Rule 7-7"]))

    def test_it_arrives_with_whichever_comes_first(self):
        self.assertEqual(
            es.first_covering_rank(EITHER, {"Rule 1-1": 9, "Rule 1-2": 2}), 2)

    def test_one_missing_member_does_not_block_the_other_set(self):
        self.assertEqual(
            es.first_covering_rank(EITHER, {"Rule 1-1": None, "Rule 1-2": 6}), 6)


class BothTogether(unittest.TestCase):
    """The case this file exists for."""

    def test_one_of_them_is_not_enough(self):
        self.assertFalse(es.is_covered(BOTH, ["Rule 1-1"]))

    def test_the_other_alone_is_not_enough_either(self):
        self.assertFalse(es.is_covered(BOTH, ["Rule 1-2"]))

    def test_a_high_rank_for_one_does_not_cover_the_set(self):
        self.assertFalse(es.covered_at(BOTH, {"Rule 1-1": 1, "Rule 1-2": None}, 20))

    def test_together_they_are(self):
        self.assertTrue(es.is_covered(BOTH, ["Rule 1-2", "Rule 1-1"]))

    def test_the_set_arrives_with_its_LAST_member(self):
        self.assertEqual(
            es.first_covering_rank(BOTH, {"Rule 1-1": 2, "Rule 1-2": 11}), 11)

    def test_a_window_below_that_does_not_carry_it(self):
        ranks = {"Rule 1-1": 2, "Rule 1-2": 11}

        self.assertFalse(es.covered_at(BOTH, ranks, 10))
        self.assertTrue(es.covered_at(BOTH, ranks, 11))


class CasesThatNameNoEvidence(unittest.TestCase):
    """An absence question asks nothing of retrieval and must not be scored
    as a retrieval failure."""

    def test_they_are_excluded_from_recall(self):
        self.assertEqual(es.recall_at([([], {}), (ALONE, {"Rule 1-1": 1})], 5), 1.0)

    def test_an_empty_set_is_not_satisfied_by_nothing(self):
        self.assertFalse(es.is_covered([[]], []))

    def test_recall_is_none_when_nothing_is_scoreable(self):
        self.assertIsNone(es.recall_at([([], {})], 5))


class RankMetrics(unittest.TestCase):

    def test_recall_counts_complete_sets_only(self):
        cases = [(BOTH, {"Rule 1-1": 1, "Rule 1-2": 8}),
                 (ALONE, {"Rule 1-1": 2})]

        self.assertEqual(es.recall_at(cases, 5), 0.5)
        self.assertEqual(es.recall_at(cases, 8), 1.0)

    def test_the_reciprocal_rank_is_of_the_complete_set(self):
        """Not of the first useful hit: a conclusion resting on two
        provisions is not half-supported by one."""
        self.assertAlmostEqual(
            es.mean_reciprocal_rank([(BOTH, {"Rule 1-1": 1, "Rule 1-2": 4})]),
            0.25)

    def test_an_uncovered_case_contributes_nothing(self):
        self.assertEqual(
            es.mean_reciprocal_rank([(BOTH, {"Rule 1-1": 1, "Rule 1-2": None})]),
            0.0)


if __name__ == "__main__":
    unittest.main()
