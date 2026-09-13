"""When an empty structure registry is mistaken for an empty standard.

The last failing supported control reads structure_count: 0 and concludes that
the standard defines no word positions. Two numbered requirements state them,
one search away. So the trigger has to be precise about which failure it is:
the model did use a tool, it did get an answer, and the answer was empty --
which is not the zero-tool case FT3E covers, and not a gap in what the guard
can see. These tests pin the boundary in both directions, because a policy that
fired whenever a registry came back empty would start searching on behalf of
turns that had already looked.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evidence_bootstrap
import evidence_recovery
from evidence_guard import EvidenceLedger, guard
from evidence_recovery import (
    EMPTY_STRUCTURE_RECOVERY, ZERO_TOOL_BOOTSTRAP, is_empty_structure, route,
    should_recover,
)

QUESTION = ("In the bound standard TICKER-TIME-9 2020-R2026: List the word "
            "number and bit range for each of the Coarse Time and the Fine "
            "Time.")
EMPTY = {"standard_id": "TICKER-TIME-9", "revision": "2020-R2026",
         "structure_count": 0, "structures": []}
FULL = {"standard_id": "TICKER-TIME-9", "revision": "2020-R2026",
        "structure_count": 1,
        "structures": [{"definition_id": "bfd-0ccd61a481e290d5",
                        "field_count": 2}]}
NOT_FOUND_EMPTY = {"error": "STRUCTURE_NOT_FOUND",
                   "recovery": {"valid_definition_ids": []}}
NOT_FOUND_WITH_OTHERS = {"error": "STRUCTURE_NOT_FOUND",
                         "recovery": {"valid_definition_ids": ["bfd-abc12345"]}}


def call(tool, *, origin="MODEL", empty=False):
    return {"tool": tool, "origin": origin, "empty_structure": empty}


def structure(empty=True, origin="MODEL"):
    return call("standard.get_structure", origin=origin, empty=empty)


class EmptyResult(unittest.TestCase):
    """What counts as a registry with nothing usable in it."""

    def test_a_zero_count_listing_is_empty(self):
        self.assertTrue(is_empty_structure(EMPTY))

    def test_a_listing_with_a_structure_is_not(self):
        self.assertFalse(is_empty_structure(FULL))

    def test_a_missing_id_in_an_empty_revision_is_empty(self):
        self.assertTrue(is_empty_structure(NOT_FOUND_EMPTY))

    def test_a_misspelled_id_alongside_real_ones_is_not(self):
        self.assertFalse(is_empty_structure(NOT_FOUND_WITH_OTHERS))

    def test_something_that_is_not_a_structure_result_is_not(self):
        for payload in (None, "", [], {}, {"results": []}):
            self.assertFalse(is_empty_structure(payload))


class Trigger(unittest.TestCase):
    """The one situation the policy exists for."""

    def test_an_empty_registry_then_a_premature_answer_recovers(self):
        calls = [structure(), structure()]

        self.assertTrue(should_recover(QUESTION, calls, answer="Nothing is "
                                                              "defined."))

    def test_a_usable_structure_does_not_recover(self):
        calls = [structure(empty=False)]

        self.assertFalse(should_recover(QUESTION, calls, answer="bits 15..0"))

    def test_a_search_after_the_empty_result_does_not_recover(self):
        calls = [structure(), call("standard.search")]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing."))

    def test_a_fetch_after_the_empty_result_does_not_recover(self):
        calls = [structure(), call("standard.fetch")]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing."))

    def test_a_search_before_the_empty_result_still_recovers(self):
        # The registry was the last thing it consulted and the last thing it
        # heard was "nothing"; that is the conclusion being drawn.
        calls = [call("standard.search"), structure()]

        self.assertTrue(should_recover(QUESTION, calls, answer="Nothing."))

    def test_it_fires_at_most_once(self):
        calls = [structure()]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing.",
                                        already_fired=True))

    def test_an_empty_answer_does_not_recover(self):
        calls = [structure()]

        for answer in ("", "   ", None):
            self.assertFalse(should_recover(QUESTION, calls, answer=answer))

    def test_an_unbound_turn_does_not_recover(self):
        calls = [structure()]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing.",
                                        bound=False))

    def test_a_bound_turn_recovers_whatever_its_wording(self):
        """Bound is the decision now — standard_scope.engages() made it before
        this module ran. See test_evidence_bootstrap for why deciding twice
        answered a specification question from the source code."""
        calls = [structure()]

        self.assertTrue(should_recover("which G codes are in modal group 1?",
                                       calls, answer="From interp_array.cc…"))
        self.assertFalse(should_recover("Thanks, that is all for today.",
                                        calls, answer="You are welcome.",
                                        bound=False))

    def test_no_tool_call_at_all_does_not_recover(self):
        self.assertFalse(should_recover(QUESTION, [], answer="Nothing."))


class Routing(unittest.TestCase):
    """What the recovery call actually asks for."""

    def test_it_searches_with_the_question_verbatim(self):
        name, arguments = route(QUESTION)

        self.assertEqual(name, "standard.search")
        self.assertEqual(arguments, {"query": QUESTION})

    def test_it_never_reaches_for_an_identifier(self):
        name, arguments = route("Give the layout of bfd-0ccd61a481e290d5.")

        self.assertEqual(name, "standard.search")
        self.assertNotIn("definition_id", arguments)


class Distinct(unittest.TestCase):
    """FT3E and FT3R answer different failures and never both answer one."""

    def test_the_zero_tool_bootstrap_is_unchanged(self):
        self.assertTrue(evidence_bootstrap.should_bootstrap(
            QUESTION, [], answer="Nothing."))
        self.assertFalse(evidence_bootstrap.should_bootstrap(
            QUESTION, [{"tool": "standard.get_structure"}], answer="Nothing."))

    def test_recovery_does_not_fire_on_the_zero_tool_case(self):
        self.assertFalse(should_recover(QUESTION, [], answer="Nothing."))

    def test_the_bootstrap_does_not_fire_on_the_empty_registry_case(self):
        calls = [{"tool": "standard.get_structure"}]

        self.assertFalse(evidence_bootstrap.should_bootstrap(
            QUESTION, calls, answer="Nothing."))

    def test_a_bootstrap_issued_structure_call_does_not_arm_recovery(self):
        # Otherwise one policy would recover the other's turn for the same
        # failure: the model itself has still called nothing.
        calls = [structure(origin=ZERO_TOOL_BOOTSTRAP)]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing."))

    def test_a_recovery_issued_call_does_not_arm_a_second_recovery(self):
        calls = [structure(),
                 call("standard.search", origin=EMPTY_STRUCTURE_RECOVERY)]

        self.assertFalse(should_recover(QUESTION, calls, answer="Nothing."))

    def test_the_two_origins_are_distinguishable(self):
        self.assertNotEqual(EMPTY_STRUCTURE_RECOVERY, ZERO_TOOL_BOOTSTRAP)
        self.assertNotIn("MODEL", evidence_recovery.INJECTED)


class Downstream(unittest.TestCase):
    """The rest of the pipeline still runs on what recovery produced."""

    def test_the_model_may_still_fetch_after_recovery(self):
        calls = [structure(),
                 call("standard.search", origin=EMPTY_STRUCTURE_RECOVERY),
                 call("standard.fetch")]

        self.assertFalse(should_recover(QUESTION, calls, answer="word 1"))

    def test_the_guard_still_validates_the_resumed_answer(self):
        ledger = EvidenceLedger().observe_fetch({"unit": {
            "source_id": "std-" + "c" * 32, "section": "7.2.2", "page": 85,
            "text": "Rule 7.2.2-1: The Coarse Time shall be carried in word 1 "
                    "of the Epoch Field.", "content_type": "requirement"}})
        guarded, violations, replaced = guard(
            "The standard does not define the word number for Coarse Time.",
            ledger)

        self.assertTrue(replaced)
        self.assertTrue(violations)
        self.assertIn("Coarse Time — word 1", guarded)

    def test_the_recovery_result_is_not_mutated_by_being_read(self):
        payload = dict(EMPTY)
        before = repr(payload)
        is_empty_structure(payload)

        self.assertEqual(repr(payload), before)


if __name__ == "__main__":
    unittest.main()
