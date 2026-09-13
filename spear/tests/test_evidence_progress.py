"""Whether a round of navigation added anything, and when to stop asking.

The policy this pins has to be hard to trigger. A session that is still
finding normative material must never be cut off, so any new source, unit,
fact, resolved label or citation puts the counter back to zero, and one
repeated call is not a stall. The other half is the ending: a loop that stops
without an answer must still say what it read and what stayed open, and must
say it from the evidence rather than from a guess.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evidence_progress
from evidence_guard import EvidenceLedger
from evidence_progress import (
    BUDGET, EXHAUSTED, NO_PROGRESS_LIMIT, ProgressTracker, bounded_absence,
    evidence_signature, should_terminate,
)

SEARCH = {"results": [{"source_id": "std-" + "a" * 32, "section": "9.4",
                       "page": 153, "snippet": "Elevation Angle"}]}
FETCH = {"unit": {"source_id": "std-" + "b" * 32, "section": "7.2.2",
                  "page": 85, "content_type": "requirement",
                  "text": "Rule 7.2.2-1: The Coarse Time shall be carried in "
                          "word 1 of the Epoch Field."},
         "citation": {"source_id": "std-" + "b" * 32}}
STRUCTURE = {"definition_id": "bfd-89d36dde83caaff6",
             "citation_source_ids": ["std-" + "c" * 32],
             "fields": [{"display_label": "Stage 1 Gain", "msb": 15, "lsb": 0,
                         "width": 16, "word_index": 1}],
             "unresolved": [{"label": "Reserved"}],
             "words": [{"word_index": 1, "word_width": 32}]}
REFUSAL = {"error": "INVALID_DEFINITION_ID", "detail": "malformed"}


def result(name, payload):
    return (name, json.dumps(payload, sort_keys=True), payload)


class Signature(unittest.TestCase):
    """What counts as evidence, and what is only presentation."""

    def test_a_search_signs_its_sources(self):
        found = evidence_signature(*result("standard.search", SEARCH))

        self.assertIn("source:std-" + "a" * 32, found)

    def test_a_fetch_signs_its_unit_and_the_facts_it_states(self):
        found = evidence_signature(*result("standard.fetch", FETCH))

        self.assertIn("unit:std-" + "b" * 32, found)
        self.assertTrue(any(value.startswith("fact:FIELD_WORD_INDEX")
                            for value in found))

    def test_a_structure_signs_what_it_settles_and_leaves_open(self):
        ledger = EvidenceLedger()
        found = evidence_signature(*result("standard.get_structure", STRUCTURE),
                                   ledger=ledger)

        self.assertIn("established:stage1gain", found)
        self.assertIn("unresolved:reserved", found)

    def test_a_refusal_establishes_nothing(self):
        self.assertEqual(
            evidence_signature(*result("standard.get_structure", REFUSAL)),
            set())

    def test_reordering_a_payload_signs_identically(self):
        shuffled = {"results": list(SEARCH["results"]), "extra": "wrapper"}
        self.assertEqual(
            evidence_signature(*result("standard.search", SEARCH)),
            evidence_signature(*result("standard.search", shuffled)))

    def test_the_same_unit_reached_two_ways_signs_the_same_source(self):
        by_search = evidence_signature(*result("standard.search", {
            "results": [{"source_id": "std-" + "b" * 32}]}))
        by_fetch = evidence_signature(*result("standard.fetch", FETCH))

        self.assertTrue(by_search & by_fetch)

    def test_the_payload_is_not_mutated(self):
        payload = json.loads(json.dumps(FETCH))
        before = repr(payload)
        evidence_signature("standard.fetch", json.dumps(payload), payload)

        self.assertEqual(repr(payload), before)


class Counting(unittest.TestCase):
    """When the counter climbs, and when it must go back to zero."""

    def test_a_new_source_is_progress(self):
        tracker = ProgressTracker()

        self.assertTrue(tracker.observe_round(1, [result("standard.search",
                                                         SEARCH)]))
        self.assertEqual(tracker.consecutive_no_progress, 0)

    def test_the_same_result_twice_is_not(self):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.search", SEARCH)])

        self.assertFalse(tracker.observe_round(2, [result("standard.search",
                                                          SEARCH)]))
        self.assertEqual(tracker.consecutive_no_progress, 1)

    def test_one_repeat_does_not_exhaust(self):
        tracker = ProgressTracker()

        for index in range(2):
            tracker.observe_round(index + 1, [result("standard.search",
                                                     SEARCH)])

        self.assertFalse(tracker.exhausted())

    def test_three_flat_rounds_exhaust(self):
        tracker = ProgressTracker()

        for index in range(1 + NO_PROGRESS_LIMIT):
            tracker.observe_round(index + 1, [result("standard.search",
                                                     SEARCH)])

        self.assertTrue(tracker.exhausted())

    def test_late_progress_resets_the_counter(self):
        tracker = ProgressTracker()

        for index in range(3):
            tracker.observe_round(index + 1, [result("standard.search",
                                                     SEARCH)])

        self.assertEqual(tracker.consecutive_no_progress, 2)
        tracker.observe_round(4, [result("standard.fetch", FETCH)])

        self.assertEqual(tracker.consecutive_no_progress, 0)
        self.assertFalse(tracker.exhausted())

    def test_a_reset_survives_further_repetition(self):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.search", SEARCH)])
        tracker.observe_round(2, [result("standard.search", SEARCH)])
        tracker.observe_round(3, [result("standard.fetch", FETCH)])
        tracker.observe_round(4, [result("standard.fetch", FETCH)])

        self.assertEqual(tracker.consecutive_no_progress, 1)
        self.assertFalse(tracker.exhausted())

    def test_a_round_with_no_evidence_tool_is_not_counted(self):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.search", SEARCH)])
        tracker.observe_round(2, [("some.other.tool", "{}", {})])

        self.assertEqual(tracker.consecutive_no_progress, 0)
        self.assertEqual(tracker.evidence_rounds, 1)

    def test_the_round_table_records_every_evidence_round(self):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.search", SEARCH)])
        tracker.observe_round(2, [result("standard.search", SEARCH)])

        self.assertEqual([row["progress"] for row in tracker.by_round],
                         [True, False])
        self.assertEqual(tracker.by_round[1]["consecutive_no_progress"], 1)


class Trigger(unittest.TestCase):
    """The policy's own preconditions."""

    def flat(self):
        tracker = ProgressTracker()

        for index in range(1 + NO_PROGRESS_LIMIT):
            tracker.observe_round(index + 1, [result("standard.search",
                                                     SEARCH)])

        return tracker

    def test_it_fires_on_a_stalled_unanswered_turn(self):
        self.assertTrue(should_terminate(
            self.flat(), calls=[{"tool": "standard.search"}]))

    def test_it_does_not_fire_once_the_model_has_answered(self):
        self.assertFalse(should_terminate(
            self.flat(), answer="Bits 15..0.",
            calls=[{"tool": "standard.search"}]))

    def test_it_does_not_fire_without_a_real_tool_call(self):
        self.assertFalse(should_terminate(self.flat(), calls=[]))

    def test_it_does_not_fire_on_an_unbound_turn(self):
        self.assertFalse(should_terminate(
            self.flat(), bound=False, calls=[{"tool": "standard.search"}]))

    def test_it_does_not_fire_while_progress_continues(self):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.search", SEARCH)])
        tracker.observe_round(2, [result("standard.fetch", FETCH)])

        self.assertFalse(should_terminate(
            tracker, calls=[{"tool": "standard.search"}]))


class Rendering(unittest.TestCase):
    """The ending, when the model produced none."""

    def ledger(self):
        return EvidenceLedger().observe(STRUCTURE)

    def rendered(self, reason=EXHAUSTED):
        tracker = ProgressTracker()
        tracker.observe_round(1, [result("standard.get_structure", STRUCTURE)])

        return bounded_absence(
            "What are the exact physical bit positions of the Elevation Angle "
            "and Azimuthal Angle fields?", self.ledger(), tracker,
            reason=reason, rounds=14, sources=["std-" + "c" * 32])

    def test_it_is_never_empty(self):
        self.assertTrue(self.rendered().strip())

    def test_it_states_what_the_evidence_established(self):
        self.assertIn("Stage 1 Gain — bits 15..0 of word 1", self.rendered())

    def test_it_states_what_stayed_open(self):
        self.assertIn("Reserved", self.rendered())
        self.assertIn("no established position", self.rendered())

    def test_it_says_the_evidence_does_not_settle_the_question(self):
        self.assertIn("does not establish the requested bit positions",
                      self.rendered())

    def test_it_disclaims_inference(self):
        self.assertIn("No position, range or width is inferred", self.rendered())

    def test_it_keeps_its_citations(self):
        self.assertIn("std-" + "c" * 32, self.rendered())

    def test_it_is_not_merely_not_found(self):
        self.assertGreater(len(self.rendered()), 200)

    def test_it_invents_no_range_for_the_requested_fields(self):
        rendered = self.rendered()

        self.assertNotIn("Elevation Angle — bits", rendered)
        self.assertNotIn("Azimuthal Angle — bits", rendered)

    def test_it_distinguishes_the_two_ways_of_stopping(self):
        self.assertIn("returned no evidence that had not already been read",
                      self.rendered(EXHAUSTED))
        self.assertIn("after 14 rounds of retrieval", self.rendered(BUDGET))


if __name__ == "__main__":
    unittest.main()


class ProseQuestionEnding(unittest.TestCase):
    """A bounded ending answers the question that was asked.

    "How should the ACK be managed" ended with "does not establish the
    requested bit positions", which reads as an answer to some other question.
    """

    def test_a_question_that_asked_for_no_positions_is_not_told_about_them(self):
        from evidence_guard import EvidenceLedger
        from evidence_progress import ProgressTracker, bounded_absence

        text = bounded_absence(
            "How should the ACK be managed when using VITA49.2 Commands packets?",
            EvidenceLedger(), ProgressTracker(), reason=BUDGET, rounds=14)

        self.assertNotIn("bit positions", text)
        self.assertIn("does not settle the question as asked", text)

    def test_a_question_about_a_field_still_is(self):
        from evidence_guard import EvidenceLedger
        from evidence_progress import ProgressTracker, bounded_absence

        text = bounded_absence(
            "Give the bit range of the Coarse Time field.",
            EvidenceLedger(), ProgressTracker(), reason=BUDGET, rounds=14)

        self.assertIn("bit positions", text)
