"""Investigation has to converge, not merely stop short of a stall.

The gate that came before this made a turn safe: nothing was written before it
had read both sides and planned. It did not make a turn cheap. Measured over
four runs of one two-turn workflow: 585 tool calls, of which 302 were distinct
code reads and only 17 were exact repeats — so the waste was never in calling
the same thing twice. It was in reading the same LINES through different
windows. Between 54% and 62% of every ranged read landed entirely on lines the
turn had already been shown.

That is what these tests hold: a window already in evidence is answered from
evidence, a write puts it back in play, and the interventions that notice a
turn going nowhere fire early and few rather than late and often.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import progress_monitor
import rag_chat
import tool_router
import work_phase

SOURCE = "src/link/handshake.c"


def context():
    return SimpleNamespace(cache={})


def wrote(ctx, times=1):
    """Whatever the router would have done to the generation after a write."""
    ledger = ctx.cache.setdefault(tool_router._REPEAT_KEY, {"generation": 0})
    ledger["generation"] += times


class LinesAlreadyShownAreNotShownAgain(unittest.TestCase):
    def test_the_first_read_of_a_window_runs(self):
        self.assertEqual(
            rag_chat._already_in_evidence(context(), f"sed -n '1,200p' {SOURCE}"),
            "")

    def test_the_same_window_again_is_answered_from_evidence(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")
        note = rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        self.assertIn("ALREADY IN EVIDENCE", note)
        self.assertIn(SOURCE, note)

    def test_a_narrower_window_inside_one_already_read_is_too(self):
        """This is the case the router's argument cache cannot see."""
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        self.assertIn("ALREADY IN EVIDENCE",
                      rag_chat._already_in_evidence(ctx, f"sed -n '50,150p' {SOURCE}"))

    def test_a_window_that_extends_past_it_runs(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        self.assertEqual(
            rag_chat._already_in_evidence(ctx, f"sed -n '200,400p' {SOURCE}"), "")

    def test_another_file_runs(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        self.assertEqual(
            rag_chat._already_in_evidence(ctx, "sed -n '1,200p' src/link/frame.c"),
            "")

    def test_a_grep_is_never_suppressed(self):
        """Searching the same file for a different pattern is a new question."""
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        for command in (f"grep -n answer {SOURCE}", f"grep -rn answer src/",
                        f"cat {SOURCE}"):
            with self.subTest(command=command):
                self.assertEqual(rag_chat._already_in_evidence(ctx, command), "")

    def test_a_write_puts_every_region_back_in_play(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")
        wrote(ctx)

        self.assertEqual(
            rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}"), "")

    def test_and_the_region_settles_again_afterwards(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")
        wrote(ctx)
        rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}")

        self.assertIn("ALREADY IN EVIDENCE",
                      rag_chat._already_in_evidence(ctx, f"sed -n '1,200p' {SOURCE}"))

    def test_the_note_says_what_to_do_instead(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"head -n 100 {SOURCE}")
        note = rag_chat._already_in_evidence(ctx, f"head -n 100 {SOURCE}")

        self.assertIn("Widen the range", note)

    def test_head_and_sed_over_the_same_lines_are_one_region(self):
        ctx = context()
        rag_chat._already_in_evidence(ctx, f"sed -n '1,100p' {SOURCE}")

        self.assertIn("ALREADY IN EVIDENCE",
                      rag_chat._already_in_evidence(ctx, f"head -n 100 {SOURCE}"))


class TheEvidenceModelBacksIt(unittest.TestCase):
    """The blocks come from the existing progress model, not a second one."""

    def test_a_ranged_read_yields_blocks(self):
        self.assertTrue(progress_monitor.read_evidence(f"sed -n '1,200p' {SOURCE}"))

    def test_an_unranged_command_yields_none(self):
        self.assertEqual(progress_monitor.read_evidence(f"grep -n x {SOURCE}"), ())


class SynthesisIsEarlyAndBounded(unittest.TestCase):
    """PHASE 13: one early intervention beats three late ones."""

    def ledger(self):
        found = work_phase.WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        found.observe_call(implementation_paths={SOURCE})

        return found

    def test_a_short_barren_sequence_is_enough(self):
        self.assertLessEqual(work_phase.REPETITION_LIMIT, 2)

    def test_the_interventions_are_few(self):
        self.assertLessEqual(work_phase.MAX_SYNTHESES, 2)

    def test_it_fires_after_the_threshold_and_not_before(self):
        found = self.ledger()

        for _ in range(work_phase.REPETITION_LIMIT - 1):
            found.observe_call()
            self.assertFalse(found.repeating)

        found.observe_call()

        self.assertTrue(found.repeating)

    def test_it_stops_firing_once_spent(self):
        found = self.ledger()
        fired = 0

        for _ in range(work_phase.REPETITION_LIMIT * 20):
            found.observe_call()

            if found.repeating:
                found.force_synthesis()
                fired += 1

        self.assertEqual(fired, work_phase.MAX_SYNTHESES)

    def test_real_progress_resets_it(self):
        found = self.ledger()

        for index in range(work_phase.REPETITION_LIMIT * 4):
            found.observe_call(implementation_paths={f"src/f{index}.c"})
            self.assertFalse(found.repeating)


class ASettledPlanItemStopsTheReading(unittest.TestCase):
    """PHASE 10: an item with both halves and a validation is done being
    investigated; re-reading its area is barren by construction."""

    def planned(self):
        found = work_phase.WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        found.observe_call(implementation_paths={SOURCE})
        found.record_plan([{
            "requirement": "r", "requirement_evidence": "§4.2.1",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c in src/link/handshake.c",
            "validation": "tests/t.c: when a second request arrives on the same link, expect a second answer carrying the same identifier"}])

        return found

    def test_rereading_the_settled_file_adds_nothing(self):
        found = self.planned()

        for _ in range(work_phase.REPETITION_LIMIT):
            found.observe_call(implementation_paths={SOURCE})

        self.assertTrue(found.repeating)

    def test_a_new_file_is_still_progress(self):
        found = self.planned()
        found.observe_call(implementation_paths={"src/link/frame.c"})

        self.assertFalse(found.repeating)


class AnUnrelatedTreeCannotSatisfyTheGate(unittest.TestCase):
    """PHASE 12: the session-tree boundary, still holding."""

    def test_reading_another_corpus_leaves_the_gate_shut(self):
        found = work_phase.WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True,
                     root="/work/project")
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        found.observe_call(
            implementation_paths={"/elsewhere/other/link/handshake.c"})

        self.assertFalse(found.may_write().allowed)
        self.assertEqual(found.may_write().reason, work_phase.NO_IMPLEMENTATION)


class ARepeatedBuildFailureBecomesAReplan(unittest.TestCase):
    """PHASE 18: the third patch against the same failure is not a fix."""

    def ledger(self):
        found = work_phase.WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        found.observe_call(implementation_paths={SOURCE})
        found.record_plan([{
            "requirement": "r", "requirement_evidence": "§4.2.1",
            "current_behaviour": "b", "implementation_evidence": SOURCE,
            "gap": "g", "correction": "c in src/link/handshake.c",
            "validation": "tests/t.c: when a second request arrives on the same link, expect a second answer carrying the same identifier"}])

        return found

    def test_two_identical_failures_are_enough(self):
        found = self.ledger()
        found.note_validation("build", "failed")

        self.assertFalse(found.validation_exhausted("build"))

        found.note_validation("build", "failed")

        self.assertTrue(found.validation_exhausted("build"))

    def test_a_replan_reopens_the_question(self):
        found = self.ledger()
        found.note_validation("build", "failed")
        found.note_validation("build", "failed")
        found.invalidate(reason="the build keeps failing the same way")

        self.assertEqual(found.phase, work_phase.Phase.PLAN)


if __name__ == "__main__":
    unittest.main()
