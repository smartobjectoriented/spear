"""The proof is designed before the code is written, or the code waits.

Four measured runs of one workflow, every gate green, and the same shape each
time: the first source edit at call 37, 73, 101 and 33, and the first test edit
at 125, 102, never and 51. Test authoring was always an afterthought and twice
it never arrived — once leaving the tree not building, because a signature
change never reached the callers in the test file nobody had opened.

The field that was supposed to stop this was mandatory and non-empty, and that
was all it was. "The fix will be validated by running the existing unit
tests", "the test suite should be extended", "run ctest" — every one of them
passed. So what is checked now is whether the statement is a DESIGN: what makes
the behaviour happen, and what result shows it worked.

Synthetic throughout: a fictional document about a fictional handshake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import work_phase
from requirement_set import Disposition, Requirement, RequirementSet
from work_phase import (EXISTING_PROVEN, FOCUSED_PLANNED, NOT_PRACTICAL,
                        WorkPhaseLedger, validation_design)

SOURCE = "src/link/handshake.c"
TEST = "tests/test_handshake.c"
HANDLE = "std-0a1b2c3d4e5f"

#: A design: a scenario and an outcome.
DESIGN = (f"{TEST}: when a second request arrives on the same link, expect a "
          f"second answer carrying the same identifier")


def ledger(trigger="", read=(SOURCE,)):
    found = WorkPhaseLedger()
    found.engage(authority_bound=True, write_requested=True,
                 requirements=RequirementSet([Requirement(
                     key="Rule 4.2.1-1", source_id=HANDLE, trigger=trigger,
                     statement="the responder answers every accepted request")]))
    found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
    found.observe_call(implementation_paths=set(read))

    return found


def item(**kw):
    base = {
        "requirement": "the responder answers every accepted request",
        "requirement_evidence": "Rule 4.2.1-1",
        "current_behaviour": "it answers only the first",
        "implementation_evidence": SOURCE,
        "gap": "later requests get no answer",
        "correction": f"answer each request in handshake_accept() in {SOURCE}",
        "validation": DESIGN,
    }
    base.update(kw)

    return base


class RunTheSuiteIsNotAPlan(unittest.TestCase):
    """CASE 1 — every one of these was accepted before, verbatim, in a run."""

    def test_the_forms_a_real_run_actually_wrote_are_refused(self):
        for statement in (
                "The fix will be validated by running the existing unit tests "
                "in tests/test_command.c",
                "run ctest",
                "the test suite should be extended",
                "Run tests/test_command.c",
                "verify behaviour",
                "the existing project test suite"):
            with self.subTest(statement=statement):
                self.assertEqual(validation_design(statement), "")

    def test_the_write_is_refused_with_a_useful_reason(self):
        found = ledger()
        outcome = found.record_plan([item(validation="run the test suite")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("no validation design", outcome.report())
        self.assertIn("names no scenario and no expected result",
                      outcome.report())

    def test_and_the_gate_stays_shut(self):
        found = ledger()
        found.record_plan([item(validation="run the test suite")])

        self.assertFalse(found.may_write().allowed)


class AScenarioAndAnOutcomeAreAPlan(unittest.TestCase):
    """CASE 2."""

    def test_a_concrete_design_is_accepted(self):
        found = ledger()
        outcome = found.record_plan([item()])

        self.assertTrue(outcome.any_accepted, outcome.report())
        self.assertTrue(found.may_write().allowed)

    def test_a_scenario_without_an_outcome_is_not(self):
        self.assertEqual(
            validation_design("when a second request arrives on the link"), "")

    def test_an_outcome_without_a_scenario_is_not(self):
        self.assertEqual(
            validation_design("expect a second answer"), "")

    def test_the_item_remembers_which_kind_it_designed(self):
        found = ledger()
        found.record_plan([item()])

        self.assertEqual(found.items[0].testability, FOCUSED_PLANNED)


class LeaningOnAnExistingTest(unittest.TestCase):
    """CASES 3 and 4 — naming a file is not evidence it reaches the branch."""

    def test_a_file_the_turn_never_opened_is_not_proof(self):
        found = ledger(read=(SOURCE,))
        found.record_plan([item(
            validation=f"{TEST} already asserts that a repeat request is "
                       f"answered, so it covers this")])

        self.assertEqual(found.items[0].testability, FOCUSED_PLANNED)

    def test_a_file_the_turn_read_may_be_reused(self):
        found = ledger(read=(SOURCE, TEST))
        found.record_plan([item(
            validation=f"{TEST} already asserts that when a repeat request "
                       f"arrives the answer carries the same identifier")])

        self.assertEqual(found.items[0].testability, EXISTING_PROVEN)

    def test_a_reused_test_is_credited_once_a_run_passes(self):
        found = ledger(read=(SOURCE, TEST))
        found.record_plan([item(
            validation=f"{TEST} already asserts that when a repeat request "
                       f"arrives the answer carries the same identifier")])
        found.note_write([SOURCE])
        found.note_validation("ctest --test-dir build", "passed")

        self.assertEqual(found.unvalidated_requirements(), ())


class TheConditionalBranch(unittest.TestCase):
    """CASE 5 — the positive case can pass while the condition is ignored."""

    def test_the_positive_case_alone_is_refused_once(self):
        found = ledger(trigger="when more than one response is requested")
        outcome = found.record_plan([item(
            validation=f"{TEST}: when two are requested, expect two answers")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("the other case too", outcome.report())

    def test_asking_twice_records_the_gap_rather_than_walling_it_off(self):
        """A check that cannot be satisfied stops being a gate."""
        found = ledger(trigger="when more than one response is requested")
        design = f"{TEST}: when two are requested, expect two answers"
        found.record_plan([item(validation=design)])
        outcome = found.record_plan([item(validation=design)])

        self.assertTrue(outcome.any_accepted, outcome.report())
        self.assertIn("Rule 4.2.1-1", found.branch_gaps())
        self.assertIn("only the case where the condition holds",
                      found.branch_gaps()["Rule 4.2.1-1"])

    def test_covering_both_is_accepted(self):
        found = ledger(trigger="when more than one response is requested")
        outcome = found.record_plan([item(
            validation=f"{TEST}: when two are requested expect two answers, "
                       f"and when only one is requested expect exactly one")])

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_an_unconditional_requirement_needs_no_second_branch(self):
        found = ledger(trigger="")
        outcome = found.record_plan([item()])

        self.assertTrue(outcome.any_accepted, outcome.report())


class NoAutomatedTestIsAnAnswerWithAReason(unittest.TestCase):
    def test_a_bare_refusal_is_not_a_design(self):
        self.assertEqual(validation_design("it cannot be tested"), "")

    def test_a_reason_and_an_alternative_is(self):
        self.assertEqual(
            validation_design("no automated test can reach this because it "
                              "needs the real device; when the board is on "
                              "the bench it is checked against a capture"),
            NOT_PRACTICAL)

    def test_it_is_credited_without_a_test_file(self):
        found = ledger()
        found.record_plan([item(
            validation="no automated test can reach this because it needs "
                       "the real device; when the board is on the bench it "
                       "is checked against a capture")])
        found.note_write([SOURCE])
        found.note_validation("ctest", "passed")

        self.assertEqual(found.unvalidated_requirements(), ())


class TheCodeWaitsForItsProof(unittest.TestCase):
    """CASES 6 and 8 — one work item, two halves."""

    def planned(self):
        found = ledger()
        found.record_plan([item()])

        return found

    def test_a_source_edit_leaves_the_requirement_awaiting_validation(self):
        found = self.planned()
        found.note_write([SOURCE])

        self.assertEqual(
            found.requirements.get("Rule 4.2.1-1").disposition,
            str(Disposition.CODE_CHANGED_AWAITING_VALIDATION))
        self.assertEqual([r.key for r in found.awaiting_validation()],
                         ["Rule 4.2.1-1"])

    def test_a_passing_run_without_the_planned_test_does_not_settle_it(self):
        found = self.planned()
        found.note_write([SOURCE])
        found.note_validation("ctest", "passed")

        self.assertEqual([r.key for r in found.unvalidated_requirements()],
                         ["Rule 4.2.1-1"])

    def test_the_test_plus_a_passing_run_does(self):
        found = self.planned()
        found.note_write([SOURCE, TEST])
        found.note_validation("ctest", "passed")

        self.assertEqual(found.unvalidated_requirements(), ())
        self.assertEqual(found.requirements.get("Rule 4.2.1-1").disposition,
                         str(Disposition.CHANGE_IMPLEMENTED))

    def test_the_contract_is_not_closed_while_proof_is_owed(self):
        found = self.planned()
        found.note_write([SOURCE])

        self.assertFalse(found.contract_closed())


class ABrokenBuildProvesNothing(unittest.TestCase):
    """CASE 7 — and it sends the turn back to the item that broke it."""

    def broken(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE])
        found.note_validation("cmake --build build", "failed")

        return found

    def test_the_affected_work_item_is_named(self):
        self.assertEqual([r.key for r in self.broken().build_broken_for()],
                         ["Rule 4.2.1-1"])

    def test_nothing_is_credited(self):
        found = self.broken()

        self.assertEqual([r.key for r in found.unvalidated_requirements()],
                         ["Rule 4.2.1-1"])

    def test_the_contract_cannot_close(self):
        self.assertFalse(self.broken().contract_closed())

    def test_a_repaired_build_lets_it_close(self):
        found = self.broken()
        found.note_write([SOURCE, TEST])
        found.note_validation("cmake --build build", "passed")
        found.note_validation("ctest", "passed")

        self.assertTrue(found.contract_closed())


class RevisionsNeedSomethingNew(unittest.TestCase):
    """CASES 9 and 10."""

    def test_a_restatement_with_nothing_learned_is_refused(self):
        found = ledger()
        found.record_plan([item()])
        outcome = found.record_plan([item(disposition="satisfied_already",
                                          correction="actually it is fine")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("no new evidence", outcome.report())

    def test_a_revision_after_reading_something_is_accepted(self):
        found = ledger()
        found.record_plan([item()])
        found.observe_call(implementation_paths={"src/link/frame.c"})
        outcome = found.record_plan([item(
            correction=f"answer from handshake_worker() in {SOURCE} instead")])

        self.assertTrue(outcome.any_accepted, outcome.report())
        self.assertTrue(found.revised[-1]["new_evidence"])

    def test_a_supersede_that_names_its_evidence_is_accepted(self):
        found = ledger()
        found.record_plan([item()])
        outcome = found.record_plan(
            [item(correction=f"answer from handshake_worker() in {SOURCE}")],
            supersedes="the responder answers every accepted request",
            reason="the accept path runs on another thread",
            new_evidence="frame.c:88 dispatches to a worker")

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_one_record_survives_all_of_it(self):
        found = ledger()
        found.record_plan([item()])
        found.record_plan([item(disposition="satisfied_already")])
        found.observe_call(implementation_paths={"src/link/frame.c"})
        found.record_plan([item(correction=f"answer in {SOURCE} worker")])

        self.assertEqual(len(found.items), 1)


class AValidationAlreadyTakenIsNotRetaken(unittest.TestCase):
    """CASE 12 — the tree has not changed and neither has the answer."""

    def test_the_same_command_on_the_same_state_is_known(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE, TEST])
        found.note_validation("ctest", "passed")

        self.assertTrue(found.already_validated("ctest"))

    def test_a_write_makes_it_worth_running_again(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE, TEST])
        found.note_validation("ctest", "passed")
        found.note_write([SOURCE])

        self.assertFalse(found.already_validated("ctest"))

    def test_a_failure_is_never_reused(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE])
        found.note_validation("ctest", "failed")

        self.assertFalse(found.already_validated("ctest"))

    def test_a_different_command_is_not_covered(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE, TEST])
        found.note_validation("ctest", "passed")

        self.assertFalse(found.already_validated("cmake --build build"))


class LandingAfterClosure(unittest.TestCase):
    """CASE 11."""

    def closed(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE, TEST])
        found.note_validation("ctest", "passed")

        return found

    def test_a_closed_and_validated_contract_says_so(self):
        self.assertTrue(self.closed().contract_closed())

    def test_an_unwritten_test_keeps_it_open(self):
        found = ledger()
        found.record_plan([item()])
        found.note_write([SOURCE])
        found.note_validation("ctest", "passed")

        self.assertFalse(found.contract_closed())

    def test_an_ungoverned_turn_is_never_closed(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=False, write_requested=True)

        self.assertFalse(found.contract_closed())


if __name__ == "__main__":
    unittest.main()
