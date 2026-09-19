"""A change is not validated by a suite that never enters it.

The run that made this necessary changed acknowledgement emission, added no
test of its own, ran 198 existing checks, and reported that all of them
passed. That was true and said nothing whatever about the new behaviour. It
then closed with its own summary table declaring every clause satisfied,
printed directly above a ledger that called one of them out of scope.

So: a requirement whose behaviour changed needs validation that reaches the
changed path, and the ledger — not the prose — has the last word on whether
the turn is finished. Synthetic throughout, for the usual reason.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_runtime
from requirement_set import Disposition, Requirement, RequirementSet
from work_phase import WorkPhaseLedger

SOURCE = "src/link/handshake.c"
HANDLE = "std-0a1b2c3d4e5f"


def ledger():
    found = WorkPhaseLedger()
    found.engage(authority_bound=True, write_requested=True,
                 requirements=RequirementSet([Requirement(
                     key="Rule 4.2.1-1", source_id=HANDLE,
                     statement="the responder answers every accepted request")]))
    found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
    found.observe_call(implementation_paths={SOURCE})

    return found


def item(**kw):
    base = {
        "requirement": "the responder answers every accepted request",
        "requirement_evidence": "Rule 4.2.1-1",
        "current_behaviour": "it answers only the first",
        "implementation_evidence": SOURCE,
        "gap": "later requests get no answer",
        "correction": f"answer each request in handshake_accept() in {SOURCE}",
        "validation": "tests/test_handshake.c exercises two requests",
    }
    base.update(kw)

    return base


class AChangedBehaviourNeedsATestThatReachesIt(unittest.TestCase):
    """PHASES 9 to 11 — 198 green checks say nothing about a branch none of
    them enters."""

    def implemented(self, validation, wrote=(SOURCE,)):
        found = ledger()
        found.record_plan([item(validation=validation)])
        found.note_write(list(wrote))
        found.note_validation("ctest --test-dir build", "passed")

        return found

    def test_leaning_on_the_existing_suite_is_not_coverage(self):
        found = self.implemented("the existing project test suite")

        self.assertEqual([r.key for r in found.unvalidated_requirements()],
                         ["Rule 4.2.1-1"])

    def test_a_test_the_turn_wrote_is(self):
        found = self.implemented("tests/test_handshake.c covers both branches",
                                 wrote=(SOURCE, "tests/test_handshake.c"))

        self.assertEqual(found.unvalidated_requirements(), ())

    def test_saying_plainly_that_no_test_can_reach_it_is_accepted(self):
        found = self.implemented(
            "no automated test can reach this without real hardware")

        self.assertEqual(found.unvalidated_requirements(), ())

    def test_a_requirement_that_changed_nothing_needs_no_test(self):
        found = ledger()
        found.record_plan([item(disposition="satisfied_already",
                                correction="no change needed",
                                validation="the existing suite covers it")])

        self.assertEqual(found.unvalidated_requirements(), ())

    def test_the_closing_record_names_them(self):
        found = self.implemented("the existing project test suite")
        note = agent_runtime.requirement_matrix_note(found)

        self.assertIn("NOT VALIDATED", note)
        self.assertIn("proves nothing about them", note)


class WhatWasLeftOutOfScopeIsInTheRecord(unittest.TestCase):
    """The matrix carries the requirements that did not make the cut, so the
    narrowing is a visible decision rather than a silence."""

    def published(self):
        import requirement_set

        def record(key, section, text, force=3):
            return type("R", (), {"key": key, "section": section, "text": text,
                                  "source_id": "std-" + section,
                                  "effective_force": force})()

        policy = type("P", (), {"claim_evidence": type("E", (), {
            "provisions": type("L", (), {"records": {
                "Rule 4.2.1-1": record("Rule 4.2.1-1", "4.2.1", "a shall b"),
                "Rule 4.2.2-1": record("Rule 4.2.2-1", "4.2.2", "c shall d"),
            }})()})()})()

        return requirement_set.publish(policy, "per Rule 4.2.1-1, ...")

    def test_the_excluded_one_is_named_with_its_reason(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True,
                     requirements=self.published())
        note = agent_runtime.requirement_matrix_note(found)

        self.assertIn("NOT in scope for this turn", note)
        self.assertIn("Rule 4.2.2-1", note)


class TheLedgerHasTheLastWord(unittest.TestCase):
    """PHASES 14 to 16 — one matrix, and a status nobody can talk past."""

    def closed(self):
        found = ledger()
        found.record_plan([item(validation="tests/test_handshake.c")])
        found.note_write([SOURCE, "tests/test_handshake.c"])
        found.note_validation("ctest", "passed")

        return found

    def test_everything_closed_and_validated_reads_as_such(self):
        self.assertEqual(agent_runtime.requirement_status(self.closed()),
                         "reviewed requirements satisfied")

    def test_an_exclusion_is_named_in_the_status(self):
        found = ledger()
        found.record_plan([item(disposition="explicitly_out_of_scope",
                                correction="this build does not support it")])

        self.assertEqual(agent_runtime.requirement_status(found),
                         "supported profile validated, with exclusions")

    def test_unfinished_work_reads_as_unfinished(self):
        found = ledger()
        found.record_plan([item(disposition="undetermined",
                                gap="could not tell",
                                correction="needs more reading")])

        self.assertEqual(agent_runtime.requirement_status(found),
                         "implementation work incomplete")

    def test_a_claim_of_completion_over_an_open_ledger_is_contradicted(self):
        found = ledger()
        found.record_plan([item(disposition="undetermined",
                                gap="could not tell",
                                correction="needs more reading")])
        note = agent_runtime.requirement_matrix_note(
            found, "The implementation is complete and satisfies all the "
                   "requirements. All 198 tests pass.")

        self.assertIn("claims this work is complete", note)
        self.assertIn("take the matrix, not the claim", note)

    def test_a_claim_over_a_closed_ledger_is_left_alone(self):
        note = agent_runtime.requirement_matrix_note(
            self.closed(), "The implementation is complete.")

        self.assertNotIn("take the matrix, not the claim", note)

    def test_the_matrix_says_it_is_the_record(self):
        note = agent_runtime.requirement_matrix_note(self.closed())

        self.assertIn("This is the record", note)

    def test_the_status_line_comes_from_the_ledger(self):
        note = agent_runtime.requirement_matrix_note(self.closed())

        self.assertIn("STATUS (from the ledger): reviewed requirements "
                      "satisfied", note)

    def test_no_contract_means_no_matrix(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)

        self.assertEqual(agent_runtime.requirement_matrix_note(found), "")


if __name__ == "__main__":
    unittest.main()
