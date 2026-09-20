"""Plan locally, edit and validate locally, close globally.

The gate before this one was global: every carried requirement had to be
dispositioned before any source write. Measured on four runs of one workflow,
that meant three of four requirements blocking an edit they had nothing to do
with — and one run spent its entire turn dispositioning the contract, looping
on a single requirement, and never wrote a line.

What replaces it is not a weaker gate. It is a gate asked at the right scope:
which planned change authorises THIS edit, and has that change answered for
its own requirements. Everything the rest of the contract owes is still owed,
and is still asked for before the turn may call itself finished.

Synthetic throughout: a fictional document about a fictional handshake.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_runtime
import work_phase
from requirement_set import Disposition, Requirement, RequirementSet
from work_phase import WorkPhaseLedger

ACCEPT = "src/link/handshake.c"
QUERY = "src/link/query.c"
FRAME = "src/link/frame.c"
TEST = "tests/test_handshake.c"

#: A validation design: a scenario, an outcome, and the other branch.
DESIGN = (f"{TEST}: when a second request arrives expect a second answer, "
          f"and when none arrives expect nothing sent")

#: Three requirements. The first two govern the accept path; the third is
#: about framing and shares nothing with them.
R1 = Requirement(key="Rule 4.2.1-1", source_id="std-aaaaaaaaaaaa",
                 statement="the responder answers every request it accepts")
R2 = Requirement(key="Rule 4.2.2-1", source_id="std-bbbbbbbbbbbb",
                 statement="the answer from handshake_accept carries the id")
R3 = Requirement(key="Rule 4.9-1", source_id="std-cccccccccccc",
                 statement="a frame longer than the window is rejected")


def ledger(*requirements, read=(ACCEPT, QUERY, FRAME)):
    found = WorkPhaseLedger()
    found.engage(authority_bound=True, write_requested=True,
                 requirements=RequirementSet(list(requirements or (R1, R2, R3))))
    found.observe_call(authority_keys={"4.2.1", "4.2.2", "4.9"},
                       authority_units=3)
    found.observe_call(implementation_paths=set(read))

    return found


def item(requirement=R1, path=ACCEPT, correction=None, **kw):
    base = {
        "requirement": requirement.statement,
        "requirement_evidence": requirement.key,
        "current_behaviour": "it answers only the first",
        "implementation_evidence": path,
        "gap": "later requests get no answer",
        "correction": correction or f"answer each request in accept() in {path}",
        "validation": DESIGN,
    }
    base.update(kw)

    return base


class OneChangeIsOneWorkItem(unittest.TestCase):
    """PHASE 2 — grouped by the code it touches, never by document section."""

    def test_a_plan_opens_a_work_item(self):
        found = ledger()
        found.record_plan([item()])

        self.assertEqual(len(found.work_items), 1)
        self.assertIn("handshake.c", found.work_items[0].paths)

    def test_two_changes_to_the_same_file_are_one_item(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(R2)])

        self.assertEqual(len(found.work_items), 1)
        self.assertEqual(found.work_items[0].requirements,
                         {R1.key, R2.key})

    def test_two_changes_to_different_files_are_two_items(self):
        found = ledger()
        found.record_plan([item(R1, path=ACCEPT)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])

        self.assertEqual(len(found.work_items), 2)

    def test_a_shared_symbol_joins_them(self):
        found = ledger()
        found.record_plan([item(R1, correction=f"change accept() in {ACCEPT}")])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(R2, path=QUERY,
                                correction=f"also from accept() in {QUERY}")])

        self.assertEqual(len(found.work_items), 1)

    def test_the_test_file_does_not_group_unrelated_work(self):
        """Every change is proved in tests/; that makes them no relation."""
        found = ledger()
        found.record_plan([item(R1, path=ACCEPT)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])

        self.assertEqual(len(found.work_items), 2)


class AnUnrelatedRequirementDoesNotBlockAnEdit(unittest.TestCase):
    """CASES 1, 3 and 11 — the whole point."""

    def planned(self):
        found = ledger()
        found.record_plan([item(R1)])

        return found

    def test_the_planned_change_may_be_written(self):
        self.assertTrue(self.planned().may_write(ACCEPT).allowed)

    def test_the_others_are_still_open_in_the_ledger(self):
        found = self.planned()

        self.assertIn(R3.key,
                      [r.key for r in found.uncovered_requirements()])

    def test_and_they_do_not_appear_in_the_refusal(self):
        found = ledger()
        message = found.may_write(ACCEPT).message

        self.assertIn("not for all of them", message)

    def test_productive_work_proceeds_with_one_requirement_planned(self):
        found = ledger(R1, R2, R3)
        found.record_plan([item(R1)])
        found.note_write([ACCEPT, TEST])

        self.assertEqual(found.writes, 1)


class ACoupledRequirementMustJoin(unittest.TestCase):
    """CASE 2 — narrow scoping is not a way out."""

    def test_a_requirement_naming_the_same_symbol_joins(self):
        found = ledger()
        found.record_plan([item(
            R1, correction=f"change handshake_accept() in {ACCEPT}")])
        work = found.work_items[0]

        self.assertIn(R2.key, work.requirements)
        self.assertIn(R2.key, work.coupled)

    def test_and_it_blocks_the_edit_until_answered(self):
        found = ledger()
        found.record_plan([item(
            R1, correction=f"change handshake_accept() in {ACCEPT}")])
        decision = found.may_write(ACCEPT)

        self.assertFalse(decision.allowed)
        self.assertIn(R2.key, decision.message)

    def test_answering_it_opens_the_edit(self):
        found = ledger()
        found.record_plan([item(
            R1, correction=f"change handshake_accept() in {ACCEPT}")])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R2, disposition="satisfied_already",
            correction=f"handshake_accept() already carries it in {ACCEPT}")])

        self.assertTrue(found.may_write(ACCEPT).allowed)

    def test_an_unrelated_requirement_is_not_pulled_in(self):
        found = ledger()
        found.record_plan([item(
            R1, correction=f"change handshake_accept() in {ACCEPT}")])

        self.assertNotIn(R3.key, found.work_items[0].requirements)


class TheEditMustBelongToAPlannedChange(unittest.TestCase):
    """CASES 7 and 8 — PHASE 17."""

    def planned(self):
        found = ledger()
        found.record_plan([item(R1)])

        return found

    def test_a_file_no_plan_names_is_refused(self):
        decision = self.planned().may_write(FRAME)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, work_phase.OUTSIDE_WORK_ITEM)

    def test_the_refusal_says_what_the_planned_work_touches(self):
        self.assertIn("handshake.c", self.planned().may_write(FRAME).message)

    def test_planning_that_file_lets_it_through(self):
        found = self.planned()
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])

        self.assertTrue(found.may_write(FRAME).allowed)

    def test_a_shell_write_asks_the_general_question(self):
        """A command names no single file; any ready change authorises it."""
        self.assertTrue(self.planned().may_write("").allowed)


class ClosedWorkIsNotReopenedForFree(unittest.TestCase):
    """CASES 5 and 6 — PHASE 9."""

    def closed(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.note_write([ACCEPT, TEST])
        found.note_validation("ctest", "passed")
        found.close_work_item(found.work_items[0])

        return found

    def test_restating_the_plan_changes_nothing(self):
        found = self.closed()
        outcome = found.record_plan([item(R1)])

        self.assertFalse(outcome.any_accepted)
        self.assertTrue(found.work_items[0].closed)

    def test_new_evidence_reopens_it_explicitly(self):
        found = self.closed()
        found.invalidate(reason="the accept path runs on another thread",
                         evidence="frame.c:88 dispatches to a worker")

        self.assertFalse(found.work_items[0].closed)
        self.assertEqual(found.work_items[0].reopened, 1)

    def test_and_nothing_may_be_written_until_it_is_replanned(self):
        found = self.closed()
        found.invalidate(reason="wrong lifecycle")

        self.assertFalse(found.may_write(ACCEPT).allowed)

    def test_a_replacement_plan_opens_it_again(self):
        found = self.closed()
        found.invalidate(reason="wrong lifecycle")
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R1, correction=f"answer from accept_worker() in {ACCEPT}")])

        self.assertTrue(found.may_write(ACCEPT).allowed)


class TwoRequirementsOnePathOneTest(unittest.TestCase):
    """CASE 9."""

    def test_one_work_item_may_close_both(self):
        found = ledger(R1, R2)
        found.record_plan([item(R1)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(R2, path=ACCEPT)])

        self.assertEqual(len(found.work_items), 1)

        found.note_write([ACCEPT, TEST])
        found.note_validation("ctest", "passed")

        self.assertEqual(found.requirements.open_items(), ())


class DependenciesAreOrdered(unittest.TestCase):
    """CASE 10 — declared, not inferred."""

    def test_a_dependency_is_recorded_on_the_item(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])
        first, second = found.work_items
        second.depends_on.add(first.key)

        self.assertEqual(second.depends_on, {first.key})

    def test_an_item_with_an_open_dependency_is_reported(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])
        first, second = found.work_items
        second.depends_on.add(first.key)

        self.assertIn(first.key, found.blocked_work_items())

    def test_once_the_dependency_closes_it_is_free(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(
            R3, path=FRAME,
            correction=f"reject the long frame in frame_read() in {FRAME}")])
        first, second = found.work_items
        second.depends_on.add(first.key)
        found.close_work_item(first)

        self.assertEqual(found.blocked_work_items(), {})


class TheContractStillDecidesDone(unittest.TestCase):
    """CASES 4 and 12 — PHASES 10 and 11."""

    def one_done(self):
        found = ledger()
        found.record_plan([item(R1)])
        found.note_write([ACCEPT, TEST])
        found.note_validation("ctest", "passed")

        return found

    def test_a_closed_work_item_does_not_close_the_contract(self):
        self.assertFalse(self.one_done().contract_closed())

    def test_the_status_says_incomplete(self):
        self.assertEqual(agent_runtime.requirement_status(self.one_done()),
                         "implementation work incomplete")

    def test_the_matrix_still_names_the_open_ones(self):
        note = agent_runtime.requirement_matrix_note(self.one_done())

        self.assertIn(R3.key, note)
        self.assertIn("not closed", note)

    def test_closing_everything_closes_the_contract(self):
        found = self.one_done()
        found.observe_call(implementation_paths={"src/link/extra.c"})
        found.record_plan([item(R2, disposition="satisfied_already",
                                correction=f"already done in {ACCEPT}")])
        found.observe_call(implementation_paths={"src/link/more.c"})
        found.record_plan([item(R3, disposition="explicitly_out_of_scope",
                                path=FRAME,
                                correction="this build does not frame")])

        self.assertTrue(found.contract_closed())

    def test_a_partial_result_never_reads_as_compliant(self):
        note = agent_runtime.requirement_matrix_note(
            self.one_done(), "The implementation is complete.")

        self.assertIn("take the matrix, not the claim", note)


class TheRecordShowsTheWork(unittest.TestCase):
    def test_work_items_are_in_the_diagnostic(self):
        found = ledger()
        found.record_plan([item(R1)])
        record = found.to_dict()

        self.assertEqual(len(record["work_items"]), 1)
        self.assertIn("handshake.c", record["work_items"][0]["paths"])

    def test_it_serialises(self):
        import json

        found = ledger()
        found.record_plan([item(R1)])

        self.assertEqual(json.loads(json.dumps(found.to_dict()))["work_items"],
                         found.to_dict()["work_items"])


if __name__ == "__main__":
    unittest.main()
