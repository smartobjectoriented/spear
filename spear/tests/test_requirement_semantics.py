"""Closing a list of records is not the same as understanding the work.

The gate before this one could be satisfied by a turn that said something
about every requirement it carried. A real run did exactly that and still
produced the wrong thing: it dispositioned one rule eight times with the
answer oscillating between satisfied, undetermined and out of scope; it
accumulated seventeen logical requirements where four had been carried; it
changed acknowledged behaviour and added no test, then reported that all 198
existing checks passed; and it closed with its own summary table declaring
every clause satisfied, printed directly above a ledger that called one of
them out of scope.

Each of those is a different failure and none of them is about safety. They
are about whether the record means anything. Everything here is synthetic: a
fictional document about a fictional handshake, because a mechanism tuned on
the document that exposed it is a mechanism that works on one document.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requirement_set
import work_phase
from requirement_set import Disposition, Requirement, RequirementSet
from work_phase import WorkPhaseLedger

SOURCE = "src/link/handshake.c"
OTHER = "src/link/device.c"
HANDLE = "std-0a1b2c3d4e5f"


def R(key, **kw):
    kw.setdefault("statement", "the responder answers every accepted request")
    kw.setdefault("source_id", HANDLE)

    return Requirement(key=key, **kw)


def ledger(*items, read=(SOURCE,)):
    found = WorkPhaseLedger()
    found.engage(authority_bound=True, write_requested=True,
                 requirements=RequirementSet(list(items or (R("Rule 4.2.1-1"),))))
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
        "validation": "tests/test_handshake.c exercises two requests",
    }
    base.update(kw)

    return base


class OneProvisionIsOneRecord(unittest.TestCase):
    """PHASE 1 and 2 — the same requirement, said again, is the same
    requirement. A real run turned one rule into eight."""

    def test_the_same_provision_cited_five_ways_is_one_item(self):
        found = ledger(R("Rule 4.2.1-1", source_id=HANDLE))

        for citation in ("Rule 4.2.1-1",
                         "§4.2.1",
                         "4.2.1-1",
                         HANDLE,
                         f"SYNTHETIC-1 2020 §4.2.1, p.17, source {HANDLE}"):
            found.record_plan([item(requirement_evidence=citation)])

        self.assertEqual(len(found.items), 1)
        self.assertEqual(len(found.requirements), 1)

    def test_the_latest_disposition_is_the_one_that_stands(self):
        found = ledger()

        for disposition in ("change_planned", "satisfied_already",
                            "explicitly_out_of_scope"):
            found.record_plan([item(disposition=disposition,
                                    correction="no change needed here")])

        self.assertEqual(found.requirements.get("Rule 4.2.1-1").disposition,
                         str(Disposition.EXPLICITLY_OUT_OF_SCOPE))

    def test_the_whole_history_is_kept(self):
        found = ledger()

        for disposition in ("change_planned", "satisfied_already",
                            "explicitly_out_of_scope"):
            found.record_plan([item(disposition=disposition,
                                    correction="no change needed here")])

        self.assertEqual(
            found.requirements.revisions()["Rule 4.2.1-1"],
            ["change_planned", "satisfied_already", "explicitly_out_of_scope"])

    def test_revisions_are_reported(self):
        found = ledger()
        found.record_plan([item()])
        found.record_plan([item(disposition="satisfied_already",
                                correction="no change needed")])

        self.assertEqual(len(found.revised), 1)
        self.assertEqual(found.revised[0]["requirement"], "Rule 4.2.1-1")

    def test_free_text_is_never_the_identity(self):
        """A model rewords its own sentence between calls."""
        found = ledger()
        found.record_plan([item(requirement="answer every request")])
        found.record_plan([item(requirement="respond to each accepted one")])

        self.assertEqual(len(found.items), 1)

    def test_two_different_provisions_stay_two(self):
        found = ledger(R("Rule 4.2.1-1", source_id="std-aaaaaaaaaaaa"),
                       R("Rule 4.3-1", source_id="std-bbbbbbbbbbbb"))
        found.record_plan([item(requirement_evidence="Rule 4.2.1-1")])
        found.record_plan([item(requirement_evidence="Rule 4.3-1")])

        self.assertEqual(len(found.items), 2)

    def test_an_invented_handle_resolves_to_nothing(self):
        found = ledger(R("Rule 4.2.1-1", source_id=HANDLE))
        outcome = found.record_plan(
            [item(requirement_evidence="std-ffffffffffff")])

        self.assertFalse(outcome.any_accepted)

    def test_an_unbound_item_is_still_idempotent_by_citation(self):
        found = ledger()
        found.record_plan([item(requirement_evidence="Rule 4.2.1-1")])
        found.record_plan([item(requirement_evidence="§9.9.9",
                                requirement="something it found itself")])
        found.record_plan([item(requirement_evidence="§9.9.9",
                                requirement="reworded")])

        self.assertEqual(len(found.items), 1)


class LifecycleSemanticsSurvive(unittest.TestCase):
    """PHASES 3 to 5 — a requirement that says WHEN is not satisfied by code
    that does the right thing eventually."""

    def test_a_conditional_provision_keeps_its_condition(self):
        self.assertEqual(
            requirement_set._trigger_of(
                "the responder shall answer after the request has been "
                "validated"),
            "after the request has been validated")

    def test_an_unconditional_provision_has_none(self):
        self.assertEqual(
            requirement_set._trigger_of("the packet shall carry an identifier"),
            "")

    def test_the_condition_travels_with_the_requirement(self):
        class Record:
            key, section = "Rule 4.2.1-1", "4.2.1"
            source_id, effective_force = HANDLE, 3
            text = "the responder shall answer after validation completes"

        class Policy:
            claim_evidence = type("E", (), {"provisions": type(
                "L", (), {"records": {"Rule 4.2.1-1": Record()}})()})()

        published = requirement_set.publish(Policy(), "")

        self.assertEqual(published.items[0].trigger,
                         "after validation completes")

    def test_a_plan_that_names_no_point_in_the_code_is_refused(self):
        found = ledger(R("Rule 4.2.1-1", trigger="after validation completes"))
        outcome = found.record_plan(
            [item(correction="send the answer eventually")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("conditions its obligation", outcome.report())

    def test_a_plan_that_names_the_point_is_accepted(self):
        found = ledger(R("Rule 4.2.1-1", trigger="after validation completes"))
        outcome = found.record_plan([item(
            correction=f"answer from handshake_accept() in {SOURCE}, after "
                       f"the validation branch returns")])

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_an_unconditional_requirement_asks_for_no_binding(self):
        found = ledger(R("Rule 4.2.1-1", trigger=""))
        outcome = found.record_plan(
            [item(correction="send the answer eventually")])

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_two_stages_of_one_operation_stay_two_requirements(self):
        """Distinct lifecycle points are not collapsed by the ledger."""
        found = ledger(R("Rule 4.2.1-1", trigger="after validation completes",
                         source_id="std-aaaaaaaaaaaa"),
                       R("Rule 4.2.1-2", trigger="after execution completes",
                         source_id="std-bbbbbbbbbbbb"))
        found.record_plan([item(
            requirement_evidence="Rule 4.2.1-1",
            correction=f"answer from validate() in {SOURCE}")])

        self.assertEqual([found.key for found in found.uncovered_requirements()],
                         ["Rule 4.2.1-2"])


class NothingIsDroppedSilently(unittest.TestCase):
    """PHASE 6 — a binding provision the answer did not repeat is a decision,
    and a decision that leaves no trace reads as an oversight."""

    class Policy:
        def __init__(self):
            def record(key, section, text, force=3):
                return type("R", (), {"key": key, "section": section,
                                      "text": text, "source_id": "std-" + section,
                                      "effective_force": force})()

            self.claim_evidence = type("E", (), {"provisions": type(
                "L", (), {"records": {
                    "Rule 4.2.1-1": record("Rule 4.2.1-1", "4.2.1", "a shall b"),
                    "Rule 4.2.2-1": record("Rule 4.2.2-1", "4.2.2", "c shall d"),
                    "Rule 4.3-1": record("Rule 4.3-1", "4.3", "e shall f"),
                }})()})()

    def published(self):
        return requirement_set.publish(self.Policy(), "per Rule 4.2.1-1, ...")

    def test_the_cited_one_is_in_scope(self):
        self.assertEqual(self.published().keys, ("Rule 4.2.1-1",))

    def test_the_others_are_carried_with_a_reason(self):
        excluded = self.published().excluded()

        self.assertEqual(len(excluded), 2)

        for found in excluded:
            self.assertIn("not named in the answer", found.origin)

    def test_they_do_not_hold_the_write_gate(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True,
                     requirements=self.published())
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        found.observe_call(implementation_paths={SOURCE})

        self.assertEqual([r.key for r in found.uncovered_requirements()],
                         ["Rule 4.2.1-1"])


class MovingACallAsksAboutItsContext(unittest.TestCase):
    """PHASES 12 and 13 — what the old call site guaranteed, the new one may
    not: thread, lock, lifetime, ownership."""

    def test_a_moved_call_with_one_file_read_is_refused(self):
        found = ledger(read=(SOURCE, OTHER))
        outcome = found.record_plan([item(
            implementation_evidence=SOURCE,
            correction=f"call device_read() from the listener in {SOURCE}")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("not safe from anywhere", outcome.report())

    def test_having_read_both_ends_it_is_accepted(self):
        found = ledger(read=(SOURCE, OTHER))
        outcome = found.record_plan([item(
            implementation_evidence=f"{SOURCE} and {OTHER}",
            current_behaviour="device.c says the worker thread owns it",
            correction=f"call device_read() from the listener in {SOURCE}")])

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_an_ordinary_local_change_is_unaffected(self):
        found = ledger()
        outcome = found.record_plan([item(
            correction=f"add a field to the report in {SOURCE}")])

        self.assertTrue(outcome.any_accepted, outcome.report())


class NormativeTruthComesFromTheStandardTools(unittest.TestCase):
    """PHASE 17 — a comment explains an implementation; it does not state a
    requirement, however many section numbers it quotes."""

    def test_a_shell_result_adds_no_authority(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(implementation_paths={SOURCE})

        self.assertFalse(found.evidence.has_authority)
        self.assertEqual(found.may_write().reason, work_phase.NO_AUTHORITY)

    def test_a_citation_found_only_in_source_is_not_backed(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(implementation_paths={SOURCE})

        self.assertFalse(found.evidence.backs_requirement("Rule 4.2.1-1"))

    def test_authority_comes_from_the_retrieval_ledger(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True)
        found.observe_call(authority_keys={"4.2.1"}, authority_units=1)

        self.assertTrue(found.evidence.backs_requirement("Rule 4.2.1-1"))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
