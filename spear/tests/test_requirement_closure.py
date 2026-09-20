"""What one grounded turn establishes, the next turn has to close.

Synthetic throughout: a fictional two-clause document about a fictional
handshake. The point of that is not squeamishness. A carry-forward tuned on
the document it was written for is one that only works on that document, and
the failure being fixed here — a requirement that stops being mentioned
between one turn and the next — has nothing to do with any particular subject.
"""

import unittest

import requirement_set
import work_phase
from requirement_set import Disposition, Requirement, RequirementSet
from work_phase import Phase, WorkPhaseLedger


SOURCE = "src/link/handshake.c"
OTHER_SOURCE = "src/link/frame.c"


def R(key, section, statement="the responder does the thing", **kw):
    return Requirement(key=key, section=section, source_id="std-" + key[-3:],
                       statement=statement, **kw)


#: Three grounded requirements, as a first turn would leave them.
R1 = R("Rule 4.2.1-1", "4.2.1", "the responder answers every accepted request")
R2 = R("Rule 4.2.2-1", "4.2.2", "the answer carries the request identifier")
R3 = R("Rule 4.3-1", "4.3", "an unknown opcode is refused, not ignored")


def carried(*items):
    return RequirementSet(list(items or (R1, R2, R3)), origin="turn one")


def governed(*items, investigated=True):
    ledger = WorkPhaseLedger()
    ledger.engage(authority_bound=True, write_requested=True,
                  requirements=carried(*items))

    if investigated:
        ledger.observe_call(authority_keys={"4.2.1", "4.2.2", "4.3"},
                            authority_units=3)
        ledger.observe_call(implementation_paths={SOURCE})

    return ledger


def item_for(requirement, **kw):
    field = {
        "requirement": requirement.statement,
        "requirement_evidence": requirement.key,
        "current_behaviour": "it does not",
        "implementation_evidence": SOURCE,
        "gap": "the behaviour is absent",
        "correction": f"add it to handshake_accept() in {SOURCE}",
        "validation": "tests/test_handshake.c: when a second request arrives on the same link, expect a second answer carrying the same identifier",
    }
    field.update(kw)

    return field


class TurnOnePublishesWhatItGrounded(unittest.TestCase):
    """CASE 1 — the set travels, and it is built from evidence not prose."""

    class Record:
        def __init__(self, key, section, text, force=3):
            self.key, self.section, self.text = key, section, text
            self.source_id, self.effective_force = "std-" + section, force

    class Policy:
        def __init__(self, records):
            ledger = type("L", (), {})()
            ledger.records = {record.key: record for record in records}
            evidence = type("E", (), {})()
            evidence.provisions = ledger
            self.claim_evidence = evidence

    def policy(self):
        return self.Policy([
            self.Record("Rule 4.2.1-1", "4.2.1", "the responder shall answer"),
            self.Record("Rule 4.2.2-1", "4.2.2", "the answer shall carry the id"),
            self.Record("Rule 4.3-1", "4.3", "an unknown opcode shall be refused"),
            self.Record("Observation 4.4-1", "4.4", "this is merely noted", 0),
        ])

    def test_the_requirements_are_carried_and_the_observation_is_not(self):
        found = requirement_set.publish(self.policy(), "")

        self.assertEqual(found.keys,
                         ("Rule 4.2.1-1", "Rule 4.2.2-1", "Rule 4.3-1"))

    def test_a_withheld_answer_still_publishes_a_contract(self):
        """The two weakest measured runs had their first answer withheld."""
        found = requirement_set.publish(self.policy(), answer="")

        self.assertEqual(len(found), 3)

    def test_an_answer_that_cited_provisions_narrows_the_set_to_those(self):
        found = requirement_set.publish(
            self.policy(), "As Rule 4.2.1-1 and Rule 4.3-1 require, ...")

        self.assertEqual(found.keys, ("Rule 4.2.1-1", "Rule 4.3-1"))

    def test_what_the_answer_left_out_is_carried_as_excluded_not_dropped(self):
        """A decision that leaves no trace is an oversight to any reader."""
        found = requirement_set.publish(
            self.policy(), "As Rule 4.2.1-1 and Rule 4.3-1 require, ...")
        excluded = found.excluded()

        self.assertEqual([item.key for item in excluded], ["Rule 4.2.2-1"])
        self.assertIn("not named in the answer", excluded[0].origin)

    def test_an_excluded_requirement_does_not_hold_the_gate(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True,
                      requirements=requirement_set.publish(
                          self.policy(),
                          "As Rule 4.2.1-1 and Rule 4.3-1 require, ..."))
        ledger.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        ledger.observe_call(implementation_paths={SOURCE})

        self.assertEqual({item.key for item in ledger.uncovered_requirements()},
                         {"Rule 4.2.1-1", "Rule 4.3-1"})

    def test_nothing_grounded_publishes_nothing(self):
        self.assertEqual(len(requirement_set.publish(self.Policy([]), "")), 0)

    def test_a_missing_policy_publishes_nothing(self):
        self.assertEqual(len(requirement_set.publish(None, "anything")), 0)


class OnlyATurnThatPointedBackInheritsIt(unittest.TestCase):
    """CASE 7 — a new self-contained task owes nobody else's obligations."""

    def test_a_back_reference_carries(self):
        for question in ("update the implementation to comply with this",
                         "implement this", "apply these requirements",
                         "make the code comply with that",
                         "check the code base and do the necessary changes "
                         "to be compliant with this."):
            with self.subTest(question=question):
                self.assertTrue(requirement_set.refers_back(question))

    def test_a_self_contained_task_does_not(self):
        for question in ("add a --verbose flag to the client",
                         "fix the crash in handshake_accept",
                         "rename the retry counter"):
            with self.subTest(question=question):
                self.assertFalse(requirement_set.refers_back(question))

    def test_an_uninherited_turn_has_no_coverage_gate(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True,
                      requirements=None)
        ledger.observe_call(authority_keys={"4.2.1"}, authority_units=1)
        ledger.observe_call(implementation_paths={SOURCE})
        ledger.record_plan([item_for(R1)])

        self.assertTrue(ledger.may_write().allowed)


class NoRequirementMayDisappear(unittest.TestCase):
    """CASES 2 and 17 — the plan covers the set or it is not a plan."""

    def test_rediscovering_only_one_does_not_shrink_the_set(self):
        ledger = governed()
        ledger.record_plan([item_for(R1)])

        self.assertEqual(len(ledger.requirements), 3)
        self.assertEqual({found.key for found in ledger.uncovered_requirements()},
                         {R2.key, R3.key})

    def test_an_unrelated_undisposed_requirement_no_longer_blocks(self):
        """Plan locally, close globally: R3 stays open and stays visible."""
        ledger = governed()
        ledger.record_plan([item_for(R1)])

        self.assertTrue(ledger.may_write(SOURCE).allowed)
        self.assertIn(R3.key,
                      [found.key for found in ledger.uncovered_requirements()])

    def test_but_it_still_keeps_the_turn_from_finishing(self):
        ledger = governed()
        ledger.record_plan([item_for(R1)])
        ledger.note_write([SOURCE, "tests/test_handshake.c"])
        ledger.note_validation("ctest", "passed")

        self.assertFalse(ledger.contract_closed())

    def test_the_gate_opens_when_every_one_has_a_disposition(self):
        ledger = governed()

        for requirement in (R1, R2, R3):
            ledger.record_plan([item_for(requirement)])

        self.assertTrue(ledger.may_write().allowed)

    def test_with_nothing_planned_the_refusal_says_to_plan_one(self):
        ledger = governed()
        message = ledger.may_write().message

        self.assertIn("Nothing has been planned yet", message)
        self.assertIn("not for all of them", message)

    def test_a_plan_item_for_nothing_carried_closes_nothing(self):
        ledger = governed()
        ledger.record_plan([item_for(R1, requirement_evidence="§4.2.1")])
        ledger.record_plan([item_for(R2)])
        ledger.record_plan([
            item_for(R3, requirement_evidence="Rule 9.9.9-1")])

        self.assertIn(R3.key,
                      [found.key for found in ledger.uncovered_requirements()])


class EveryDispositionIsAnAnswerExceptSilence(unittest.TestCase):
    """CASES 3, 4 and 5."""

    def test_already_satisfied_closes_without_an_edit(self):
        ledger = governed()
        ledger.record_plan([item_for(
            R1, disposition="satisfied_already",
            current_behaviour="handshake_accept() already answers each one",
            gap="none", correction="no change needed",
            validation="the existing tests/test_handshake.c covers it")])

        self.assertEqual(ledger.requirements.get(R1.key).disposition,
                         str(Disposition.SATISFIED_ALREADY))
        self.assertNotIn(R1.key,
                         [found.key for found in ledger.uncovered_requirements()])

    def test_out_of_scope_closes_with_its_reason_on_the_record(self):
        ledger = governed()
        ledger.record_plan([item_for(
            R3, disposition="explicitly_out_of_scope",
            correction="this build does not implement opcode dispatch")])
        found = ledger.requirements.get(R3.key)

        self.assertEqual(found.disposition,
                         str(Disposition.EXPLICITLY_OUT_OF_SCOPE))
        self.assertIn("does not implement", found.note)

    def test_undetermined_is_an_answer_and_is_not_a_finish(self):
        ledger = governed()

        for requirement in (R1, R2):
            ledger.record_plan([item_for(requirement)])

        ledger.record_plan([item_for(R3, disposition="undetermined")])

        # It no longer blocks the write gate -- something was said -- and it
        # keeps the turn from being finished, which is the honest reading.
        self.assertTrue(ledger.may_write().allowed)
        self.assertIn(R3.key,
                      [found.key for found in ledger.requirements.open_items()])

    def test_naming_the_change_settles_a_claim_of_not_knowing(self):
        """The label is cheap; the seven fields are not, so the fields win."""
        ledger = governed()
        ledger.record_plan([item_for(
            R1, disposition="undetermined",
            correction="change handshake_accept() in " + SOURCE)])

        self.assertEqual(ledger.requirements.get(R1.key).disposition,
                         str(Disposition.CHANGE_PLANNED))

    def test_a_genuine_dead_end_stays_undetermined(self):
        ledger = governed()
        ledger.record_plan([item_for(
            R1, disposition="undetermined",
            gap="could not tell whether the responder is reached at all",
            correction="needs a reading of the dispatch path first")])

        self.assertEqual(ledger.requirements.get(R1.key).disposition,
                         str(Disposition.UNDETERMINED))

    def test_a_write_settles_the_label_but_not_the_proof(self):
        """A write says the code changed. It does not say it works."""
        ledger = governed()
        ledger.record_plan([item_for(
            R1, disposition="undetermined",
            gap="could not tell",
            correction="needs more reading")])
        ledger.note_write([SOURCE])

        self.assertEqual(
            ledger.requirements.get(R1.key).disposition,
            str(Disposition.CODE_CHANGED_AWAITING_VALIDATION))

    def test_an_unknown_disposition_falls_back_to_the_ordinary_case(self):
        ledger = governed()
        ledger.record_plan([item_for(R1, disposition="probably_fine")])

        self.assertEqual(ledger.requirements.get(R1.key).disposition,
                         str(Disposition.CHANGE_PLANNED))

    def test_a_planned_change_is_not_a_finished_one(self):
        ledger = governed()

        for requirement in (R1, R2, R3):
            ledger.record_plan([item_for(requirement)])

        self.assertEqual(len(ledger.requirements.open_items()), 3)

    def test_a_write_to_the_named_file_moves_it_on_but_not_to_done(self):
        ledger = governed()

        for requirement in (R1, R2, R3):
            ledger.record_plan([item_for(requirement)])

        ledger.note_write([SOURCE])

        self.assertEqual(
            {found.disposition for found in ledger.requirements},
            {str(Disposition.CODE_CHANGED_AWAITING_VALIDATION)})
        self.assertEqual(len(ledger.requirements.open_items()), 3)

    def test_the_write_and_its_validation_together_finish_it(self):
        ledger = governed()

        for requirement in (R1, R2, R3):
            ledger.record_plan([item_for(requirement)])

        ledger.note_write([SOURCE, "tests/test_handshake.c"])
        ledger.note_validation("ctest --test-dir build", "passed")

        self.assertEqual(ledger.requirements.open_items(), ())

    def test_a_write_somewhere_else_finishes_nothing(self):
        ledger = governed()
        ledger.record_plan([item_for(R1)])
        ledger.note_write(["docs/README.md"])

        self.assertEqual(ledger.requirements.get(R1.key).disposition,
                         str(Disposition.CHANGE_PLANNED))


class AnUngroundedSentenceIsNotARequirement(unittest.TestCase):
    """CASE 6 — the set is built from provisions, never from the reply."""

    def test_prose_alone_publishes_nothing(self):
        class Empty:
            claim_evidence = type("E", (), {"provisions": None})()

        found = requirement_set.publish(
            Empty(), "The responder must also log every refusal.")

        self.assertEqual(len(found), 0)

    def test_a_claim_the_ledger_does_not_hold_cannot_be_disposed(self):
        ledger = governed()

        self.assertIsNone(
            ledger.requirements.dispose("Rule 9.9.9-1", Disposition.SATISFIED_ALREADY))


class AFamilyIsTrackedBeforeThePlanCloses(unittest.TestCase):
    """PHASE 8 — inferred from the provision's own words, never hard-coded."""

    def test_a_coordinated_list_becomes_family_members(self):
        self.assertEqual(
            requirement_set._family_of(
                "one response for each of the Alpha, Bravo or Charlie bits set"),
            ("Alpha", "Bravo", "Charlie"))

    def test_ordinary_prose_declares_no_family(self):
        self.assertEqual(
            requirement_set._family_of("the packet shall carry one identifier"),
            ())

    def test_the_members_travel_with_the_requirement(self):
        found = RequirementSet([R("Rule 4.2.1-1", "4.2.1",
                                  "one for each of Alpha, Bravo or Charlie",
                                  members=("Alpha", "Bravo", "Charlie"))])

        self.assertEqual(found.items[0].members, ("Alpha", "Bravo", "Charlie"))

    def test_the_members_are_put_in_front_of_the_turn_when_it_is_blocked(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True,
                      requirements=RequirementSet([
                          R("Rule 4.2.1-1", "4.2.1",
                            "one for each of Alpha, Bravo or Charlie",
                            members=("Alpha", "Bravo", "Charlie")),
                          R("Rule 4.2.2-1", "4.2.2",
                            "the same handshake_accept path also carries the id")]))
        ledger.observe_call(authority_keys={"4.2.1", "4.2.2"}, authority_units=2)
        ledger.observe_call(implementation_paths={SOURCE})
        ledger.record_plan([item_for(
            R("Rule 4.2.2-1", "4.2.2", "x"),
            correction=f"change handshake_accept() in {SOURCE}")])
        ledger.requirements.dispose("Rule 4.2.1-1", "change_planned")
        ledger.work_items[0].requirements.add("Rule 4.2.1-1")
        message = ledger.may_write(SOURCE).message

        self.assertIn("covers each of: Alpha, Bravo, Charlie", message)

    def test_a_family_survives_serialisation(self):
        import json

        found = RequirementSet([R1, R("Rule 4.9-1", "4.9", "x",
                                      members=("Alpha", "Bravo"))])
        again = RequirementSet.from_dict(
            json.loads(json.dumps(found.to_dict())))

        self.assertEqual(again.items[1].members, ("Alpha", "Bravo"))
        self.assertEqual(again.keys, found.keys)


class CitationsFindTheRequirementTheyMean(unittest.TestCase):
    def test_the_printed_key_matches(self):
        self.assertEqual(carried().match("Rule 4.2.1-1").key, R1.key)

    def test_the_bare_ordinal_matches(self):
        self.assertEqual(carried().match("see 4.2.1-1 above").key, R1.key)

    def test_the_section_matches(self):
        self.assertEqual(carried().match("§4.3").key, R3.key)

    def test_a_containing_section_matches(self):
        self.assertEqual(carried().match("§4.2").key, R1.key)

    def test_the_shape_a_model_actually_writes_matches(self):
        """Not the bare key: the document, the section, the page, the handle.

        Every plan item of one measured run cited its provision this way and
        not one of them matched, so nothing was dispositioned, the gate never
        opened and the turn was refused ten edits with the work correctly
        planned.
        """
        found = RequirementSet([
            Requirement(key="Rule 4.2.1-1", section="",
                        source_id="std-0a1b2c3d4e5f")])

        for citation in (
                "SYNTHETIC-1 2020 §4.2.1, p.17, source std-0a1b2c3d4e5f",
                "SYNTHETIC-1 2020 §4.2.1, p.17",
                "source std-0a1b2c3d4e5f",
                "§4.2",
                "Rule 4.2.1-1"):
            with self.subTest(citation=citation):
                self.assertIsNotNone(found.match(citation))

    def test_a_handle_nobody_retrieved_matches_nothing(self):
        found = RequirementSet([
            Requirement(key="Rule 4.2.1-1", source_id="std-0a1b2c3d4e5f")])

        self.assertIsNone(found.match("source std-ffffffffffff"))

    def test_something_nobody_carried_matches_nothing(self):
        self.assertIsNone(carried().match("Rule 9.9.9-1"))

    def test_no_citation_at_all_matches_nothing(self):
        self.assertIsNone(carried().match("the usual requirement"))


class TheMatrixIsTheReview(unittest.TestCase):
    """PHASE 15 — a line per requirement, whatever became of it."""

    def test_every_requirement_has_a_row(self):
        ledger = governed()
        ledger.record_plan([item_for(R1)])
        rows = ledger.requirements.matrix()

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["requirement"] for row in rows},
                         {R1.key, R2.key, R3.key})

    def test_a_row_carries_its_evidence_and_its_state(self):
        ledger = governed()
        ledger.record_plan([item_for(R1, disposition="satisfied_already")])
        row = next(r for r in ledger.requirements.matrix()
                   if r["requirement"] == R1.key)

        self.assertEqual(row["disposition"], str(Disposition.SATISFIED_ALREADY))
        self.assertTrue(row["evidence"])

    def test_the_note_names_the_open_ones(self):
        import agent_runtime

        ledger = governed()
        ledger.record_plan([item_for(R1)])
        note = agent_runtime.requirement_matrix_note(ledger)

        self.assertIn("REQUIREMENT MATRIX", note)
        self.assertIn(R2.key, note)
        self.assertIn("not closed", note)

    def test_there_is_no_note_without_a_contract(self):
        import agent_runtime

        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True)

        self.assertEqual(agent_runtime.requirement_matrix_note(ledger), "")


if __name__ == "__main__":
    unittest.main()
