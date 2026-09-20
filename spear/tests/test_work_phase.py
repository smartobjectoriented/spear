"""The lifecycle of a turn that changes code to satisfy an authoritative source.

Everything here is synthetic. The "standard" is a two-clause document about a
fictional handshake, the "implementation" is a made-up file, and the point of
that is not squeamishness: a gate tuned on the document it was written for is
a gate that only works on that document. If these fixtures were real, a test
could pass because the rule happened to fit the text rather than because the
mechanism is sound.
"""

import unittest

import work_phase
from work_phase import GapItem, Phase, WorkPhaseLedger


#: A fictional document, and the two provisions this suite plans against.
CLAUSE_A = "4.2.1"
CLAUSE_B = "4.2.2"

#: A fictional implementation.
SOURCE = "src/link/handshake.c"
OTHER_SOURCE = "src/link/frame.c"


def engaged():
    ledger = WorkPhaseLedger()
    ledger.engage(authority_bound=True, write_requested=True)

    return ledger


def investigated(ledger=None):
    """A ledger that has read both sides and nothing more."""
    ledger = ledger or engaged()
    ledger.observe_call(authority_keys={CLAUSE_A, CLAUSE_B}, authority_units=2)
    ledger.observe_call(implementation_paths={SOURCE})

    return ledger


def good_item(**overrides):
    item = {
        "requirement": "the responder answers every request it accepts",
        "requirement_evidence": f"Rule {CLAUSE_A}-1",
        "current_behaviour": "the responder answers only the first request",
        "implementation_evidence": SOURCE,
        "gap": "subsequent requests are dropped without an answer",
        "correction": "answer each accepted request in handshake_accept()",
        "validation": "tests/test_handshake.c: when a second request arrives on the same link, expect a second answer carrying the same identifier",
    }
    item.update(overrides)

    return item


class TheGateIsOnlyOnTheShapeItGoverns(unittest.TestCase):
    """Case 11 and case 12: everything else keeps the workflow it had."""

    def test_a_pure_code_task_is_not_governed(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=False, write_requested=True)

        self.assertFalse(ledger.engaged)
        self.assertTrue(ledger.may_write().allowed)

    def test_a_pure_question_about_the_document_is_not_governed(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=False)

        self.assertFalse(ledger.engaged)
        self.assertTrue(ledger.may_write().allowed)
        self.assertEqual(ledger.phase, Phase.INVESTIGATE)

    def test_neither_half_is_not_governed(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=False, write_requested=False)

        self.assertFalse(ledger.engaged)
        self.assertTrue(ledger.may_write().allowed)

    def test_a_disengaged_ledger_never_leaves_investigate(self):
        """No phase machinery runs for a turn the lifecycle does not govern."""
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=False)
        ledger.observe_call(authority_keys={CLAUSE_A}, authority_units=1)
        ledger.observe_call(implementation_paths={SOURCE})
        ledger.begin_review()
        ledger.finish()

        self.assertEqual(ledger.phase, Phase.INVESTIGATE)


class WhatTheSessionAlreadyEstablished(unittest.TestCase):
    """Two prompts: "what does it require?", then "make the change"."""

    def carried(self):
        found = WorkPhaseLedger()
        found.engage(authority_bound=True, write_requested=True,
                     prior_authority=(CLAUSE_A, CLAUSE_B))

        return found

    def test_a_clause_read_in_the_previous_turn_may_be_planned_against(self):
        ledger = self.carried()
        ledger.observe_call(implementation_paths={SOURCE})
        outcome = ledger.record_plan([good_item()])

        self.assertTrue(outcome.any_accepted, outcome.report())

    def test_the_code_must_still_be_read_in_this_one(self):
        ledger = self.carried()

        self.assertFalse(ledger.may_write().allowed)
        self.assertEqual(ledger.may_write().reason,
                         work_phase.NO_IMPLEMENTATION)

    def test_a_clause_nobody_ever_read_is_still_refused(self):
        ledger = self.carried()
        ledger.observe_call(implementation_paths={SOURCE})
        outcome = ledger.record_plan([good_item(
            requirement_evidence="Rule 9.9.9-1")])

        self.assertFalse(outcome.any_accepted)


class NoEditBeforeInvestigationIsComplete(unittest.TestCase):
    """Cases 1, 2 and 3."""

    def test_nothing_read_at_all(self):
        ledger = engaged()
        decision = ledger.may_write()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, work_phase.NO_AUTHORITY)
        self.assertIn(work_phase.WRITE_BLOCKED, decision.message)

    def test_requirements_read_but_no_code(self):
        ledger = engaged()
        ledger.observe_call(authority_keys={CLAUSE_A}, authority_units=1)
        decision = ledger.may_write()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, work_phase.NO_IMPLEMENTATION)
        self.assertIn("no source file has been read", decision.message)

    def test_code_read_but_no_requirements(self):
        ledger = engaged()
        ledger.observe_call(implementation_paths={SOURCE})
        decision = ledger.may_write()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, work_phase.NO_AUTHORITY)
        self.assertIn("nothing to be compliant with", decision.message)

    def test_both_halves_open_the_planning_phase_and_no_more(self):
        ledger = investigated()

        self.assertTrue(ledger.investigation_complete)
        self.assertEqual(ledger.phase, Phase.PLAN)
        self.assertFalse(ledger.may_write().allowed)
        self.assertEqual(ledger.may_write().reason, work_phase.NO_PLAN)


class APlanIsEvidenceOrItIsNothing(unittest.TestCase):
    """Cases 4 and 5."""

    def test_a_complete_evidence_backed_item_opens_the_gate(self):
        ledger = investigated()
        outcome = ledger.record_plan([good_item()])

        self.assertTrue(outcome.any_accepted)
        self.assertEqual(outcome.rejected, ())
        self.assertEqual(ledger.phase, Phase.EDIT)
        self.assertTrue(ledger.may_write().allowed)

    def test_a_vague_plan_maps_no_requirement_to_a_change(self):
        ledger = investigated()
        outcome = ledger.record_plan([{
            "requirement": "be compliant",
            "correction": "I will update the handshake handling",
        }])

        self.assertFalse(outcome.any_accepted)
        self.assertFalse(ledger.may_write().allowed)
        self.assertEqual(ledger.phase, Phase.PLAN)
        self.assertIn("missing", outcome.report())

    def test_an_item_with_no_validation_is_refused(self):
        ledger = investigated()
        outcome = ledger.record_plan([good_item(validation="")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("validation", outcome.report())

    def test_a_citation_nobody_retrieved_is_refused(self):
        """The fluent item is the dangerous one: it looks exactly right."""
        ledger = investigated()
        outcome = ledger.record_plan([good_item(
            requirement_evidence="Rule 9.9.9-1")])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("is not among the evidence this session holds",
                      outcome.report())
        self.assertIn(CLAUSE_A, outcome.report())
        self.assertFalse(ledger.may_write().allowed)

    def test_a_file_nobody_opened_is_refused(self):
        ledger = investigated()
        outcome = ledger.record_plan([good_item(
            implementation_evidence=OTHER_SOURCE)])

        self.assertFalse(outcome.any_accepted)
        self.assertIn("was read this turn", outcome.report())

    def test_one_good_item_among_bad_ones_still_opens_the_gate(self):
        ledger = investigated()
        outcome = ledger.record_plan([
            good_item(requirement_evidence="Rule 9.9.9-1"),
            good_item(requirement="the responder rejects an unknown opcode"),
        ])

        self.assertEqual(len(outcome.accepted), 1)
        self.assertEqual(len(outcome.rejected), 1)
        self.assertTrue(ledger.may_write().allowed)

    def test_the_refusal_names_what_was_wrong_with_the_last_attempt(self):
        ledger = investigated()
        ledger.record_plan([good_item(requirement_evidence="Rule 9.9.9-1")])

        self.assertIn("Last plan item refused", ledger.may_write().message)


class CitationsAreMatchedAgainstWhatWasRead(unittest.TestCase):
    def test_a_provision_ordinal_is_backed_by_its_section(self):
        ledger = investigated()

        self.assertTrue(ledger.evidence.backs_requirement(f"Rule {CLAUSE_A}-7"))

    def test_a_broader_section_is_backed_by_a_narrower_reading(self):
        """Having read §4.2.1, a turn has the text it needs to cite §4.2."""
        ledger = engaged()
        ledger.observe_call(authority_keys={CLAUSE_A}, authority_units=1)

        self.assertTrue(ledger.evidence.backs_requirement("§4.2"))

    def test_a_narrower_section_is_not_backed_by_a_broader_reading(self):
        """Reading a chapter heading is not reading the rule under it."""
        ledger = engaged()
        ledger.observe_call(authority_keys={"4"}, authority_units=1)

        self.assertFalse(ledger.evidence.backs_requirement("Rule 4.2.1-1"))

    def test_an_opaque_handle_matches_exactly(self):
        ledger = engaged()
        ledger.observe_call(authority_keys={"std-0a1b2c3d4e"}, authority_units=1)

        self.assertTrue(ledger.evidence.backs_requirement("std-0a1b2c3d4e"))
        self.assertFalse(ledger.evidence.backs_requirement("std-ffffffffff"))

    def test_code_outside_the_session_tree_is_not_this_implementation(self):
        """Several corpora attached means a great deal of unrelated C."""
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True,
                      root="/work/project")
        ledger.observe_call(authority_keys={CLAUSE_A}, authority_units=1)
        ledger.observe_call(
            implementation_paths={"/elsewhere/other/link/handshake.c"})

        self.assertFalse(ledger.evidence.has_implementation)
        self.assertEqual(ledger.may_write().reason,
                         work_phase.NO_IMPLEMENTATION)

    def test_code_inside_the_session_tree_counts(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=True, write_requested=True,
                      root="/work/project")
        ledger.observe_call(
            implementation_paths={"/work/project/" + SOURCE, SOURCE})

        self.assertTrue(ledger.evidence.has_implementation)

    def test_with_no_boundary_stated_every_path_counts(self):
        ledger = engaged()
        ledger.observe_call(implementation_paths={"/anywhere/at/all.c"})

        self.assertTrue(ledger.evidence.has_implementation)

    def test_a_file_matches_however_the_path_is_spelled(self):
        ledger = engaged()
        ledger.observe_call(implementation_paths={"./" + SOURCE})

        self.assertTrue(ledger.evidence.backs_implementation(SOURCE))
        self.assertTrue(ledger.evidence.backs_implementation("handshake.c"))
        self.assertFalse(ledger.evidence.backs_implementation(OTHER_SOURCE))


class NewEvidenceSendsTheTurnBackToPlan(unittest.TestCase):
    """Case 6."""

    def setUp(self):
        self.ledger = investigated()
        self.ledger.record_plan([good_item()])
        self.ledger.note_write()

    def test_the_phase_returns_to_plan(self):
        self.ledger.invalidate(
            reason="the accept path runs on the receive thread",
            invalidated=good_item()["requirement"],
            evidence=f"{SOURCE}:88 dispatches to a worker")

        self.assertEqual(self.ledger.phase, Phase.PLAN)
        self.assertFalse(self.ledger.may_write().allowed)

    def test_the_reason_the_item_and_the_evidence_are_all_recorded(self):
        self.ledger.invalidate(
            reason="the value is already carried by the existing header",
            invalidated=good_item()["requirement"],
            evidence=f"{SOURCE}:12 defines the field")
        replan = self.ledger.replans[-1]

        self.assertIn("already carried", replan.reason)
        self.assertEqual(replan.invalidated, good_item()["requirement"])
        self.assertIn(":12", replan.evidence)

    def test_a_replacement_plan_reopens_the_gate(self):
        self.ledger.invalidate(reason="wrong lifecycle",
                               invalidated=good_item()["requirement"])
        self.ledger.record_plan([good_item(
            requirement="the responder answers on the worker thread",
            correction="answer from handshake_worker()")])

        self.assertTrue(self.ledger.may_write().allowed)

    def test_superseding_through_the_plan_call_records_the_replan(self):
        self.ledger.record_plan(
            [good_item(requirement="the responder answers once per request")],
            supersedes=good_item()["requirement"],
            reason="the first item duplicated an existing check",
            new_evidence=f"{SOURCE}:40 already validates this")

        self.assertEqual(len(self.ledger.replans), 1)
        self.assertEqual(len(self.ledger.items), 1)
        self.assertEqual(self.ledger.items[0].requirement,
                         "the responder answers once per request")

    def test_the_first_write_phase_survives_a_replan(self):
        """The diagnostic records where the FIRST write landed, once."""
        self.ledger.invalidate(reason="wrong lifecycle")
        self.ledger.record_plan([good_item()])
        self.ledger.note_write()

        self.assertEqual(self.ledger.first_write_phase, str(Phase.EDIT))
        self.assertEqual(self.ledger.writes, 2)


class ExplorationThatAddsNothingIsStopped(unittest.TestCase):
    """Case 7."""

    def test_equivalent_calls_in_a_row_trip_the_control(self):
        ledger = investigated()

        for _ in range(work_phase.REPETITION_LIMIT):
            ledger.observe_call(authority_keys={CLAUSE_A},
                                implementation_paths={SOURCE})

        self.assertTrue(ledger.repeating)

    def test_the_threshold_is_small(self):
        """Not eighty calls. The failure is visible long before that."""
        self.assertLessEqual(work_phase.REPETITION_LIMIT, 5)

    def test_one_new_file_resets_it(self):
        ledger = investigated()
        ledger.observe_call(implementation_paths={SOURCE})
        ledger.observe_call(implementation_paths={SOURCE})
        ledger.observe_call(implementation_paths={OTHER_SOURCE})

        self.assertFalse(ledger.repeating)
        self.assertEqual(ledger.consecutive_without_evidence, 0)

    def test_one_new_clause_resets_it(self):
        ledger = investigated()
        ledger.observe_call(authority_keys={CLAUSE_A})
        ledger.observe_call(authority_keys={CLAUSE_A})
        ledger.observe_call(authority_keys={"4.3.9"})

        self.assertFalse(ledger.repeating)

    def test_the_demand_is_not_repeated_without_end(self):
        """Eight copies of one sentence is wallpaper, and it is paid for."""
        ledger = investigated()
        forced = 0

        for _ in range(work_phase.REPETITION_LIMIT * 20):
            ledger.observe_call()

            if ledger.repeating:
                ledger.force_synthesis()
                forced += 1

        self.assertEqual(forced, work_phase.MAX_SYNTHESES)

    def test_forcing_synthesis_clears_the_count_and_is_recorded(self):
        ledger = investigated()

        for _ in range(work_phase.REPETITION_LIMIT):
            ledger.observe_call()

        ledger.force_synthesis()

        self.assertFalse(ledger.repeating)
        self.assertEqual(ledger.syntheses_forced, 1)

    def test_recording_a_plan_clears_it_too(self):
        ledger = investigated()

        for _ in range(work_phase.REPETITION_LIMIT):
            ledger.observe_call()

        ledger.record_plan([good_item()])

        self.assertFalse(ledger.repeating)


class InvestigateHasAnEnd(unittest.TestCase):
    """Both halves read, nothing planned, and the reading carries on."""

    def test_nothing_is_owed_before_investigation_is_complete(self):
        ledger = engaged()
        ledger.observe_call(authority_keys={CLAUSE_A}, authority_units=1)

        for _ in range(work_phase.PLAN_PATIENCE * 2):
            ledger.observe_call(authority_keys={CLAUSE_A})

        self.assertFalse(ledger.owes_a_plan())

    def test_a_plan_is_owed_after_enough_further_reading(self):
        ledger = investigated()

        for index in range(work_phase.PLAN_PATIENCE):
            self.assertFalse(ledger.owes_a_plan(),
                             f"demanded after {index} calls")
            ledger.observe_call(implementation_paths={f"src/f{index}.c"})

        self.assertTrue(ledger.owes_a_plan())

    def test_nothing_is_owed_once_something_is_planned(self):
        ledger = investigated()
        ledger.record_plan([good_item()])

        for _ in range(work_phase.PLAN_PATIENCE * 2):
            ledger.observe_call(implementation_paths={OTHER_SOURCE})

        self.assertFalse(ledger.owes_a_plan())

    def test_the_reminder_is_bounded(self):
        ledger = investigated()
        demanded = 0

        for _ in range(work_phase.PLAN_PATIENCE * 10):
            ledger.observe_call(implementation_paths={SOURCE})
            demanded += bool(ledger.owes_a_plan())

        self.assertEqual(demanded, work_phase.MAX_PLAN_DEMANDS)

    def test_asking_twice_and_being_ignored_narrows_the_round(self):
        ledger = investigated()

        self.assertFalse(ledger.plan_demand_ignored)

        for _ in range(work_phase.PLAN_PATIENCE * 4):
            ledger.observe_call(implementation_paths={SOURCE})
            ledger.owes_a_plan()

        self.assertTrue(ledger.plan_demand_ignored)

    def test_a_plan_ends_the_narrowing(self):
        ledger = investigated()

        for _ in range(work_phase.PLAN_PATIENCE * 4):
            ledger.observe_call(implementation_paths={SOURCE})
            ledger.owes_a_plan()

        ledger.record_plan([good_item()])

        self.assertFalse(ledger.plan_demand_ignored)

    def test_an_ungoverned_turn_is_owed_nothing(self):
        ledger = WorkPhaseLedger()
        ledger.engage(authority_bound=False, write_requested=True)

        self.assertFalse(ledger.owes_a_plan())


class AFailingCommandIsNotRunForever(unittest.TestCase):
    """Case 8."""

    def test_the_same_failure_twice_is_enough(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()

        self.assertFalse(ledger.validation_exhausted("build"))
        ledger.note_validation("build", "failed")
        self.assertFalse(ledger.validation_exhausted("build"))
        ledger.note_validation("build", "failed")
        self.assertTrue(ledger.validation_exhausted("build"))

    def test_the_threshold_is_per_command(self):
        ledger = investigated()
        ledger.note_validation("build", "failed")
        ledger.note_validation("build", "failed")

        self.assertTrue(ledger.validation_exhausted("build"))
        self.assertFalse(ledger.validation_exhausted("test"))

    def test_a_pass_forgets_the_earlier_failures(self):
        ledger = investigated()
        ledger.note_validation("build", "failed")
        ledger.note_validation("build", "passed")
        ledger.note_validation("build", "failed")

        self.assertFalse(ledger.validation_exhausted("build"))


class CompilingIsNotTesting(unittest.TestCase):
    """Case 9."""

    def test_an_edited_tree_with_no_validation_has_none(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()

        self.assertFalse(ledger.validated)
        self.assertEqual(ledger.phase, Phase.EDIT)

    def test_a_validation_that_ran_moves_the_turn_into_test(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation("ctest --test-dir build", "passed")

        self.assertEqual(ledger.phase, Phase.TEST)
        self.assertTrue(ledger.validated)

    def test_a_validation_that_could_not_run_is_not_a_pass(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation("ctest --test-dir build", "not_run")

        self.assertFalse(ledger.validated)

    def test_a_requirement_no_validation_covers_is_reported(self):
        ledger = investigated()
        ledger.record_plan([good_item(
            validation="when a reviewer walks the diff, expect the new branch "
                       "to be visible")])
        ledger.note_write()
        ledger.note_validation("ctest --test-dir build", "passed")

        self.assertEqual([item.requirement for item in ledger.unresolved_items()],
                         [good_item()["requirement"]])

    def test_a_command_that_answers_the_items_own_description_covers_it(self):
        """"tests/test_handshake.c" against a ctest that ran is one claim."""
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation(
            "ctest --test-dir build --tests-regex test_handshake", "passed")

        self.assertEqual(ledger.unresolved_items(), ())

    def test_a_failing_command_covers_nothing(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation(
            "ctest --test-dir build --tests-regex test_handshake", "failed")

        self.assertEqual(len(ledger.unresolved_items()), 1)

    def test_a_validation_that_names_what_it_covers_resolves_it(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation("ctest", "passed",
                               covers=[good_item()["requirement"]])

        self.assertEqual(ledger.unresolved_items(), ())

    def test_writes_stay_available_during_test(self):
        """A failing suite is answered by a fix, not by a closed gate."""
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()
        ledger.note_validation("ctest", "failed")

        self.assertTrue(ledger.may_write().allowed)


class TheFinalReviewIsReadOnly(unittest.TestCase):
    """Case 10, on the side the lifecycle owns."""

    def setUp(self):
        self.ledger = investigated()
        self.ledger.record_plan([good_item()])
        self.ledger.note_write()
        self.ledger.note_validation("ctest", "passed")
        self.ledger.begin_review()

    def test_the_phase_is_review(self):
        self.assertEqual(self.ledger.phase, Phase.REVIEW)

    def test_a_write_during_review_is_refused(self):
        decision = self.ledger.may_write()

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, work_phase.REVIEW_IS_READ_ONLY)

    def test_a_review_finding_reopens_the_gate_through_a_plan(self):
        self.ledger.record_plan([good_item(
            requirement="the responder reports an unknown opcode")])

        self.assertEqual(self.ledger.phase, Phase.EDIT)
        self.assertTrue(self.ledger.may_write().allowed)


class TheRecordIsInspectable(unittest.TestCase):
    """A diagnostic has to be able to read what happened without the log."""

    def test_the_first_write_phase_is_named(self):
        ledger = investigated()
        ledger.record_plan([good_item()])
        ledger.note_write()

        self.assertEqual(ledger.to_dict()["first_write_phase"], "edit")

    def test_refusals_are_counted(self):
        ledger = engaged()
        ledger.refuse_write()
        ledger.refuse_write()

        self.assertEqual(ledger.to_dict()["write_refusals"], 2)

    def test_the_whole_record_is_serialisable(self):
        import json

        ledger = investigated()
        ledger.record_plan([good_item(), good_item(requirement_evidence="9.9")])
        ledger.note_write()
        ledger.note_validation("ctest", "passed")
        ledger.invalidate(reason="new evidence")
        record = ledger.to_dict()

        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(record["plan_items"][0]["requirement"],
                         good_item()["requirement"])
        self.assertEqual(len(record["rejected_items"]), 1)
        self.assertEqual(len(record["replans"]), 1)

    def test_every_field_of_an_item_is_required(self):
        """The shape is the contract; a field quietly dropped is a gap lost."""
        self.assertEqual(
            GapItem.FIELDS,
            ("requirement", "requirement_evidence", "current_behaviour",
             "implementation_evidence", "gap", "correction", "validation"))
        self.assertEqual(len(GapItem().missing()), len(GapItem.FIELDS))


if __name__ == "__main__":
    unittest.main()
