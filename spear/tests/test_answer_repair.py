"""Withholding is safe and it is not the goal.

A turn that retrieved the right provision, quoted it accurately and then
opened with the opposite conclusion has everything needed to answer. Refusing
it protects the reader and teaches nothing. So one rewrite is allowed -- and
the same guards run again on what comes back.

What these tests are really about is the four rules that keep the repair from
being a way around the gate.
"""

from __future__ import annotations

import unittest

import answer_repair as ar
import normative_claims as nc


def evidence(*units):
    found = nc.NormativeEvidence()
    for index, (text, modality, content_type, section) in enumerate(units):
        found.observe({"section": section, "source_id": f"std-{index:032d}",
                       "modality": modality, "content_type": content_type,
                       "text": text})
    return found


RULE = ("Rule 5.2-1: A Gadget Reply shall carry only one tag.",
        "SHALL", "REQUIREMENT", "5.2")
RECOMMEND = ("Recommendation 5.3-1: A Handler should retry once.",
             "SHOULD", "RECOMMENDATION", "5.3")
FLAGS = ("The field carries the TagA, TagB and TagC bits.", "NONE", "TABLE", "5.4")


class OnlySomeFailuresAreRepairable(unittest.TestCase):
    def test_a_contradiction_is_repairable(self):
        self.assertTrue(ar.is_repairable(
            [{"kind": nc.INCOHERENT_CONCLUSION}]))

    def test_strengthening_and_a_bad_identifier_are_repairable(self):
        self.assertTrue(ar.is_repairable(
            [{"kind": nc.STRENGTHENED_MODALITY},
             {"kind": nc.UNGROUNDED_IDENTIFIER}]))

    def test_an_unsupported_bound_is_not_repairable(self):
        """A maximum nothing states is not a wording problem, and asking the
        model to rewrite it is how an abstention becomes an invention."""
        self.assertFalse(ar.is_repairable(
            [{"kind": nc.UNSUPPORTED_CARDINALITY}]))

    def test_a_mixed_set_containing_an_unrepairable_finding_is_refused(self):
        self.assertFalse(ar.is_repairable(
            [{"kind": nc.INCOHERENT_CONCLUSION},
             {"kind": nc.UNSUPPORTED_CARDINALITY}]))

    def test_no_findings_means_nothing_to_repair(self):
        self.assertFalse(ar.is_repairable([]))


class ThePromptConstrainsRatherThanAnswers(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence(RULE, RECOMMEND)
        self.problems = [{"kind": nc.INCOHERENT_CONCLUSION}]

    def prompt(self):
        return ar.prompt("May a Reply carry two tags?",
                         "Yes, a Reply may carry two tags.",
                         self.evidence, self.problems)

    def test_it_names_every_permitted_provision_with_its_modality(self):
        text = self.prompt()

        self.assertIn("Rule 5.2-1", text)
        self.assertIn("modality: SHALL", text)
        self.assertIn("Recommendation 5.3-1", text)
        self.assertIn("modality: SHOULD", text)

    def test_it_carries_the_rejected_conclusion_and_the_reason(self):
        text = self.prompt()

        self.assertIn("Yes, a Reply may carry two tags.", text)
        self.assertIn("contradicts", text)

    def test_it_forbids_further_retrieval_in_words(self):
        self.assertIn("cannot retrieve more", self.prompt())

    def test_it_offers_the_option_of_saying_the_evidence_does_not_settle_it(self):
        self.assertIn("do not settle", self.prompt())

    def test_it_does_not_contain_an_answer(self):
        """Nothing here knows the right answer, and the prompt must not
        pretend to."""
        text = self.prompt().lower()

        self.assertNotIn("the correct answer is", text)
        self.assertNotIn("you should answer", text)


class TheRepairIsBounded(unittest.TestCase):
    def test_an_unrepairable_finding_never_calls_the_model(self):
        calls = []

        result = ar.attempt("q", "a", evidence(FLAGS),
                            [{"kind": nc.UNSUPPORTED_CARDINALITY}],
                            ask=lambda text: calls.append(text) or "anything")

        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_exactly_one_call_is_made(self):
        calls = []

        ar.attempt("q", "a", evidence(RULE), [{"kind": nc.INCOHERENT_CONCLUSION}],
                   ask=lambda text: calls.append(text) or "No. Rule 5.2-1 applies.")

        self.assertEqual(len(calls), 1)

    def test_a_failing_model_withholds_rather_than_raising(self):
        def boom(_):
            raise RuntimeError("the endpoint is down")

        self.assertIsNone(ar.attempt("q", "a", evidence(RULE),
                                     [{"kind": nc.INCOHERENT_CONCLUSION}], ask=boom))

    def test_an_empty_repair_is_no_repair(self):
        self.assertIsNone(ar.attempt("q", "a", evidence(RULE),
                                     [{"kind": nc.INCOHERENT_CONCLUSION}],
                                     ask=lambda _: "   "))


class TheSameGuardsJudgeTheRepair(unittest.TestCase):
    """The repair is validated by the identical pipeline, not a relaxed one."""

    def test_a_good_repair_passes_the_guard_that_rejected_the_original(self):
        found = evidence(RULE)
        original = "Yes, a Reply may carry two tags. Rule 5.2-1 governs this."
        _, problems, fired = nc.guard(original, found,
                                      question="May a Reply carry two tags?")
        self.assertTrue(fired)

        repaired = ar.attempt("May a Reply carry two tags?", original, found,
                              problems,
                              ask=lambda _: "No. Rule 5.2-1 requires a Reply to "
                                            "carry only one tag.")
        _, again, fired_again = nc.guard(repaired, found,
                                         question="May a Reply carry two tags?")

        self.assertFalse(fired_again, again)

    def test_a_repair_that_repeats_the_defect_still_fails(self):
        found = evidence(RULE)
        problems = [{"kind": nc.INCOHERENT_CONCLUSION}]

        repaired = ar.attempt("May a Reply carry two tags?", "Yes.", found,
                              problems,
                              ask=lambda _: "Yes, two tags are fine. "
                                            "Rule 5.2-1 governs this.")
        _, _, fired = nc.guard(repaired, found,
                               question="May a Reply carry two tags?")

        self.assertTrue(fired)

    def test_a_repair_inventing_an_identifier_still_fails(self):
        found = evidence(RULE)

        repaired = ar.attempt("q", "Yes.", found,
                              [{"kind": nc.INCOHERENT_CONCLUSION}],
                              ask=lambda _: "No. Rule 5.2-1 and the TagQ bit apply.")
        _, problems, fired = nc.guard(repaired, found)

        self.assertTrue(fired)
        self.assertIn(nc.UNGROUNDED_IDENTIFIER,
                      {problem["kind"] for problem in problems})


if __name__ == "__main__":
    unittest.main()


class WhatWithheldAnAnswerIsReportable(unittest.TestCase):
    """A battery that cannot see which check fired can say a turn withheld
    and not why. The repair is the same: one constrained rewrite either
    happened or it did not, and "the answer changed" is not evidence of it.
    """

    def trace(self):
        import standard_answer_policy

        policy = standard_answer_policy.policy_for(
            {"standard_id": "ACME-1", "revision": "2030",
             "corpus_manifest_sha256": "0" * 64,
             "index_fingerprint": "1" * 64,
             "retrieval_fingerprint": "2" * 64},
            "How many replies are required?")
        policy.finalize("Up to three replies are required.", rounds=1)

        return policy.trace()

    def test_the_claims_guard_reports_whether_it_fired(self):
        self.assertIn("claims_guard_triggered", self.trace())

    def test_its_findings_are_reported_by_kind(self):
        found = self.trace()

        self.assertEqual(
            [item["kind"] for item in found["claim_findings"]],
            [] if not found["claims_guard_triggered"] else
            [item["kind"] for item in found["claim_findings"]])

    def test_the_repair_reports_both_attempt_and_outcome(self):
        """Attempted-and-refused is a different fact from never attempted,
        and an evaluation that conflates them cannot tell a guard that held
        from a rewrite that was never offered."""
        found = self.trace()

        self.assertIn("repair_attempted", found)
        self.assertIn("repair_accepted", found)

    def test_no_repair_is_attempted_without_a_way_to_ask(self):
        self.assertFalse(self.trace()["repair_attempted"])


class ARepairThatStopsAnsweringIsNotARepair(unittest.TestCase):
    """Passing the guards is not the test. A draft that asserts nothing
    passes every one of them.

    Measured: a turn quoted the governing rule verbatim and concluded
    correctly from it. One finding -- an unglossed field name in a
    parenthesis -- sent it to repair, and the rewrite replied that the
    provisions did not settle the question. Nothing was left to object to,
    so it was accepted, and a correct answer was replaced by a refusal over
    a hyphen.
    """

    ANSWERED = ("The bit shall be set to 0 when acknowledgement is wanted in "
                "all cases. Rule 4.2-1 states this directly.")
    DECLINED = ("The provisions above do not settle the question. None of "
                "them specifies a mandatory action.")
    WORDING_ONLY = [{"kind": nc.UNGROUNDED_IDENTIFIER, "identifier": "Not-Ack"}]

    def test_an_answering_draft_is_recognised(self):
        self.assertTrue(ar.answers(self.ANSWERED))

    def test_a_declining_draft_is_recognised(self):
        self.assertFalse(ar.answers(self.DECLINED))

    def test_a_declination_after_a_leading_caveat_is_still_a_declination(self):
        """The rewrite that prompted this led with a line of its own
        instructions before abandoning the answer."""
        self.assertFalse(ar.answers(
            "A permission is not a requirement.\n\n" + self.DECLINED))

    def test_an_answer_may_still_mention_what_is_unsettled_later(self):
        self.assertTrue(ar.answers(
            self.ANSWERED + " Whether a timeout applies is not established "
            "by these provisions."))

    def test_answering_to_declining_is_rejected(self):
        self.assertEqual(
            ar.outcome(self.ANSWERED, self.DECLINED, self.WORDING_ONLY),
            ar.REPAIR_WITHHELD)

    def test_answering_to_answering_is_accepted(self):
        self.assertEqual(
            ar.outcome(self.ANSWERED, self.ANSWERED, self.WORDING_ONLY),
            ar.REPAIR_ANSWERED)

    def test_a_draft_that_never_answered_may_still_decline(self):
        """Genuinely insufficient evidence must keep being allowed to say so."""
        self.assertEqual(
            ar.outcome(self.DECLINED, self.DECLINED, self.WORDING_ONLY),
            ar.REPAIR_FAILED)

    def test_an_empty_repair_is_a_failure_not_a_withhold(self):
        self.assertEqual(
            ar.outcome(self.ANSWERED, "", self.WORDING_ONLY),
            ar.REPAIR_FAILED)


class TheRepairPromptAsksForTheSmallestChange(unittest.TestCase):

    def prompt_for(self, problems):
        return ar.prompt("Does it apply?", "Rule 4.2-1 applies.",
                                    evidence(RULE), problems)

    def test_a_wording_only_finding_says_the_conclusion_stood(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "Not-Ack"}])

        self.assertIn("Your conclusion was not rejected", text)

    def test_a_conclusion_bearing_finding_does_not_say_that(self):
        """A conclusion the guards rejected must remain changeable."""
        text = self.prompt_for([{"kind": nc.STRENGTHENED_MODALITY,
                                 "claimed": "requirement",
                                 "supported": "recommendation",
                                 "sentence": "It is required."}])

        self.assertNotIn("Your conclusion was not rejected", text)

    def test_it_forbids_retreating_to_cannot_determine(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "Not-Ack"}])

        self.assertIn("do not settle the question", text)
        self.assertIn("Do NOT replace a supported answer", text)

    def test_it_still_permits_a_genuine_abstention(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "Not-Ack"}])

        self.assertIn("genuinely do not answer it", text)

    def test_it_forbids_strengthening_force(self):
        self.assertIn("Do not strengthen a provision's force",
                      self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                        "identifier": "Not-Ack"}]))

    def test_it_forbids_new_evidence(self):
        self.assertIn("you cannot retrieve more",
                      self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                        "identifier": "Not-Ack"}]))


class FindingsAreSortedByWhatTheyBearOn(unittest.TestCase):

    def test_wording_findings_leave_the_conclusion_alone(self):
        self.assertEqual(
            ar.PRESENTATION_ONLY,
            {nc.UNGROUNDED_IDENTIFIER, nc.AMBIGUOUS_CITATION})

    def test_conclusion_findings_may_change_it(self):
        self.assertEqual(
            ar.CONCLUSION_BEARING,
            {nc.INCOHERENT_CONCLUSION, nc.STRENGTHENED_MODALITY,
             nc.MISATTRIBUTED_FORCE})

    def test_every_repairable_finding_is_classified(self):
        """A new repairable finding must be placed deliberately, not default
        into whichever class happens to be checked first."""
        self.assertEqual(
            ar.REPAIRABLE,
            ar.PRESENTATION_ONLY | ar.CONCLUSION_BEARING)


class ARepairMustNotNameTheIdentifierAgain(unittest.TestCase):
    """An ungrounded name is not fixed by re-hyphenating it.

    Measured: a turn answered a scope question correctly and closed with an
    elaboration naming a field the evidence never mentions. The repair kept
    the elaboration, named the field again, and the turn withheld -- a
    correct conclusion lost to a sentence that was never needed.
    """

    FLAGGED = [{"kind": nc.UNGROUNDED_IDENTIFIER, "identifier": "SelD"}]
    SUPPORTED = ("No, Rule 5.2-1 does not constrain the request. It applies "
                 "to the reply.")

    def test_naming_it_again_is_refused(self):
        again = self.SUPPORTED + " It may set SelA, SelB and SelD."

        self.assertEqual(ar.outcome(self.SUPPORTED, again, self.FLAGGED),
                         ar.REPAIR_UNGROUNDED)

    def test_a_respelling_is_the_same_name(self):
        """"Sel-D" and "SelD" are one identifier; a rewrite that merely
        re-hyphenates has corrected nothing."""
        self.assertEqual(
            ar.outcome(self.SUPPORTED, self.SUPPORTED + " It may set Sel-D.",
                       self.FLAGGED),
            ar.REPAIR_UNGROUNDED)

    def test_dropping_the_elaboration_is_accepted(self):
        self.assertEqual(ar.outcome(self.SUPPORTED, self.SUPPORTED, self.FLAGGED),
                         ar.REPAIR_ANSWERED)

    def test_a_grounded_generic_category_is_accepted(self):
        generic = self.SUPPORTED + " It may set any combination of selectors."

        self.assertEqual(ar.outcome(self.SUPPORTED, generic, self.FLAGGED),
                         ar.REPAIR_ANSWERED)

    def test_another_ungrounded_synonym_is_still_caught_by_the_guards(self):
        """Not by this check -- it only knows the flagged name -- but the
        repaired draft is re-guarded, and a new ungrounded identifier is a
        new finding."""
        swapped = self.SUPPORTED + " It may set the SelE selector."
        found = nc.identifier_findings(swapped, evidence(RULE))

        self.assertIn(nc.UNGROUNDED_IDENTIFIER,
                      [item["kind"] for item in found])

    def test_deleting_the_conclusion_is_still_over_correction(self):
        self.assertEqual(
            ar.outcome(self.SUPPORTED,
                       "The provisions do not settle the question.",
                       self.FLAGGED),
            ar.REPAIR_WITHHELD)

    def test_an_essential_identifier_is_not_replaced_by_invention(self):
        """When the flagged name IS the answer, a rewrite that keeps it is
        refused and the turn withholds. What must not happen is a grounded-
        looking substitute being invented in its place."""
        essential = "The field is named SelD."

        self.assertEqual(ar.outcome(essential, "The field is named SelD.",
                                    self.FLAGGED),
                         ar.REPAIR_UNGROUNDED)


class TheRepairIsToldWhichNamesExist(unittest.TestCase):

    def prompt_for(self, problems, *units):
        return ar.prompt("Does it apply?", "Rule 5.2-1 applies to SelD.",
                         evidence(*(units or (FLAGS,))), problems)

    def test_the_grounded_spellings_are_shown(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "SelD"}])

        self.assertIn("TagA", text)
        self.assertIn("spelled as", text)

    def test_they_are_bounded_not_offered(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "SelD"}])

        self.assertIn("add none of these merely to be more explicit", text)

    def test_evidence_with_no_identifiers_says_so(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "SelD"}], RECOMMEND)

        self.assertIn("names no field identifiers", text)

    def test_a_different_finding_does_not_list_identifiers(self):
        text = self.prompt_for([{"kind": nc.INCOHERENT_CONCLUSION,
                                 "sentence": "Yes."}])

        self.assertNotIn("spelled as", text)

    def test_the_instruction_forbids_variants_and_synonyms(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "SelD"}])

        self.assertIn("variant spelling", text)
        self.assertIn("synonym", text)

    def test_it_says_what_to_do_when_the_name_is_essential(self):
        text = self.prompt_for([{"kind": nc.UNGROUNDED_IDENTIFIER,
                                 "identifier": "SelD"}])

        self.assertIn("evidence does not support the point", text)


class OneAttemptStillMeansOne(unittest.TestCase):

    def test_a_refused_repair_is_not_retried(self):
        asked = []

        def ask(text):
            asked.append(text)
            return "The provisions do not settle the question."

        ar.attempt("Q?", "Rule 5.2-1 applies.", evidence(RULE),
                   [{"kind": nc.UNGROUNDED_IDENTIFIER, "identifier": "SelD"}],
                   ask=ask)

        self.assertEqual(len(asked), 1)
