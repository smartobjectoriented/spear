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
