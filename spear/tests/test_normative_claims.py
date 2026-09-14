"""Retrieval is not agreement.

Every guard before this one asks whether supporting evidence was retrieved.
On one battery three failures passed all of them, because in each case the
right clause HAD been retrieved and was cited nearby:

  * an answer opened "Yes, ... may carry more than one", quoted the rule
    saying it shall carry only one, and concluded correctly two sentences on;
  * an answer named a field occurring nowhere in the document;
  * an answer asked for a maximum the document never states counted the
    request flags and reported their number.

The fixtures here are invented -- a SYNTH widget with selectors, a gadget
with flags -- because the checks are not about any standard. Where a real
identifier appears it is because the test is about identifier SHAPE, and a
shape test needs specimens.
"""

from __future__ import annotations

import unittest

import normative_claims as nc


def evidence(*units):
    """units: (text, modality, content_type, section)"""
    found = nc.NormativeEvidence()
    for text, modality, content_type, section in units:
        found.units.append(nc.EvidenceUnit(
            section=section, source_id="std-" + "0" * 32,
            modality=modality, content_type=content_type, text=text))
    return found


REQUIRE_ONE = ("Rule 4.2.1.1-2: A Widget Reply shall have only one of the "
               "selectors SelA, SelB and SelC set to 1.",
               "SHALL", "REQUIREMENT", "4.2.1.1")
RECOMMEND = ("Recommendation 4.3-1: A Widget Handler should behave similarly "
             "in Mode 1 and Mode 2.", "SHOULD", "RECOMMENDATION", "4.3")
PERMIT = ("Permission 4.4-1: A Widget Request may name any combination of "
          "selectors.", "MAY", "REQUIREMENT", "4.4")
INFORM = ("Observation 4.1-1: the constraint is on the reply, not the request.",
          "NONE", "INFORMATIVE", "4.1")


class EvidenceReadsItsOwnWords(unittest.TestCase):
    def test_stored_modality_is_used(self):
        self.assertEqual(evidence(REQUIRE_ONE).level(), nc.REQUIREMENT)
        self.assertEqual(evidence(RECOMMEND).level(), nc.RECOMMENDATION)

    def test_the_words_win_when_the_store_disagrees(self):
        """Units typed INFORMATIVE carry `shall`, and one typed REQUIREMENT
        holds a Permission. The type is an extraction artefact; the words are
        the document."""
        mistyped = evidence(("Rule 9-1: a Gadget shall emit one frame.",
                             "NONE", "INFORMATIVE", "9"))

        self.assertEqual(mistyped.level(), nc.REQUIREMENT)

    def test_a_permission_keeps_its_modality_despite_its_type(self):
        """The fixture is typed REQUIREMENT and its modality is MAY. The unit
        establishes a permission: the type does not promote it."""
        self.assertEqual(evidence(PERMIT).level(), nc.PERMISSION)


class AnInventedIdentifierIsRejected(unittest.TestCase):
    """The B5 fabrication: a field named in the answer and in no unit."""

    def test_an_identifier_absent_from_the_evidence_is_flagged(self):
        found = nc.identifier_findings(
            "The reply may also carry status indicators such as SelQ.",
            evidence(REQUIRE_ONE))

        self.assertEqual([item["identifier"] for item in found], ["SelQ"])

    def test_identifiers_the_evidence_uses_are_accepted(self):
        found = nc.identifier_findings(
            "Only one of SelA, SelB or SelC may be set.", evidence(REQUIRE_ONE))

        self.assertEqual(found, [])

    def test_ordinary_english_is_not_an_identifier(self):
        """A conservative shape: a capitalised word starting a sentence must
        never read as a field name."""
        prose = ("Yes. The Widget Reply is constrained. Only one selector is "
                 "set. However, Mode 1 and Mode 2 differ. See Section 4.")

        self.assertEqual(nc.identifier_findings(prose, evidence(REQUIRE_ONE)), [])

    def test_project_metadata_may_ground_an_identifier(self):
        found = nc.identifier_findings("The GadgetX register is read.",
                                       evidence(REQUIRE_ONE),
                                       permitted=("GadgetX",))

        self.assertEqual(found, [])

    def test_nothing_is_flagged_when_no_evidence_was_retrieved(self):
        """With no units, every identifier would be ungrounded; that is a
        retrieval problem and a different guard's business."""
        self.assertEqual(nc.identifier_findings("SelQ and SelZ.",
                                                nc.NormativeEvidence()), [])

    def test_the_real_shapes_this_was_written_for(self):
        """Specimens, not a standard: internal capital, or a hyphenated part."""
        for token in ("AckV", "ReqX", "SchX", "Req-V", "Ctrl-P"):
            with self.subTest(token=token):
                self.assertTrue(nc._IDENTIFIER.fullmatch(token), token)

        for word in ("Yes", "The", "Acknowledge", "Controller", "However"):
            with self.subTest(word=word):
                self.assertIsNone(nc._IDENTIFIER.fullmatch(word), word)

    def test_a_bare_acronym_is_deliberately_not_policed(self):
        """An acronym is defined once, in a clause this turn probably did not
        retrieve. The first version flagged CAM on an otherwise correct answer
        and withheld it -- a worse failure than missing a fabricated acronym
        would be. So the shape is narrowed to field names on purpose."""
        for acronym in ("CAM", "NACK", "CIF"):
            with self.subTest(acronym=acronym):
                self.assertIsNone(nc._IDENTIFIER.fullmatch(acronym), acronym)

        self.assertEqual(
            nc.identifier_findings("The CAM field carries NACK.",
                                   evidence(REQUIRE_ONE)), [])


class ModalityIsNotStrengthened(unittest.TestCase):
    """The E1 failure: a recommendation answered as an obligation."""

    def test_a_recommendation_may_not_become_a_requirement(self):
        found = nc.modality_findings(
            "Yes, a Widget Handler is required to behave the same way.",
            evidence(RECOMMEND), question="Is a Widget Handler required to?")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["claimed"], "requirement")
        self.assertEqual(found[0]["supported"], "recommendation")

    def test_a_bare_yes_inherits_the_question_s_modality(self):
        """No modal word appears in the answer at all; the promotion is in
        answering yes to a question about obligation."""
        found = nc.modality_findings("Yes.", evidence(RECOMMEND),
                                     question="Is this behaviour required?")

        self.assertEqual(found[0]["claimed"], "requirement")

    def test_expected_to_is_a_requirement_claim(self):
        found = nc.modality_findings("A Handler is expected to do so.",
                                     evidence(RECOMMEND), question="")

        self.assertEqual(len(found), 1)

    def test_restating_the_recommendation_is_allowed(self):
        found = nc.modality_findings(
            "A Widget Handler should behave the same way.", evidence(RECOMMEND),
            question="Is a Widget Handler required to?")

        self.assertEqual(found, [])

    def test_a_permission_may_not_become_a_recommendation(self):
        found = nc.modality_findings("A Request should name one selector.",
                                     evidence(INFORM), question="")

        self.assertEqual(found[0]["claimed"], "recommendation")

    def test_a_requirement_conclusion_on_requirement_evidence_passes(self):
        found = nc.modality_findings("A Reply shall set only one selector.",
                                     evidence(REQUIRE_ONE), question="")

        self.assertEqual(found, [])


class TheConclusionMayNotContradictTheEvidence(unittest.TestCase):
    """The B5 shape: right quotation, wrong opening."""

    def test_an_affirmative_opening_over_restrictive_evidence_is_caught(self):
        """The answer names the rule it rests on, as a real answer does, and
        that rule says the opposite of its opening."""
        found = nc.coherence_findings(
            "Yes, a Widget Reply may carry more than one selector. "
            "Rule 4.2.1.1-2 governs this.",
            evidence(REQUIRE_ONE), question="May it carry more than one?")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], nc.INCOHERENT_CONCLUSION)

    def test_an_answer_that_reverses_itself_is_caught_without_evidence(self):
        """Later prose containing the correct statement does not repair an
        opening that says the opposite."""
        answer = ("Yes, more than one selector may be set. Rule 4.2.1.1-2 "
                  "states the reply shall have only one selector set. So a "
                  "single reply shall not carry more than one.")
        found = nc.coherence_findings(answer, nc.NormativeEvidence())

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["source"], "self")

    def test_a_denial_over_restrictive_evidence_is_correct(self):
        found = nc.coherence_findings(
            "No. A Widget Reply shall carry only one selector.",
            evidence(REQUIRE_ONE))

        self.assertEqual(found, [])

    def test_an_affirmative_over_permissive_evidence_is_correct(self):
        found = nc.coherence_findings(
            "Yes, a Widget Request may name any combination.", evidence(PERMIT))

        self.assertEqual(found, [])

    def test_prose_without_a_yes_or_no_opening_is_left_alone(self):
        found = nc.coherence_findings(
            "A Widget Reply carries one selector.", evidence(REQUIRE_ONE))

        self.assertEqual(found, [])


class StructuralEnumerationIsNotCardinality(unittest.TestCase):
    """The G1 failure: three flags counted into a maximum of three."""

    FLAGS = ("The Request field carries the SelA, SelB and SelC bits.",
             "NONE", "TABLE", "4.5")
    BOUND = ("Rule 4.6-1: a Handler shall generate at most two replies.",
             "SHALL", "REQUIREMENT", "4.6")

    def test_counting_flags_does_not_establish_a_maximum(self):
        found = nc.cardinality_findings(
            "The maximum number of replies is three.", evidence(self.FLAGS),
            question="What maximum number of replies may a Handler generate?")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["asserted"], "three")

    def test_a_clause_that_states_a_bound_supports_one(self):
        found = nc.cardinality_findings(
            "The maximum is two.", evidence(self.BOUND),
            question="What is the maximum number of replies?")

        self.assertEqual(found, [])

    def test_a_question_that_asks_for_no_bound_is_not_policed(self):
        found = nc.cardinality_findings(
            "The reply carries one selector.", evidence(self.FLAGS),
            question="Which selector does the reply carry?")

        self.assertEqual(found, [])

    def test_an_answer_that_states_no_number_is_not_policed(self):
        found = nc.cardinality_findings(
            "The standard states no maximum.", evidence(self.FLAGS),
            question="What is the maximum number of replies?")

        self.assertEqual(found, [])


class TheGateWithholdsRatherThanShows(unittest.TestCase):
    def test_a_clean_answer_passes_through_unchanged(self):
        answer = "A Widget Reply shall set only one of SelA, SelB or SelC."
        out, problems, fired = nc.guard(answer, evidence(REQUIRE_ONE))

        self.assertEqual(out, answer)
        self.assertEqual(problems, [])
        self.assertFalse(fired)

    def test_a_failing_answer_is_replaced_and_says_why(self):
        out, problems, fired = nc.guard(
            "Yes, a Widget Reply may carry more than one selector such as "
            "SelQ. Rule 4.2.1.1-2 governs this.",
            evidence(REQUIRE_ONE), question="May it carry more than one?")

        self.assertTrue(fired)
        self.assertIn("withheld", out)
        # The note names the offending identifier -- that is the point of it --
        # but the ANSWER's claim about it is gone.
        self.assertIn("SelQ: named in the answer", out)
        self.assertNotIn("may carry more than one", out)
        self.assertIn("§4.2.1.1", out)
        self.assertGreaterEqual(len(problems), 2)

    def test_the_principal_conclusion_skips_harness_prefaces(self):
        answer = ("No source file was read this turn: nothing in this answer "
                  "assesses the implementation.\nYes, it may.")

        self.assertEqual(nc.principal(answer), "Yes, it may.")


if __name__ == "__main__":
    unittest.main()


class ModalityIsJudgedAgainstWhatIsCited(unittest.TestCase):
    """Taking the strongest retrieved unit let a recommendation be promoted.

    A turn that reads a whole section reads its `shall` rules and its `should`
    recommendation together. If the bar is the maximum, every requirement
    claim in that turn is licensed -- including one about the recommendation.
    """

    SECTION = (("Rule 4.3-1: A Handler shall parse the request.",
                "SHALL", "REQUIREMENT", "4.3"),
               ("Recommendation 4.3-2: A Handler should behave similarly in "
                "Mode 1 and Mode 2.", "SHOULD", "RECOMMENDATION", "4.3"))

    def test_a_requirement_claim_citing_the_recommendation_is_caught(self):
        found = nc.modality_findings(
            "Yes, per Recommendation 4.3-2 a Handler is required to.",
            evidence(*self.SECTION), question="Is a Handler required to?")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["supported"], "recommendation")

    def test_a_requirement_claim_citing_the_rule_passes(self):
        found = nc.modality_findings(
            "Per Rule 4.3-1 a Handler shall parse the request.",
            evidence(*self.SECTION), question="")

        self.assertEqual(found, [])


class ABoundMustNameItsOwnValue(unittest.TestCase):
    """"only one" in a neighbouring rule is a bound of one, not of three."""

    MIXED = (("Rule 5.1-1: A Reply shall carry only one selector.",
              "SHALL", "REQUIREMENT", "5.1"),
             ("The field carries the SelA, SelB and SelC bits.",
              "NONE", "TABLE", "5.2"))

    def test_a_neighbouring_bound_of_one_does_not_support_three(self):
        found = nc.cardinality_findings(
            "The maximum number of replies is three.", evidence(*self.MIXED),
            question="What is the maximum number of replies?")

        self.assertEqual(len(found), 1)

    def test_a_bound_naming_the_claimed_value_supports_it(self):
        found = nc.cardinality_findings(
            "The maximum is one.", evidence(*self.MIXED),
            question="What is the maximum number of selectors?")

        self.assertEqual(found, [])


class ANumberAloneDoesNotIdentifyAProvision(unittest.TestCase):
    """A document may number its Rules and its Recommendations independently
    inside one section, so `Recommendation 4.3-2` and `Rule 4.3-2` are two
    provisions with the same number and different modality. Matching on the
    number alone pulled the Rule's unit in beside the Recommendation's and
    promoted a `should` to a `shall`.
    """

    BOTH = (("Rule 4.3-2: A Handler shall parse the request.",
             "SHALL", "REQUIREMENT", "4.3"),
            ("Recommendation 4.3-2: A Handler should behave similarly in "
             "Mode 1 and Mode 2.", "SHOULD", "RECOMMENDATION", "4.3"))

    def test_the_kind_is_kept_when_the_answer_gives_it(self):
        self.assertEqual(nc._citations("per **Recommendation 4.3-2** (p.9)"),
                         ["Recommendation 4.3-2"])

    def test_a_bare_number_is_kept_bare(self):
        self.assertEqual(nc._citations("see 4.3.1.2"), ["4.3.1.2"])

    def test_citing_the_recommendation_does_not_reach_the_rule(self):
        found = nc.modality_findings(
            "Yes, per Recommendation 4.3-2 a Handler is required to.",
            evidence(*self.BOTH), question="Is a Handler required to?")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["supported"], "recommendation")

    def test_citing_the_rule_reaches_the_rule(self):
        found = nc.modality_findings(
            "Per Rule 4.3-2 a Handler shall parse the request.",
            evidence(*self.BOTH), question="")

        self.assertEqual(found, [])


class AffirmingOnPermissiveEvidenceIsAllowed(unittest.TestCase):
    """The first version of the coherence check withheld two correct answers.

    A turn that reads a whole chapter reads restrictive rules beside
    permissive ones. "Affirmative opening while SOME retrieved rule
    restricts something" refused an answer about what a request may ask for
    because of a rule about what the reply may carry. So the check fires
    only when every provision the answer CITES is restrictive.
    """

    MIXED = (("Rule 4.2-1: A Reply shall carry only one selector.",
              "SHALL", "REQUIREMENT", "4.2"),
             ("Permission 4.4-1: A Request may name any combination of "
              "selectors.", "MAY", "REQUIREMENT", "4.4"))

    def test_an_answer_resting_on_the_permission_may_say_yes(self):
        found = nc.coherence_findings(
            "Yes. Permission 4.4-1 lets a Request name any combination.",
            evidence(*self.MIXED), question="May a Request name several?")

        self.assertEqual(found, [])

    def test_an_answer_resting_only_on_the_restriction_may_not(self):
        found = nc.coherence_findings(
            "Yes, a Reply may carry several. Rule 4.2-1 is the governing rule.",
            evidence(*self.MIXED), question="May a Reply carry several?")

        self.assertEqual(len(found), 1)

    def test_an_uncited_affirmation_is_left_to_the_self_check(self):
        found = nc.coherence_findings("Yes, it may.", evidence(*self.MIXED))

        self.assertEqual(found, [])


class OneFieldMaySpellItselfTwoWays(unittest.TestCase):
    """A document writes `Req-V` in a heading and `ReqV` in a table row, and
    an answer may pick either. Comparing literal spellings flagged three real
    fields as fabricated and withheld a correct answer."""

    SPELLINGS = ("The Controllee reads ReqV, ReqX and ReqS from the field.",
                 "NONE", "TABLE", "4.9")

    def test_the_hyphenated_spelling_is_grounded_by_the_joined_one(self):
        found = nc.identifier_findings(
            "The Controller sets Req-V, Req-X and Req-S.",
            evidence(self.SPELLINGS))

        self.assertEqual(found, [])

    def test_a_different_field_is_still_rejected(self):
        found = nc.identifier_findings("The Controller sets Req-Q.",
                                       evidence(self.SPELLINGS))

        self.assertEqual([item["identifier"] for item in found], ["Req-Q"])
