"""A modal verb is not a normative force, and a printed role is not one either.

Three things were one thing. What a document CALLS a provision, what modal
verbs appear inside it, and what it may be used as evidence for are separate
facts, and conflating them fails in both directions:

* an Observation whose prose says "must" was read as a binding requirement,
  so a turn concluded that the Observation itself imposed the obligation --
  a Rule elsewhere does, and the Observation is describing it;
* a Rule that says "should" would, on the opposite mistake, carry a "shall"
  because of what it is called.

Every fixture here is invented. Nothing in this file or in the code it
exercises knows any particular standard: which ceiling belongs to which role
is declared by a document's own profile.
"""

from __future__ import annotations

import unittest

import normative_force as nf
import provision_identity as pi

UNIT = {"source_id": "std-" + "1" * 32, "section": "5.2", "page": 3,
        "content_type": "TEXT", "modality": "NONE"}


def provisions(text):
    """The records a unit of this text yields, as the runtime derives them."""
    return [record for record in pi.records_from_unit(dict(UNIT, text=text))
            if isinstance(record.key, pi.ProvisionKey)]


def force_of(text):
    return provisions(text)[0].effective_force


def name(level):
    return nf.FORCE_NAME[level]


class AnObservationDescribesAnObligationWithoutImposingOne(unittest.TestCase):
    """Case 1 and case 2. The modal is real; whose modal it is, is the point."""

    PLAIN = "Observation 5.2-3: The device must be configured before first use."
    REPORTING = ("Observation 5.2-4: Rule 5.2-1 requires the device to be "
                 "configured before first use.")

    def test_its_own_must_grounds_nothing_binding(self):
        self.assertEqual(name(force_of(self.PLAIN)), "informative")

    def test_the_must_is_not_discarded_only_reclassified(self):
        """The word is evidence about the document and may steer retrieval
        toward the provision that does impose it."""
        record = provisions(self.PLAIN)[0]

        self.assertEqual(record.lexical_modalities, ("must",))
        self.assertEqual([item.status for item in record.force.lexical],
                         [nf.REPORTED])

    def test_a_reported_requirement_says_why_it_is_reported(self):
        record = provisions(self.REPORTING)[0]

        self.assertEqual(name(record.effective_force), "informative")
        self.assertIn("subject of this statement",
                      " ".join(item.why for item in record.force.reported))

    def test_the_role_comes_from_the_label_not_the_words(self):
        self.assertEqual(provisions(self.PLAIN)[0].role, nf.OBSERVATION)


class TheRoleIsACeilingNotAFloor(unittest.TestCase):
    """Cases 3 to 6. Weak wording is not promoted by a strong role, and a
    strong word is not promoted by a weak one."""

    def test_a_recommendation_supports_a_recommendation(self):
        self.assertEqual(
            name(force_of("Recommendation 5.2-5: The implementation should retry.")),
            "recommendation")

    def test_a_recommendation_does_not_support_a_requirement(self):
        record = provisions("Recommendation 5.2-5: The implementation should retry.")[0]

        self.assertLess(record.effective_force, nf.REQUIREMENT_FORCE)

    def test_a_permission_supports_a_permission(self):
        self.assertEqual(
            name(force_of("Permission 5.2-6: The implementation may retry.")),
            "permission")

    def test_a_permission_does_not_support_a_recommendation(self):
        record = provisions("Permission 5.2-6: The implementation may retry.")[0]

        self.assertLess(record.effective_force, nf.RECOMMENDATION_FORCE)

    def test_a_rule_that_says_shall_supports_a_requirement(self):
        self.assertEqual(
            name(force_of("Rule 5.2-7: The implementation shall retry.")),
            "requirement")

    def test_a_rule_that_says_should_is_not_promoted_to_shall(self):
        """Being called a Rule is authority to impose, not a statement that
        this one did."""
        record = provisions("Rule 5.2-8: The implementation should retry.")[0]

        self.assertEqual(name(record.effective_force), "recommendation")
        self.assertEqual(name(record.normative_authority), "requirement")


class AQuotationIsNotAStatement(unittest.TestCase):
    """Case 7. Informative prose that quotes a requirement is reporting it."""

    QUOTING = ('Observation 5.2-9: The earlier clause reads "the device shall '
               'retry on failure" and is unchanged in this revision.')

    def test_the_quoted_shall_does_not_bind(self):
        self.assertEqual(name(force_of(self.QUOTING)), "informative")

    def test_the_quotation_is_named_as_the_reason(self):
        record = provisions(self.QUOTING)[0]

        self.assertIn("quotation",
                      " ".join(item.why for item in record.force.reported))

    def test_a_quotation_inside_a_rule_is_reported_too(self):
        """Authority is not the only signal: a Rule quoting some other text is
        still quoting."""
        record = provisions('Rule 5.2-10: Clause 4 reads "the device shall '
                            'retry" and applies here.')[0]

        self.assertLess(record.effective_force, nf.REQUIREMENT_FORCE)


class OneUnitMayCarryBoth(unittest.TestCase):
    """Case 8. The container is not the provision: only the Rule binds."""

    TEXT = ("Observation 5.2-11: The device must be configured before use. "
            "Rule 5.2-11: The device shall report its configuration.")

    def test_the_two_provisions_are_assessed_separately(self):
        found = {record.role: name(record.effective_force)
                 for record in provisions(self.TEXT)}

        self.assertEqual(found, {nf.OBSERVATION: "informative",
                                 nf.RULE: "requirement"})

    def test_only_one_of_them_can_ground_a_binding_claim(self):
        binding = [record for record in provisions(self.TEXT)
                   if record.effective_force >= nf.REQUIREMENT_FORCE]

        self.assertEqual([record.key.kind for record in binding], ["Rule"])


class UnlabelledNormativeTextIsNotDiscarded(unittest.TestCase):
    """Case 9. A document's normative prose is frequently unnumbered, and a
    ceiling on UNLABELLED would silently delete it."""

    BODY = dict(UNIT, content_type="REQUIREMENT",
                text="The device shall report its configuration on request.")

    def record(self):
        return pi.records_from_unit(dict(self.BODY))[0]

    def test_it_still_establishes_a_requirement(self):
        self.assertEqual(name(self.record().effective_force), "requirement")

    def test_its_role_is_unlabelled_rather_than_absent(self):
        self.assertEqual(self.record().role, nf.UNLABELLED)

    def test_a_table_row_keeps_its_authority(self):
        """A bit-assignment row IS the requirement a Rule points at."""
        row = pi.records_from_unit(dict(
            UNIT, text="20 ReqV Set to 1: a Validation Acknowledge shall be sent."))[0]

        self.assertEqual(name(row.effective_force), "requirement")


class ATaxonomyIsDeclaredNotAssumed(unittest.TestCase):
    """What a role MEANS is a property of a document. Another standard may
    give the same word a different weight, and the guard must not care."""

    def test_a_standard_may_declare_its_own_ceilings(self):
        odd = nf.RoleTaxonomy(name="odd",
                              ceilings={nf.OBSERVATION: nf.REQUIREMENT_FORCE})
        found = nf.assess("The device must be configured.",
                          role=nf.OBSERVATION, taxonomy=odd)

        self.assertEqual(name(found.effective), "requirement")

    def test_the_default_is_the_conventional_reading(self):
        self.assertEqual(nf.taxonomy_for("a-standard-nobody-declared"),
                         nf.CONVENTIONAL)

    def test_a_declared_taxonomy_is_found_by_standard_id(self):
        odd = nf.RoleTaxonomy(name="odd")
        nf.register_taxonomy("TEST-STANDARD-1", odd)

        self.assertIs(nf.taxonomy_for("TEST-STANDARD-1"), odd)

    def test_an_unknown_role_is_not_silenced(self):
        """A role nobody declared is a role whose semantics nobody wrote
        down. Treating it as informative would discard normative text on the
        strength of an unrecognised word."""
        found = nf.assess("The device shall retry.", role="SOMETHING-NEW")

        self.assertEqual(name(found.effective), "requirement")


if __name__ == "__main__":
    unittest.main()


class AReportingVerbMustActuallyReport(unittest.TestCase):
    """The signal is a verb WITH its clause, not a word that can be a verb.

    Matching the bare verbs read the noun "state" as the verb "states", so
    "State and Event Indicators and Enable bits shall be positioned ..." was
    a reported requirement and 45 ordinary Rules of one document grounded
    nothing at all. A guard that silently demotes half a standard is worse
    than the promotion it was written to stop.
    """

    def test_a_noun_that_looks_like_a_reporting_verb_is_not_one(self):
        record = provisions("Rule 5.2-12: State and Event Indicators and "
                            "Enable bits shall be positioned as shown.")[0]

        self.assertEqual(name(record.effective_force), "requirement")

    def test_another_noun_in_the_same_family(self):
        record = provisions("Rule 5.2-13: Bits 15 to 10 of the Sea State "
                            "field shall be User Defined.")[0]

        self.assertEqual(name(record.effective_force), "requirement")

    def test_a_rule_citing_a_section_still_imposes(self):
        """Citational language attaches to a reference. A provision that
        cites a section while stating its own obligation is still stating
        it."""
        record = provisions("Rule 5.2-14: The field shall be encoded in "
                            "accordance with Section 4.1.")[0]

        self.assertEqual(name(record.effective_force), "requirement")

    def test_the_verb_with_its_clause_still_reports(self):
        record = provisions("Rule 5.2-15: Section 4 states that the field "
                            "shall be encoded as an integer.")[0]

        self.assertEqual(name(record.effective_force), "informative")

    def test_a_cited_provision_that_is_only_a_modifier_still_imposes(self):
        """"for Rule 11-1 shall be" cites a rule and then states this one's
        own obligation. Demoting on the citation's presence alone took 45
        ordinary rules of one document down with it."""
        record = provisions("Rule 5.2-16: The representation of the size for "
                            "Rule 5.2-1 shall be two consecutive words.")[0]

        self.assertEqual(name(record.effective_force), "requirement")

    def test_a_cited_provision_as_the_subject_reports(self):
        record = provisions("Observation 5.2-17: Rule 5.2-1 shall hold "
                            "unless a Timestamp Adjustment has been sent.")[0]

        self.assertEqual(name(record.effective_force), "informative")

    def test_an_attributed_condition_does_not_reach_the_obligation(self):
        """"If X indicates that Y, the field shall Z" attributes the
        CONDITION and then states this provision's own obligation. Letting
        the verb govern the whole sentence demoted six rules of one
        document."""
        record = provisions("Rule 5.2-18: If the code indicates that leap "
                            "seconds are not applicable, the field shall be "
                            "encoded as an integer.")[0]

        self.assertEqual(name(record.effective_force), "requirement")

    def test_a_permission_survives_an_attributed_condition_too(self):
        record = provisions("Permission 5.2-19: If the code indicates that "
                            "duplication is used, an implementation may "
                            "observe the second later.")[0]

        self.assertEqual(name(record.effective_force), "permission")
