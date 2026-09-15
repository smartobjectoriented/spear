"""A label is not always a declaration, and a colon is not always there.

The first version of this rule treated the colon as decisive: a labelled
provision followed by ":" declared it, anything else referred to it. Measured
on one bound standard that was right 1332 times and wrong three: Definition
3.5.1-7, Definition 3.5.1-8 and Observation 9.7-1 carry no colon anywhere,
and the rule discarded all three in silence. A provision that vanishes is
worse than one that needs review.

So classification uses several signals and has three outcomes, not two. What
it may never do is throw away a candidate it could not classify.
"""

from __future__ import annotations

import unittest

import provision_identity as pi


def classify(text):
    match = pi._LABEL.search(text)
    assert match, text
    return pi.classify_label(text, match)


class AColonDeclares(unittest.TestCase):
    def test_a_label_with_a_colon_is_a_declaration(self):
        self.assertEqual(classify("Rule 5.2-1: A Reply shall carry one tag."),
                         pi.DECLARATION)

    def test_a_declaration_at_the_start_of_a_unit(self):
        self.assertEqual(classify("Rule 5.2-1: A Reply shall stop."),
                         pi.DECLARATION)

    def test_a_declaration_after_other_prose(self):
        self.assertEqual(
            classify("The regulations follow. Rule 5.2-1: A Reply shall stop."),
            pi.DECLARATION)


class CitationLanguageRefers(unittest.TestCase):
    def test_a_label_after_a_preposition_is_a_reference(self):
        for lead in ("as required by", "pursuant to", "see", "in violation of",
                     "according to", "governed by", "together with"):
            with self.subTest(lead=lead):
                self.assertEqual(classify(f"The reply is bounded {lead} "
                                          "Rule 5.2-1 and stops there."),
                                 pi.REFERENCE)

    def test_a_label_reporting_what_a_provision_does_is_a_reference(self):
        """"Rule 5.2-1 says that ..." points at a provision. A provision does
        not open by reporting itself."""
        for verb in ("says", "implies", "requires", "establishes", "permits",
                     "differs", "applies", "and"):
            with self.subTest(verb=verb):
                self.assertEqual(classify(f"Rule 5.2-1 {verb} the reply is one."),
                                 pi.REFERENCE)

    def test_a_mid_sentence_label_is_a_reference(self):
        self.assertEqual(classify("the reply Rule 5.2-1 bounded"), pi.REFERENCE)


class ADroppedColonIsKeptNotDiscarded(unittest.TestCase):
    """The failure this exists to prevent."""

    def test_a_declaration_whose_colon_was_dropped_is_a_candidate(self):
        self.assertEqual(
            classify("The data is named herein. Definition 3.5.1-7 Extension "
                     "Data are data representing information."),
            pi.CANDIDATE)

    def test_a_candidate_is_not_discarded(self):
        unit = {"source_id": "std-" + "a" * 32, "section": "3.5.1", "page": 46,
                "modality": "NONE", "content_type": "EXAMPLE",
                "text": "So named. Definition 3.5.1-7 Extension Data are data "
                        "representing any information."}
        records = pi.records_from_unit(unit)

        self.assertEqual([record.key for record in records][:1],
                         [pi.ProvisionKey("3.5.1", pi.DEFINITION, 7)])
        self.assertEqual(records[0].declaration_status, pi.CANDIDATE)

    def test_a_reference_is_not_derived_as_a_provision(self):
        unit = {"source_id": "std-" + "b" * 32, "section": "5.2", "page": 9,
                "modality": "SHALL", "content_type": "REQUIREMENT",
                "text": "The reply is bounded pursuant to Rule 5.2-1 and stops."}
        keys = [record.key for record in pi.records_from_unit(unit)
                if isinstance(record.key, pi.ProvisionKey)]

        self.assertEqual(keys, [])


class ACandidateMayNotGroundAClaim(unittest.TestCase):
    """Preserved and surfaced, but never promoted into evidence."""

    UNIT = {"source_id": "std-" + "c" * 32, "section": "3.5.1", "page": 46,
            "modality": "NONE", "content_type": "EXAMPLE",
            "text": "So named. Definition 3.5.1-7 Extension Data are data."}
    DECLARED = {"source_id": "std-" + "d" * 32, "section": "5.2", "page": 9,
                "modality": "SHALL", "content_type": "REQUIREMENT",
                "text": "Rule 5.2-1: A Reply shall carry one tag."}

    def ledger(self, *units):
        found = pi.ProvisionLedger()
        for unit in units:
            found.observe(unit)
        return found

    def test_a_candidate_does_not_satisfy_has(self):
        found = self.ledger(self.UNIT)

        self.assertFalse(found.has(pi.ProvisionKey("3.5.1", pi.DEFINITION, 7)))

    def test_a_candidate_does_not_resolve_a_citation(self):
        found = self.ledger(self.UNIT)

        self.assertIsNone(found.resolve("Definition 3.5.1-7"))

    def test_a_declaration_does_satisfy_has(self):
        found = self.ledger(self.DECLARED)

        self.assertTrue(found.has(pi.ProvisionKey("5.2", pi.RULE, 1)))

    def test_a_candidate_is_exposed_for_diagnostics(self):
        """Keyed by INSTANCE: a candidate is one declaration among possibly
        several sharing a printed label, so the diagnostic names which."""
        found = self.ledger(self.UNIT, self.DECLARED)
        candidates = found.candidates()

        self.assertEqual([record.citation_key for record in candidates.values()],
                         [pi.ProvisionKey("3.5.1", pi.DEFINITION, 7)])
        self.assertTrue(all(isinstance(identity, pi.ProvisionInstanceId)
                            for identity in candidates))


class TheExtractionBoundaryCase(unittest.TestCase):
    def test_a_label_split_from_its_body_still_yields_a_record(self):
        """The fragment declares the provision even where the body is absent;
        it is a candidate rather than nothing."""
        unit = {"source_id": "std-" + "e" * 32, "section": "9.7", "page": 205,
                "modality": "NONE", "content_type": "REQUIREMENT",
                "text": "At the Reference Point. Observation 9.7-1 The "
                        "Timestamp Adjustment is negative."}
        records = [r for r in pi.records_from_unit(unit)
                   if isinstance(r.key, pi.ProvisionKey)]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].declaration_status, pi.CANDIDATE)


if __name__ == "__main__":
    unittest.main()


class ASentenceRunningThroughALabelIsAReference(unittest.TestCase):
    """A document mentions a provision mid-clause in ways a colon rule and a
    verb list both miss: a comma straight after the label, or a word closed up
    against it with no space. Neither declares anything, and treating them as
    declarations invented a reused printed citation and a candidate."""

    def test_a_comma_after_the_label_is_a_reference(self):
        self.assertEqual(
            classify("Permission 5.2-1, coupled with the other rules, applies."),
            pi.REFERENCE)

    def test_a_word_closed_up_against_the_label_is_a_reference(self):
        self.assertEqual(classify("Permission 5.2-1permits the repetition of it."),
                         pi.REFERENCE)

    def test_a_reporting_have_is_a_reference(self):
        self.assertEqual(classify("Rule 5.2-12 has four possible readings."),
                         pi.REFERENCE)

    def test_a_colon_declaration_is_untouched(self):
        self.assertEqual(classify("Permission 5.2-1: A Request may name any tag."),
                         pi.DECLARATION)

    def test_a_dropped_colon_declaration_is_still_a_candidate(self):
        self.assertEqual(
            classify("So named. Definition 5.2-7 Extension Data are data."),
            pi.CANDIDATE)


class ADeclarationBesideItsReferenceIsOneProvision(unittest.TestCase):
    """One printed label, one declaration and one mention: one instance, and
    no ambiguity for a citation to resolve."""

    DECL = {"source_id": "std-" + "7" * 32, "section": "5.2", "page": 9,
            "content_type": "REQUIREMENT", "modality": "MAY",
            "text": "Permission 5.2-1: A Request may name any tag."}
    REF = {"source_id": "std-" + "8" * 32, "section": "5.2", "page": 9,
           "content_type": "TEXT", "modality": "NONE",
           "text": "Permission 5.2-1, coupled with the other rules, applies here."}

    def ledger(self):
        found = pi.ProvisionLedger()
        found.observe(self.DECL); found.observe(self.REF)
        return found

    def test_only_the_declaration_becomes_a_provision(self):
        key = pi.ProvisionKey("5.2", pi.PERMISSION, 1)

        self.assertEqual(len(self.ledger().instances_for(key)), 1)

    def test_the_citation_resolves_without_ambiguity(self):
        found = self.ledger()

        self.assertEqual(found.resolve("Permission 5.2-1"),
                         pi.ProvisionKey("5.2", pi.PERMISSION, 1))

    def test_the_reference_text_survives_as_evidence(self):
        """Nothing is discarded: the sentence is still a unit, it simply
        declares no provision."""
        records = pi.records_from_unit(self.REF)
        provisions = [r for r in records if isinstance(r.key, pi.ProvisionKey)
                      and r.key.ordinal is not None]

        self.assertEqual(provisions, [])


class ASubLetteredOrdinalStillDeclares(unittest.TestCase):
    """The label regex stops at the digits, so a document that numbers a
    provision "5.2-1a" leaves a letter before the colon. Read as a run-on
    word that is the shape of a reference, and the provision would vanish."""

    def test_the_colon_after_the_suffix_is_still_decisive(self):
        self.assertEqual(classify("Rule 5.2-1a: A Request shall name one tag."),
                         pi.DECLARATION)

    def test_a_word_running_into_the_label_is_still_a_reference(self):
        self.assertEqual(classify("Permission 5.2-1permits the repetition."),
                         pi.REFERENCE)
