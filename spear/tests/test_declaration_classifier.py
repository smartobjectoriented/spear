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
