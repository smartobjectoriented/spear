"""A bare label is not a normative identity.

A standards document numbers each KIND of provision independently, so one
section and ordinal can name several provisions with different modality:

    Rule 5.2-1            shall ...
    Recommendation 5.2-1  should ...
    Permission 5.2-1      may ...

Measured on the bound standard: 308 of 806 bare labels carry more than one
kind, and one carries four. Everything that keyed identity on the bare label
was therefore wrong in more than a third of cases -- including a ledger whose
coverage test proved only that a SECTION had been retrieved. That is how a
retrieved `shall` Rule came to license a requirement claim about the `should`
Recommendation beside it.

The fixtures are a SYNTH gadget, invented. The collision is the point, not the
document it was found in.
"""

from __future__ import annotations

import unittest

import provision_identity as pi

SECTION = "5.2"

#: One unit carrying three provisions that share section and ordinal -- the
#: shape that makes a bare label useless. Stored SHALL, because the Rule in it
#: is what the extractor typed the container from.
COLLIDING_UNIT = {
    "source_id": "std-" + "a" * 32, "section": SECTION, "page": 42,
    "modality": "SHALL", "content_type": "REQUIREMENT",
    "text": ("The regulations below are with respect to the Gadget Request. "
             "Rule 5.2-1: A Gadget Reply shall carry one tag. "
             "Recommendation 5.2-1: A Gadget Handler should retry once. "
             "Permission 5.2-1: A Gadget Request may name any tag."),
}

OTHER_UNIT = {
    "source_id": "std-" + "b" * 32, "section": SECTION, "page": 43,
    "modality": "SHALL", "content_type": "REQUIREMENT",
    "text": "Rule 5.2-2: A Gadget Reply shall echo the request id.",
}


def ledger(*units):
    found = pi.ProvisionLedger()
    for unit in units:
        found.observe(unit)
    return found


class AKeyIsThreeThings(unittest.TestCase):
    def test_kind_distinguishes_otherwise_identical_keys(self):
        rule = pi.ProvisionKey(SECTION, pi.RULE, 1)
        recommendation = pi.ProvisionKey(SECTION, pi.RECOMMENDATION, 1)

        self.assertNotEqual(rule, recommendation)
        self.assertEqual(rule.bare_label, recommendation.bare_label)

    def test_the_bare_label_is_not_the_identity(self):
        """Both render the same careless citation."""
        self.assertEqual(pi.ProvisionKey(SECTION, pi.RULE, 1).bare_label, "5.2-1")

    def test_a_missing_ordinal_is_explicit_not_a_section(self):
        key = pi.ProvisionKey(SECTION, pi.SCOPE_PREAMBLE, None)

        self.assertIsNone(key.ordinal)
        self.assertNotEqual(key, pi.ProvisionKey(SECTION, pi.RULE, None))
        self.assertEqual(str(key), "ScopePreamble §5.2")

    def test_keys_are_hashable_and_orderable(self):
        keys = {pi.ProvisionKey(SECTION, pi.RULE, 1),
                pi.ProvisionKey(SECTION, pi.RULE, 1)}

        self.assertEqual(len(keys), 1)
        self.assertEqual(len(sorted([pi.ProvisionKey(SECTION, pi.RULE, 2),
                                     pi.ProvisionKey(SECTION, pi.RULE, 1)])), 2)


class OneUnitYieldsSeveralProvisions(unittest.TestCase):
    def test_three_colliding_provisions_are_three_records(self):
        records = {record.key for record in pi.records_from_unit(COLLIDING_UNIT)}

        for kind in (pi.RULE, pi.RECOMMENDATION, pi.PERMISSION):
            self.assertIn(pi.ProvisionKey(SECTION, kind, 1), records)

    def test_each_record_keeps_its_parent_unit(self):
        for record in pi.records_from_unit(COLLIDING_UNIT):
            self.assertEqual(record.source_id, COLLIDING_UNIT["source_id"])
            self.assertEqual(record.page, 42)

    def test_modality_comes_from_the_provision_not_the_container(self):
        """The unit is stored SHALL because a Rule sits in it. Inheriting that
        reports a permission as a requirement."""
        records = {record.key.kind: record
                   for record in pi.records_from_unit(COLLIDING_UNIT)}

        self.assertEqual(records[pi.RULE].modality, "SHALL")
        self.assertEqual(records[pi.RECOMMENDATION].modality, "SHOULD")
        self.assertEqual(records[pi.PERMISSION].modality, "MAY")
        self.assertEqual(records[pi.PERMISSION].unit_modality, "SHALL")

    def test_each_provision_hashes_its_own_text(self):
        records = {record.key.kind: record
                   for record in pi.records_from_unit(COLLIDING_UNIT)}
        digests = {kind: record.text_sha256 for kind, record in records.items()}

        self.assertEqual(len(set(digests.values())), len(digests))

    def test_the_lead_in_becomes_a_scope_preamble_not_a_rule(self):
        keys = [record.key for record in pi.records_from_unit(COLLIDING_UNIT)]
        preamble = [key for key in keys if key.kind == pi.SCOPE_PREAMBLE]

        self.assertEqual(len(preamble), 1)
        self.assertIsNone(preamble[0].ordinal)


class RetrievingOneDoesNotCoverAnother(unittest.TestCase):
    def test_the_rule_does_not_cover_the_recommendation(self):
        only_rule = ledger(OTHER_UNIT)

        self.assertTrue(only_rule.has(pi.ProvisionKey(SECTION, pi.RULE, 2)))
        self.assertFalse(only_rule.has(
            pi.ProvisionKey(SECTION, pi.RECOMMENDATION, 2)))

    def test_the_recommendation_does_not_cover_the_rule(self):
        one = {**OTHER_UNIT, "text": "Recommendation 5.2-9: A Handler should wait."}
        found = ledger(one)

        self.assertTrue(found.has(
            pi.ProvisionKey(SECTION, pi.RECOMMENDATION, 9)))
        self.assertFalse(found.has(pi.ProvisionKey(SECTION, pi.RULE, 9)))

    def test_section_coverage_is_weaker_and_says_so(self):
        """Something from the section was read. That is not the provision."""
        found = ledger(OTHER_UNIT)

        self.assertTrue(found.covers_section(SECTION))
        self.assertFalse(found.has(pi.ProvisionKey(SECTION, pi.RULE, 1)))


class ABareReferenceIsAmbiguous(unittest.TestCase):
    def setUp(self):
        self.found = ledger(COLLIDING_UNIT, OTHER_UNIT)

    def test_a_bare_colliding_label_raises_rather_than_picking(self):
        with self.assertRaises(pi.AmbiguousReference) as caught:
            self.found.resolve("5.2-1")

        self.assertEqual(len(caught.exception.candidates), 3)

    def test_a_bare_unique_label_resolves(self):
        self.assertEqual(self.found.resolve("5.2-2"),
                         pi.ProvisionKey(SECTION, pi.RULE, 2))

    def test_an_explicit_rule_resolves_to_the_rule(self):
        self.assertEqual(self.found.resolve("Rule 5.2-1"),
                         pi.ProvisionKey(SECTION, pi.RULE, 1))

    def test_an_explicit_recommendation_resolves_to_the_recommendation(self):
        self.assertEqual(self.found.resolve("Recommendation 5.2-1"),
                         pi.ProvisionKey(SECTION, pi.RECOMMENDATION, 1))

    def test_the_two_explicit_forms_differ(self):
        self.assertNotEqual(self.found.resolve("Rule 5.2-1"),
                            self.found.resolve("Recommendation 5.2-1"))

    def test_a_provision_never_retrieved_resolves_to_nothing(self):
        self.assertIsNone(self.found.resolve("Rule 9.9-9"))

    def test_references_in_prose_separate_resolved_from_ambiguous(self):
        resolved, ambiguous = self.found.references_in(
            "Per Rule 5.2-1 and 5.2-1 and Rule 5.2-2 the reply is bounded.")

        self.assertIn(pi.ProvisionKey(SECTION, pi.RULE, 1), resolved)
        self.assertIn(pi.ProvisionKey(SECTION, pi.RULE, 2), resolved)
        self.assertEqual(len(ambiguous), 1)

    def test_a_bare_section_is_not_read_as_a_citation(self):
        resolved, ambiguous = self.found.references_in("See section 5.2 for more.")

        self.assertEqual((resolved, ambiguous), ([], []))


class TheCollisionIsDetectable(unittest.TestCase):
    def test_kinds_sharing_one_ordinal_are_reported(self):
        found = ledger(COLLIDING_UNIT)

        self.assertEqual(found.kinds_for(SECTION, 1),
                         [pi.PERMISSION, pi.RECOMMENDATION, pi.RULE])

    def test_a_unique_ordinal_reports_one_kind(self):
        self.assertEqual(ledger(OTHER_UNIT).kinds_for(SECTION, 2), [pi.RULE])


if __name__ == "__main__":
    unittest.main()
