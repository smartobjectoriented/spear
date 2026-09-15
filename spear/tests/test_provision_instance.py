"""The printed label is what the document calls a provision, not its identity.

VITA 49.2 declares Rule 5.1.6-2 twice in section 5.1.6 -- page 75 about
Enable bits, page 76 about Packet Class documentation. Same section, same
heading path, same printed label, both proper declarations, different
normative content. Thirty-seven such labels exist in that one document.

So ProvisionKey stays a faithful record of the printed citation and stops
being treated as unique, and ProvisionInstanceId identifies the declaration.
The fixtures below are invented; the collision is not.
"""

from __future__ import annotations

import unittest

import provision_identity as pi

SECTION, BINDING = "5.2", {"standard_id": "SYNTH-1", "revision": "2026"}


def unit(source_id, text, *, page=1, content_type="REQUIREMENT", modality="SHALL"):
    return {"source_id": source_id, "section": SECTION, "page": page,
            "content_type": content_type, "modality": modality, "text": text,
            **BINDING}


#: One printed label, two declarations, different content.
A = unit("std-" + "a" * 32, "Rule 5.2-1: A Gadget Reply shall carry one tag.", page=75)
B = unit("std-" + "b" * 32, "Rule 5.2-1: The Class documentation shall specify "
                            "the tag width.", page=76)
UNIQUE = unit("std-" + "c" * 32, "Rule 5.2-9: A Gadget Reply shall stop.", page=77)


def ledger(*units):
    found = pi.ProvisionLedger()
    for item in units:
        found.observe(item)
    return found


def record(unit_dict):
    return pi.records_from_unit(unit_dict)[0]


class OneLabelTwoInstances(unittest.TestCase):
    def test_they_share_a_citation_key(self):
        self.assertEqual(record(A).citation_key, record(B).citation_key)
        self.assertEqual(str(record(A).citation_key), "Rule 5.2-1")

    def test_they_have_different_instance_ids(self):
        self.assertNotEqual(record(A).instance_id, record(B).instance_id)

    def test_an_instance_id_is_deterministic(self):
        self.assertEqual(record(A).instance_id, record(A).instance_id)

    def test_an_instance_id_changes_when_the_declaration_changes(self):
        moved = dict(A, text="Rule 5.2-1: A Gadget Reply shall carry two tags.")

        self.assertNotEqual(record(A).instance_id, record(moved).instance_id)

    def test_identical_text_in_two_units_is_two_instances(self):
        """Same words, different provenance: still two declarations until a
        human says which is the duplicate."""
        twin = dict(A, source_id="std-" + "d" * 32, page=80)

        self.assertNotEqual(record(A).instance_id, record(twin).instance_id)


class ARepeatedLabelIsAmbiguous(unittest.TestCase):
    def setUp(self):
        self.both = ledger(A, B)
        self.one = ledger(UNIQUE)

    def test_the_printed_label_alone_does_not_resolve(self):
        with self.assertRaises(pi.AmbiguousReference):
            self.both.resolve("Rule 5.2-1")

    def test_a_unique_label_still_resolves(self):
        self.assertEqual(self.one.resolve("Rule 5.2-9"),
                         pi.ProvisionKey(SECTION, pi.RULE, 9))

    def test_a_repeated_label_cannot_ground_a_claim(self):
        """`has` is the grounding question and must say no; `covers_citation`
        reports the weaker fact that something with that label was read."""
        key = pi.ProvisionKey(SECTION, pi.RULE, 1)

        self.assertFalse(self.both.has(key))
        self.assertTrue(self.both.covers_citation(key))

    def test_a_unique_label_grounds_normally(self):
        key = pi.ProvisionKey(SECTION, pi.RULE, 9)

        self.assertTrue(self.one.has(key))


class RetrievingOneDoesNotEstablishTheOther(unittest.TestCase):
    def test_covers_instance_distinguishes_them(self):
        only_a = ledger(A)

        self.assertTrue(only_a.covers_instance(record(A).instance_id))
        self.assertFalse(only_a.covers_instance(record(B).instance_id))

    def test_instances_for_lists_every_declaration(self):
        found = ledger(A, B).instances_for(pi.ProvisionKey(SECTION, pi.RULE, 1))

        self.assertEqual(len(found), 2)
        self.assertEqual({item.page for item in found}, {75, 76})

    def test_modality_of_one_cannot_licence_the_other(self):
        """A permits, B requires; grounding must not borrow across them."""
        permissive = unit("std-" + "e" * 32,
                          "Rule 5.2-1: A Gadget Request may name any tag.",
                          page=75, modality="MAY")
        found = ledger(permissive, B)
        instances = found.instances_for(pi.ProvisionKey(SECTION, pi.RULE, 1))

        self.assertEqual({item.modality for item in instances}, {"MAY", "SHALL"})
        self.assertFalse(found.has(pi.ProvisionKey(SECTION, pi.RULE, 1)))


class ApprovalBindsToOneInstance(unittest.TestCase):
    def test_approving_one_does_not_approve_the_other(self):
        import approval_binding as ab

        approvals = ab.load({"approvals": [{
            "identity": {"section": SECTION, "provision_type": pi.RULE, "ordinal": 1},
            "carrying_source_id": A["source_id"],
            "unit_text_sha256": record(A).unit_text_sha256,
            "review_status": "HUMAN_APPROVED"}]})
        report = ab.verify(approvals, [record(A), record(B)])

        self.assertEqual(report[0]["status"], ab.HUMAN_APPROVED)
        self.assertEqual(report[0]["instance_id"], str(record(A).instance_id))

    def test_an_approval_that_cannot_pick_one_is_reported_not_guessed(self):
        import approval_binding as ab

        approvals = ab.load({"approvals": [{
            "identity": {"section": SECTION, "provision_type": pi.RULE, "ordinal": 1},
            "review_status": "HUMAN_APPROVED"}]})
        report = ab.verify(approvals, [record(A), record(B)])

        self.assertEqual(report[0]["status"], ab.AMBIGUOUS_INSTANCE)


class RenderingKeepsThePrintedLabel(unittest.TestCase):
    """A human citation must stay the document's own, with just enough
    locator to tell two of them apart. No renumbering, no opaque hash."""

    def test_the_default_rendering_is_the_printed_label(self):
        self.assertEqual(record(A).rendered_citation(), "Rule 5.2-1")

    def test_disambiguation_adds_a_page_not_a_new_number(self):
        self.assertEqual(record(A).rendered_citation(disambiguate=True),
                         "Rule 5.2-1 (p. 75)")
        self.assertEqual(record(B).rendered_citation(disambiguate=True),
                         "Rule 5.2-1 (p. 76)")

    def test_the_digest_is_not_the_user_facing_citation(self):
        self.assertNotIn("@", record(A).rendered_citation(disambiguate=True))


class APageSplitIsOneProvision(unittest.TestCase):
    """A declaration continued across an extraction boundary is ONE instance
    with two provenance spans, not two competing declarations."""

    #: The shape the extractor actually leaves: one provision's text broken
    #: across a page, both fragments carrying the same derived key because
    #: neither half re-declares the label.
    FIRST = unit("std-" + "f" * 32, "A Gadget Reply shall carry the tag, the "
                                    "width and the", page=90)
    SECOND = unit("std-" + "0" * 32, "count of the items it describes.", page=91)

    def test_the_two_fragments_become_one_instance(self):
        joined, keys = pi.reconstruct_page_splits(
            pi.records_from_unit(self.FIRST) + pi.records_from_unit(self.SECOND))
        rules = [r for r in joined if isinstance(r.key, pi.ProvisionKey)]

        self.assertEqual(len(rules), 1, [str(r.key) for r in rules])
        self.assertEqual(len(rules[0].spans), 2)
        self.assertEqual([span["page"] for span in rules[0].spans], [90, 91])
        self.assertEqual(keys, [rules[0].key])

    def test_a_second_declaration_is_not_absorbed_as_a_continuation(self):
        """Both fragments declare the label, so they are two provisions."""
        joined, keys = pi.reconstruct_page_splits(
            pi.records_from_unit(A) + pi.records_from_unit(B))

        self.assertEqual(len(joined), 2)
        self.assertEqual(keys, [])


class ACandidateKeepsItsInstanceAndItsStatus(unittest.TestCase):
    CAND = unit("std-" + "1" * 32, "So named. Definition 5.2-7 Extension Data "
                                   "are data representing information.",
                content_type="EXAMPLE", modality="NONE", page=46)

    def test_a_candidate_has_an_instance_id(self):
        found = [r for r in pi.records_from_unit(self.CAND)
                 if isinstance(r.key, pi.ProvisionKey)]

        self.assertEqual(found[0].declaration_status, pi.CANDIDATE)
        self.assertIsInstance(found[0].instance_id, pi.ProvisionInstanceId)

    def test_a_candidate_is_visible_but_grounds_nothing(self):
        found = ledger(self.CAND)
        key = pi.ProvisionKey(SECTION, pi.DEFINITION, 7)

        self.assertTrue(found.candidates())
        self.assertFalse(found.has(key))
        self.assertFalse(found.covers_citation(key))


class StructuralEvidenceStaysOutOfCitationResolution(unittest.TestCase):
    ROW = {"source_id": "std-" + "2" * 32, "section": SECTION, "page": 5,
           "content_type": "UNKNOWN", "modality": "NONE",
           "text": "20 TagV Tag Validation Set to 1: request it", **BINDING}

    def test_a_table_row_is_not_a_numbered_provision(self):
        found = pi.records_from_unit(self.ROW, table="5.2-1")[0]

        self.assertIsInstance(found.key, pi.TableRowKey)
        self.assertNotIsInstance(found.key, pi.ProvisionKey)

    def test_a_row_does_not_resolve_through_the_citation_resolver(self):
        self.assertIsNone(ledger(self.ROW).resolve("5.2-20"))


if __name__ == "__main__":
    unittest.main()


class ASnippetAndItsFullUnitAreOneInstance(unittest.TestCase):
    """The same declaration reaches a turn twice and not always whole.

    `search` returns a truncated snippet and `fetch` the full unit. Keyed on
    text alone those are two digests and so a false ambiguity -- which made a
    turn refuse a citation the document does not actually reuse.
    """

    FULL = unit("std-" + "3" * 32,
                "Rule 5.2-5: A Gadget Reply shall carry the tag and the width "
                "and the count of the items it describes.", page=30)
    SNIPPET = unit("std-" + "3" * 32,
                   "Rule 5.2-5: A Gadget Reply shall carry the tag", page=30)

    def test_two_readings_of_one_declaration_are_one_instance(self):
        found = ledger(self.SNIPPET, self.FULL)
        key = pi.ProvisionKey(SECTION, pi.RULE, 5)

        self.assertEqual(len(found.instances_for(key)), 1)
        self.assertTrue(found.has(key))

    def test_the_fuller_reading_wins(self):
        found = ledger(self.SNIPPET, self.FULL)
        record, = found.instances_for(pi.ProvisionKey(SECTION, pi.RULE, 5))

        self.assertIn("count of the items", record.text)

    def test_order_does_not_matter(self):
        first = ledger(self.FULL, self.SNIPPET)
        record, = first.instances_for(pi.ProvisionKey(SECTION, pi.RULE, 5))

        self.assertIn("count of the items", record.text)

    def test_the_same_label_in_a_DIFFERENT_unit_is_still_two(self):
        elsewhere = dict(self.FULL, source_id="std-" + "4" * 32, page=31)
        found = ledger(self.FULL, elsewhere)

        self.assertEqual(
            len(found.instances_for(pi.ProvisionKey(SECTION, pi.RULE, 5))), 2)
