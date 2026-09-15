"""Approval attaches to a provision, never to the unit that carried it.

One extracted unit carries a Permission, an Observation and a Rule sharing a
section and an ordinal. A reviewer who approved the Rule approved one of the
three; keying on the parent source_id -- or on the bare label all three would
render -- approves the other two by accident.

The fixtures are invented. The collision they reproduce is not.
"""

from __future__ import annotations

import unittest

import approval_binding as ab
import provision_identity as pi

UNIT = {
    "source_id": "std-" + "c" * 32, "section": "5.2", "page": 7,
    "modality": "SHALL", "content_type": "REQUIREMENT",
    "text": ("Rule 5.2-1: A Gadget Reply shall carry one tag. "
             "Permission 5.2-1: A Gadget Request may name any tag. "
             "Observation 5.2-1: the constraint is on the reply."),
}


def store(unit=UNIT):
    return {record.key: record for record in pi.records_from_unit(unit)}


def approval(kind="Rule", ordinal=1, source_id=UNIT["source_id"], digest=None):
    records = store()
    record = records[pi.ProvisionKey("5.2", kind, ordinal)]
    return {"approvals": [{
        "identity": {"section": "5.2", "provision_type": kind, "ordinal": ordinal},
        "carrying_source_id": source_id,
        "unit_text_sha256": digest if digest is not None else record.unit_text_sha256,
        "review_status": "HUMAN_APPROVED"}]}


class ApprovingOneDoesNotApproveItsNeighbours(unittest.TestCase):
    def test_the_rule_is_approved(self):
        report = ab.verify(ab.load(approval()), store())

        self.assertEqual([row["status"] for row in report], [ab.HUMAN_APPROVED])

    def test_the_permission_sharing_the_unit_stays_unreviewed(self):
        approvals, records = ab.load(approval()), store()

        self.assertEqual(
            ab.status_of(pi.ProvisionKey("5.2", pi.PERMISSION, 1),
                         approvals, records), ab.UNREVIEWED)

    def test_the_observation_sharing_the_ordinal_stays_unreviewed(self):
        approvals, records = ab.load(approval()), store()

        self.assertEqual(
            ab.status_of(pi.ProvisionKey("5.2", pi.OBSERVATION, 1),
                         approvals, records), ab.UNREVIEWED)

    def test_approval_is_not_inferred_from_the_parent_unit(self):
        """Every provision here has the same source_id; only one is approved."""
        approvals, records = ab.load(approval()), store()
        approved = [key for key in records
                    if ab.status_of(key, approvals, records) == ab.HUMAN_APPROVED]

        self.assertEqual(approved, [pi.ProvisionKey("5.2", pi.RULE, 1)])


class AnApprovalGoesStaleWhenTheTextMoves(unittest.TestCase):
    def test_a_changed_unit_text_is_stale_not_approved(self):
        report = ab.verify(ab.load(approval(digest="0" * 64)), store())

        self.assertEqual(report[0]["status"], ab.STALE_APPROVAL)

    def test_a_provision_that_vanished_is_a_mismatch(self):
        gone = {**UNIT, "text": "Rule 5.2-9: A Gadget Reply shall stop."}
        report = ab.verify(ab.load(approval()), store(gone))

        self.assertEqual(report[0]["status"], ab.EXTRACTION_MISMATCH)

    def test_a_provision_carried_by_another_unit_is_a_mismatch(self):
        report = ab.verify(ab.load(approval(source_id="std-" + "9" * 32)), store())

        self.assertEqual(report[0]["status"], ab.EXTRACTION_MISMATCH)

    def test_nothing_is_approved_without_a_record(self):
        self.assertEqual(
            ab.status_of(pi.ProvisionKey("9.9", pi.RULE, 1), {}, store()),
            ab.UNREVIEWED)


class ATableRowKeepsItsOwnOrdinal(unittest.TestCase):
    """Four approved rows of one table collapsed onto a single identity when
    their ordinal was a phrase rather than a number."""

    def test_a_phrase_ordinal_keeps_its_trailing_number(self):
        keys = {ab.key_of({"identity": {"section": "8.3.1",
                                        "provision_type": pi.TABLE_ROW,
                                        "ordinal": f"Table 8.3.1-1 bit {bit}"}})
                for bit in (21, 20, 19, 18)}

        self.assertEqual(len(keys), 4)
        self.assertIn(pi.ProvisionKey("8.3.1", pi.TABLE_ROW, 20), keys)

    def test_a_table_row_unit_becomes_a_structural_record(self):
        row = {"source_id": "std-" + "d" * 32, "section": "8.3.1", "page": 1,
               "modality": "NONE", "content_type": "UNKNOWN",
               "text": "20 ReqV Request Validation Set to 1: request it"}
        records = pi.records_from_unit(row, table="8.3.1-1")

        self.assertEqual([record.key for record in records],
                         [pi.TableRowKey("8.3.1", "8.3.1-1", 20)])

    def test_a_row_with_no_table_context_is_still_unique(self):
        """Derivation of one unit cannot know its table; the fallback keys it
        by its own unit so two rows never merge."""
        row = {"source_id": "std-" + "f" * 32, "section": "8.3.1",
               "modality": "NONE", "content_type": "UNKNOWN",
               "text": "20 ReqV Request Validation"}

        self.assertEqual(pi.records_from_unit(row)[0].key.table,
                         "unit:std-" + "f" * 32)


if __name__ == "__main__":
    unittest.main()
