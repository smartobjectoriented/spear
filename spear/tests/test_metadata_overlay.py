"""A reviewer may correct metadata without anyone editing the extraction."""

from __future__ import annotations

import unittest

import metadata_overlay as mo
import provision_identity as pi

UNIT = {"source_id": "std-" + "e" * 32, "section": "5.4", "page": 3,
        "modality": "NONE", "content_type": "INFORMATIVE",
        "text": "Rule 5.4-1: A Gadget shall emit one frame."}


def record():
    return pi.records_from_unit(UNIT)[0]


def overlay(field="content_type", value="REQUIREMENT", digest=None):
    return mo.load({"corrections": [{
        "identity": {"section": "5.4", "kind": pi.RULE, "ordinal": 1},
        "field": field, "value": value, "reviewer": "reviewer",
        "reviewed_at": "2026-09-15",
        "provision_text_sha256": digest if digest is not None else record().text_sha256,
        "reason": "the page says shall"}]})


class AnEmptyOverlayChangesNothing(unittest.TestCase):
    def test_effective_metadata_is_the_extraction(self):
        self.assertEqual(mo.effective(record(), {}),
                         {"content_type": "INFORMATIVE", "modality": "SHALL"})

    def test_an_empty_overlay_reports_nothing(self):
        self.assertEqual(mo.report({}, {}), [])


class ACorrectionAppliesOnlyWhileItIsLive(unittest.TestCase):
    def test_a_live_correction_is_applied(self):
        self.assertEqual(mo.effective(record(), overlay())["content_type"],
                         "REQUIREMENT")

    def test_a_correction_against_other_text_is_stale_and_ignored(self):
        stale = overlay(digest="0" * 64)

        self.assertEqual(mo.effective(record(), stale)["content_type"],
                         "INFORMATIVE")
        self.assertEqual(mo.report(stale, {record().key: record()})[0]["status"],
                         mo.STALE)

    def test_a_correction_for_a_vanished_provision_is_reported(self):
        self.assertEqual(mo.report(overlay(), {})[0]["status"],
                         mo.UNKNOWN_PROVISION)

    def test_a_correction_carries_its_provenance(self):
        entry = mo.report(overlay(), {record().key: record()})[0]

        self.assertEqual(entry["reviewer"], "reviewer")
        self.assertEqual(entry["status"], mo.ACTIVE)


class TextIsNeverOverlaid(unittest.TestCase):
    """A correction that could rewrite a clause would be a way of answering
    from something the document does not say."""

    def test_only_metadata_fields_are_correctable(self):
        self.assertEqual(mo.CORRECTABLE, {"content_type", "modality"})

    def test_a_text_correction_is_refused_at_load(self):
        self.assertEqual(
            mo.load({"corrections": [{"identity": {"section": "5.4",
                                                   "kind": pi.RULE, "ordinal": 1},
                                      "field": "text", "value": "anything"}]}),
            {})

    def test_the_provision_text_is_untouched_by_an_applied_correction(self):
        before = record().text

        mo.effective(record(), overlay())

        self.assertEqual(record().text, before)


if __name__ == "__main__":
    unittest.main()
