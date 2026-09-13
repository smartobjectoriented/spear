"""Where an answer says its evidence came from, checked against where it came from.

The failing control answers correctly and then cites a PDF on somebody's
filesystem that no tool ever mentioned. Every deterministic layer before this
one is about structure, so the invented half sails through looking like
provenance. The rule here is not that a path looks wrong -- it is that the
tools did not produce it -- and the remedy is surgical: an ungrounded reference
is cut, never replaced with a guess about which source backs which sentence.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import provenance_guard
from provenance_guard import (
    FABRICATED_PATH, FABRICATED_SOURCE_ID, FABRICATED_URL, ProvenanceLedger,
    findings, guard, sanitize,
)

REAL = "std-5287d23d6932688558e46a628f8aaec2"
OTHER = "std-e1d69bac5d63695f703b9e201c727b5d"
STRUCTURE = "bfd-89d36dde83caaff6"
FAKE_ID = "std-deadbeefdeadbeefdeadbeefdeadbeef"
FAKE_URL = ("file:///home/user/verdin/standard/TICKER-TIME-9/2020-R2026/"
            "TICKER-TIME-9_2020-R2026.pdf#page=85")

RESULT = ('{"unit": {"source_id": "' + REAL + '", "section": "7.2.2", '
          '"page": 85}, "citation": {"source_id": "' + REAL + '", '
          '"rendered": "[TICKER-TIME-9 2020-R2026 \\u00a77.2.2, p.85, source '
          + REAL + ']"}}')


def ledger(*texts):
    found = ProvenanceLedger()

    for text in texts or (RESULT,):
        found.observe(text)

    return found


class Contract(unittest.TestCase):
    """What the tools actually put in front of the model."""

    def test_a_returned_source_id_is_known(self):
        self.assertIn(REAL, ledger().source_ids)

    def test_a_structure_id_is_recorded_too(self):
        found = ledger('{"definition_id": "' + STRUCTURE + '"}')

        self.assertIn(STRUCTURE, found.source_ids)

    def test_nothing_is_derived_from_a_result(self):
        found = ledger()

        self.assertEqual(found.urls, set())
        self.assertEqual(found.paths, set())

    def test_a_url_a_tool_did_return_is_recorded(self):
        found = ledger('{"link": "https://vita.com/spec.pdf"}')

        self.assertIn("https://vita.com/spec.pdf", found.urls)

    def test_an_empty_result_records_nothing(self):
        self.assertEqual(ledger("").source_ids, set())


class Detection(unittest.TestCase):
    """Which values are challenged, and which are left alone."""

    def test_a_returned_id_passes(self):
        self.assertEqual(findings(f"See {REAL} for the rule.", ledger()), [])

    def test_an_unseen_id_fails(self):
        found = findings(f"See {FAKE_ID}.", ledger())

        self.assertEqual([item["kind"] for item in found],
                         [FABRICATED_SOURCE_ID])

    def test_a_returned_url_passes(self):
        found = ledger('{"link": "https://vita.com/spec.pdf"}')

        self.assertEqual(findings("See https://vita.com/spec.pdf.", found), [])

    def test_a_returned_url_passes_mid_sentence_too(self):
        found = ledger('{"link": "https://vita.com/spec.pdf"}')

        self.assertEqual(
            findings("See https://vita.com/spec.pdf, section 7.", found), [])

    def test_a_fabricated_url_is_still_caught_before_a_full_stop(self):
        found = findings("See https://example.org/spec.pdf.", ledger())

        self.assertEqual(found[0]["value"], "https://example.org/spec.pdf")

    def test_an_unseen_https_url_fails(self):
        found = findings("See https://example.org/spec.pdf.", ledger())

        self.assertEqual([item["kind"] for item in found], [FABRICATED_URL])

    def test_an_unseen_file_url_fails(self):
        found = findings(f"See {FAKE_URL}.", ledger())

        self.assertEqual([item["kind"] for item in found], [FABRICATED_URL])

    def test_an_unseen_home_path_fails(self):
        found = findings("Stored at /home/user/verdin/spec.pdf.", ledger())

        self.assertEqual([item["kind"] for item in found], [FABRICATED_PATH])

    def test_a_windows_path_fails(self):
        found = findings(r"Stored at C:\standards\vita.pdf.", ledger())

        self.assertEqual([item["kind"] for item in found], [FABRICATED_PATH])

    def test_a_grounded_page_reference_passes(self):
        answer = f"Rule 7.2.2-1, §7.2.2, p.85 ({REAL})."

        self.assertEqual(findings(answer, ledger()), [])

    def test_ordinary_prose_with_a_slash_is_not_provenance(self):
        for answer in ("The MSB/LSB order is unspecified.",
                       "Use the read/write path in section 7.",
                       "See figure 9.4.1.1-1 and/or the rule above.",
                       "A relative include like ./config.h is fine."):
            with self.subTest(answer=answer):
                self.assertEqual(findings(answer, ledger()), [])

    def test_a_code_example_with_a_relative_path_is_untouched(self):
        answer = '#include "vita/packet.h"\nopen("data/frame.bin");'

        self.assertEqual(findings(answer, ledger()), [])


class Sanitizing(unittest.TestCase):
    """What survives the cut."""

    def test_a_clean_answer_is_byte_identical(self):
        answer = f"Coarse Time is carried in word 1 ({REAL})."
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertEqual(cleaned, answer)
        self.assertEqual(removed, [])

    def test_a_link_keeps_its_text_and_loses_its_target(self):
        answer = f"- Coarse Time, word 1 ([§7.2.2, p.85, {REAL}]({FAKE_URL}))"
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertNotIn("file://", cleaned)
        self.assertIn(REAL, cleaned)
        self.assertIn("§7.2.2, p.85", cleaned)
        self.assertEqual(retained, [REAL])

    def test_the_substantive_answer_is_untouched(self):
        answer = (f"Coarse Time is carried in word 1 of the Epoch Field "
                  f"([{REAL}]({FAKE_URL})). The standard does not specify the "
                  f"bit range.")
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertIn("Coarse Time is carried in word 1 of the Epoch Field",
                      cleaned)
        self.assertIn("The standard does not specify the bit range.", cleaned)

    def test_every_fabricated_url_goes(self):
        answer = "\n".join(f"- item {index} ([{REAL}]({FAKE_URL}))"
                           for index in range(3))
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertNotIn("file://", cleaned)
        self.assertEqual(cleaned.count(REAL), 3)

    def test_a_fabricated_id_and_a_fabricated_path_are_both_caught(self):
        answer = f"See {FAKE_ID} at /home/user/spec.pdf and {REAL}."
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertNotIn(FAKE_ID, cleaned)
        self.assertNotIn("/home/user/spec.pdf", cleaned)
        self.assertIn(REAL, cleaned)

    def test_no_replacement_reference_is_invented(self):
        answer = f"Coarse Time is in word 1 ([see the spec]({FAKE_URL}))."
        cleaned, removed, retained = sanitize(answer, ledger())

        self.assertNotIn(REAL, cleaned)
        self.assertNotIn("Sources:", cleaned)

    def test_the_guard_reports_what_it_removed(self):
        answer = f"Word 1 ([{REAL}]({FAKE_URL}))."
        cleaned, problems, removed, fired = guard(answer, ledger())

        self.assertTrue(fired)
        self.assertEqual([item["kind"] for item in problems], [FABRICATED_URL])
        self.assertTrue(removed)


class Untouched(unittest.TestCase):
    """The renderings the earlier layers produce must pass through unchanged."""

    def rendering_ledger(self):
        return ledger(RESULT, '{"sources": ["std-' + "b" * 32 + '"]}',
                      '{"definition_id": "' + STRUCTURE + '"}')

    def unchanged(self, answer):
        cleaned, problems, removed, fired = guard(answer, self.rendering_ledger())

        self.assertFalse(fired, msg=f"guard fired on: {answer[:60]}")
        self.assertEqual(cleaned, answer)

    def test_the_h4_rendering_passes_through(self):
        self.unchanged(
            "Established by the normative evidence:\n"
            "- Phase Offset, Radians — bits 15..0 of word 1\n\n"
            "Unresolved:\n- Reserved — normative position and width are not "
            "established.\n\nArithmetic observation (not a normative "
            "assignment):\n- word 1: bits 31..16 are currently unclaimed by "
            "the established fields.\n\nSources: std-" + "b" * 32)

    def test_the_f2_geometry_rendering_passes_through(self):
        self.unchanged(
            "Established by the normative evidence:\n"
            "- Horizontal Beamwidth, Degrees — the diagram prints bits 15..0 "
            "within this field's own cell (page 153)\n"
            "- Vertical Beamwidth, Degrees — the diagram prints bits 15..0 "
            "within this field's own cell (page 153)\n\nSources: std-"
            + "b" * 32)

    def test_the_f1_bounded_rendering_passes_through(self):
        self.unchanged(
            "Requested: Elevation Angle, Azimuthal Angle.\n\n"
            "The retrieved normative evidence does not establish the requested "
            "bit positions. Navigation stopped after 14 rounds of retrieval; "
            "76 distinct pieces of evidence were read in total.\n\n"
            "No position, range or width is inferred here beyond what the "
            "evidence states.\n\nSources read: std-" + "b" * 32)

    def test_the_entailment_rendering_passes_through(self):
        self.unchanged(
            "Established by the normative evidence:\n"
            "- Redundancy Field — a 12-octet record\n"
            "- Primary Value — octet offset 0, length 4 octets\n"
            "- Reserved — octets 4..7, length 4 octets; the evidence gives no "
            "offset for it, and this is the only interval its stated length "
            "can occupy\n\nSources: std-" + "b" * 32)

    def test_a_structure_answer_keeps_its_definition_id(self):
        self.unchanged(f"Approved structure {STRUCTURE} is incomplete.")

    def test_the_answer_is_not_mutated_in_place(self):
        answer = f"Word 1 ([{REAL}]({FAKE_URL}))."
        before = answer
        guard(answer, self.rendering_ledger())

        self.assertEqual(answer, before)

    def test_a_tool_result_is_not_mutated_by_being_observed(self):
        text = RESULT
        before = text
        ProvenanceLedger().observe(text)

        self.assertEqual(text, before)


if __name__ == "__main__":
    unittest.main()
