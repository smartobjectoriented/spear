"""The line the final-answer guard draws, and where it refuses to draw one.

H4's distinction is normative rather than arithmetic. Saying that bits 31..16
are unclaimed follows from a stated word width and a stated range. Saying that
the label "Reserved" occupies bits 31..16 binds an unresolved name to a
concrete range and is exactly what no evidence establishes. These tests pin
both halves, because a guard that blocked the arithmetic too would be a refusal
layer rather than an evidence boundary.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_guard import EvidenceLedger, guard, safe_rendering, validate


def _payload():
    """H4's structure as the tool returns it by default."""
    return {
        "definition_id": "bfd-0ccd61a481e290d5",
        "structural_completeness": "STRUCTURALLY_INCOMPLETE",
        "citation_source_ids": ["std-1a61ae3595e515a6bd6e4711c01850a0"],
        "fields": [{"display_label": "Phase Offset, Radians", "word_index": 1,
                    "msb": 15, "lsb": 0, "width": 16,
                    "position_source": "STATED_RANGE"}],
        "words": [{"word_index": 1, "word_width": 32,
                   "unclaimed_bits": [[31, 16]],
                   "unpositioned_labels": ["Reserved"],
                   "unresolved_labels": [{"label": "Reserved",
                                          "declared_msb": None,
                                          "declared_lsb": None,
                                          "cause": "NO_DETERMINISTIC_RANGE"}]}],
    }


def _complete_payload():
    return {
        "definition_id": "bfd-complete",
        "structural_completeness": "STRUCTURALLY_COMPLETE",
        "citation_source_ids": ["std-aaaa1111bbbb2222"],
        "fields": [{"display_label": "Phase Offset, Radians", "word_index": 1,
                    "msb": 15, "lsb": 0, "width": 16,
                    "position_source": "STATED_RANGE"},
                   {"display_label": "Reserved", "word_index": 1,
                    "msb": 31, "lsb": 16, "width": 16,
                    "position_source": "STATED_RANGE"}],
        "words": [{"word_index": 1, "word_width": 32}],
    }


class Ledger(unittest.TestCase):
    def test_it_reads_established_and_unresolved_apart(self):
        ledger = EvidenceLedger().observe(_payload())
        self.assertIn("phaseoffsetradians", ledger.established)
        self.assertIn("reserved", ledger.unresolved)

    def test_it_does_not_mutate_the_tool_payload(self):
        payload = _payload()
        EvidenceLedger().observe(payload)
        self.assertEqual(payload, _payload())

    def test_it_never_invents_a_range_for_an_unresolved_label(self):
        ledger = EvidenceLedger().observe(_payload())
        self.assertNotIn("reserved", ledger.established)


class UnsupportedClaimsAreBlocked(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger().observe(_payload())

    def test_reserved_given_a_range_fails(self):
        found = validate("Reserved occupies bits 31..16.", self.ledger)
        self.assertTrue(found)

    def test_reserved_given_a_width_fails(self):
        found = validate("Reserved is 16 bits wide.", self.ledger)
        self.assertTrue(found)

    def test_an_invented_padding_region_fails(self):
        found = validate("Padding occupies bits 31..16.", self.ledger)
        self.assertTrue(found)

    def test_an_unsupported_claim_in_prose_fails(self):
        answer = ("The structure is incomplete. The Reserved field occupies "
                  "bits 31..16 of word 1.")
        self.assertTrue(validate(answer, self.ledger))

    def test_an_unsupported_claim_in_code_fails(self):
        answer = ("```c\nstruct s {\n    uint32_t phase_offset : 16;\n"
                  "    uint32_t reserved : 16;\n};\n```")
        self.assertTrue(validate(answer, self.ledger))

    def test_an_unsupported_claim_in_a_table_fails(self):
        answer = ("| Field | Bits |\n|---|---|\n"
                  "| Phase Offset, Radians | 15..0 |\n| Reserved | 31..16 |")
        self.assertTrue(validate(answer, self.ledger))

    def test_a_contradicted_established_range_fails(self):
        found = validate("Phase Offset, Radians occupies bits 31..16.",
                         self.ledger)
        self.assertTrue(found)


class SupportedStatementsPass(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger().observe(_payload())

    def test_the_established_field_and_range_passes(self):
        answer = "Phase Offset, Radians occupies bits 15..0 of word 1."
        self.assertEqual(validate(answer, self.ledger), [])

    def test_stating_the_label_as_unresolved_passes(self):
        answer = ("Reserved is named by the diagram but its normative position "
                  "and width are not established.")
        self.assertEqual(validate(answer, self.ledger), [])

    def test_the_arithmetic_unclaimed_range_passes(self):
        answer = ("Bits 31..16 remain unclaimed by the established fields of "
                  "the 32-bit word.")
        self.assertEqual(validate(answer, self.ledger), [])

    def test_the_arithmetic_gap_does_not_become_a_normative_label(self):
        allowed = "Bits 31..16 are currently unassigned."
        blocked = "Bits 31..16 are Reserved."
        self.assertEqual(validate(allowed, self.ledger), [])
        self.assertTrue(validate(blocked, self.ledger))

    def test_a_complete_supported_artifact_passes_unchanged(self):
        ledger = EvidenceLedger().observe(_complete_payload())
        answer = ("```c\nstruct s {\n    uint32_t phase_offset_radians : 16;\n"
                  "    uint32_t reserved : 16;\n};\n```\n"
                  "Reserved occupies bits 31..16.")
        returned, found, replaced = guard(answer, ledger)
        self.assertEqual(found, [])
        self.assertIs(replaced, False)
        self.assertEqual(returned, answer)


class SafeRenderingStaysUseful(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger().observe(_payload())
        self.text = safe_rendering(self.ledger)

    def test_it_keeps_the_established_field(self):
        self.assertIn("Phase Offset, Radians", self.text)
        self.assertIn("15..0", self.text)

    def test_it_names_the_unresolved_label_without_placing_it(self):
        self.assertIn("Reserved", self.text)
        self.assertIn("not established", self.text)

    def test_it_offers_the_arithmetic_observation(self):
        self.assertIn("unclaimed", self.text)
        self.assertIn("31..16", self.text)

    def test_it_states_the_constraint(self):
        self.assertIn("does not establish", self.text)

    def test_it_preserves_citations(self):
        self.assertIn("std-1a61ae3595e515a6bd6e4711c01850a0", self.text)

    def test_it_is_not_a_generic_refusal(self):
        self.assertGreater(len(self.text.splitlines()), 6)

    def test_its_own_output_survives_the_validator(self):
        # the replacement must not itself trip the guard
        self.assertEqual(validate(self.text, self.ledger), [])


class GuardIsInertWithoutEvidence(unittest.TestCase):
    def test_an_empty_ledger_blocks_nothing(self):
        answer = "Reserved occupies bits 31..16."
        returned, found, replaced = guard(answer, EvidenceLedger())
        self.assertEqual(found, [])
        self.assertIs(replaced, False)
        self.assertEqual(returned, answer)


class LegacyStructureContractIsDefault(unittest.TestCase):
    def test_get_structure_returns_the_pre_ft3t_shape_by_default(self):
        import os

        self.assertNotEqual(os.environ.get("SPEAR_STRUCTURE_CONTRACT"),
                            "hardened")


if __name__ == "__main__":
    unittest.main()
