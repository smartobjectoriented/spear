"""A placement nobody stated, accepted only when nothing else was possible.

The rule this pins is uniqueness, not plausibility. A twelve-octet record with
two four-octet members at 0 and 8 and a third whose length is four has exactly
one interval left, and refusing octets 4..7 there would be refusing arithmetic.
H4 looks superficially similar -- sixteen spare bits, one unresolved label --
and is not: the label's width was never established, so the constraints do not
pick a placement and the model choosing one is invention. These tests hold both
sides, and the difference between them is a single established number.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence_guard import EvidenceLedger, explain, guard, validate
from structural_entailment import (
    DETERMINISTIC_ENTAILMENT, EXPLICIT, MEMBERS_DO_NOT_ACCOUNT_FOR_EXTENT,
    NO_CANDIDATE_WIDTH, NO_CONTAINER_EXTENT, OVERLAPPING_MEMBERS,
    PLACEMENT_NOT_UNIQUE, REJECTED, Member, unique_placement,
)

RECORD = ("Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
          "twelve-octet record carrying the Primary Value and the Secondary "
          "Value, each four octets, together with four octets reserved to the "
          "transport.")
UNSIZED = ("Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
           "twelve-octet record carrying the Primary Value and the Secondary "
           "Value, each four octets, together with an area reserved to the "
           "transport.")
OFFSET_A = ("Rule 7.4.2-1: The Primary Value shall begin at octet offset 0 of "
            "the Redundancy Field.")
OFFSET_B = ("Rule 7.4.2-2: The Secondary Value shall begin at octet offset 8 "
            "of the Redundancy Field.")


def unit(text, index=0):
    return {"unit": {"source_id": f"std-{index:032d}", "section": "7.4",
                     "page": 90, "text": text, "content_type": "requirement"}}


def ledger(*texts):
    found = EvidenceLedger()

    for index, text in enumerate(texts):
        found.observe_fetch(unit(text, index))

    return found


def paths(answer, item):
    return {row["path"] for row in explain(answer, item)}


class Proof(unittest.TestCase):
    """The rule itself, on intervals alone."""

    def test_a_single_gap_of_the_right_size_is_entailed(self):
        proof = unique_placement(
            Member("Reserved", None, 4),
            [Member("A", 0, 4), Member("B", 8, 4), Member("Reserved", None, 4)],
            12)

        self.assertTrue(proof.entailed)
        self.assertEqual(proof.interval, (4, 7))
        self.assertEqual(proof.reason, DETERMINISTIC_ENTAILMENT)

    def test_an_unknown_candidate_width_is_never_entailed(self):
        proof = unique_placement(Member("Reserved", None, None),
                                 [Member("Phase Offset", 0, 16)], 32)

        self.assertFalse(proof.entailed)
        self.assertEqual(proof.reason, NO_CANDIDATE_WIDTH)

    def test_two_possible_gaps_are_not_entailed(self):
        proof = unique_placement(
            Member("R", None, 4),
            [Member("A", 4, 4), Member("B", None, 4), Member("C", None, 4)], 16)

        self.assertFalse(proof.entailed)
        self.assertEqual(proof.reason, PLACEMENT_NOT_UNIQUE)
        self.assertGreater(len(proof.candidates), 1)

    def test_a_gap_wider_than_the_candidate_is_not_entailed(self):
        # One hole of four octets and a member two wide: three placements.
        proof = unique_placement(
            Member("R", None, 2),
            [Member("A", 0, 4), Member("B", 8, 4), Member("R", None, 2)], 12)

        self.assertFalse(proof.entailed)

    def test_an_unknown_container_extent_is_never_entailed(self):
        proof = unique_placement(Member("R", None, 4),
                                 [Member("A", 0, 4), Member("B", 8, 4)], None)

        self.assertFalse(proof.entailed)
        self.assertEqual(proof.reason, NO_CONTAINER_EXTENT)

    def test_overlapping_established_members_fail_closed(self):
        proof = unique_placement(Member("R", None, 4),
                                 [Member("A", 0, 6), Member("B", 4, 4),
                                  Member("R", None, 4)], 12)

        self.assertFalse(proof.entailed)
        self.assertEqual(proof.reason, OVERLAPPING_MEMBERS)

    def test_members_that_do_not_account_for_the_container_fail(self):
        proof = unique_placement(
            Member("R", None, 4),
            [Member("A", 0, 4), Member("B", 8, 4), Member("R", None, 4)], 16)

        self.assertFalse(proof.entailed)
        self.assertEqual(proof.reason, MEMBERS_DO_NOT_ACCOUNT_FOR_EXTENT)

    def test_the_rule_never_prefers_the_lowest_offset(self):
        proof = unique_placement(
            Member("R", None, 4),
            [Member("A", 8, 4), Member("B", None, 4), Member("R", None, 4)], 12)

        self.assertFalse(proof.entailed)


class Cmdstat(unittest.TestCase):
    """The authorized case, end to end through the guard."""

    def setUp(self):
        self.ledger = ledger(RECORD, OFFSET_A, OFFSET_B)

    def test_the_entailed_placement_is_accepted(self):
        answer = "Reserved occupies octets 4..7 of the Redundancy Field."

        self.assertEqual(validate(answer, self.ledger), [])
        self.assertEqual(paths(answer, self.ledger),
                         {DETERMINISTIC_ENTAILMENT})

    def test_the_path_is_recorded_as_entailment_not_as_explicit(self):
        rows = explain("Reserved occupies octets 4..7.", self.ledger)

        self.assertEqual(rows[0]["path"], DETERMINISTIC_ENTAILMENT)
        self.assertEqual(rows[0]["proof"]["interval"], [4, 7])
        self.assertEqual(rows[0]["proof"]["extent"], 12)

    def test_a_stated_offset_is_still_recorded_as_explicit(self):
        rows = explain("The Primary Value begins at octet offset 0.",
                       self.ledger)

        self.assertEqual(rows[0]["path"], EXPLICIT)

    def test_the_impossible_placement_is_still_refused(self):
        answer = "Reserved occupies octets 12..15 of the Redundancy Field."
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertIn("outside the established 12-octet record",
                      violations[0]["reason"])

    def test_a_placement_other_than_the_entailed_one_is_refused(self):
        violations = validate("Reserved occupies octets 0..3.", self.ledger)

        self.assertTrue(violations)

    def test_nothing_is_entailed_when_the_width_is_not_stated(self):
        answer = "Reserved occupies octets 4..7."
        violations = validate(answer, ledger(UNSIZED, OFFSET_A, OFFSET_B))

        self.assertTrue(violations)
        self.assertIn("none is forced", violations[0]["reason"])


class HeldOpen(unittest.TestCase):
    """H4, which the rule must go on refusing."""

    def structure_ledger(self):
        return EvidenceLedger().observe({
            "definition_id": "bfd-0ccd61a481e290d5",
            "structural_completeness": "STRUCTURALLY_INCOMPLETE",
            "citation_source_ids": ["std-" + "b" * 32],
            "fields": [{"display_label": "Phase Offset, Radians", "msb": 15,
                        "lsb": 0, "width": 16, "word_index": 1,
                        "normative_source_id": "std-" + "b" * 32}],
            "unresolved": [{"label": "Reserved"}],
            "words": [{"word_index": 1, "word_width": 32}]})

    def test_the_reserved_range_is_still_refused(self):
        violations = validate("Reserved occupies bits 31..16 of word 1.",
                              self.structure_ledger())

        self.assertTrue(violations)

    def test_the_c_member_form_is_still_refused(self):
        answer = ("struct s {\n    uint32_t phase_offset : 16;\n"
                  "    uint32_t reserved : 16;\n};")

        self.assertTrue(validate(answer, self.structure_ledger()))

    def test_the_arithmetic_statement_still_passes(self):
        answer = "Bits 31..16 of word 1 remain unclaimed by the stated fields."

        self.assertEqual(validate(answer, self.structure_ledger()), [])

    def test_no_octet_path_is_offered_for_a_structure_only_ledger(self):
        self.assertEqual(explain("Reserved occupies bits 31..16.",
                                 self.structure_ledger()), [])

    def test_the_h4_rendering_is_unchanged(self):
        from evidence_guard import safe_rendering
        rendered = safe_rendering(self.structure_ledger())

        self.assertIn("Reserved — normative position and width are not "
                      "established.", rendered)
        self.assertIn("word 1: bits 31..16 are currently unclaimed", rendered)


class Extraction(unittest.TestCase):
    """The wordings the extractor was missing."""

    def setUp(self):
        self.ledger = ledger(RECORD, OFFSET_A, OFFSET_B)

    def _claim(self, answer):
        rows = explain(answer, self.ledger)

        return [row for row in rows if "reserved" in row["label"].lower()]

    def test_an_offset_first_table_is_read(self):
        answer = ("| Offset | Length | Component |\n"
                  "|--------|--------|-----------|\n"
                  "| 12     | 4      | Reserved to transport |\n")
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertIn("outside", violations[0]["reason"])

    def test_a_field_first_table_is_still_read(self):
        answer = ("| Component | Octet Offset | Length |\n"
                  "|---|---|---|\n"
                  "| Reserved to transport | 12 | 4 |\n")

        self.assertTrue(validate(answer, self.ledger))

    def test_a_table_with_no_label_heading_is_not_guessed_at(self):
        answer = ("| A | B | C |\n|---|---|---|\n| 12 | 4 | Reserved |\n")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_bare_from_offset_prose_is_read(self):
        answer = ("Reserved to the transport: 4 octets, filling the remaining "
                  "space from offset 12 to 15")
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)

    def test_offset_and_length_prose_is_read(self):
        answer = "Reserved begins at offset 12 and has length 4."

        self.assertTrue(validate(answer, self.ledger))

    def test_the_colon_form_is_read(self):
        answer = "Reserved: offset 12, length 4"

        self.assertTrue(validate(answer, self.ledger))

    def test_octets_to_form_is_read(self):
        answer = "Reserved occupies octets 12 to 15."

        self.assertTrue(validate(answer, self.ledger))


class ArithmeticFraming(unittest.TestCase):
    """A hole may be observed; a label may not be hidden behind the word."""

    def setUp(self):
        self.ledger = ledger(UNSIZED, OFFSET_A, OFFSET_B)

    def test_an_unlabelled_gap_is_an_observation(self):
        answer = "Octets 4..7 remain unclaimed by the positioned components."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_remaining_does_not_excuse_a_labelled_placement(self):
        answer = "Reserved fills the remaining octets 4..7."
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)

    def test_free_does_not_excuse_a_labelled_placement(self):
        answer = "Reserved takes the free space at octets 4..7."

        self.assertTrue(validate(answer, self.ledger))

    def test_the_same_wording_is_accepted_once_the_width_makes_it_unique(self):
        answer = "Reserved fills the remaining octets 4..7."

        self.assertEqual(validate(answer, ledger(RECORD, OFFSET_A, OFFSET_B)),
                         [])

    def test_an_unnamed_region_outside_the_record_is_refused(self):
        answer = "The trailer sits at octets 16..19, beyond the record."
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertIn("outside", violations[0]["reason"])


class GuardBehaviour(unittest.TestCase):
    """What actually reaches the reader."""

    def test_an_entailed_answer_is_not_replaced(self):
        answer = ("The Redundancy Field is 12 octets. Primary Value at octet "
                  "offset 0, length 4. Secondary Value at octet offset 8, "
                  "length 4. Reserved occupies octets 4..7.")
        guarded, violations, replaced = guard(
            answer, ledger(RECORD, OFFSET_A, OFFSET_B))

        self.assertFalse(replaced)
        self.assertEqual(violations, [])
        self.assertEqual(guarded, answer)

    def test_an_impossible_answer_is_replaced_with_the_grounded_rendering(self):
        answer = "Reserved occupies octets 12..15, completing the record."
        guarded, violations, replaced = guard(
            answer, ledger(RECORD, OFFSET_A, OFFSET_B))

        self.assertTrue(replaced)
        self.assertIn("Primary Value — octet offset 0", guarded)
        self.assertIn("Reserved — octets 4..7", guarded)
        self.assertNotIn("12..15", guarded)


if __name__ == "__main__":
    unittest.main()
