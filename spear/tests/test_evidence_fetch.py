"""Facts a fetched requirement states, and the ones it refuses to state.

The failing control is told a twelve-octet record and two of its three octet
offsets, and then places the third at octet 12 -- past the end of the record it
has just quoted. Nothing in the evidence puts it anywhere. So the ledger has to
be able to hold "length four, offset not given" without ever completing it, and
the guard has to refuse both the impossible placement and the plausible one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evidence_fetch
from evidence_guard import (
    CONTRADICTS_ESTABLISHED_EVIDENCE, EvidenceLedger, _DENIAL_ONLY, guard,
    safe_rendering, validate,
)

RECORD = ("Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
          "twelve-octet record carrying the Primary Value and the Secondary "
          "Value, each four octets, together with four octets reserved to the "
          "transport.")
OFFSET_A = ("Rule 7.4.2-1: The Primary Value shall begin at octet offset 0 of "
            "the Redundancy Field.")
OFFSET_B = ("Rule 7.4.2-2: The Secondary Value shall begin at octet offset 8 "
            "of the Redundancy Field.")
WORDS = ("Rule 7.2.1-1: The Epoch Field shall consist of three 32-bit words, "
         "two of which carry the Coarse Time and the Fine Time.")
WORD_A = ("Rule 7.2.2-1: The Coarse Time shall be carried in word 1 of the "
          "Epoch Field.")
WORD_B = ("Rule 7.2.2-2: The Fine Time shall be carried in word 3 of the "
          "Epoch Field.")


def unit(text, source_id="std-" + "a" * 32, section="7.4.1", page=90):
    return {"unit": {"source_id": source_id, "section": section, "page": page,
                     "text": text, "content_type": "requirement"}}


def octet_ledger(*texts):
    ledger = EvidenceLedger()

    for index, text in enumerate(texts):
        ledger.observe_fetch(unit(text, source_id=f"std-{index:032d}"))

    return ledger


class FetchFacts(unittest.TestCase):
    """What one requirement is read to say."""

    def test_offset_and_length_enter_the_ledger_with_provenance(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)

        self.assertEqual(ledger.octet_fields["primaryvalue"]["offset"], 0)
        self.assertEqual(ledger.octet_fields["primaryvalue"]["length"], 4)
        self.assertEqual(ledger.record["octets"], 12)

        where = ledger.octet_fields["primaryvalue"]["provenance"]

        self.assertEqual(where["statement"], "7.4.2-1")
        self.assertEqual(where["section"], "7.4.1")
        self.assertEqual(where["page"], 90)
        self.assertTrue(where["source_id"].startswith("std-"))
        self.assertIn("octet offset 0", where["span"])

    def test_a_length_without_an_offset_stays_without_one(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)

        self.assertEqual(ledger.octet_fields["reserved"]["length"], 4)
        self.assertIsNone(ledger.octet_fields["reserved"]["offset"])

    def test_the_ledger_does_not_complete_the_record_for_itself(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)
        offsets = {key: item["offset"]
                   for key, item in ledger.octet_fields.items()}

        self.assertEqual(offsets, {"primaryvalue": 0, "secondaryvalue": 8,
                                   "reserved": None})

    def test_word_indices_enter_the_ledger(self):
        ledger = octet_ledger(WORDS, WORD_A, WORD_B)

        self.assertEqual(ledger.word_fields["coarsetime"]["word_index"], 1)
        self.assertEqual(ledger.word_fields["finetime"]["word_index"], 3)
        self.assertEqual(ledger.record["words"], 3)
        self.assertEqual(ledger.record["word_width"], 32)

    def test_word_two_is_not_invented(self):
        ledger = octet_ledger(WORDS, WORD_A, WORD_B)

        self.assertNotIn(2, [item["word_index"]
                             for item in ledger.word_fields.values()])

    def test_only_a_requirement_establishes_anything(self):
        described = ("Observation 7.4.2-3: The Primary Value shall begin at "
                     "octet offset 4 of the Redundancy Field.")

        self.assertEqual(evidence_fetch.unit_facts(unit(described)), ())

    def test_an_unnumbered_sentence_states_nothing(self):
        loose = "The Primary Value shall begin at octet offset 4."

        self.assertEqual(evidence_fetch.unit_facts(unit(loose)), ())

    def test_a_payload_that_is_not_a_fetch_result_is_ignored(self):
        for payload in (None, {}, {"error": "SOURCE_NOT_FOUND"}, [1, 2]):
            self.assertEqual(evidence_fetch.unit_facts(payload), ())


class OctetClaims(unittest.TestCase):
    """What an answer may and may not say about where octets sit."""

    def setUp(self):
        self.ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)

    def test_the_established_offsets_pass(self):
        answer = ("| Component | Octet Offset | Length |\n"
                  "|---|---|---|\n"
                  "| Primary Value | 0 | 4 |\n"
                  "| Secondary Value | 8 | 4 |")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_reserved_at_octet_twelve_fails(self):
        answer = ("| Component | Octet Offset | Length |\n"
                  "|---|---|---|\n"
                  "| Reserved (transport) | 12 | 4 |")
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"], "OCTET_OFFSET")

    def test_reserved_at_octet_four_is_now_entailed(self):
        # FT3F refused this: the evidence does not say octet 4. FT3F2 accepts
        # it, on the authorized rule that a placement counts when it is the
        # unique solution to the constraints already established -- here a
        # twelve-octet record with two four-octet members at 0 and 8 and a
        # third whose length is four. The refusal this replaces was correct
        # under the older rule and is kept below with the width removed.
        answer = "The Reserved region begins at octet offset 4."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_reserved_at_octet_four_still_fails_without_a_stated_width(self):
        ledger = octet_ledger(
            "Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
            "twelve-octet record carrying the Primary Value and the Secondary "
            "Value, each four octets, together with an area reserved to the "
            "transport.", OFFSET_A, OFFSET_B)
        violations = validate("The Reserved region begins at octet offset 4.",
                              ledger)

        self.assertTrue(violations)
        self.assertIn("none is forced", violations[0]["reason"])

    def test_a_placement_past_the_record_fails_whatever_it_is_called(self):
        ledger = octet_ledger(RECORD)
        answer = "The Transport Trailer begins at octet offset 16."
        violations = validate(answer, ledger)

        self.assertTrue(violations)
        self.assertIn("12-octet record", violations[0]["reason"])

    def test_moving_an_established_field_fails(self):
        answer = "The Secondary Value begins at octet offset 4."
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertIn("contradicts", violations[0]["reason"])

    def test_the_unclaimed_octets_may_be_stated_as_arithmetic(self):
        answer = ("Octets 4..7 are not claimed by either positioned "
                  "component; the evidence leaves them unassigned.")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_an_empty_ledger_blocks_nothing(self):
        answer = "The Reserved region begins at octet offset 4."

        self.assertEqual(validate(answer, EvidenceLedger()), [])


class Contradiction(unittest.TestCase):
    """The one new failure class: denying what the evidence states."""

    def setUp(self):
        self.ledger = octet_ledger(WORDS, WORD_A, WORD_B)

    def test_a_correct_word_fact_with_an_open_bit_caveat_passes(self):
        answer = ("Coarse Time is carried in word 1. Fine Time is carried in "
                  "word 3. The Epoch Field consists of three 32-bit words. "
                  "The bit ranges within those words are not specified.")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_the_established_words_pass(self):
        answer = "Coarse Time is carried in word 1 and Fine Time in word 3."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_denying_the_established_word_fails(self):
        answer = ("The standard does not define the word number for Coarse "
                  "Time.")
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"],
                         CONTRADICTS_ESTABLISHED_EVIDENCE)

    def test_omitting_a_field_is_not_a_contradiction(self):
        answer = "Coarse Time is carried in word 1."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_denying_something_other_than_the_position_is_not_one_either(self):
        answer = ("The standard does not specify the content or format of the "
                  "Coarse Time and Fine Time fields beyond their size and "
                  "position.")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_denying_the_bit_range_leaves_the_word_number_standing(self):
        # FT4D. The guard used to read this as retracting "word 1" because the
        # sentence carries a denial and names the field. It denies the bits.
        answer = ("Coarse Time is carried in word 1 and Fine Time in word 3, "
                  "but the standard does not specify the bit ranges (e.g., "
                  "msb/lsb) for Coarse Time or Fine Time within their "
                  "respective words.")

        self.assertEqual(validate(answer, self.ledger), [])

    def test_denying_the_word_number_still_fires(self):
        answer = ("The standard does not specify the word number for Coarse "
                  "Time.")
        violations = validate(answer, self.ledger)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"],
                         CONTRADICTS_ESTABLISHED_EVIDENCE)

    def test_denying_bits_inside_a_named_word_passes(self):
        answer = "The bit positions within word 3 are not established."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_a_denial_of_an_established_bit_range_still_fires(self):
        ledger = EvidenceLedger().observe({
            "definition_id": "bfd-1", "citation_source_ids": [],
            "fields": [{"display_label": "Field X", "msb": 15, "lsb": 0,
                        "width": 16, "word_index": 1}],
            "words": [{"word_index": 1, "word_width": 32}]})
        violations = validate("The bit range for Field X is not specified.",
                              ledger)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"],
                         CONTRADICTS_ESTABLISHED_EVIDENCE)

    def test_a_denial_with_no_resolvable_dimension_contradicts_nothing(self):
        # Conservative by choice: matching a denial to a fact on the strength
        # of the shared label alone is the over-reach FT4D removes.
        answer = "The standard does not specify anything about Coarse Time."

        self.assertEqual(validate(answer, self.ledger), [])

    def test_a_blanket_refusal_that_denies_the_word_still_fires(self):
        answer = ("There is no defined word number for Coarse Time or Fine "
                  "Time in this revision.")

        self.assertTrue(validate(answer, self.ledger))

    def test_the_guard_and_the_oracle_agree_on_the_ft3g2_wordings(self):
        import sys
        sys.path.insert(0, str(ROOT / "eval" / "abstention"))
        import normative_dimensions as nd

        for tail, dimension in (
                ("the bit ranges are not specified", nd.BITS),
                ("the bit ranges are not established", nd.BITS),
                ("the evidence does not state the bit ranges", nd.BITS),
                ("the bit positions within each word are undefined", nd.BITS),
                ("there is no defined bit range for either field", nd.BITS),
                ("the word number for Coarse Time is not specified", nd.WORD),
                ("the evidence does not state which word carries Coarse Time",
                 nd.WORD)):
            with self.subTest(tail=tail):
                match = _DENIAL_ONLY.search(tail)

                self.assertIsNotNone(match, tail)
                self.assertEqual(nd.dimension_of(tail, match.end()),
                                 dimension)

    def test_a_denial_naming_no_established_field_does_not_fire(self):
        answer = "The standard does not define the word number for Drift Rate."

        self.assertEqual(validate(answer, self.ledger), [])


class Rendering(unittest.TestCase):
    """What replaces an answer the evidence does not support."""

    def test_the_octet_rendering_keeps_every_supported_fact(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)
        rendered = safe_rendering(ledger)

        self.assertIn("Primary Value — octet offset 0, length 4 octets",
                      rendered)
        self.assertIn("Secondary Value — octet offset 8, length 4 octets",
                      rendered)
        self.assertIn("12-octet record", rendered)

    def test_the_octet_rendering_states_the_entailed_interval(self):
        rendered = safe_rendering(octet_ledger(RECORD, OFFSET_A, OFFSET_B))

        self.assertIn("Reserved — octets 4..7", rendered)
        self.assertIn("only interval its stated length can occupy", rendered)
        self.assertNotIn("Reserved — octet offset", rendered)

    def test_the_octet_rendering_leaves_an_unforced_region_open(self):
        ledger = octet_ledger(
            "Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
            "twelve-octet record carrying the Primary Value and the Secondary "
            "Value, each four octets, together with an area reserved to the "
            "transport.", OFFSET_A, OFFSET_B)

        self.assertNotIn("Reserved — octets", safe_rendering(ledger))

    def test_no_arithmetic_gap_is_offered_once_every_octet_is_accounted(self):
        # Reserved now occupies 4..7 by entailment, so there is no hole left
        # to observe. A gap sentence here would contradict the line above it.
        rendered = safe_rendering(octet_ledger(RECORD, OFFSET_A, OFFSET_B))

        self.assertNotIn("fall outside the components", rendered)

    def test_an_unaccounted_gap_is_still_stated_as_arithmetic(self):
        ledger = octet_ledger(
            "Rule 7.4.1-1: The Redundancy Field shall be serialized as a "
            "twelve-octet record carrying the Primary Value and the Secondary "
            "Value, each four octets, together with an area reserved to the "
            "transport.", OFFSET_A, OFFSET_B)
        rendered = safe_rendering(ledger)

        self.assertIn("Arithmetic observation (not a normative assignment):",
                      rendered)
        self.assertIn("octets 4..7 fall outside", rendered)

    def test_the_rendering_carries_the_fetch_citations(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)
        rendered = safe_rendering(ledger)

        for source_id in ledger.sources:
            self.assertIn(source_id, rendered)

    def test_the_word_rendering_does_not_invent_the_spare_word(self):
        rendered = safe_rendering(octet_ledger(WORDS, WORD_A, WORD_B))

        self.assertIn("Coarse Time — word 1", rendered)
        self.assertIn("Fine Time — word 3", rendered)
        self.assertNotIn("word 2", rendered)

    def test_the_guard_returns_the_answer_untouched_when_it_is_supported(self):
        ledger = octet_ledger(RECORD, OFFSET_A, OFFSET_B)
        answer = "Primary Value begins at octet offset 0, length four octets."
        guarded, violations, replaced = guard(answer, ledger)

        self.assertFalse(replaced)
        self.assertEqual(violations, [])
        self.assertEqual(guarded, answer)


class Unchanged(unittest.TestCase):
    """FT3V and FT3E behaviour, pinned where FT3F could have moved it."""

    def structure_ledger(self):
        return EvidenceLedger().observe({
            "definition_id": "bfd-0ccd61a481e290d5",
            "structural_completeness": "STRUCTURALLY_INCOMPLETE",
            "citation_source_ids": ["std-" + "b" * 32],
            "fields": [{"display_label": "Stage 1 Gain", "msb": 15, "lsb": 0,
                        "width": 16, "word_index": 1,
                        "normative_source_id": "std-" + "b" * 32}],
            "unresolved": [{"label": "Reserved"}],
            "words": [{"word_index": 1, "word_width": 32}]})

    def test_a_get_structure_only_ledger_gains_no_fetch_state(self):
        ledger = self.structure_ledger()

        self.assertEqual(ledger.octet_fields, {})
        self.assertEqual(ledger.word_fields, {})
        self.assertEqual(ledger.record, {})

    def test_h4_still_fails_on_an_invented_reserved_range(self):
        answer = "Reserved occupies bits 31..16 of word 1."
        violations = validate(answer, self.structure_ledger())

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"], "FIELD_RANGE")

    def test_h4_still_allows_the_arithmetic_statement(self):
        answer = "Bits 31..16 of word 1 remain unclaimed by the stated fields."

        self.assertEqual(validate(answer, self.structure_ledger()), [])

    def test_the_h4_rendering_is_what_it_was(self):
        rendered = safe_rendering(self.structure_ledger())

        self.assertIn("Stage 1 Gain — bits 15..0 of word 1", rendered)
        self.assertIn("Reserved — normative position and width are not "
                      "established.", rendered)
        self.assertIn("word 1: bits 31..16 are currently unclaimed", rendered)

    def test_the_bootstrap_is_untouched_by_the_ledger(self):
        import evidence_bootstrap

        self.assertFalse(evidence_bootstrap.should_bootstrap(
            "Give the octet offset of every field.",
            [{"tool": "standard.fetch"}], answer="something"))
        self.assertTrue(evidence_bootstrap.should_bootstrap(
            "Give the octet offset of every field.", [], answer="something"))

    def test_a_fetch_payload_is_not_mutated_by_being_read(self):
        payload = unit(RECORD)
        before = repr(payload)
        EvidenceLedger().observe_fetch(payload)

        self.assertEqual(repr(payload), before)


if __name__ == "__main__":
    unittest.main()
