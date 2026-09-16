"""Two cells side by side, and the word they are not halves of.

A packet diagram prints each cell's range inside that cell. Flattened to one
line of text, "Horizontal Beamwidth (15..0) ... Vertical Beamwidth (15..0)"
reads like the two halves of a 32-bit word, and the model completes it by
putting Vertical at 31..16. The figure says no such thing: it prints two
sixteen-bit quantities in separate columns. These tests hold that distinction
from both ends -- a cell's own range may be restated, and may not be moved --
and pin that nothing else in the stack moved with it.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import diagram_geometry
from diagram_geometry import Cell, cells_for_unit, independent
from evidence_guard import (
    CELL_LOCAL_RANGE_USED_AS_GLOBAL, EvidenceLedger, guard, safe_rendering,
    validate,
)

SID, REV = "ANSI-VITA-49.2", "2017-R2024"
#: The shape the recovery exists for: a packet diagram flattened into one
#: line, each cell still carrying its own range. Written out here rather than
#: read from a store, because a source_id names a unit inside ONE extraction
#: and nothing else. Pinning one made these tests skip in silence the moment
#: the store was re-extracted -- and the store they were pinned to had since
#: been replaced by one that does not flatten the row at all, so the
#: protection stopped being exercised exactly when it stopped being needed
#: for that corpus. It is still needed for every corpus that does flatten.
FLATTENED = {
    "source_id": "std-" + "b" * 32, "standard_id": SID, "revision": REV,
    "section": "9.4.1.5", "page": 153, "content_type": "UNKNOWN",
    "text": "1                Horizontal Beamwidth (15..0), Degrees"
            "                                      Vertical Beamwidth "
            "(15..0), Degrees",
}


def horizontal(msb=15, lsb=0):
    return Cell(label="Horizontal Beamwidth, Degrees",
                text="Horizontal Beamwidth (15..0), Degrees", msb=msb, lsb=lsb,
                page=153, block=21, line=1,
                bbox=(130.689, 565.596, 246.78, 574.397), source_id="std-a",
                range_text="(15..0)")


def vertical(msb=15, lsb=0):
    return Cell(label="Vertical Beamwidth, Degrees",
                text="Vertical Beamwidth (15..0), Degrees", msb=msb, lsb=lsb,
                page=153, block=29, line=5,
                bbox=(358.302, 565.596, 467.248, 574.397), source_id="std-a",
                range_text="(15..0)")


def two_cells():
    return EvidenceLedger().observe_cells([horizontal(), vertical()])


class Recovery(unittest.TestCase):
    """The cells, read back out of the store's own layout artifact."""

    def setUp(self):
        self.unit = dict(FLATTENED)

    def test_the_flattened_row_resolves_to_two_cells(self):
        cells = cells_for_unit(self.unit, standard_id=SID, revision=REV)

        self.assertEqual([cell.label for cell in cells],
                         ["Horizontal Beamwidth, Degrees",
                          "Vertical Beamwidth, Degrees"])

    def test_each_cell_keeps_its_own_local_range(self):
        cells = cells_for_unit(self.unit, standard_id=SID, revision=REV)

        for cell in cells:
            self.assertEqual((cell.msb, cell.lsb), (15, 0))

    def test_the_cells_are_printed_in_separate_columns(self):
        cells = cells_for_unit(self.unit, standard_id=SID, revision=REV)

        self.assertEqual(len(independent(cells)), 1)

    def test_provenance_survives(self):
        cell = cells_for_unit(self.unit, standard_id=SID, revision=REV)[0]

        self.assertEqual(cell.provenance["page"], 153)
        self.assertEqual(cell.provenance["source_id"], FLATTENED["source_id"])
        self.assertEqual(len(cell.provenance["bbox"]), 4)
        self.assertIn("Horizontal Beamwidth", cell.provenance["text"])

    def test_a_unit_with_no_page_resolves_to_nothing(self):
        unit = dict(self.unit)
        unit.pop("page")

        self.assertEqual(cells_for_unit(unit, standard_id=SID, revision=REV),
                         ())

    def test_the_layout_artifact_is_only_read(self):
        path = diagram_geometry.layout_path(SID, REV)
        before = path.stat().st_mtime_ns, path.stat().st_size
        cells_for_unit(self.unit, standard_id=SID, revision=REV)

        self.assertEqual((path.stat().st_mtime_ns, path.stat().st_size),
                         before)

    def test_the_fetched_unit_is_not_mutated(self):
        before = repr(self.unit)
        cells_for_unit(self.unit, standard_id=SID, revision=REV)

        self.assertEqual(repr(self.unit), before)


class Independence(unittest.TestCase):
    """What two cells printing the same range do and do not mean."""

    def test_both_cells_keep_their_own_range(self):
        ledger = two_cells()

        self.assertEqual(ledger.cell_fields["horizontalbeamwidthdegrees"]
                         ["msb"], 15)
        self.assertEqual(ledger.cell_fields["verticalbeamwidthdegrees"]["msb"],
                         15)

    def test_no_global_position_is_derived_for_either(self):
        ledger = two_cells()

        self.assertEqual(ledger.established, {})
        self.assertEqual(ledger.word_fields, {})

    def test_two_local_ranges_are_not_concatenated(self):
        rendered = safe_rendering(two_cells())

        self.assertNotIn("31..16", rendered)

    def test_reading_order_alone_establishes_nothing(self):
        # The second cell is second only in the flattened text. Ordering is
        # not a position, and the ledger records none.
        ledger = EvidenceLedger().observe_cells([vertical(), horizontal()])

        self.assertEqual(sorted(ledger.cell_fields), sorted(
            two_cells().cell_fields))


class Claims(unittest.TestCase):
    """What an answer may say about a cell."""

    def test_restating_the_horizontal_cell_range_passes(self):
        answer = "The Horizontal Beamwidth occupies bits 15..0 of its cell."

        self.assertEqual(validate(answer, two_cells()), [])

    def test_restating_the_vertical_cell_range_passes(self):
        answer = "The Vertical Beamwidth occupies bits 15..0."

        self.assertEqual(validate(answer, two_cells()), [])

    def test_putting_vertical_in_the_upper_half_fails(self):
        violations = validate("Vertical Beamwidth occupies bits 31..16.",
                              two_cells())

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"],
                         CELL_LOCAL_RANGE_USED_AS_GLOBAL)

    def test_putting_horizontal_in_the_upper_half_fails_too(self):
        self.assertTrue(validate("Horizontal Beamwidth occupies bits 31..16.",
                                 two_cells()))

    def test_the_half_wording_without_numbers_fails(self):
        violations = validate(
            "Vertical Beamwidth occupies the upper physical half.",
            two_cells())

        self.assertTrue(violations)
        self.assertIn("upper half", violations[0]["detail"])

    def test_markdown_emphasis_does_not_hide_the_claim(self):
        # The live answer writes "**Vertical Beamwidth**: occupies bits
        # 31..16". The emphasis sits between the name and the range, and
        # before FT4B's normalisation the label was read as "occupies".
        answer = "- **Vertical Beamwidth**: occupies bits 31..16 (upper half)"
        violations = validate(answer, two_cells())

        self.assertTrue(violations)
        self.assertEqual(violations[0]["kind"],
                         CELL_LOCAL_RANGE_USED_AS_GLOBAL)

    def test_markdown_emphasis_does_not_hide_a_half_claim(self):
        answer = ("**Horizontal Beamwidth** occupies the **upper half** of the "
                  "32-bit word.")
        violations = validate(answer, two_cells())

        self.assertTrue(violations)
        self.assertIn("upper half", violations[0]["detail"])

    def test_either_direction_of_the_complement_is_refused(self):
        for answer in ("Horizontal Beamwidth occupies bits 31..16 and Vertical "
                       "Beamwidth occupies bits 15..0.",
                       "Vertical Beamwidth occupies bits 31..16 and Horizontal "
                       "Beamwidth occupies bits 15..0."):
            with self.subTest(answer=answer[:40]):
                self.assertTrue(validate(answer, two_cells()))

    def test_emphasis_keeps_the_arithmetic_reading_intact(self):
        answer = "**Octets 4..7** remain unclaimed by the positioned fields."

        self.assertEqual(validate(answer, two_cells()), [])

    def structure_and_cell(self):
        """The same field positioned by a structure and printed in a diagram."""
        ledger = EvidenceLedger().observe({
            "definition_id": "bfd-x", "citation_source_ids": ["std-" + "a" * 32],
            "fields": [{"display_label": "Stage 2 Gain, dB", "msb": 31,
                        "lsb": 16, "width": 16, "word_index": 1,
                        "normative_source_id": "std-" + "a" * 32}],
            "words": [{"word_index": 1, "word_width": 32}]})
        ledger.observe_cells([Cell("Stage 2 Gain, dB",
                                   "Stage 2 Gain (15..0), dB", 15, 0, 168, 12,
                                   1, (100.0, 500.0, 260.0, 512.0),
                                   "std-" + "a" * 32, "(15..0)")])

        return ledger

    def test_a_structure_established_position_outranks_the_cell_print(self):
        # A production session reads the approved structure and the diagram row
        # of the same field. The structure positions it at 31..16; the cell
        # prints the value-local (15..0). Before PROD1 the cell rule refused the
        # structure's own answer, which is the G1/G2 question verbatim.
        ledger = EvidenceLedger().observe({
            "definition_id": "bfd-x", "citation_source_ids": ["std-" + "a" * 32],
            "fields": [{"display_label": "Stage 2 Gain, dB", "msb": 31,
                        "lsb": 16, "width": 16, "word_index": 1,
                        "normative_source_id": "std-" + "a" * 32}],
            "words": [{"word_index": 1, "word_width": 32}]})
        ledger.observe_cells([Cell("Stage 2 Gain, dB",
                                   "Stage 2 Gain (15..0), dB", 15, 0, 168, 12,
                                   1, (100.0, 500.0, 260.0, 512.0),
                                   "std-" + "a" * 32, "(15..0)")])

        self.assertEqual(
            validate("Stage 2 Gain is physically in bits 31..16 of word 1.",
                     ledger), [])

    def test_a_range_the_structure_does_not_establish_is_still_refused(self):
        ledger = EvidenceLedger().observe({
            "definition_id": "bfd-x", "citation_source_ids": ["std-" + "a" * 32],
            "fields": [{"display_label": "Stage 2 Gain, dB", "msb": 31,
                        "lsb": 16, "width": 16, "word_index": 1,
                        "normative_source_id": "std-" + "a" * 32}],
            "words": [{"word_index": 1, "word_width": 32}]})
        ledger.observe_cells([Cell("Stage 2 Gain, dB",
                                   "Stage 2 Gain (15..0), dB", 15, 0, 168, 12,
                                   1, (100.0, 500.0, 260.0, 512.0),
                                   "std-" + "a" * 32, "(15..0)")])

        self.assertTrue(validate("Stage 2 Gain is in bits 23..8 of word 1.",
                                 ledger))

    def test_the_established_half_may_be_named_in_words(self):
        # "Stage 2 Gain occupies the upper half" restates bits 31..16 of a
        # 32-bit word. The numeric branch already allowed the equivalent
        # sentence; the half wording carries no numbers and was still refused.
        self.assertEqual(
            validate("Stage 2 Gain occupies the upper half of word 1.",
                     self.structure_and_cell()), [])

    def test_the_wrong_half_is_still_refused(self):
        self.assertTrue(validate("Stage 2 Gain occupies the lower half of "
                                 "word 1.", self.structure_and_cell()))

    def test_the_f2_complement_is_unaffected_by_that_precedence(self):
        self.assertTrue(validate("Vertical Beamwidth occupies bits 31..16.",
                                 two_cells()))

    def test_an_empty_cell_ledger_blocks_nothing(self):
        self.assertEqual(
            validate("Vertical Beamwidth occupies bits 31..16.",
                     EvidenceLedger()), [])


class Rendering(unittest.TestCase):
    """What replaces an answer that moved a cell."""

    def test_both_cell_facts_survive(self):
        rendered = safe_rendering(two_cells())

        self.assertIn("Horizontal Beamwidth, Degrees — the diagram prints bits "
                      "15..0", rendered)
        self.assertIn("Vertical Beamwidth, Degrees — the diagram prints bits "
                      "15..0", rendered)

    def test_it_says_the_halves_are_not_established(self):
        rendered = safe_rendering(two_cells())

        self.assertIn("does not establish that either field occupies the upper "
                      "or the lower half", rendered)

    def test_it_keeps_the_citation(self):
        self.assertIn("std-a", safe_rendering(two_cells()))

    def test_it_invents_no_relationship_between_the_cells(self):
        rendered = safe_rendering(two_cells())

        for tell in ("31..16", "concatenat", "first half", "second half"):
            self.assertNotIn(tell, rendered)

    def test_the_guard_replaces_the_moved_claim(self):
        guarded, violations, replaced = guard(
            "Vertical Beamwidth occupies bits 31..16 of the word.",
            two_cells())

        self.assertTrue(replaced)
        self.assertNotIn("31..16", guarded)


class Untouched(unittest.TestCase):
    """Everything the geometry adapter must not have moved."""

    def structure_ledger(self):
        return EvidenceLedger().observe({
            "definition_id": "bfd-0ccd61a481e290d5",
            "structural_completeness": "STRUCTURALLY_INCOMPLETE",
            "citation_source_ids": ["std-" + "b" * 32],
            "fields": [{"display_label": "Phase Offset, Radians", "msb": 15,
                        "lsb": 0, "width": 16, "word_index": 1}],
            "unresolved": [{"label": "Reserved"}],
            "words": [{"word_index": 1, "word_width": 32}]})

    def test_h4_still_refuses_the_invented_reserved_range(self):
        self.assertTrue(validate("Reserved occupies bits 31..16 of word 1.",
                                 self.structure_ledger()))

    def test_h4_rendering_gains_no_cell_section(self):
        rendered = safe_rendering(self.structure_ledger())

        self.assertNotIn("diagram prints bits", rendered)
        self.assertIn("Reserved — normative position and width are not "
                      "established.", rendered)

    def test_a_structure_ledger_holds_no_cell_facts(self):
        self.assertEqual(self.structure_ledger().cell_fields, {})

    def test_octet_entailment_is_unchanged(self):
        from structural_entailment import Member, unique_placement
        proof = unique_placement(
            Member("Reserved", None, 4),
            [Member("A", 0, 4), Member("B", 8, 4), Member("Reserved", None, 4)],
            12)

        self.assertTrue(proof.entailed)
        self.assertEqual(proof.interval, (4, 7))

    def test_the_zero_tool_bootstrap_is_unchanged(self):
        import evidence_bootstrap

        self.assertTrue(evidence_bootstrap.should_bootstrap(
            "Give the bit range of the Beamwidth field.", [],
            answer="something"))

    def test_the_empty_structure_recovery_is_unchanged(self):
        import evidence_recovery

        calls = [{"tool": "standard.get_structure", "origin": "MODEL",
                  "empty_structure": True}]

        self.assertTrue(evidence_recovery.should_recover(
            "Give the word number of the Coarse Time field.", calls,
            answer="Nothing is defined."))

    def test_the_round_budget_rendering_is_unchanged(self):
        from evidence_progress import ProgressTracker, bounded_absence
        rendered = bounded_absence("What bit positions does X occupy?",
                                   self.structure_ledger(), ProgressTracker(),
                                   reason="ROUND_BUDGET", rounds=14)

        self.assertIn("does not establish the requested bit positions",
                      rendered)


if __name__ == "__main__":
    unittest.main()
