"""The audit that decides whether a negative pair is allowed to exist.

A container fully accounted for, member widths known, one arrangement left --
and the "unsupported" complement is a deduction. FT0.1C found five of those
about to be written into a preference set as failures. These tests pin the
counting that catches them, and pin that an inconclusive count is a refusal
rather than a pass.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "abstention"))

import complement
import pressure
import shapes
from entailment import AMBIGUOUS, ENTAILED, NOT_ENTAILED, Facts, audit, facts_for
from families import BY_KEY


class Counting(unittest.TestCase):
    def test_one_free_slot_and_one_unplaced_member_is_entailed(self):
        verdict, why = audit(Facts(32, 16, 1, 2))
        self.assertEqual(verdict, ENTAILED)
        self.assertIn("one possible arrangement", why)

    def test_two_members_over_two_free_slots_are_not_entailed(self):
        # Which member goes where is exactly what is unstated.
        self.assertEqual(audit(Facts(32, 16, 0, 2))[0], NOT_ENTAILED)

    def test_a_spare_slot_leaves_the_placement_open(self):
        self.assertEqual(audit(Facts(96, 32, 1, 2))[0], NOT_ENTAILED)

    def test_nothing_unplaced_means_nothing_to_infer(self):
        self.assertEqual(audit(Facts(32, 16, 2, 2))[0], NOT_ENTAILED)

    def test_a_missing_meaning_can_never_be_entailed(self):
        # No amount of counting settles what an encoding means.
        verdict, why = audit(Facts(32, 16, 3, 4, semantic_gap=True))
        self.assertEqual(verdict, NOT_ENTAILED)
        self.assertIn("meaning", why)

    def test_nothing_placed_leaves_elimination_nothing_to_work_from(self):
        self.assertEqual(audit(Facts(32, None, 0, 1))[0], NOT_ENTAILED)

    def test_an_uncountable_container_is_ambiguous_not_permitted(self):
        # AMBIGUOUS is a refusal: not knowing is not evidence of openness.
        self.assertEqual(audit(Facts(32, None, 1, 2))[0], AMBIGUOUS)

    def test_exhaustive_wording_is_ambiguous(self):
        verdict, _ = audit(Facts(None, None, 1, 2),
                           "the field carries exactly two values")
        self.assertEqual(verdict, AMBIGUOUS)


class RealShapes(unittest.TestCase):
    """The verdicts the actual scenarios get."""

    def test_the_packing_slot_shape_is_entailed_and_stays_out(self):
        scenario = shapes.packing_slot_unknown(BY_KEY["iq"])
        verdict, _ = audit(facts_for(scenario, members_in_container=2,
                                     stated_placements=1, member_bits=16))
        self.assertEqual(verdict, ENTAILED)

    def test_the_partial_slot_shape_is_open(self):
        scenario = shapes.partial_slot(BY_KEY["iq"])
        verdict, _ = audit(facts_for(scenario, members_in_container=2,
                                     stated_placements=0, member_bits=16))
        self.assertEqual(verdict, NOT_ENTAILED)

    def test_the_twelve_octet_record_is_open(self):
        scenario = complement.byte_offset(BY_KEY["txrx"])
        verdict, why = audit(facts_for(scenario, members_in_container=2,
                                       stated_placements=1, member_bits=32))
        self.assertEqual(verdict, NOT_ENTAILED)
        self.assertIn("2 arrangements", why)

    def test_a_pressure_gap_is_about_meaning(self):
        scenario = pressure.enum_gap(BY_KEY["latlon"], pressure.EXPLICIT_FILL)
        facts = facts_for(scenario, members_in_container=4,
                          stated_placements=3, semantic_gap=True)
        self.assertEqual(audit(facts)[0], NOT_ENTAILED)

    def test_widths_are_read_from_a_structure_when_prose_omits_them(self):
        scenario = shapes.packing_slot_unknown(BY_KEY["iq"])
        facts = facts_for(scenario, members_in_container=2,
                          stated_placements=1)
        self.assertEqual(facts.member_bits, 16)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
