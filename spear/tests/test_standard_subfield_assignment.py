"""Two 16-bit quantities in one word, and the rule that says which half is which.

A label reading (15..0) twice in the same word is not two fields fighting over
the same bits; it is two subfields of one word, each numbering its own bits
from zero. The diagram never says which half each one occupies. Some sections
of the standard do, in prose, and that prose is the only authority this phase
accepts. Where it is absent or incomplete the pair stays refused.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_subfield_assignment import (
    AMBIGUOUS_NAME_MATCH, CROSS_REFERENCE_UNRESOLVED, NO_ASSIGNMENT,
    NORMATIVE_SUBFIELD_ASSIGNMENT, PARTIAL_ASSIGNMENT, SLOT_CONFLICT,
    STATED_RANGE_CONFLICT, AssignmentForm, PackingSlot,
    normalize_quantity, packed_slot_projection, subfield_assignments,
)

SECTION = "9.9.9"
SOURCE = "std-" + "0" * 32


class _Unit:
    """The smallest thing the extractor reads: one canonical unit."""

    def __init__(self, text, *, section=SECTION, source_id=SOURCE,
                 content_type="REQUIREMENT", page=1):
        self.text = text
        self.section = section
        self.source_id = source_id
        self.page = page

        class _Value:
            def __init__(self, v): self.value = v
        self.content_type = _Value(content_type)
        self.modality = _Value("SHALL")


class Normalization(unittest.TestCase):
    """Bounded normalization, never fuzzy matching."""

    def test_case_and_punctuation_are_removed(self):
        self.assertEqual(normalize_quantity("Stage 1 Gain (15..0), dB"),
                         normalize_quantity("stage 1 gain, DB"))

    def test_written_ordinals_become_digits(self):
        self.assertEqual(normalize_quantity("input third order intercept"),
                         normalize_quantity("Input 3 Order Intercept"))
        self.assertEqual(normalize_quantity("Input 2nd Order Intercept"),
                         normalize_quantity("input second order intercept"))

    def test_distinct_quantities_stay_distinct(self):
        self.assertNotEqual(normalize_quantity("Stage 1 Gain"),
                            normalize_quantity("Stage 2 Gain"))
        self.assertNotEqual(normalize_quantity("input second order intercept"),
                            normalize_quantity("input third order intercept"))


class Extraction(unittest.TestCase):
    """The three sentence shapes the standard actually uses."""

    def test_a_subject_named_assignment(self):
        found, _ = subfield_assignments([_Unit(
            "The Alpha Level value shall be expressed in two's-complement "
            "format in the lower 16 bits of the Level field.")], section=SECTION)
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].slot, PackingSlot.LOW_SLOT)
        self.assertIs(found[0].form, AssignmentForm.SUBJECT_NAMED)
        self.assertEqual(found[0].source_id, SOURCE)

    def test_a_portion_named_assignment_with_explicit_bits(self):
        found, _ = subfield_assignments([_Unit(
            "The upper portion (bits 31..16) of the Span field shall express "
            "the peak excursion of the device.")], section=SECTION)
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].slot, PackingSlot.HIGH_SLOT)
        self.assertIs(found[0].form, AssignmentForm.PORTION_NAMED)
        self.assertEqual((found[0].msb, found[0].lsb), (31, 16))

    def test_a_respectively_assignment_names_both(self):
        found, _ = subfield_assignments([_Unit(
            "The Level Field contains two 16-bit subfields, Alpha Level and "
            "Beta Level, which occupy the lower and upper 16 bits of the "
            "Level Field, respectively.")], section=SECTION)
        self.assertEqual(len(found), 2)
        slots = {tuple(item.normalized): item.slot for item in found}
        self.assertIs(slots[normalize_quantity("Alpha Level")],
                      PackingSlot.LOW_SLOT)
        self.assertIs(slots[normalize_quantity("Beta Level")],
                      PackingSlot.HIGH_SLOT)

    def test_an_alias_definition_is_recorded_separately(self):
        _, aliases = subfield_assignments([_Unit(
            "The CMR field shall express the common-mode range of the "
            "receiver from the Reference Point.")], section=SECTION)
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0].normalized, normalize_quantity("CMR"))
        self.assertEqual(aliases[0].means,
                         normalize_quantity("common mode range"))

    def test_prose_from_another_section_is_not_read(self):
        found, _ = subfield_assignments([_Unit(
            "The Other Thing shall be expressed in the upper 16 bits of it.",
            section="1.2.3")], section=SECTION)
        self.assertEqual(found, ())

    def test_non_normative_prose_is_not_read(self):
        found, _ = subfield_assignments([_Unit(
            "The Alpha Level value is in the lower 16 bits of the Level field.",
            content_type="UNKNOWN")], section=SECTION)
        self.assertEqual(found, ())


def diagram(*labels, word=1, widths=None):
    """A minimal bitfield/table payload holding one word of named labels."""
    cells, rows = [], []
    for index, text in enumerate(labels):
        cells.append({
            "cell_id": f"cel-{index:016d}", "row_index": 1, "column_index": index,
            "page": 1, "bbox": [100.0 + index * 200, 100.0,
                                260.0 + index * 200, 112.0],
            "text": text, "source_ids": [SOURCE],
            "provenance": "DIRECT_TEXT_MATCH", "semantic": True})
    rows.append({"row_index": 1, "page": 1, "bbox": [60.0, 100.0, 560.0, 112.0],
                 "cells": cells, "header_candidate": "FALSE"})
    table = {"table_id": "tbl-packed", "rows": rows,
             "page_start": 1, "page_end": 1,
             "bbox": [60.0, 60.0, 560.0, 120.0]}
    bitfield = {"bitfield_id": "bit-" + "0" * 16, "table_id": "tbl-packed",
                "page": 1, "bit_labels": [], "spans": []}
    associations = {c["cell_id"]: word for c in cells}
    return bitfield, table, associations


class Projection(unittest.TestCase):
    """Turning an assignment into a physical slot, or refusing to."""

    def project(self, labels, sentences, *, associations=None, word=1):
        bitfield, table, assoc = diagram(*labels, word=word)
        if associations is not None:
            assoc = {c: associations[i] for i, c in enumerate(sorted(assoc))}
        units = [_Unit(s) for s in sentences]
        return packed_slot_projection(bitfield, table, units,
                                      associations=assoc, section=SECTION)

    CLEAN = ["Alpha Value (15..0), dB", "Beta Value (15..0), dB"]
    PROSE = ["The Alpha Value shall be expressed in the upper 16 bits of the F.",
             "The Beta Value shall be expressed in the lower 16 bits of the F."]

    def test_a_clean_two_field_assignment_projects(self):
        found, refusals = self.project(self.CLEAN, self.PROSE)
        self.assertEqual(refusals, ())
        self.assertEqual(len(found), 2)
        high = next(v for v in found.values() if v.slot is PackingSlot.HIGH_SLOT)
        low = next(v for v in found.values() if v.slot is PackingSlot.LOW_SLOT)
        self.assertEqual((high.msb, high.lsb), (31, 16))
        self.assertEqual((low.msb, low.lsb), (15, 0))
        self.assertEqual((high.declared_msb, high.declared_lsb), (15, 0))
        self.assertEqual((low.declared_msb, low.declared_lsb), (15, 0))

    def test_the_projection_names_its_authority_and_source(self):
        found, _ = self.project(self.CLEAN, self.PROSE)
        for item in found.values():
            self.assertEqual(item.authority, NORMATIVE_SUBFIELD_ASSIGNMENT)
            self.assertEqual(item.source_id, SOURCE)

    def test_both_members_share_one_packing_group(self):
        found, _ = self.project(self.CLEAN, self.PROSE)
        ids = {item.packing_group_id for item in found.values()}
        self.assertEqual(len(ids), 1)
        self.assertTrue(next(iter(ids)).startswith("pkg-"))

    def test_the_group_is_not_a_single_composite_value(self):
        found, _ = self.project(self.CLEAN, self.PROSE)
        for item in found.values():
            # Each quantity stays its own 16-bit value. Nothing here says the
            # two of them are one 32-bit number.
            self.assertEqual(item.value_width, 16)
            self.assertNotIn("segment", json.dumps(item.to_dict()))

    def test_explicit_physical_bits_in_prose_project(self):
        found, refusals = self.project(self.CLEAN, [
            "The upper portion (bits 31..16) of the F field shall express the "
            "Alpha Value of the device.",
            "The lower portion (bits 15..0) of the F field shall express the "
            "Beta Value of the device."])
        self.assertEqual(refusals, ())
        high = next(v for v in found.values() if v.slot is PackingSlot.HIGH_SLOT)
        self.assertEqual((high.msb, high.lsb), (31, 16))

    def test_sentence_order_does_not_decide_the_slots(self):
        forward, _ = self.project(self.CLEAN, self.PROSE)
        backward, _ = self.project(self.CLEAN, list(reversed(self.PROSE)))
        self.assertEqual(
            {c: (v.slot.value, v.msb, v.lsb) for c, v in forward.items()},
            {c: (v.slot.value, v.msb, v.lsb) for c, v in backward.items()})

    def test_an_alias_lets_a_defined_name_match(self):
        found, refusals = self.project(
            ["Common Mode Range (15..0), dB", "Drift Margin (15..0), dB"],
            ["The CMR field shall express the common-mode range of the "
             "receiver from the Reference Point.",
             "The CMR field shall be expressed in the upper 16 bits of the F.",
             "The Drift Margin field shall be expressed in the lower 16 bits "
             "of the Drift Margin field."])
        self.assertEqual(refusals, ())
        high = next(v for v in found.values() if v.slot is PackingSlot.HIGH_SLOT)
        self.assertEqual(high.alias_of, normalize_quantity("CMR"))


class Refusals(Projection):
    """Every way the authority is not good enough."""

    def test_both_assigned_to_the_same_slot(self):
        found, refusals = self.project(self.CLEAN, [
            "The Alpha Value shall be expressed in the upper 16 bits of the F.",
            "The Beta Value shall be expressed in the upper 16 bits of the F."])
        self.assertEqual(found, {})
        self.assertIn(SLOT_CONFLICT, {r["reason"] for r in refusals})

    def test_only_one_quantity_assigned(self):
        found, refusals = self.project(self.CLEAN, [self.PROSE[0]])
        self.assertEqual(found, {})
        self.assertIn(PARTIAL_ASSIGNMENT, {r["reason"] for r in refusals})

    def test_no_prose_at_all(self):
        found, refusals = self.project(self.CLEAN, [])
        self.assertEqual(found, {})
        self.assertIn(NO_ASSIGNMENT, {r["reason"] for r in refusals})

    def test_a_name_matching_both_candidates(self):
        found, refusals = self.project(
            ["Gain (15..0), dB", "Gain (15..0), dB"],
            ["The Gain shall be expressed in the upper 16 bits of the F.",
             "The Gain shall be expressed in the lower 16 bits of the F."])
        self.assertEqual(found, {})
        self.assertTrue({AMBIGUOUS_NAME_MATCH, SLOT_CONFLICT}
                        & {r["reason"] for r in refusals})

    def test_an_incomplete_singular_rule_is_not_a_mapping(self):
        # The shape that defeats a mapping: one sentence puts "the value" in
        # the lower half and never says which of the two quantities it means.
        found, refusals = self.project(
            ["Horizontal Spanwidth (15..0), Degrees",
             "Vertical Spanwidth (15..0), Degrees"],
            ["The Span width value shall be expressed in two's-complement "
             "format in the lower 16 bits of the Span width field."])
        self.assertEqual(found, {})
        self.assertTrue({PARTIAL_ASSIGNMENT, NO_ASSIGNMENT}
                        & {r["reason"] for r in refusals})

    def test_a_stated_physical_range_wins_over_prose(self):
        found, refusals = self.project(
            ["Alpha Value (31..16), dB", "Beta Value (15..0), dB"],
            ["The Alpha Value shall be expressed in the lower 16 bits of the F.",
             "The Beta Value shall be expressed in the upper 16 bits of the F."])
        self.assertEqual(found, {})
        self.assertIn(STATED_RANGE_CONFLICT, {r["reason"] for r in refusals})

    def test_three_labels_in_one_word(self):
        found, refusals = self.project(
            ["Alpha Value (15..0), dB", "Beta Value (15..0), dB",
             "Gamma Value (15..0), dB"], self.PROSE)
        self.assertEqual(found, {})

    def test_one_label_only(self):
        found, refusals = self.project(["Alpha Value (15..0), dB"], self.PROSE)
        self.assertEqual(found, {})

    def test_labels_in_different_words(self):
        found, refusals = self.project(self.CLEAN, self.PROSE,
                                       associations=[1, 2])
        self.assertEqual(found, {})


class CrossReference(Projection):
    """One hop, explicit, and inspectable."""

    def project_chain(self, local, referenced):
        bitfield, table, assoc = diagram(*self.CLEAN)
        units = [_Unit(local, section=SECTION, source_id="std-" + "1" * 32)]
        units += [_Unit(s, section="9.5.3", source_id="std-" + "2" * 32)
                  for s in referenced]
        return packed_slot_projection(bitfield, table, units,
                                      associations=assoc, section=SECTION)

    def test_a_referenced_rule_supplies_the_assignment(self):
        found, refusals = self.project_chain(
            "The Widget Subfield shall have the format shown in Figure 1 and "
            "follow the regulations of the Level Field in Section 9.5.3.",
            self.PROSE)
        self.assertEqual(refusals, ())
        self.assertEqual(len(found), 2)
        for item in found.values():
            self.assertEqual(item.cross_reference, "9.5.3")
            self.assertEqual(item.source_id, "std-" + "2" * 32)
            self.assertEqual(item.reference_source_id, "std-" + "1" * 32)

    def test_a_reference_to_a_section_with_no_assignment_refuses(self):
        found, refusals = self.project_chain(
            "The Widget Subfield shall follow the regulations of Section 9.5.3.",
            ["The Gain Field is described elsewhere."])
        self.assertEqual(found, {})
        self.assertTrue({CROSS_REFERENCE_UNRESOLVED, NO_ASSIGNMENT}
                        & {r["reason"] for r in refusals})

    def test_a_chain_is_not_followed_twice(self):
        bitfield, table, assoc = diagram(*self.CLEAN)
        units = [_Unit("Follow the regulations in Section 9.5.3.",
                       section=SECTION, source_id="std-" + "1" * 32),
                 _Unit("Follow the regulations in Section 9.9.1.",
                       section="9.5.3", source_id="std-" + "2" * 32)]
        units += [_Unit(s, section="9.9.1", source_id="std-" + "3" * 32)
                  for s in self.PROSE]
        found, refusals = packed_slot_projection(
            bitfield, table, units, associations=assoc, section=SECTION)
        self.assertEqual(found, {})


class Determinism(Projection):
    def test_the_projection_is_stable(self):
        first, _ = self.project(self.CLEAN, self.PROSE)
        second, _ = self.project(self.CLEAN, self.PROSE)
        self.assertEqual(
            json.dumps({c: v.to_dict() for c, v in first.items()}, sort_keys=True),
            json.dumps({c: v.to_dict() for c, v in second.items()}, sort_keys=True))

    def test_the_group_id_does_not_depend_on_time_or_path(self):
        found, _ = self.project(self.CLEAN, self.PROSE)
        group = next(iter(found.values())).packing_group_id
        for absent in ("/", "20", "tmp"):
            self.assertNotIn(absent, group[4:])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
