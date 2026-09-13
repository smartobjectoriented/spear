"""A 64-bit value split over two words, and the many ways that is not proven.

Every diagram here is synthetic. The point of the phase is that a label saying
63..32 is understood without being believed, so most of these cases exist to
check that a pair is refused and the label stays exactly where it was.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, CoordinateDomain, PositionSource, SpanRole,
    StandardBitfieldApproval, UnresolvedCause, associate_words,
    promote_bitfield, review_evidence, review_evidence_fingerprint,
    unresolved_label_of,
)
from standard_structure import ProvenanceQuality
from standard_value_pair import (
    ADJACENT_WORD_PAIR, AMBIGUOUS_SEGMENTS, NO_LOW_SEGMENT, PROVENANCE_NOT_DIRECT,
    SEGMENT_WIDTH_UNSUPPORTED, SUPPORTED_VALUE_WIDTH, SegmentRole,
    WORDS_NOT_ADJACENT, WORD_IDENTITY_NOT_EXPLICIT, WORD_ORDER_REVERSED,
    normalize_value_base, pair_segments, value_local_pairs,
)
from standard_word_association import field_candidates
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, SID, REV, bitfield, cell, ruler_row, table, word_row,
)

STRUCTURE = "c" * 64


def diagram(*words, word_column: bool = True):
    """One ruler row and one band holding the given words, one per line."""
    rows = (ruler_row(0),
            word_row(1, tuple(words), word_column=word_column))
    source = table(rows)
    return bitfield(source), source


def pairs_of(*words, **kwargs):
    candidate, source = diagram(*words, **kwargs)
    return value_local_pairs(candidate.to_dict(), source.to_dict())


def refusal_reasons(refusals):
    return {item["reason"] for item in refusals}


HIGH = "Bandwidth (63..32), Hz"
LOW = "Bandwidth (31..0), Hz"


class ProvenPair(unittest.TestCase):
    """The one shape this phase supports."""

    def setUp(self):
        self.pairs, self.refusals = pairs_of((1, (HIGH,)), (2, (LOW,)))

    def test_a_clean_high_and_low_half_pair(self):
        self.assertEqual(len(self.pairs), 1)
        self.assertEqual(self.refusals, ())

    def test_the_value_is_sixty_four_bits_wide(self):
        self.assertEqual(self.pairs[0].value_width, SUPPORTED_VALUE_WIDTH)
        self.assertEqual(self.pairs[0].segment_count, 2)

    def test_the_high_segment_keeps_its_declared_range(self):
        high = self.pairs[0].high
        self.assertEqual((high.declared_msb, high.declared_lsb), (63, 32))
        self.assertIs(high.segment_role, SegmentRole.HIGH)
        self.assertEqual(high.segment_index, 0)

    def test_the_low_segment_keeps_its_declared_range(self):
        low = self.pairs[0].low
        self.assertEqual((low.declared_msb, low.declared_lsb), (31, 0))
        self.assertIs(low.segment_role, SegmentRole.LOW)
        self.assertEqual(low.segment_index, 1)

    def test_only_the_high_segment_is_value_local(self):
        self.assertTrue(self.pairs[0].high.value_local)
        self.assertFalse(self.pairs[0].low.value_local)
        self.assertIs(self.pairs[0].high.coordinate_domain,
                      CoordinateDomain.VALUE_LOCAL)
        self.assertIs(self.pairs[0].low.coordinate_domain,
                      CoordinateDomain.WORD_LOCAL)

    def test_both_words_are_explicit_and_adjacent(self):
        self.assertEqual(self.pairs[0].high.word_index, 1)
        self.assertEqual(self.pairs[0].low.word_index, 2)

    def test_one_group_covers_both_segments(self):
        self.assertEqual(len(self.pairs[0].cell_ids), 2)
        self.assertTrue(self.pairs[0].group_id.startswith("vgr-"))
        self.assertEqual({item.cell_id for item in self.pairs[0].segments},
                         set(self.pairs[0].cell_ids))

    def test_the_projection_names_its_evidence(self):
        self.assertEqual(self.pairs[0].projection_source, ADJACENT_WORD_PAIR)

    def test_the_pairing_is_deterministic(self):
        again, _ = pairs_of((1, (HIGH,)), (2, (LOW,)))
        self.assertEqual(json.dumps(self.pairs[0].to_dict(), sort_keys=True),
                         json.dumps(again[0].to_dict(), sort_keys=True))

    def test_no_packet_global_offset_is_introduced(self):
        # Every number in the pair is either a value coordinate or a word
        # coordinate. Nothing counts bits from the start of a packet.
        payload = json.dumps(self.pairs[0].to_dict())
        for absent in ("packet", "offset", "byte"):
            self.assertNotIn(absent, payload.lower())


class BaseNames(unittest.TestCase):
    def test_the_range_token_is_removed_and_the_unit_kept(self):
        self.assertEqual(normalize_value_base(HIGH, range_text="(63..32)"),
                         "BANDWIDTH HZ")
        self.assertEqual(normalize_value_base(LOW, range_text="(31..0)"),
                         "BANDWIDTH HZ")

    def test_two_halves_of_one_quantity_normalize_alike(self):
        self.assertEqual(
            normalize_value_base("Sample Rate (63..32), Hz", range_text="(63..32)"),
            normalize_value_base("Sample Rate (31..0), Hz", range_text="(31..0)"))


class Deferred(unittest.TestCase):
    """Everything the rule refuses, and the reason it gives."""

    def test_a_high_half_with_no_low_half(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, ("Something Else (31..0)",)))
        self.assertEqual(pairs, ())
        self.assertIn(NO_LOW_SEGMENT, refusal_reasons(refusals))

    def test_a_different_base_label_is_not_a_partner(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, ("Sample Rate (31..0), Hz",)))
        self.assertEqual(pairs, ())
        self.assertIn(NO_LOW_SEGMENT, refusal_reasons(refusals))

    def test_an_incompatible_unit_is_not_a_partner(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, ("Bandwidth (31..0), dB",)))
        self.assertEqual(pairs, ())
        self.assertIn(NO_LOW_SEGMENT, refusal_reasons(refusals))

    def test_a_non_adjacent_partner(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, ("Filler (31..0)",)),
                                   (3, (LOW,)))
        self.assertEqual(pairs, ())
        self.assertIn(WORDS_NOT_ADJACENT, refusal_reasons(refusals))

    def test_the_low_half_may_not_precede_the_high_half(self):
        pairs, refusals = pairs_of((1, (LOW,)), (2, (HIGH,)))
        self.assertEqual(pairs, ())
        self.assertIn(WORD_ORDER_REVERSED, refusal_reasons(refusals))

    def test_a_sixty_two_bit_high_half_never_becomes_sixty_three(self):
        pairs, refusals = pairs_of((1, ("Bandwidth (62..32), Hz",)), (2, (LOW,)))
        self.assertEqual(pairs, ())
        self.assertIn(SEGMENT_WIDTH_UNSUPPORTED, refusal_reasons(refusals))

    def test_a_thirty_bit_low_half_is_not_a_partner(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, ("Bandwidth (30..0), Hz",)))
        self.assertEqual(pairs, ())
        self.assertIn(NO_LOW_SEGMENT, refusal_reasons(refusals))

    def test_two_low_halves_are_ambiguous(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, (LOW,)), (3, (LOW,)))
        self.assertEqual(pairs, ())
        self.assertIn(AMBIGUOUS_SEGMENTS, refusal_reasons(refusals))

    def test_two_high_halves_are_ambiguous(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, (HIGH,)), (3, (LOW,)))
        self.assertEqual(pairs, ())
        self.assertIn(AMBIGUOUS_SEGMENTS, refusal_reasons(refusals))

    def test_a_three_segment_group_is_ambiguous(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, (LOW,)), (3, (HIGH,)))
        self.assertEqual(pairs, ())
        self.assertIn(AMBIGUOUS_SEGMENTS, refusal_reasons(refusals))

    def test_a_geometric_word_identity_is_not_enough(self):
        pairs, refusals = pairs_of((1, (HIGH,)), (2, (LOW,)), word_column=False)
        self.assertEqual(pairs, ())
        self.assertIn(WORD_IDENTITY_NOT_EXPLICIT, refusal_reasons(refusals))

    def test_inherited_provenance_is_not_enough(self):
        rows = (ruler_row(0),
                word_row(1, ((1, (HIGH,)), (2, (LOW,)))))
        source = table(rows)
        payload = source.to_dict()
        for row in payload["rows"]:
            for item in row["cells"]:
                if item["text"] == HIGH:
                    item["provenance"] = ProvenanceQuality.ROW_INHERITED.value
        pairs, refusals = value_local_pairs(
            bitfield(source).to_dict(), payload)
        self.assertEqual(pairs, ())
        self.assertIn(PROVENANCE_NOT_DIRECT, refusal_reasons(refusals))

    def test_repeated_word_local_ranges_are_untouched(self):
        # The historical family: several 15..0 labels in one diagram. They are
        # a different problem and this phase must not reach them.
        pairs, refusals = pairs_of((1, ("Angle (15..0), degrees",)),
                                   (2, ("Angle (15..0), degrees",)))
        self.assertEqual(pairs, ())
        self.assertEqual(refusals, ())

    def test_an_ordinary_word_local_diagram_produces_nothing(self):
        pairs, refusals = pairs_of((1, ("Count (31..0)",)), (2, ("Flags (7..0)",)))
        self.assertEqual(pairs, ())
        self.assertEqual(refusals, ())

    def test_a_label_with_no_range_produces_nothing(self):
        pairs, refusals = pairs_of((1, ("Reserved",)), (2, ("TSI",)))
        self.assertEqual(pairs, ())
        self.assertEqual(refusals, ())


class Projection(unittest.TestCase):
    """What a promoted pair looks like, and what it refuses to claim."""

    def promote(self, *words, roles=None):
        candidate, source = diagram(*words)
        payload, table_payload = candidate.to_dict(), source.to_dict()
        pairs, _ = value_local_pairs(payload, table_payload)
        segments = {cell_id: segment for pair in pairs
                    for cell_id, segment in pair.by_cell().items()}
        originals = field_candidates(payload, table_payload,
                                     value_local_cells=frozenset(segments))
        roles = tuple(roles or [SpanRole.FIELD.value] * len(originals))
        approval = StandardBitfieldApproval(
            candidate_id=candidate.bitfield_id, verdict="PASS",
            reviewer="test-operator", reviewed_at="1970-01-01T00:00:00+00:00",
            structure_fingerprint=STRUCTURE, span_roles=roles,
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=(),
            review_evidence_fingerprint=review_evidence_fingerprint(
                candidate.bitfield_id, STRUCTURE, table=table_payload,
                originals=originals, roles=roles,
                associations=associate_words(table_payload), reviewed_links=(),
                value_segments=segments))
        return promote_bitfield(
            candidate, source, approval, standard_id=SID, revision=REV,
            corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
            structure_fingerprint=STRUCTURE)

    def setUp(self):
        self.definition, self.blocked = self.promote((1, (HIGH,)), (2, (LOW,)))
        self.assertIsNone(self.blocked)
        self.payload = self.definition.to_dict()
        self.high = next(f for f in self.payload["fields"] if f["word_index"] == 1)
        self.low = next(f for f in self.payload["fields"] if f["word_index"] == 2)

    def test_both_halves_are_promoted(self):
        self.assertEqual(len(self.payload["fields"]), 2)

    def test_the_high_segment_occupies_its_whole_word(self):
        self.assertEqual((self.high["msb"], self.high["lsb"]), (31, 0))
        self.assertEqual(self.high["width"], 32)

    def test_the_high_segment_still_declares_value_bits(self):
        self.assertEqual((self.high["declared_msb"], self.high["declared_lsb"]),
                         (63, 32))
        self.assertEqual(self.high["coordinate_domain"],
                         CoordinateDomain.VALUE_LOCAL.value)

    def test_the_high_segment_never_claims_the_label_said_word_bits(self):
        self.assertEqual(self.high["position_source"],
                         PositionSource.VALUE_LOCAL_PROJECTION.value)
        self.assertNotEqual(self.high["position_source"],
                            PositionSource.STATED_RANGE.value)
        self.assertEqual(self.high["projection_source"], ADJACENT_WORD_PAIR)

    def test_the_low_segment_is_an_ordinary_stated_range(self):
        self.assertEqual((self.low["msb"], self.low["lsb"]), (31, 0))
        self.assertEqual((self.low["declared_msb"], self.low["declared_lsb"]),
                         (31, 0))
        self.assertEqual(self.low["coordinate_domain"],
                         CoordinateDomain.WORD_LOCAL.value)
        self.assertEqual(self.low["position_source"],
                         PositionSource.STATED_RANGE.value)

    def test_both_halves_share_one_value_group(self):
        self.assertEqual(self.high["value_group_id"], self.low["value_group_id"])
        self.assertIsNotNone(self.high["value_group_id"])
        self.assertEqual(self.high["value_width"], SUPPORTED_VALUE_WIDTH)
        self.assertEqual((self.high["segment_index"], self.low["segment_index"]),
                         (0, 1))

    def test_the_definition_is_structurally_complete(self):
        self.assertEqual(self.payload["structural_completeness"],
                         "STRUCTURALLY_COMPLETE")
        self.assertEqual(self.payload["unresolved_words"], [])

    def test_an_ordinary_field_declares_its_own_word_bits(self):
        definition, blocked = self.promote((1, ("Count (31..0)",)),
                                           (2, ("Flags (7..0)",)))
        self.assertIsNone(blocked)
        for field in definition.to_dict()["fields"]:
            self.assertEqual(field["coordinate_domain"],
                             CoordinateDomain.WORD_LOCAL.value)
            self.assertEqual(field["declared_msb"], field["msb"])
            self.assertEqual(field["declared_lsb"], field["lsb"])
            self.assertIsNone(field["value_group_id"])

    def test_an_unpaired_high_half_stays_unresolved(self):
        definition, blocked = self.promote((1, (HIGH,)), (2, ("Count (31..0)",)))
        self.assertIsNone(blocked)
        payload = definition.to_dict()
        self.assertEqual(len(payload["fields"]), 1)
        word = next(item for item in payload["unresolved_words"]
                    if item["word_index"] == 1)
        self.assertEqual(word["causes"], [UnresolvedCause.VALUE_LOCAL_DEFERRED.value])

    def test_serialization_is_deterministic(self):
        again, _ = self.promote((1, (HIGH,)), (2, (LOW,)))
        self.assertEqual(json.dumps(self.payload, sort_keys=True),
                         json.dumps(again.to_dict(), sort_keys=True))
        self.assertEqual(self.definition.definition_id, again.definition_id)


class UnresolvedCauseModel(unittest.TestCase):
    """Why a label has no position, without parsing the label."""

    def label_for(self, text):
        _, source = diagram((1, (text,)), (2, ("Count (31..0)",)))
        found = next(item for row in source.to_dict()["rows"]
                     for item in row["cells"] if item["text"] == text)
        return unresolved_label_of(found, ruler_labels=tuple(range(31, -1, -1)))

    def test_a_label_with_no_range_is_not_understood(self):
        found = self.label_for("Reserved")
        self.assertIs(found.cause, UnresolvedCause.NO_DETERMINISTIC_RANGE)
        self.assertIsNone(found.declared_msb)

    def test_a_value_local_label_is_understood_and_deferred(self):
        found = self.label_for(HIGH)
        self.assertIs(found.cause, UnresolvedCause.VALUE_LOCAL_DEFERRED)
        self.assertEqual((found.declared_msb, found.declared_lsb), (63, 32))

    def test_the_cause_never_requires_reading_the_label(self):
        for text in ("Reserved", HIGH, "TSI"):
            found = self.label_for(text)
            self.assertIn(found.cause, tuple(UnresolvedCause))
            self.assertEqual(found.to_dict()["cause"], found.cause.value)


class ReviewEvidence(unittest.TestCase):
    """A reviewer accepting a projection accepts the transformation."""

    def evidence(self, *words, segments=True):
        candidate, source = diagram(*words)
        payload, table_payload = candidate.to_dict(), source.to_dict()
        pairs, _ = value_local_pairs(payload, table_payload)
        found = {cell_id: segment for pair in pairs
                 for cell_id, segment in pair.by_cell().items()}
        originals = field_candidates(
            payload, table_payload, value_local_cells=frozenset(found))
        return review_evidence(
            candidate.bitfield_id, STRUCTURE, table=table_payload,
            originals=originals, roles=[SpanRole.FIELD.value] * len(originals),
            associations=associate_words(table_payload), reviewed_links=(),
            value_segments=found if segments else None)

    def test_a_projected_candidate_binds_its_declared_range(self):
        found = self.evidence((1, (HIGH,)), (2, (LOW,)))
        projected = [item for item in found["candidates"] if "value" in item]
        self.assertEqual(len(projected), 2)
        self.assertIn([63, 32, 0, "BANDWIDTH HZ"],
                      [item["value"] for item in projected])

    def test_an_ordinary_candidate_is_shaped_exactly_as_before(self):
        found = self.evidence((1, ("Count (31..0)",)), (2, ("Flags (7..0)",)))
        for item in found["candidates"]:
            self.assertEqual(sorted(item), ["cell", "role", "text", "word"])

    def test_a_diagram_with_no_pair_hashes_as_it_always_did(self):
        with_segments = self.evidence((1, ("Count (31..0)",)),
                                      (2, ("Flags (7..0)",)), segments=True)
        without = self.evidence((1, ("Count (31..0)",)),
                                (2, ("Flags (7..0)",)), segments=False)
        self.assertEqual(with_segments, without)

    def test_a_growing_candidate_set_changes_the_evidence(self):
        paired = self.evidence((1, (HIGH,)), (2, (LOW,)))
        unpaired = self.evidence((1, (HIGH,)), (2, ("Count (31..0)",)))
        self.assertNotEqual(len(paired["candidates"]), len(unpaired["candidates"]))


class NoMutation(unittest.TestCase):
    """Discovery reads; it never writes."""

    def test_pairing_leaves_its_inputs_alone(self):
        candidate, source = diagram((1, (HIGH,)), (2, (LOW,)))
        payload, table_payload = candidate.to_dict(), source.to_dict()
        before = json.dumps([payload, table_payload], sort_keys=True)
        value_local_pairs(payload, table_payload)
        pair_segments(payload, table_payload)
        self.assertEqual(before,
                         json.dumps([payload, table_payload], sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
