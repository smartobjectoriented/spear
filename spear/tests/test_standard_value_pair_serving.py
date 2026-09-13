"""What a model caller sees once a 64-bit value spans two words.

The contract has to carry two coordinate systems at once without letting a
caller mistake one for the other, and it has to keep working for every
definition written before either existed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, CoordinateDomain, PositionSource, SpanRole,
    StandardBitfieldApproval, associate_words, promote_bitfield,
    review_evidence_fingerprint,
)
from standard_structure_access import StandardStructureAccess
from standard_value_pair import SUPPORTED_VALUE_WIDTH, value_local_pairs
from standard_word_association import field_candidates
from tests.standard_word_fixture import (
    FINGERPRINT, LAYOUT, REV, SID, bitfield, ruler_row, table, word_row,
)

STRUCTURE = "c" * 64
HIGH = "Bandwidth (63..32), Hz"
LOW = "Bandwidth (31..0), Hz"


def promoted(*words):
    source = table((ruler_row(0), word_row(1, tuple(words))))
    candidate = bitfield(source)
    payload, table_payload = candidate.to_dict(), source.to_dict()
    pairs, _ = value_local_pairs(payload, table_payload)
    segments = {cell_id: segment for pair in pairs
                for cell_id, segment in pair.by_cell().items()}
    originals = field_candidates(payload, table_payload,
                                 value_local_cells=frozenset(segments))
    roles = tuple([SpanRole.FIELD.value] * len(originals))
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
    definition, blocked = promote_bitfield(
        candidate, source, approval, standard_id=SID, revision=REV,
        corpus_fingerprint=FINGERPRINT, layout_fingerprint=LAYOUT,
        structure_fingerprint=STRUCTURE)
    return definition, approval, payload, table_payload, blocked


class ServedContract(unittest.TestCase):
    """The rendered shape, taken straight from the serving path."""

    def setUp(self):
        definition, approval, payload, table_payload, blocked = promoted(
            (1, (HIGH,)), (2, (LOW,)))
        self.assertIsNone(blocked)
        self.served = StandardStructureAccess.__new__(
            StandardStructureAccess)._render(
                definition.to_dict(), approval, None, payload, table_payload,
                word_width=32)
        self.high = next(f for f in self.served["fields"] if f["word_index"] == 1)
        self.low = next(f for f in self.served["fields"] if f["word_index"] == 2)

    def test_both_coordinate_systems_are_served(self):
        self.assertEqual((self.high["msb"], self.high["lsb"]), (31, 0))
        self.assertEqual((self.high["declared_msb"], self.high["declared_lsb"]),
                         (63, 32))

    def test_the_coordinate_domain_is_named(self):
        self.assertEqual(self.high["coordinate_domain"],
                         CoordinateDomain.VALUE_LOCAL.value)
        self.assertEqual(self.low["coordinate_domain"],
                         CoordinateDomain.WORD_LOCAL.value)

    def test_a_caller_cannot_read_the_projection_as_a_stated_range(self):
        self.assertEqual(self.high["position_source"],
                         PositionSource.VALUE_LOCAL_PROJECTION.value)

    def test_the_paired_segment_is_reachable_from_the_group(self):
        groups = self.served["value_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["value_width"], SUPPORTED_VALUE_WIDTH)
        self.assertEqual([item["field_id"] for item in groups[0]["segments"]],
                         [self.high["field_id"], self.low["field_id"]])
        self.assertEqual([item["segment_index"] for item in groups[0]["segments"]],
                         [0, 1])

    def test_the_group_carries_the_reason_it_is_deterministic(self):
        self.assertIsNotNone(self.served["value_groups"][0]["projection_source"])

    def test_both_words_are_fully_covered(self):
        for word in self.served["words"]:
            self.assertEqual(word["covered_bits"], [[31, 0]])
            self.assertEqual(word["unclaimed_bits"], [])
            self.assertEqual(word["structural_status"], "resolved")

    def test_the_structure_is_complete(self):
        self.assertEqual(self.served["structural_completeness"],
                         "STRUCTURALLY_COMPLETE")
        self.assertEqual(self.served["unresolved_word_count"], 0)

    def test_no_invented_field_appears(self):
        self.assertEqual(len(self.served["fields"]), 2)
        self.assertEqual([f["semantic_role"] for f in self.served["fields"]],
                         ["FIELD", "FIELD"])

    def test_serving_is_stable(self):
        definition, approval, payload, table_payload, _ = promoted(
            (1, (HIGH,)), (2, (LOW,)))
        again = StandardStructureAccess.__new__(
            StandardStructureAccess)._render(
                definition.to_dict(), approval, None, payload, table_payload,
                word_width=32)
        self.assertEqual(json.dumps(self.served, sort_keys=True),
                         json.dumps(again, sort_keys=True))


class BackwardCompatibleServing(unittest.TestCase):
    """A definition written before any of this still renders."""

    def test_an_ordinary_field_declares_its_own_word_bits(self):
        definition, approval, payload, table_payload, blocked = promoted(
            (1, ("Count (31..0)",)), (2, ("Flags (7..0)",)))
        self.assertIsNone(blocked)
        served = StandardStructureAccess.__new__(
            StandardStructureAccess)._render(
                definition.to_dict(), approval, None, payload, table_payload,
                word_width=32)
        self.assertEqual(served["value_groups"], [])
        for field in served["fields"]:
            self.assertEqual(field["declared_msb"], field["msb"])
            self.assertEqual(field["declared_lsb"], field["lsb"])
            self.assertEqual(field["coordinate_domain"],
                             CoordinateDomain.WORD_LOCAL.value)
            self.assertIsNone(field["value_group_id"])

    def test_a_payload_without_the_new_keys_still_renders(self):
        # Exactly the shape a store written before STD2E holds. Serving must
        # not require a rebuild before it can answer.
        definition, approval, payload, table_payload, _ = promoted(
            (1, ("Count (31..0)",)), (2, ("Flags (7..0)",)))
        old = definition.to_dict()
        for field in old["fields"]:
            for key in ("coordinate_domain", "declared_msb", "declared_lsb",
                        "value_group_id", "value_width", "segment_index",
                        "segment_count", "projection_source"):
                field.pop(key, None)
        for word in old["unresolved_words"]:
            word.pop("unresolved_labels", None)
            word.pop("causes", None)
        served = StandardStructureAccess.__new__(
            StandardStructureAccess)._render(
                old, approval, None, payload, table_payload, word_width=32)
        self.assertEqual(served["value_groups"], [])
        for field in served["fields"]:
            self.assertEqual(field["declared_msb"], field["msb"])
            self.assertEqual(field["coordinate_domain"],
                             CoordinateDomain.WORD_LOCAL.value)

    def test_an_unresolved_word_serves_its_causes(self):
        definition, approval, payload, table_payload, blocked = promoted(
            (1, (HIGH,)), (2, ("Count (31..0)",)))
        self.assertIsNone(blocked)
        served = StandardStructureAccess.__new__(
            StandardStructureAccess)._render(
                definition.to_dict(), approval, None, payload, table_payload,
                word_width=32)
        word = next(item for item in served["words"] if item["word_index"] == 1)
        self.assertEqual(word["unresolved_causes"], ["VALUE_LOCAL_DEFERRED"])
        self.assertEqual([item["declared_msb"] for item in word["unresolved_labels"]],
                         [63])
        # The plain label list a reader may already depend on is untouched.
        self.assertEqual(word["unpositioned_labels"], [HIGH])


class Citations(unittest.TestCase):
    """A projected field cites the source that states 63..32."""

    def test_the_citation_is_the_cell_that_states_the_value_range(self):
        definition, _approval, _payload, table_payload, blocked = promoted(
            (1, (HIGH,)), (2, (LOW,)))
        self.assertIsNone(blocked)
        cells = {item["cell_id"]: item for row in table_payload["rows"]
                 for item in row["cells"]}
        high = next(f for f in definition.to_dict()["fields"]
                    if f["word_index"] == 1)
        self.assertEqual(high["source_cell_ids"], [cells[
            next(c for c, v in cells.items() if v["text"] == HIGH)]["cell_id"]])
        cited = cells[high["source_cell_ids"][0]]
        self.assertEqual(cited["text"], HIGH)
        self.assertIn("63..32", cited["text"])
        self.assertEqual(high["supporting_source_ids"],
                         list(cited["source_ids"]))

    def test_no_citation_claims_the_source_stated_word_bits(self):
        definition, _a, _p, table_payload, _b = promoted((1, (HIGH,)), (2, (LOW,)))
        cells = {item["cell_id"]: item for row in table_payload["rows"]
                 for item in row["cells"]}
        high = next(f for f in definition.to_dict()["fields"]
                    if f["word_index"] == 1)
        cited = cells[high["source_cell_ids"][0]]["text"]
        self.assertNotIn("(31..0)", cited)
        # The range the label actually printed, never the projected one.
        self.assertEqual(high["stated_range_text"], "(63..32)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
