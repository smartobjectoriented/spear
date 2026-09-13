"""What a caller is told when it asks for a structure that does not exist.

A refusal that only says "not found" leaves a caller guessing, and a caller
that guesses identifiers spends its whole budget guessing. The refusal has to
say what a valid identifier looks like and how to obtain one -- without ever
accepting a guess, matching a name, or reinterpreting a source id as a
definition id.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_retrieval import rebuild_lexical_index
from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, SpanRole, StandardBitfieldApproval,
    associate_words, build_semantics, review_evidence_fingerprint,
)
from standard_semantic_store import (
    StandardApprovalStore, StandardSemanticStore, build_semantic_manifest,
)
from standard_store import StandardStore
from standard_structure import extract_structures, structure_fingerprint
from standard_structure_access import (
    INVALID_DEFINITION_ID, STRUCTURE_NOT_FOUND, StandardStructureAccess,
    StructureAccessError,
)
from standard_structure_store import StandardStructureStore, build_manifest
from standard_tools import StandardToolService
from standard_value_pair import value_local_pairs
from standard_word_association import field_candidates
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext
from tracing import NullTraceRecorder, TraceEmitter

SID, REV = "SEM", "R1"
OPERATOR = "test-operator"


class _Served(unittest.TestCase):
    """One approved, persisted structure to ask for -- and to ask around."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "bitfield.pdf"
        pdf.write_bytes(semantic_bitfield_pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.store, SID, REV)
        self.units = self.store.load_units(SID, REV)
        self.artifact = extract_layout(pdf,
                                       pdf_sha256=self.manifest.source_pdf_sha256)
        (self.store.revision_dir(SID, REV) / "layout.json").write_text(
            json.dumps(self.artifact))
        self.structures = extract_structures(
            self.units, self.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256)
        StandardStructureStore(self.store).save(build_manifest(
            self.structures, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version), self.structures)
        self.fingerprint = structure_fingerprint(self.structures)
        self.candidate = self.structures.bitfields[0]
        self.table = next(item for item in self.structures.tables
                          if item.table_id == self.candidate.table_id)
        pairs, _ = value_local_pairs(self.candidate.to_dict(), self.table.to_dict())
        originals = field_candidates(
            self.candidate.to_dict(), self.table.to_dict(),
            value_local_cells=frozenset(c for p in pairs for c in p.cell_ids))
        roles = tuple([SpanRole.FIELD.value] * len(originals))
        approval = StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict="PASS",
            reviewer=OPERATOR, reviewed_at="1970-01-01T00:00:00+00:00",
            structure_fingerprint=self.fingerprint, span_roles=roles,
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=(),
            review_evidence_fingerprint=review_evidence_fingerprint(
                self.candidate.bitfield_id, self.fingerprint,
                table=self.table.to_dict(), originals=originals, roles=roles,
                associations=associate_words(self.table.to_dict()),
                reviewed_links=()))
        path = StandardApprovalStore(self.store).path(SID, REV)
        path.write_text(json.dumps({
            "schema_version": APPROVAL_SCHEMA_VERSION, "standard_id": SID,
            "revision": REV, "structure_fingerprint": self.fingerprint,
            "updated_at": "1970-01-01T00:00:00+00:00",
            "rows": [approval.to_dict()]}))
        approvals, _ = StandardApprovalStore(self.store).load(SID, REV)
        semantics = build_semantics(
            self.structures, approvals, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint, layout=self.artifact,
            units=self.units)
        StandardSemanticStore(self.store).save(build_semantic_manifest(
            semantics, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint,
            approved=len(approvals)), semantics)
        self.definition_id = semantics.bitfields[0].definition_id
        self.access = StandardStructureAccess(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def refusal(self, definition_id):
        with self.assertRaises(StructureAccessError) as caught:
            self.access.get(SID, REV, definition_id=definition_id)
        return caught.exception

    def served(self, arguments):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        context = ToolExecutionContext(
            "task_recovery01", TraceEmitter(NullTraceRecorder()), {},
            metadata={"standard_binding": self.store.binding(SID, REV).to_dict()})
        return registry.handler("standard.get_structure")(context, arguments)


class InvalidIdentifier(_Served):
    def test_a_malformed_identifier_is_refused_by_its_own_reason(self):
        # 32 hex digits where the tool hands out 16: the shape is wrong, which
        # is a different fault from a well-formed name that happens to be gone.
        found = self.refusal("bfd-" + "0" * 32)
        self.assertEqual(found.reason, INVALID_DEFINITION_ID)

    def test_a_well_formed_but_unknown_identifier_is_simply_not_found(self):
        found = self.refusal("bfd-" + "0" * 16)
        self.assertEqual(found.reason, STRUCTURE_NOT_FOUND)

    def test_both_refusals_say_how_to_recover(self):
        for asked in ("bfd-" + "0" * 32, "bfd-" + "0" * 16):
            payload = json.dumps(self.refusal(asked).to_dict()).lower()
            self.assertIn("no arguments", payload)

    def test_the_refusal_names_the_supported_identifier_forms(self):
        payload = json.dumps(self.refusal("bfd-" + "0" * 32).to_dict())
        self.assertIn("bfd-", payload)
        self.assertIn("bit-", payload)

    def test_the_refusal_lists_the_identifiers_that_do_exist(self):
        payload = self.refusal("bfd-" + "0" * 32).to_dict()
        self.assertIn(self.definition_id, json.dumps(payload))

    def test_a_source_id_is_not_a_definition_id(self):
        found = self.refusal("std-" + "a" * 32)
        self.assertEqual(found.reason, INVALID_DEFINITION_ID)
        self.assertIn("source", json.dumps(found.to_dict()).lower())

    def test_a_human_readable_name_is_not_a_definition_id(self):
        found = self.refusal("Some Field Name")
        self.assertEqual(found.reason, INVALID_DEFINITION_ID)

    def test_a_wildcard_is_not_a_definition_id(self):
        self.assertEqual(self.refusal("bfd-*").reason, INVALID_DEFINITION_ID)

    def test_nothing_is_fuzzy_matched_to_a_real_structure(self):
        # A near-miss must never resolve to the one structure that exists.
        near = self.definition_id[:-1] + ("0" if self.definition_id[-1] != "0"
                                          else "1")
        self.assertEqual(self.refusal(near).reason, STRUCTURE_NOT_FOUND)


class StillWorks(_Served):
    def test_a_valid_definition_id_still_resolves(self):
        found = self.access.get(SID, REV, definition_id=self.definition_id)
        self.assertEqual(found["definition_id"], self.definition_id)

    def test_a_valid_source_candidate_id_still_resolves(self):
        found = self.access.get(
            SID, REV, source_candidate_id=self.candidate.bitfield_id)
        self.assertEqual(found["source_candidate_id"],
                         self.candidate.bitfield_id)

    def test_listing_still_works_with_no_arguments(self):
        listing = self.access.index(SID, REV)
        self.assertIn(self.definition_id,
                      [item["definition_id"] for item in listing["structures"]])


class ServedRefusal(_Served):
    def test_the_tool_returns_the_refusal_as_data(self):
        result = self.served({"definition_id": "not-an-identifier"})
        self.assertEqual(dict(result.metadata).get("structure_refusal"),
                         INVALID_DEFINITION_ID)
        self.assertEqual(json.loads(result.text)["error"], INVALID_DEFINITION_ID)
        self.assertIn("recovery", json.loads(result.text))

    def test_the_refusal_names_its_reason_the_way_every_tool_does(self):
        # A caller should be able to detect any normative refusal by one key,
        # whichever tool produced it.
        result = self.served({"definition_id": "bfd-*"})
        self.assertEqual(dict(result.metadata).get("standard_refusal"),
                         json.loads(result.text)["error"])

    def test_the_refusal_is_read_only(self):
        result = self.served({"definition_id": "bfd-" + "0" * 32})
        self.assertFalse(result.mutation)
        self.assertEqual(result.affected_paths, ())

    def test_a_refusal_does_not_disturb_the_store(self):
        semantics = StandardSemanticStore(self.store)
        before = semantics.load(SID, REV)[0].semantic_fingerprint
        for bad in ("bfd-" + "0" * 32, "std-" + "a" * 32, "anything"):
            self.served({"definition_id": bad})
        self.assertEqual(semantics.load(SID, REV)[0].semantic_fingerprint, before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class MismatchedPairing(_Served):
    """A real definition with the wrong candidate, or the reverse.

    This used to answer "that candidate exists but no human has approved
    it" -- false, the candidate was approved for another definition -- and a
    caller told that tried every other candidate against the same definition,
    one refusal per round, until the budget was gone. Eleven rounds, once.
    """

    def test_a_real_definition_with_the_wrong_candidate_names_the_pairing(self):
        with self.assertRaises(StructureAccessError) as caught:
            self.access.get(SID, REV, definition_id=self.definition_id,
                            source_candidate_id="bit-" + "0" * 16)

        found = caught.exception
        self.assertEqual(found.reason, STRUCTURE_NOT_FOUND)
        self.assertEqual(found.recovery["approved_pairing"],
                         {"definition_id": self.definition_id,
                          "source_candidate_id": self.candidate.bitfield_id})

    def test_a_real_candidate_with_the_wrong_definition_is_not_unapproved(self):
        with self.assertRaises(StructureAccessError) as caught:
            self.access.get(SID, REV, definition_id="bfd-" + "1" * 16,
                            source_candidate_id=self.candidate.bitfield_id)

        found = caught.exception
        self.assertEqual(found.reason, STRUCTURE_NOT_FOUND)
        self.assertNotIn("no human has approved", found.detail)
        self.assertEqual(found.recovery["approved_pairing"]["definition_id"],
                         self.definition_id)

    def test_the_pairing_is_enough_to_be_served(self):
        with self.assertRaises(StructureAccessError) as caught:
            self.access.get(SID, REV, definition_id=self.definition_id,
                            source_candidate_id="bit-" + "0" * 16)

        pairing = caught.exception.recovery["approved_pairing"]
        served = self.access.get(SID, REV, **pairing)
        self.assertEqual(served["definition_id"], self.definition_id)

    def test_an_invalid_identifier_is_told_the_pairs_not_two_lists(self):
        found = self.refusal("bfd-cam-field-001")
        self.assertEqual(found.reason, INVALID_DEFINITION_ID)
        self.assertEqual(found.recovery["approved_structures"],
                         [{"definition_id": self.definition_id,
                           "source_candidate_id": self.candidate.bitfield_id}])
