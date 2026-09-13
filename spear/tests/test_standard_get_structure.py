"""STD3A: the model may read reviewed structures, and nothing else.

Every fixture here is invented. No licensed normative text appears in this file.
The tool builds nothing: a candidate no person approved has no route through it,
and a structure that describes half a word says so rather than filling it in.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_prose_range import StandardReviewedLink
from standard_semantic import (
    StandardBitfieldApproval, build_semantics, normative_links_for,
    promote_bitfield, review_evidence_fingerprint,
)
from standard_semantic_store import (
    StandardApprovalStore, StandardSemanticStore, build_semantic_manifest,
)
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore
from standard_structure import extract_structures, structure_fingerprint
from standard_structure_access import (
    AMBIGUOUS_STRUCTURE, STALE_APPROVAL, STALE_REVIEW_EVIDENCE,
    STALE_SEMANTIC_STORE, STRUCTURE_INCOMPLETE, STRUCTURE_NOT_APPROVED,
    STRUCTURE_NOT_FOUND, StandardStructureAccess, StructureAccessError,
)
from standard_structure_store import StandardStructureStore, build_manifest
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from standard_word_association import associate_words, field_candidates
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext
from tracing import NullTraceRecorder, TraceEmitter

SID, REV = "SEM", "R1"
OPERATOR = "test-operator"


class _Served(unittest.TestCase):
    """One approved definition, built and persisted the ordinary way."""

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
        self.artifact = extract_layout(pdf, pdf_sha256=self.manifest.source_pdf_sha256)
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
        self.candidate = self.structures.bitfields[0]
        self.table = next(item for item in self.structures.tables
                          if item.table_id == self.candidate.table_id)
        self.fingerprint = structure_fingerprint(self.structures)
        self.access = StandardStructureAccess(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def approve(self, roles=None, reviewer=OPERATOR, evidence=True):
        originals = field_candidates(self.candidate.to_dict(), self.table.to_dict())
        roles = tuple(roles or ["FIELD"] * len(originals))
        approval = StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict="ACCEPTABLE_WARNING",
            reviewer=reviewer, reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=self.fingerprint, span_roles=roles,
            review_evidence_fingerprint=(review_evidence_fingerprint(
                self.candidate.bitfield_id, self.fingerprint,
                table=self.table.to_dict(), originals=originals, roles=roles,
                associations=associate_words(self.table.to_dict()),
                reviewed_links=()) if evidence else None))
        return approval

    def persist(self, approval):
        StandardApprovalStore._atomic_write = getattr(
            StandardApprovalStore, "_atomic_write", None)
        from standard_semantic_store import _atomic_write
        from standard_schema import canonical_json
        path = StandardApprovalStore(self.store).path(SID, REV)
        _atomic_write(path, canonical_json({
            "schema_version": approval.schema_version, "standard_id": SID,
            "revision": REV, "structure_fingerprint": approval.structure_fingerprint,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "rows": [approval.to_dict()]}))
        semantics = build_semantics(
            self.structures, {approval.candidate_id: approval},
            standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint, layout=self.artifact,
            units=self.units)
        StandardSemanticStore(self.store).save(build_semantic_manifest(
            semantics, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint, approved=1), semantics)
        return semantics

    def served(self, **kwargs):
        return self.access.get(SID, REV, **kwargs)

    def refusal(self, **kwargs):
        with self.assertRaises(StructureAccessError) as caught:
            self.access.get(SID, REV, **kwargs)
        return caught.exception.reason


class ApprovedStructureTests(_Served):
    def test_an_approved_structure_is_returned_with_its_fields(self):
        semantics = self.persist(self.approve())
        definition = semantics.bitfields[0]
        served = self.served(definition_id=definition.definition_id)
        self.assertEqual(served["definition_id"], definition.definition_id)
        self.assertEqual(served["field_count"], len(definition.fields))
        self.assertEqual(served["human_verdict"], "ACCEPTABLE_WARNING")
        self.assertEqual(served["reviewer"], OPERATOR)
        self.assertEqual(served["structural_completeness"],
                         definition.structural_completeness.value)

    def test_a_structure_can_be_found_by_its_source_candidate(self):
        semantics = self.persist(self.approve())
        served = self.served(source_candidate_id=self.candidate.bitfield_id)
        self.assertEqual(served["definition_id"],
                         semantics.bitfields[0].definition_id)

    def test_the_index_lists_only_approved_structures(self):
        self.persist(self.approve())
        index = self.access.index(SID, REV)
        self.assertEqual(index["approved_structure_count"], 1)
        self.assertEqual(index["structures"][0]["source_candidate_id"],
                         self.candidate.bitfield_id)
        self.assertNotIn("fields", index["structures"][0])

    def test_identifiers_and_output_are_stable_across_calls(self):
        semantics = self.persist(self.approve())
        first = self.served(definition_id=semantics.bitfields[0].definition_id)
        second = self.served(definition_id=semantics.bitfields[0].definition_id)
        self.assertEqual(first, second)
        self.assertEqual([f["field_id"] for f in first["fields"]],
                         [f.field_id for f in semantics.bitfields[0].fields])

    def test_every_field_keeps_its_canonical_sources(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        for field, origin in zip(served["fields"], sorted(
                semantics.bitfields[0].fields,
                key=lambda f: (f.word_index, -f.msb))):
            self.assertEqual(tuple(field["canonical_source_ids"]),
                             origin.supporting_source_ids)
            self.assertTrue(field["source_cell_ids"])
            self.assertEqual(field["position_source"], origin.position_source.value)

    def test_the_result_offers_what_a_citation_needs(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        known = {unit.source_id for unit in self.units}
        self.assertTrue(served["citation_source_ids"])
        self.assertTrue(set(served["citation_source_ids"]) <= known)


class RefusalTests(_Served):
    def test_an_unapproved_candidate_is_never_served(self):
        # Nothing persisted: the candidate exists, no person approved it.
        self.assertEqual(
            self.refusal(source_candidate_id=self.candidate.bitfield_id),
            STRUCTURE_NOT_APPROVED)

    def test_an_unknown_identity_is_not_found(self):
        self.persist(self.approve())
        self.assertEqual(self.refusal(definition_id="bfd-0000000000000000"),
                         STRUCTURE_NOT_FOUND)

    def test_a_mismatched_pair_of_identities_finds_nothing(self):
        semantics = self.persist(self.approve())
        self.assertEqual(
            self.refusal(definition_id=semantics.bitfields[0].definition_id,
                         source_candidate_id="bit-0000000000000000"),
            STRUCTURE_NOT_FOUND)

    def _resave_semantic_manifest(self, **changes):
        directory = StandardSemanticStore(self.store).directory(SID, REV)
        path = directory / "manifest.json"
        raw = json.loads(path.read_text())
        raw.update(changes)
        path.write_text(json.dumps(raw))

    def test_a_corpus_that_moved_since_the_build_refuses_the_store(self):
        self.persist(self.approve())
        self._resave_semantic_manifest(corpus_fingerprint="0" * 64)
        self.assertEqual(self.refusal(source_candidate_id=self.candidate.bitfield_id),
                         STALE_SEMANTIC_STORE)

    def test_a_layout_that_moved_since_the_build_refuses_the_store(self):
        self.persist(self.approve())
        self._resave_semantic_manifest(layout_fingerprint="1" * 64)
        self.assertEqual(self.refusal(source_candidate_id=self.candidate.bitfield_id),
                         STALE_SEMANTIC_STORE)

    def test_a_geometry_that_moved_since_the_build_refuses_the_store(self):
        self.persist(self.approve())
        self._resave_semantic_manifest(structure_fingerprint="2" * 64)
        self.assertEqual(self.refusal(source_candidate_id=self.candidate.bitfield_id),
                         STALE_SEMANTIC_STORE)

    def test_a_withdrawn_approval_stops_the_structure_being_served(self):
        semantics = self.persist(self.approve())
        path = StandardApprovalStore(self.store).path(SID, REV)
        raw = json.loads(path.read_text())
        raw["rows"] = []
        path.write_text(json.dumps(raw))
        self.assertEqual(
            self.refusal(definition_id=semantics.bitfields[0].definition_id),
            STRUCTURE_NOT_APPROVED)

    def test_an_approval_recorded_against_other_geometry_is_stale(self):
        semantics = self.persist(self.approve())
        path = StandardApprovalStore(self.store).path(SID, REV)
        raw = json.loads(path.read_text())
        raw["rows"][0]["structure_fingerprint"] = "3" * 64
        path.write_text(json.dumps(raw))
        self.assertEqual(
            self.refusal(definition_id=semantics.bitfields[0].definition_id),
            STALE_APPROVAL)

    def test_review_evidence_that_no_longer_reproduces_is_refused(self):
        semantics = self.persist(self.approve())
        path = StandardApprovalStore(self.store).path(SID, REV)
        raw = json.loads(path.read_text())
        raw["rows"][0]["review_evidence_fingerprint"] = "4" * 64
        path.write_text(json.dumps(raw))
        self.assertEqual(
            self.refusal(definition_id=semantics.bitfields[0].definition_id),
            STALE_REVIEW_EVIDENCE)

    def test_a_rejected_verdict_is_never_served(self):
        approval = replace(self.approve(), verdict="NEEDS_FOLLOWUP",
                           review_evidence_fingerprint=None)
        # A blocked verdict builds no definition at all, so there is nothing
        # to serve and nothing that could leak.
        semantics = self.persist(approval)
        self.assertEqual(semantics.bitfields, ())
        self.assertEqual(self.refusal(source_candidate_id=self.candidate.bitfield_id),
                         STRUCTURE_NOT_APPROVED)


class CompletenessTests(_Served):
    def test_completeness_and_structural_completeness_are_both_reported(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        self.assertIn("completeness", served)
        self.assertIn("structural_completeness", served)
        self.assertNotEqual(served["completeness"],
                            served["structural_completeness"])

    def test_every_word_reports_covered_and_unclaimed_bits(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        for word in served["words"]:
            self.assertIn("covered_bits", word)
            self.assertIn("unclaimed_bits", word)
            self.assertIn("unpositioned_labels", word)
            self.assertEqual(word["word_width"], 32)

    def test_require_complete_passes_a_complete_structure(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id,
                             require_structurally_complete=True)
        self.assertEqual(served["structural_completeness"],
                         "STRUCTURALLY_COMPLETE")

    def test_the_result_carries_evidence_and_nothing_generated(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        self.assertEqual(set(served["fields"][0]), {
            "field_id", "word_index", "word_label", "word_association_source",
            "normative_label", "display_label", "semantic_role", "msb", "lsb",
            "width", "position_source", "stated_range_text", "provenance_grades",
            "source_cell_ids", "canonical_source_ids", "normative_source_id",
            "normative_link_fingerprint", "warnings",
            # STD2E-A: what the label declares, kept apart from where the bits
            # sit in the word, plus the value a multi-word field is part of.
            "coordinate_domain", "declared_msb", "declared_lsb",
            "value_group_id", "value_width", "segment_index", "segment_count",
            "projection_source",
            # STD2E-B1: the word a quantity shares, and the slot a rule gave it.
            "packing_group_id", "packing_slot", "projection_reference"})
        self.assertEqual(set(served["words"][0]), {
            "word_index", "word_label", "word_width", "field_count",
            "covered_bits", "unclaimed_bits", "unpositioned_labels",
            "unresolved_labels", "unresolved_causes", "structural_status"})

    def test_no_field_carries_a_packet_global_offset(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        for field in served["fields"]:
            self.assertLessEqual(field["msb"], 31)
            self.assertGreaterEqual(field["lsb"], 0)
            self.assertNotIn("global_msb", field)

    def test_no_long_normative_passage_is_returned(self):
        semantics = self.persist(self.approve())
        served = self.served(definition_id=semantics.bitfields[0].definition_id)
        for field in served["fields"]:
            self.assertLessEqual(len(field["normative_label"]), 80)
            self.assertLessEqual(len(field["display_label"]), 80)


class ToolSurfaceTests(_Served):
    def context(self, binding=None):
        return ToolExecutionContext(
            "task_12345678", TraceEmitter(NullTraceRecorder()), {},
            metadata={} if binding is None else {"standard_binding": binding})

    def registry(self):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        return registry

    def test_the_tool_is_registered_read_only(self):
        registry = self.registry()
        names = {spec.name for spec in registry.list_specs()}
        self.assertEqual(names, set(STANDARD_TOOL_NAMES))
        self.assertIn("standard.get_structure", names)
        spec = next(item for item in registry.list_specs()
                    if item.name == "standard.get_structure")
        self.assertEqual(spec.mutability.value, "read_only")

    def test_validation_and_build_tools_are_still_absent(self):
        names = {spec.name for spec in self.registry().list_specs()}
        for forbidden in ("standard.validate", "standard.build_structure",
                          "standard.approve_bitfield", "standard.generate_code"):
            self.assertNotIn(forbidden, names)

    def test_the_tool_refuses_without_a_bound_standard(self):
        self.persist(self.approve())
        handler = self.registry().handler("standard.get_structure")
        with self.assertRaises(PermissionError):
            handler(self.context(), {"source_candidate_id":
                                     self.candidate.bitfield_id})

    def test_the_tool_refuses_a_revision_that_is_not_bound(self):
        self.persist(self.approve())
        binding = self.store.binding(SID, REV).to_dict()
        handler = self.registry().handler("standard.get_structure")
        with self.assertRaises(PermissionError):
            handler(self.context(binding), {"revision": "OTHER"})

    def test_a_refusal_comes_back_as_data_with_its_reason(self):
        self.persist(self.approve())
        binding = self.store.binding(SID, REV).to_dict()
        handler = self.registry().handler("standard.get_structure")
        result = handler(self.context(binding),
                         {"definition_id": "bfd-0000000000000000"})
        self.assertEqual(json.loads(result.text)["error"], STRUCTURE_NOT_FOUND)
        self.assertEqual(result.metadata["structure_refusal"], STRUCTURE_NOT_FOUND)
        self.assertFalse(result.mutation)

    def test_calling_the_tool_writes_nothing(self):
        self.persist(self.approve())
        binding = self.store.binding(SID, REV).to_dict()
        handler = self.registry().handler("standard.get_structure")
        root = self.store.revision_dir(SID, REV)
        before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
        handler(self.context(binding),
                {"source_candidate_id": self.candidate.bitfield_id})
        handler(self.context(binding), {})
        after = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
