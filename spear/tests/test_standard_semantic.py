"""STD2B: the reviewed bitfield semantic model. All fixtures are synthetic.

No licensed normative text appears in this file.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_store import StandardStore, StandardStoreError
from standard_structure import (
    ProvenanceQuality, extract_structures, structure_fingerprint,
)
from standard_structure_store import StandardStructureStore, build_manifest
from standard_semantic import (
    APPROVING_VERDICTS, BitOrder, SemanticRole, SpanRole,
    StandardBitfieldApproval, StandardSemanticError, bit_order_of,
    build_semantics, promote_bitfield, semantic_fingerprint, validate_semantics,
)
from standard_semantic_store import (
    StandardApprovalStore, StandardSemanticStore, build_semantic_manifest,
    operator_identity,
)
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes
from tool_registry import ToolRegistry

SID, REV = "SEM", "R1"
OPERATOR = "test-operator"


class _Semantic(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.pdf = root / "bitfield.pdf"
        self.pdf.write_bytes(semantic_bitfield_pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(self.store, self.pdf, standard_id=SID,
                                   revision=REV, source_origin="TEST_FIXTURE")
        self.units = self.store.load_units(SID, REV)
        self.artifact = extract_layout(
            self.pdf, pdf_sha256=self.manifest.source_pdf_sha256)
        (self.store.revision_dir(SID, REV) / "layout.json").write_text(
            json.dumps(self.artifact))
        self.structures = extract_structures(
            self.units, self.artifact, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256)
        self.structure_store = StandardStructureStore(self.store)
        self.structure_store.save(build_manifest(
            self.structures, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version), self.structures)
        self.candidate = self.structures.bitfields[0]
        self.table = next(item for item in self.structures.tables
                          if item.table_id == self.candidate.table_id)
        self.fingerprints = dict(
            standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=structure_fingerprint(self.structures))
        self.approvals = StandardApprovalStore(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def approval(self, verdict="PASS", roles=None, reviewer=OPERATOR, **kwargs):
        return StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict=verdict,
            reviewer=reviewer, reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=self.fingerprints["structure_fingerprint"],
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(self.candidate.spans)),
            **kwargs)

    def promote(self, approval):
        return promote_bitfield(self.candidate, self.table, approval,
                                **self.fingerprints)


class EndToEndTests(_Semantic):
    def test_two_reviewed_spans_become_fields_with_derived_positions(self):
        definition, blocked = self.promote(self.approval())
        self.assertIsNone(blocked)
        self.assertEqual(definition.bit_order, BitOrder.MSB_TO_LSB)
        self.assertEqual(definition.visible_bit_labels,
                         (31, 30, 29, 28, 27, 26, 25, 24))
        fields = {field.label: field for field in definition.fields}
        self.assertEqual(set(fields), {"FIELD_A_ALPHA_ONEX", "FIELD_B_BRAVO_TWO"})
        alpha = fields["FIELD_A_ALPHA_ONEX"]
        bravo = fields["FIELD_B_BRAVO_TWO"]
        self.assertEqual((alpha.msb, alpha.lsb, alpha.width), (31, 28, 4))
        self.assertEqual((bravo.msb, bravo.lsb, bravo.width), (27, 24, 4))
        for field in (alpha, bravo):
            self.assertEqual(field.semantic_role, SemanticRole.FIELD)
            self.assertTrue(field.supporting_source_ids)
            self.assertTrue(field.source_cell_ids)
            self.assertNotIn(ProvenanceQuality.UNRESOLVED.value,
                             field.provenance_grades)

    def test_the_definition_records_who_approved_it_and_against_what(self):
        definition, _ = self.promote(self.approval())
        self.assertEqual(definition.approved_by, OPERATOR)
        self.assertEqual(definition.approval_verdict, "PASS")
        self.assertEqual(definition.structure_fingerprint,
                         self.fingerprints["structure_fingerprint"])
        self.assertEqual(definition.source_bitfield_candidate_id,
                         self.candidate.bitfield_id)
        self.assertTrue(definition.semantic_fingerprint)

    def test_identifiers_are_deterministic(self):
        first, _ = self.promote(self.approval())
        second, _ = self.promote(self.approval())
        self.assertEqual(first.definition_id, second.definition_id)
        self.assertEqual([f.field_id for f in first.fields],
                         [f.field_id for f in second.fields])
        self.assertTrue(first.definition_id.startswith("bfd-"))
        self.assertTrue(first.fields[0].field_id.startswith("fld-"))

    def test_validation_accepts_a_well_formed_definition(self):
        semantics = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        validate_semantics(
            semantics, structures=self.structures,
            source_ids=frozenset(unit.source_id for unit in self.units),
            corpus_fingerprint=self.fingerprints["corpus_fingerprint"],
            layout_fingerprint=self.fingerprints["layout_fingerprint"],
            structure_fingerprint=self.fingerprints["structure_fingerprint"])
        self.assertEqual(len(semantics.bitfields), 1)
        self.assertEqual(len(semantics.bitfields[0].fields), 2)
        self.assertEqual(semantics.packets, ())

    def test_a_packet_appears_only_when_a_reviewer_named_one(self):
        without = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        self.assertEqual(without.packets, ())
        with_identity = build_semantics(
            self.structures,
            {self.candidate.bitfield_id: self.approval(
                packet_identity="Example Packet")}, **self.fingerprints)
        self.assertEqual(len(with_identity.packets), 1)
        packet = with_identity.packets[0]
        self.assertEqual(packet.identity, "Example Packet")
        self.assertEqual(packet.bitfield_definition_ids,
                         (with_identity.bitfields[0].definition_id,))


class ReviewGateTests(_Semantic):
    def test_an_unreviewed_candidate_is_not_promoted(self):
        definition, blocked = self.promote(None)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "NO_HUMAN_REVIEW")

    def test_a_rejecting_verdict_is_not_promoted(self):
        for verdict in ("FAIL", "NEEDS_FOLLOWUP"):
            definition, blocked = self.promote(self.approval(verdict=verdict))
            self.assertIsNone(definition)
            self.assertEqual(blocked["reason"], "HUMAN_REVIEW_REJECTED")

    def test_both_approving_verdicts_are_accepted(self):
        for verdict in APPROVING_VERDICTS:
            definition, blocked = self.promote(self.approval(verdict=verdict))
            self.assertIsNotNone(definition, verdict)
            self.assertIsNone(blocked)

    def test_an_assistant_identity_can_never_approve(self):
        for name in ("claude", "claude-assistant", "Claude Assistant",
                     "assistant", "model", "llm"):
            with self.assertRaises(StandardSemanticError, msg=name):
                self.approval(reviewer=name)

    def test_operator_identity_refuses_an_assistant_name(self):
        self.assertEqual(operator_identity("daniel"), "daniel")
        with self.assertRaises(StandardSemanticError):
            operator_identity("claude-assistant")

    def test_an_approval_recorded_against_other_geometry_is_stale(self):
        stale = replace(self.approval(), structure_fingerprint="b" * 64)
        definition, blocked = self.promote(stale)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_APPROVAL")


class SpanClassificationTests(_Semantic):
    def test_an_unclassified_span_blocks_the_whole_definition(self):
        definition, blocked = self.promote(
            self.approval(roles=["FIELD", "UNKNOWN"]))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPAN_UNCLASSIFIED")

    def test_a_structural_or_row_label_is_not_promoted_as_a_field(self):
        for role in ("STRUCTURAL_LABEL", "ROW_LABEL", "IGNORE"):
            definition, _ = self.promote(self.approval(roles=["FIELD", role]))
            self.assertEqual(len(definition.fields), 1, role)
            self.assertEqual(definition.fields[0].label, "FIELD_A_ALPHA_ONEX")

    def test_a_reviewed_reserved_span_becomes_a_reserved_field(self):
        definition, _ = self.promote(self.approval(roles=["FIELD", "RESERVED"]))
        roles = {field.label: field.semantic_role for field in definition.fields}
        self.assertEqual(roles["FIELD_B_BRAVO_TWO"], SemanticRole.RESERVED)
        self.assertEqual(roles["FIELD_A_ALPHA_ONEX"], SemanticRole.FIELD)

    def test_a_candidate_with_no_field_span_produces_nothing(self):
        definition, blocked = self.promote(
            self.approval(roles=["ROW_LABEL", "ROW_LABEL"]))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "NO_PROMOTABLE_SPAN")

    def test_the_approval_must_classify_every_span(self):
        with self.assertRaises(StandardSemanticError):
            StandardBitfieldApproval(
                candidate_id=self.candidate.bitfield_id, verdict="PASS",
                reviewer=OPERATOR, reviewed_at="2026-01-01T00:00:00+00:00",
                structure_fingerprint=self.fingerprints["structure_fingerprint"],
                span_roles=("NOT_A_ROLE",))
        definition, blocked = self.promote(self.approval(roles=["FIELD"]))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPANS_NOT_CLASSIFIED")


class BitOrderAndDerivationTests(_Semantic):
    def test_bit_order_is_read_from_the_ruler(self):
        self.assertEqual(bit_order_of([31, 30, 29, 28]), BitOrder.MSB_TO_LSB)
        self.assertEqual(bit_order_of([0, 1, 2, 3]), BitOrder.LSB_TO_MSB)
        self.assertEqual(bit_order_of([3, 7, 1]), BitOrder.UNKNOWN)
        self.assertEqual(bit_order_of([5]), BitOrder.UNKNOWN)

    def test_an_ambiguous_ruler_is_never_promoted(self):
        broken = replace(self.candidate, bit_labels=tuple(
            replace(label, value=value) for label, value
            in zip(self.candidate.bit_labels, (31, 30, 12, 28, 27, 26, 25, 24))))
        definition, blocked = promote_bitfield(
            broken, self.table, self.approval(), **self.fingerprints)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "AMBIGUOUS_BIT_ORDER")

    def test_a_broken_label_range_is_not_turned_into_a_field(self):
        span = self.candidate.spans[0]
        broken = replace(self.candidate, spans=(
            replace(span, covered_labels=(31, 29)),) + self.candidate.spans[1:])
        definition, blocked = promote_bitfield(
            broken, self.table, self.approval(), **self.fingerprints)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPAN_NOT_CONTIGUOUS")

    def test_gaps_between_fields_are_preserved_not_filled(self):
        definition, _ = self.promote(self.approval(roles=["FIELD", "IGNORE"]))
        self.assertEqual(len(definition.fields), 1)
        self.assertTrue(any("does not cover every ruler position" in item
                            for item in definition.warnings))

    def test_overlapping_fields_block_the_definition(self):
        span = self.candidate.spans[1]
        overlapped = replace(self.candidate, spans=(
            self.candidate.spans[0],
            replace(span, covered_labels=(30, 29, 28, 27))))
        definition, blocked = promote_bitfield(
            overlapped, self.table, self.approval(), **self.fingerprints)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "OVERLAPPING_FIELDS")

    def test_the_word_index_of_each_field_is_kept(self):
        definition, _ = self.promote(self.approval())
        self.assertTrue(all(field.word_index == 1 for field in definition.fields))

    def test_no_masks_shifts_or_code_are_produced(self):
        definition, _ = self.promote(self.approval())

        def keys(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    yield name
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        present = set(keys(definition.to_dict()))
        for forbidden in ("mask", "shift", "c_type", "ctype", "struct",
                          "union", "encoder", "decoder", "language"):
            self.assertNotIn(forbidden, present)
        # Positions are derived; the bit-level encoding of them is not.
        self.assertEqual({field.width for field in definition.fields}, {4})


class ProvenanceGateTests(_Semantic):
    def test_a_field_resting_on_unresolved_provenance_is_blocked(self):
        row = self.table.rows[1]
        stripped = replace(self.table, rows=(
            self.table.rows[0],
            replace(row, cells=tuple(
                replace(cell, source_ids=(), semantic=True,
                        provenance=ProvenanceQuality.UNRESOLVED)
                for cell in row.cells))) + self.table.rows[2:])
        definition, blocked = promote_bitfield(
            self.candidate, stripped, self.approval(), **self.fingerprints)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "UNRESOLVED_PROVENANCE")

    def test_inherited_provenance_is_accepted_once_a_person_approved_it(self):
        for grade in (ProvenanceQuality.ROW_INHERITED,
                      ProvenanceQuality.TABLE_REGION_INHERITED):
            row = self.table.rows[1]
            table = replace(self.table, rows=(
                self.table.rows[0],
                replace(row, cells=tuple(replace(cell, provenance=grade)
                                         for cell in row.cells)))
                + self.table.rows[2:])
            definition, blocked = promote_bitfield(
                self.candidate, table, self.approval(), **self.fingerprints)
            self.assertIsNotNone(definition, grade)
            self.assertIn(grade.value, definition.fields[0].provenance_grades)

    def test_a_field_citing_an_unknown_source_fails_validation(self):
        semantics = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        definition = semantics.bitfields[0]
        forged = replace(semantics, bitfields=(replace(
            definition, fields=(replace(definition.fields[0],
                                        supporting_source_ids=("std-" + "f" * 32,)),)
            + definition.fields[1:]),))
        with self.assertRaises(StandardSemanticError):
            validate_semantics(
                forged, structures=self.structures,
                source_ids=frozenset(unit.source_id for unit in self.units),
                corpus_fingerprint=self.fingerprints["corpus_fingerprint"],
                layout_fingerprint=self.fingerprints["layout_fingerprint"],
                structure_fingerprint=self.fingerprints["structure_fingerprint"])


class SemanticStoreTests(_Semantic):
    def build(self):
        semantics = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        manifest = build_semantic_manifest(
            semantics, approved=1, created_at="2026-01-01T00:00:00+00:00",
            **self.fingerprints)
        return manifest, semantics

    def test_definitions_are_stored_privately_and_reload(self):
        store = StandardSemanticStore(self.store)
        manifest, semantics = self.build()
        directory = store.save(manifest, semantics)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        for name in ("manifest.json", "definitions.json"):
            self.assertEqual((directory / name).stat().st_mode & 0o777, 0o600)
        loaded, payload = store.load(SID, REV)
        self.assertEqual(loaded.semantic_fingerprint,
                         semantic_fingerprint(semantics))
        self.assertEqual(loaded.bitfield_definition_count, 1)
        self.assertEqual(loaded.field_definition_count, 2)
        self.assertEqual(store.state(SID, REV), "READY")

    def test_a_manifest_that_does_not_describe_its_definitions_is_refused(self):
        store = StandardSemanticStore(self.store)
        manifest, semantics = self.build()
        with self.assertRaises(StandardSemanticError):
            store.save(replace(manifest, semantic_fingerprint="c" * 64), semantics)

    def test_the_store_is_stale_when_any_upstream_fingerprint_moves(self):
        store = StandardSemanticStore(self.store)
        manifest, semantics = self.build()
        directory = store.save(manifest, semantics)
        for field, message in (("corpus_fingerprint", "canonical corpus"),
                               ("layout_fingerprint", "layout"),
                               ("structure_fingerprint", "geometry")):
            raw = json.loads((directory / "manifest.json").read_text())
            raw[field] = "d" * 64
            (directory / "manifest.json").write_text(json.dumps(raw))
            with self.assertRaises(StandardSemanticError) as raised:
                store.load(SID, REV)
            self.assertIn(message, str(raised.exception))
            store.save(manifest, semantics)

    def test_traversal_and_symlink_escape_are_refused(self):
        store = StandardSemanticStore(self.store)
        for standard_id in ("../../etc", "..", "a/b"):
            with self.assertRaises((StandardStoreError, StandardSemanticError)):
                store.directory(standard_id, REV)
        manifest, semantics = self.build()
        directory = store.save(manifest, semantics)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        escaped = directory / "escape.json"
        escaped.symlink_to(outside)
        from standard_semantic_store import _atomic_write
        with self.assertRaises(StandardSemanticError):
            _atomic_write(escaped, b"{}")


class ApprovalStoreTests(_Semantic):
    def test_an_approval_is_stamped_with_the_live_fingerprint(self):
        approval = self.approvals.approve(
            SID, REV, self.candidate.bitfield_id, verdict="PASS",
            span_roles=["FIELD", "FIELD"], reviewer=OPERATOR)
        self.assertEqual(approval.structure_fingerprint,
                         self.fingerprints["structure_fingerprint"])
        accepted, refused = self.approvals.load(SID, REV)
        self.assertEqual(list(accepted), [self.candidate.bitfield_id])
        self.assertEqual(refused, [])

    def test_an_unknown_candidate_or_wrong_span_count_is_refused(self):
        with self.assertRaises(StandardSemanticError):
            self.approvals.approve(SID, REV, "bit-" + "0" * 16, verdict="PASS",
                                   span_roles=["FIELD"], reviewer=OPERATOR)
        with self.assertRaises(StandardSemanticError):
            self.approvals.approve(SID, REV, self.candidate.bitfield_id,
                                   verdict="PASS", span_roles=["FIELD"],
                                   reviewer=OPERATOR)

    def test_a_replayed_approval_is_refused_after_the_geometry_moves(self):
        self.approvals.approve(SID, REV, self.candidate.bitfield_id,
                               verdict="PASS", span_roles=["FIELD", "FIELD"],
                               reviewer=OPERATOR)
        path = self.approvals.path(SID, REV)
        raw = json.loads(path.read_text())
        raw["rows"][0]["structure_fingerprint"] = "e" * 64
        path.write_text(json.dumps(raw))
        accepted, refused = self.approvals.load(SID, REV)
        self.assertEqual(accepted, {})
        self.assertEqual(refused[0]["reason"], "STALE_APPROVAL")

    def test_a_reviewer_injected_into_the_file_cannot_be_an_assistant(self):
        self.approvals.approve(SID, REV, self.candidate.bitfield_id,
                               verdict="PASS", span_roles=["FIELD", "FIELD"],
                               reviewer=OPERATOR)
        path = self.approvals.path(SID, REV)
        raw = json.loads(path.read_text())
        raw["rows"][0]["reviewer"] = "claude-assistant"
        path.write_text(json.dumps(raw))
        accepted, refused = self.approvals.load(SID, REV)
        self.assertEqual(accepted, {})
        self.assertEqual(refused[0]["reason"], "MALFORMED")

    def test_a_candidate_id_that_is_not_a_bitfield_id_is_refused(self):
        for value in ("tbl-0123456789abcdef", "../../etc", "bit-zz"):
            with self.assertRaises(StandardSemanticError):
                StandardBitfieldApproval(
                    candidate_id=value, verdict="PASS", reviewer=OPERATOR,
                    reviewed_at="2026-01-01T00:00:00+00:00",
                    structure_fingerprint=self.fingerprints["structure_fingerprint"])


class ToolSurfaceTests(_Semantic):
    def test_no_structure_or_validation_tool_is_registered(self):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        names = {spec.name for spec in registry.list_specs()}
        self.assertEqual(names, set(STANDARD_TOOL_NAMES))
        for forbidden in ("standard.validate",
                          "standard.build_structure", "standard.approve_bitfield"):
            self.assertNotIn(forbidden, names)
        import standard_semantic, standard_semantic_store
        for module in (standard_semantic, standard_semantic_store):
            self.assertFalse(hasattr(module, "register"))

    def test_semantic_promotion_needs_no_vector_index(self):
        self.assertFalse((self.store.revision_dir(SID, REV) / "indexes"
                          / "vector").exists())
        definition, _ = self.promote(self.approval())
        self.assertIsNotNone(definition)

    def test_the_canonical_corpus_is_untouched(self):
        before = self.manifest.corpus_manifest_sha256
        self.promote(self.approval())
        build_semantics(self.structures,
                        {self.candidate.bitfield_id: self.approval()},
                        **self.fingerprints)
        self.assertEqual(self.store.verify_corpus(SID, REV).corpus_manifest_sha256,
                         before)

    def test_a_table_candidate_that_is_not_a_bitfield_is_never_promoted(self):
        semantics = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        promoted = {item.source_table_id for item in semantics.bitfields}
        bitfield_tables = {item.table_id for item in self.structures.bitfields}
        self.assertTrue(promoted <= bitfield_tables)
        self.assertLess(len(promoted), len(self.structures.tables) + 1)
