"""STD2C: a field takes its bits from the text before it takes them from a box.

Every fixture here is invented. No licensed normative text appears in this file.
The geometry in these fixtures is deliberately misleading, because that is what
STD2B-R found real geometry to be: a centred label inset from the field it names.
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_semantic import (
    DefinitionCompleteness, PositionSource, SemanticRole, SpanRole,
    StandardBitfieldApproval, StandardSemanticError, build_semantics,
    promote_bitfield, resolve_position, semantic_fingerprint, validate_semantics,
)
from standard_semantic_store import (
    StandardApprovalStore, StandardSemanticStore, build_semantic_manifest,
)
from standard_semantic import PARTIAL_RULER_WARNING
from standard_stated_range import GEOMETRY_RANGE_CONFLICT
from standard_store import StandardStore
from standard_structure import extract_structures, structure_fingerprint
from standard_structure_store import StandardStructureStore, build_manifest
from standard_word_association import field_candidates
from tests.standard_geometry_fixture import (
    semantic_bitfield_pdf_bytes, stated_range_pdf_bytes,
)

SID, REV = "SEM", "R1"
OPERATOR = "test-operator"


class _Fixture(unittest.TestCase):
    """The STD2B layout, whose ruler runs 31..24 and whose spans are measured."""

    pdf_bytes = staticmethod(semantic_bitfield_pdf_bytes)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "bitfield.pdf"
        pdf.write_bytes(self.pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                                   source_origin="TEST_FIXTURE")
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
        self.fingerprints = dict(
            standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=structure_fingerprint(self.structures))

    def tearDown(self):
        self.temp.cleanup()

    # -- fixture surgery, so the geometry can be made to disagree on purpose --

    def restate(self, *labels, rows=None):
        """Rewrite each span's label, and the cell it came from, together."""
        spans, cells = [], {}
        for index, span in enumerate(self.candidate.spans):
            text = labels[index] if index < len(labels) else span.text
            row = span.row_index if rows is None else rows[index]
            cells[(span.bbox, span.text)] = (text, row)
            spans.append(replace(span, text=text, row_index=row))
        moved: dict[int, list] = {}
        rebuilt = []
        for row in self.table.rows:
            keep = []
            for cell in row.cells:
                key = (cell.bbox, cell.text)
                if key not in cells:
                    keep.append(cell)
                    continue
                text, target = cells[key]
                # A span that now belongs to another word moves its cell with
                # it, because a cell lives in exactly one row of the table.
                moved.setdefault(target, []).append(
                    replace(cell, text=text, row_index=target))
            rebuilt.append(replace(row, cells=tuple(keep)))
        rows = {row.row_index: row for row in rebuilt}
        for index, arrivals in moved.items():
            base = rows.get(index, replace(rebuilt[-1], row_index=index, cells=()))
            rows[index] = replace(base, cells=tuple(
                sorted(base.cells + tuple(arrivals),
                       key=lambda cell: cell.column_index)))
        self.table = replace(self.table, rows=tuple(
            rows[key] for key in sorted(rows)))
        self.candidate = replace(self.candidate, spans=tuple(spans))

    def approval(self, verdict="PASS", roles=None, reviewer=OPERATOR, **kwargs):
        return StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict=verdict,
            reviewer=reviewer, reviewed_at="2026-01-01T00:00:00+00:00",
            structure_fingerprint=self.fingerprints["structure_fingerprint"],
            span_roles=tuple(roles if roles is not None
                             else ["FIELD"] * len(self.candidates())),
            **kwargs)

    def candidates(self):
        """Every cell that could name a field, measured or stated."""
        return field_candidates(self.candidate.to_dict(), self.table.to_dict())

    def promote(self, approval=None):
        return promote_bitfield(self.candidate, self.table,
                                approval or self.approval(), **self.fingerprints)


class PositionAuthorityTests(_Fixture):
    def test_a_stated_range_gives_the_position_despite_a_misleading_box(self):
        # Each label's box measures four bits; each label states eight.
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        fields = {field.label: field for field in definition.fields}
        alpha = fields["FIELD_A (31-24)"]
        bravo = fields["FIELD_B (23-16)"]
        self.assertEqual((alpha.msb, alpha.lsb, alpha.width), (31, 24, 8))
        self.assertEqual((bravo.msb, bravo.lsb, bravo.width), (23, 16, 8))
        for field in (alpha, bravo):
            self.assertEqual(field.position_source, PositionSource.STATED_RANGE)
            self.assertEqual(field.semantic_role, SemanticRole.FIELD)

    def test_geometry_still_gives_the_position_when_nothing_is_stated(self):
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        for field in definition.fields:
            self.assertEqual(field.position_source, PositionSource.GEOMETRIC_RULER)
        fields = {field.label: field for field in definition.fields}
        self.assertEqual((fields["FIELD_A_ALPHA_ONEX"].msb,
                          fields["FIELD_A_ALPHA_ONEX"].lsb), (31, 28))

    def test_a_measured_position_says_when_its_ruler_was_only_a_fragment(self):
        # The fixture ruler runs 31..24, which is a piece of a word, not a word.
        definition, _ = self.promote()
        for field in definition.fields:
            self.assertEqual(field.position_source, PositionSource.GEOMETRIC_RULER)
            self.assertIn(PARTIAL_RULER_WARNING, field.warnings)

    def test_a_stated_position_carries_no_partial_ruler_warning(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, _ = self.promote()
        for field in definition.fields:
            self.assertNotIn(PARTIAL_RULER_WARNING, field.warnings)

    def test_a_conflict_resolves_to_the_stated_range_and_is_recorded(self):
        position, why, blocking = resolve_position(
            "SUBFIELD_LENGTH (23-12)", cell_id="cel-1",
            source_ids=("std-1",), provenance="DIRECT_TEXT_MATCH",
            covered_labels=(17, 16, 15, 14), ruler_labels=tuple(range(31, -1, -1)))
        self.assertEqual(why, "")
        self.assertFalse(blocking)
        self.assertEqual((position.msb, position.lsb, position.width), (23, 12, 12))
        self.assertEqual(position.source, PositionSource.STATED_RANGE)
        self.assertEqual((position.geometry_msb, position.geometry_lsb), (17, 14))
        self.assertIn(GEOMETRY_RANGE_CONFLICT, position.warnings)

    def test_the_two_authorities_are_never_averaged_or_intersected(self):
        position, _, _ = resolve_position(
            "FIELD (23-12)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH", covered_labels=(17, 16, 15, 14),
            ruler_labels=tuple(range(31, -1, -1)))
        self.assertEqual(set(position.covered), set(range(12, 24)))
        self.assertNotIn(position.width, (4, 8))

    def test_the_conflict_survives_into_the_stored_field(self):
        self.restate("FIELD_A (31-16)", "FIELD_B (15-0)")
        definition, _ = self.promote()
        for field in definition.fields:
            self.assertEqual(field.position_source, PositionSource.STATED_RANGE)
            self.assertTrue(field.stated_range_text)
            self.assertIsNotNone(field.geometry_msb)
            self.assertNotEqual((field.msb, field.lsb),
                                (field.geometry_msb, field.geometry_lsb))

    def test_the_normative_label_is_stored_whole_and_a_display_label_derived(self):
        self.restate("Alpha Size (31-24)", "Beta Count (23-16)")
        definition, _ = self.promote()
        fields = {field.display_label: field for field in definition.fields}
        self.assertEqual(set(fields), {"Alpha Size", "Beta Count"})
        self.assertEqual(fields["Alpha Size"].label, "Alpha Size (31-24)")
        self.assertEqual(fields["Alpha Size"].normalized_identifier, "ALPHA_SIZE")

    def test_a_word_count_label_never_becomes_a_position(self):
        self.restate("BLOCK_HEADER (1 Word)", "FIELD_B (23-16)")
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        # The word count states nothing, so that span falls back to its box.
        by_label = {field.label: field for field in definition.fields}
        self.assertEqual(by_label["BLOCK_HEADER (1 Word)"].position_source,
                         PositionSource.GEOMETRIC_RULER)
        self.assertEqual(by_label["FIELD_B (23-16)"].position_source,
                         PositionSource.STATED_RANGE)


class CompletenessTests(_Fixture):
    def test_a_definition_whose_fields_all_resolve_is_complete(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, _ = self.promote()
        self.assertEqual(definition.completeness, DefinitionCompleteness.COMPLETE)
        self.assertEqual(definition.unresolved_field_count, 0)
        self.assertEqual(len(definition.fields), 2)

    def test_a_field_with_no_authority_is_left_out_and_marked_partial(self):
        # A range wider than the word states nothing word-local, and this span
        # covers no ruler label to fall back to.
        self.restate("FIELD_A (63..32)", "FIELD_B (23-16)")
        self.candidate = replace(self.candidate, spans=(
            replace(self.candidate.spans[0], covered_labels=()),
            self.candidate.spans[1]))
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        self.assertEqual(definition.completeness, DefinitionCompleteness.PARTIAL)
        self.assertEqual(definition.unresolved_field_count, 1)
        self.assertEqual([field.label for field in definition.fields],
                         ["FIELD_B (23-16)"])
        self.assertTrue(any("field candidate 0" in item
                            for item in definition.warnings))

    def test_a_missing_field_is_never_invented(self):
        self.restate("FIELD_A (63..32)", "FIELD_B (23-16)")
        self.candidate = replace(self.candidate, spans=(
            replace(self.candidate.spans[0], covered_labels=()),
            self.candidate.spans[1]))
        definition, _ = self.promote()
        self.assertNotIn("RESERVED", [field.semantic_role.value
                                      for field in definition.fields])
        self.assertEqual(len(definition.fields), 1)

    def test_a_candidate_with_no_resolvable_field_is_blocked_not_partial(self):
        self.restate("FIELD_A (63..32)", "FIELD_B (63..32)")
        self.candidate = replace(self.candidate, spans=tuple(
            replace(span, covered_labels=()) for span in self.candidate.spans))
        definition, blocked = self.promote()
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "NO_RESOLVED_FIELD_POSITION")


class RangeLayoutTests(_Fixture):
    def test_two_stated_fields_in_one_word_may_not_overlap(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (27-16)", rows=[1, 1])
        definition, blocked = self.promote()
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "OVERLAPPING_FIELDS")

    def test_the_same_local_range_in_two_words_is_not_an_overlap(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (31-24)", rows=[1, 2])
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        self.assertEqual({field.word_index for field in definition.fields}, {1, 2})
        for field in definition.fields:
            self.assertEqual((field.msb, field.lsb), (31, 24))

    def test_gaps_between_stated_fields_are_left_as_the_document_wrote_them(self):
        self.restate("FIELD_A (31-28)", "FIELD_B (25-24)", rows=[1, 1])
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        covered = set()
        for field in definition.fields:
            covered |= set(field.covered_bit_labels)
        self.assertNotIn(27, covered)
        self.assertNotIn(26, covered)
        self.assertEqual(len(definition.fields), 2)

    def test_a_range_outside_a_whole_ruler_does_not_become_a_field(self):
        wide = tuple(range(31, -1, -1))
        position, why, blocking = resolve_position(
            "FIELD (47-40)", cell_id="cel-1", source_ids=("std-1",),
            provenance="DIRECT_TEXT_MATCH", covered_labels=(), ruler_labels=wide)
        self.assertIsNone(position)
        self.assertFalse(blocking)
        self.assertIn("refused", why)


class GateTests(_Fixture):
    def test_a_stated_range_does_not_bypass_human_review(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, blocked = promote_bitfield(
            self.candidate, self.table, None, **self.fingerprints)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "NO_HUMAN_REVIEW")

    def test_an_assistant_may_not_approve_a_stated_range_definition(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        with self.assertRaises(StandardSemanticError):
            self.approval(reviewer="claude-assistant")

    def test_a_stated_range_still_needs_a_canonical_source(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        stripped = []
        for row in self.table.rows:
            stripped.append(replace(row, cells=tuple(
                replace(cell, source_ids=()) if cell.text.startswith("FIELD_A")
                else cell for cell in row.cells)))
        self.table = replace(self.table, rows=tuple(stripped))
        definition, blocked = self.promote()
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "UNRESOLVED_PROVENANCE")

    def test_a_stale_approval_still_blocks_a_stated_range_definition(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        approval = replace(self.approval(), structure_fingerprint="0" * 64)
        definition, blocked = self.promote(approval)
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "STALE_APPROVAL")

    def test_an_unclassified_span_still_blocks(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, blocked = self.promote(
            self.approval(roles=["FIELD", SpanRole.UNKNOWN.value]))
        self.assertIsNone(definition)
        self.assertEqual(blocked["reason"], "SPAN_UNCLASSIFIED")

    def test_validation_accepts_a_stated_position_off_a_fragment_ruler(self):
        self.restate("FIELD_A (15-8)", "FIELD_B (7-0)")
        semantics = build_semantics(
            self.structures, {self.candidate.bitfield_id: self.approval()},
            **self.fingerprints)
        # build_semantics reads the unmutated structures, so promote directly.
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        for field in definition.fields:
            self.assertFalse(set(field.covered_bit_labels)
                             <= set(definition.visible_bit_labels))
        validate_semantics(
            replace(semantics, bitfields=(definition,)),
            structures=self.structures,
            source_ids=frozenset(unit.source_id for unit in self.units),
            **{key: value for key, value in self.fingerprints.items()
               if key.endswith("_fingerprint")})

    def test_validation_still_rejects_a_measured_position_off_the_ruler(self):
        definition, _ = self.promote()
        broken = replace(definition, fields=(
            replace(definition.fields[0], msb=63, lsb=60,
                    covered_bit_labels=(60, 61, 62, 63), width=4),
        ) + definition.fields[1:])
        semantics = build_semantics(
            self.structures, {}, **self.fingerprints)
        with self.assertRaises(StandardSemanticError):
            validate_semantics(
                replace(semantics, bitfields=(broken,)),
                structures=self.structures,
                source_ids=frozenset(unit.source_id for unit in self.units),
                **{key: value for key, value in self.fingerprints.items()
                   if key.endswith("_fingerprint")})


class DeterminismTests(_Fixture):
    def test_a_stated_range_definition_is_byte_for_byte_reproducible(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        first, _ = self.promote()
        second, _ = self.promote()
        self.assertEqual(first.definition_id, second.definition_id)
        self.assertEqual(first.semantic_fingerprint, second.semantic_fingerprint)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_the_position_source_survives_the_dictionary_round_trip(self):
        self.restate("FIELD_A (31-24)", "FIELD_B (23-16)")
        definition, _ = self.promote()
        value = definition.to_dict()
        self.assertEqual(value["completeness"], "COMPLETE")
        self.assertTrue(all(field["position_source"] == "STATED_RANGE"
                            for field in value["fields"]))
        self.assertTrue(all(field["stated_range_text"] for field in value["fields"]))

    def test_a_field_id_distinguishes_a_stated_position_from_a_measured_one(self):
        measured, _ = self.promote()
        self.restate("FIELD_A_ALPHA_ONEX (31-28)", "FIELD_B_BRAVO_TWO (27-24)")
        stated, _ = self.promote()
        # Same bits, different authority: the identity must not collide.
        self.assertEqual((measured.fields[0].msb, measured.fields[0].lsb),
                         (stated.fields[0].msb, stated.fields[0].lsb))
        self.assertNotEqual(measured.fields[0].field_id, stated.fields[0].field_id)

    def test_an_empty_semantic_set_is_still_deterministic(self):
        semantics = build_semantics(self.structures, {}, **self.fingerprints)
        self.assertEqual(semantics.bitfields, ())
        self.assertEqual(semantic_fingerprint(semantics),
                         semantic_fingerprint(build_semantics(
                             self.structures, {}, **self.fingerprints)))


class ExtractionPipelineTests(_Fixture):
    """A stated range read out of a real extracted page, not a hand-built cell."""

    pdf_bytes = staticmethod(stated_range_pdf_bytes)

    def test_a_stated_range_survives_extraction_and_outranks_the_measurement(self):
        span = self.candidate.spans[0]
        self.assertIn("(15..12)", span.text)
        # What the page measures is not what the label says.
        self.assertNotEqual(
            (max(span.covered_labels), min(span.covered_labels)), (15, 12))
        definition, blocked = self.promote()
        self.assertIsNone(blocked)
        field = definition.fields[0]
        self.assertEqual((field.msb, field.lsb, field.width), (15, 12, 4))
        self.assertEqual(field.position_source, PositionSource.STATED_RANGE)
        self.assertTrue(field.supporting_source_ids)

    def test_the_geometry_store_is_not_rewritten_by_reading_stated_ranges(self):
        before = StandardStructureStore(self.store).load_manifest(SID, REV)
        self.promote()
        after = StandardStructureStore(self.store).load_manifest(SID, REV)
        self.assertEqual(before.structure_fingerprint, after.structure_fingerprint)
        self.assertEqual(before.corpus_fingerprint, after.corpus_fingerprint)
        self.assertEqual(before.layout_fingerprint, after.layout_fingerprint)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
