"""The operator command that carries existing approvals into the history log.

It reports by default and writes only when told to, because the thing it
touches is the record of human decisions and there is no undo for it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_approval_history import (
    ApprovalEventType, EvidenceStatus, StandardApprovalHistory,
)
from standard_commands import (
    StandardCommandError, StandardOperator, handle_standard_command,
    standard_help_lines, standard_usage,
)
from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, StandardBitfieldApproval, build_semantics,
    semantic_fingerprint,
)
from standard_semantic_store import StandardApprovalStore
from standard_store import StandardStore
from standard_structure import extract_structures, structure_fingerprint
from standard_structure_store import StandardStructureStore, build_manifest
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes

SID, REV = "SEM", "R1"


class _Migration(unittest.TestCase):
    """A store holding one approval written the way the old code wrote it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "bitfield.pdf"
        pdf.write_bytes(semantic_bitfield_pdf_bytes())
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(self.store, pdf, standard_id=SID,
                                   revision=REV, source_origin="TEST_FIXTURE")
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
        self.approvals = StandardApprovalStore(self.store)
        self.history = StandardApprovalHistory(self.store)
        self.operator = StandardOperator(self.store)
        from standard_value_pair import value_local_pairs
        from standard_word_association import field_candidates
        pairs, _ = value_local_pairs(self.candidate.to_dict(),
                                     self.table.to_dict())
        self.width = len(field_candidates(
            self.candidate.to_dict(), self.table.to_dict(),
            value_local_cells=frozenset(cell for pair in pairs
                                        for cell in pair.cell_ids)))

    def tearDown(self):
        self.temp.cleanup()

    def legacy(self, *, evidence="e" * 64, reviewer="rossierd",
               approved_at="2026-08-31T14:10:47+00:00"):
        """An approvals file exactly as the version before the log wrote it."""
        approval = StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id,
            verdict="ACCEPTABLE_WARNING", reviewer=reviewer,
            reviewed_at=approved_at, structure_fingerprint=self.fingerprint,
            span_roles=tuple(["FIELD"] * self.width),
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=(),
            review_evidence_fingerprint=evidence)
        row = approval.to_dict()
        row.pop("event_id", None)
        row.pop("supersedes", None)
        self.approvals.path(SID, REV).write_text(json.dumps({
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "standard_id": SID, "revision": REV,
            "structure_fingerprint": self.fingerprint,
            "updated_at": approved_at, "rows": [row]}))
        return approval

    def run_command(self, tail=""):
        return handle_standard_command(
            f"/standard migrate-approval-history {SID} {REV} {tail}".strip(),
            self.operator)

    def snapshot(self):
        paths = [self.approvals.path(SID, REV), self.history.path(SID, REV)]
        return [(p.read_bytes() if p.exists() else None) for p in paths]


class Discoverability(_Migration):
    def test_the_action_appears_in_usage(self):
        self.assertIn("migrate-approval-history", standard_usage())

    def test_the_action_appears_in_help_with_its_flag(self):
        line = next(item for item in standard_help_lines()
                    if "migrate-approval-history" in item)
        self.assertEqual(
            line, "  /standard migrate-approval-history <id> <revision> [--apply]")

    def test_help_is_generated_from_one_table(self):
        # The usage string and the help lines must not drift apart, so both
        # are derived rather than written twice.
        for name in ("migrate-approval-history", "approval-history"):
            self.assertIn(name, standard_usage())
            self.assertTrue(any(name in item for item in standard_help_lines()))


class Arguments(_Migration):
    def test_a_missing_revision_is_rejected(self):
        with self.assertRaises(StandardCommandError) as caught:
            handle_standard_command(
                f"/standard migrate-approval-history {SID}", self.operator)
        self.assertIn("usage:", str(caught.exception))

    def test_a_missing_identifier_is_rejected(self):
        with self.assertRaises(StandardCommandError):
            handle_standard_command(
                "/standard migrate-approval-history", self.operator)

    def test_a_surplus_argument_is_rejected(self):
        with self.assertRaises(StandardCommandError):
            self.run_command("extra")

    def test_an_unknown_flag_is_rejected(self):
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command("--force")
        self.assertIn("unknown option --force", str(caught.exception))

    def test_an_unknown_flag_is_rejected_even_beside_apply(self):
        with self.assertRaises(StandardCommandError):
            self.run_command("--apply --yes")


class DryRun(_Migration):
    def test_the_default_writes_nothing(self):
        self.legacy()
        before = self.snapshot()
        self.run_command()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.history.path(SID, REV).exists())

    def test_the_plan_is_reported(self):
        self.legacy()
        output = self.run_command()
        self.assertIn("Active approvals                     : 1", output)
        self.assertIn("Approvals requiring migration        : 1", output)
        self.assertIn("Events to create                     : 1", output)
        self.assertIn("Active pointers to attach            : 1", output)
        self.assertIn("Existing historical events           : 0", output)
        self.assertIn(self.candidate.bitfield_id, output)
        self.assertIn("rossierd", output)
        self.assertIn("2026-08-31T14:10:47+00:00", output)
        self.assertIn("Nothing was written", output)

    def test_a_legacy_approval_with_no_evidence_is_counted_and_marked(self):
        self.legacy(evidence=None)
        output = self.run_command()
        self.assertIn("Legacy approvals with no review evidence: 1", output)
        self.assertIn("<none>", output)
        self.assertIn("predates evidence binding", output)

    def test_no_full_evidence_payload_is_dumped(self):
        self.legacy()
        self.assertNotIn('"candidates"', self.run_command())

    def test_a_store_with_nothing_to_migrate_says_so(self):
        output = self.run_command()
        self.assertIn("Approvals requiring migration        : 0", output)
        self.assertIn("already recorded", output)


class Apply(_Migration):
    def test_apply_creates_the_events(self):
        self.legacy()
        output = self.run_command("--apply")
        self.assertIn("Initial events created: 1", output)
        self.assertIn("Active approvals linked: 1", output)
        self.assertIn("Existing events preserved: 0", output)
        self.assertEqual(len(self.history.events(SID, REV)), 1)

    def test_apply_links_the_active_approval(self):
        self.legacy()
        self.run_command("--apply")
        stored, _ = self.approvals.load(SID, REV)
        active = stored[self.candidate.bitfield_id]
        self.assertEqual(active.event_id,
                         self.history.events(SID, REV)[0].event_id)
        self.assertEqual(self.history.verify(SID, REV, stored), [])

    def test_the_event_is_an_initial_migrated_decision(self):
        self.legacy()
        self.run_command("--apply")
        event = self.history.events(SID, REV)[0]
        self.assertIs(event.event_type,
                      ApprovalEventType.MIGRATED_INITIAL_APPROVAL)
        self.assertIsNone(event.supersedes)

    def test_the_original_reviewer_is_preserved(self):
        legacy = self.legacy(reviewer="rossierd")
        self.run_command("--apply")
        self.assertEqual(self.history.events(SID, REV)[0].reviewer, "rossierd")

    def test_the_original_timestamp_is_preserved(self):
        legacy = self.legacy()
        self.run_command("--apply")
        self.assertEqual(self.history.events(SID, REV)[0].approved_at,
                         legacy.reviewed_at)

    def test_the_original_evidence_is_preserved(self):
        legacy = self.legacy()
        self.run_command("--apply")
        self.assertEqual(
            self.history.events(SID, REV)[0].review_evidence_fingerprint,
            legacy.review_evidence_fingerprint)

    def test_a_legacy_null_evidence_stays_null(self):
        self.legacy(evidence=None)
        self.run_command("--apply")
        event = self.history.events(SID, REV)[0]
        self.assertIsNone(event.review_evidence_fingerprint)
        self.assertIs(event.evidence_status,
                      EvidenceStatus.PREDATES_EVIDENCE_BINDING)

    def test_the_output_does_not_claim_a_rebuild(self):
        self.legacy()
        output = self.run_command("--apply").lower()
        for absent in ("rebuilt", "build-structure", "definition"):
            self.assertNotIn(absent, output)


class Idempotence(_Migration):
    def test_a_second_apply_creates_nothing(self):
        self.legacy()
        self.run_command("--apply")
        first = self.history.events(SID, REV)
        output = self.run_command("--apply")
        self.assertIn("Initial events created: 0", output)
        self.assertIn("Existing events preserved: 1", output)
        self.assertIn("Nothing to migrate", output)
        self.assertEqual(len(self.history.events(SID, REV)), 1)

    def test_a_second_apply_preserves_the_event_byte_for_byte(self):
        self.legacy()
        self.run_command("--apply")
        before = self.history.path(SID, REV).read_bytes()
        self.run_command("--apply")
        self.assertEqual(self.history.path(SID, REV).read_bytes(), before)

    def test_a_dry_run_after_apply_reports_nothing_to_do(self):
        self.legacy()
        self.run_command("--apply")
        output = self.run_command()
        self.assertIn("Approvals requiring migration        : 0", output)
        self.assertIn("Existing historical events           : 1", output)


class FailClosed(_Migration):
    def test_a_corrupt_history_blocks_the_dry_run(self):
        self.legacy()
        self.history.path(SID, REV).write_text("{not json\n")
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command()
        self.assertIn("migration refused", str(caught.exception))

    def test_a_corrupt_history_blocks_apply_and_writes_nothing(self):
        self.legacy()
        self.history.path(SID, REV).write_text("{not json\n")
        before = self.snapshot()
        with self.assertRaises(StandardCommandError):
            self.run_command("--apply")
        self.assertEqual(self.snapshot(), before)

    def test_a_broken_active_pointer_blocks_apply(self):
        self.legacy()
        self.run_command("--apply")
        self.history.path(SID, REV).unlink()
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command("--apply")
        self.assertIn("ACTIVE_EVENT_MISSING", str(caught.exception))

    def test_a_diverged_active_approval_blocks_apply(self):
        self.legacy()
        self.run_command("--apply")
        raw = json.loads(self.approvals.path(SID, REV).read_text("utf-8"))
        raw["rows"][0]["verdict"] = "FAIL"
        self.approvals.path(SID, REV).write_text(json.dumps(raw))
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command("--apply")
        self.assertIn("ACTIVE_EVENT_DIVERGED", str(caught.exception))

    def test_a_record_from_another_revision_blocks_apply(self):
        self.legacy()
        self.run_command("--apply")
        event = self.history.events(SID, REV)[0].to_dict()
        event["revision"] = "OTHER"
        self.history.path(SID, REV).write_text(json.dumps(event) + "\n")
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command("--apply")
        self.assertIn("belongs to", str(caught.exception))

    def test_a_duplicate_event_blocks_apply(self):
        self.legacy()
        self.run_command("--apply")
        line = self.history.path(SID, REV).read_text("utf-8").strip()
        self.history.path(SID, REV).write_text(line + "\n" + line + "\n")
        with self.assertRaises(StandardCommandError) as caught:
            self.run_command("--apply")
        self.assertIn("recorded twice", str(caught.exception))


class Interoperability(_Migration):
    def test_the_history_command_sees_the_migrated_events(self):
        self.legacy()
        self.run_command("--apply")
        output = handle_standard_command(
            f"/standard approval-history {SID} {REV}", self.operator)
        self.assertIn("MIGRATED_INITIAL_APPROVAL", output)
        self.assertIn("[ACTIVE]", output)
        self.assertIn("Consistency: OK", output)
        self.assertIn(self.candidate.bitfield_id, output)

    def test_migration_leaves_the_semantic_definitions_alone(self):
        self.legacy()

        def build():
            approvals, _ = self.approvals.load(SID, REV)
            return build_semantics(
                self.structures, approvals, standard_id=SID, revision=REV,
                corpus_fingerprint=self.manifest.corpus_manifest_sha256,
                layout_fingerprint=self.artifact["layout_fingerprint"],
                structure_fingerprint=self.fingerprint,
                layout=self.artifact, units=self.units)

        before = build()
        self.run_command("--apply")
        after = build()
        self.assertEqual(semantic_fingerprint(before),
                         semantic_fingerprint(after))
        self.assertEqual(json.dumps(before.to_dict(), sort_keys=True),
                         json.dumps(after.to_dict(), sort_keys=True))

    def test_no_event_identity_reaches_the_semantic_payload(self):
        self.legacy()
        self.run_command("--apply")
        approvals, _ = self.approvals.load(SID, REV)
        semantics = build_semantics(
            self.structures, approvals, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint,
            layout=self.artifact, units=self.units)
        self.assertNotIn("evt-", json.dumps(semantics.to_dict()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
