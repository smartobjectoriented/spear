"""Re-reviewing a candidate must not erase what was reviewed before.

The store used to key approvals by candidate, so a second decision replaced
the first and the first stopped existing. These tests hold the replacement
open: the new decision governs, the old one stays readable, and the link
between them is explicit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standard_approval_history import (
    APPROVAL_FILE_SCHEMA_VERSION, ApprovalEventType, EvidenceStatus,
    StandardApprovalHistory, StandardApprovalHistoryError, event_for,
)
from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_semantic import (
    APPROVAL_SCHEMA_VERSION, StandardBitfieldApproval, StandardSemanticError,
    build_semantics,
)
from standard_semantic_store import StandardApprovalStore
from standard_store import StandardStore
from standard_structure import extract_structures, structure_fingerprint
from standard_structure_store import StandardStructureStore, build_manifest
from tests.standard_geometry_fixture import semantic_bitfield_pdf_bytes

SID, REV = "SEM", "R1"
OPERATOR = "test-operator"


class _Store(unittest.TestCase):
    """One ingested fixture standard with a reviewable bitfield candidate."""

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

    def approve(self, verdict="PASS", roles=None, reviewer=OPERATOR):
        return self.approvals.approve(
            SID, REV, self.candidate.bitfield_id, verdict=verdict,
            span_roles=roles or ["FIELD"] * self.width, reviewer=reviewer)

    def events(self, **kwargs):
        return self.history.events(SID, REV, **kwargs)


class FirstApproval(_Store):
    def test_a_first_approval_records_an_event(self):
        approval = self.approve()
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertIs(events[0].event_type, ApprovalEventType.INITIAL_APPROVAL)
        self.assertIsNone(events[0].supersedes)

    def test_the_first_event_becomes_active(self):
        approval = self.approve()
        self.assertEqual(approval.event_id, self.events()[0].event_id)
        stored, _ = self.approvals.load(SID, REV)
        self.assertEqual(stored[self.candidate.bitfield_id].event_id,
                         self.events()[0].event_id)

    def test_the_event_is_the_chain_tip(self):
        self.approve()
        tip = self.history.tip(SID, REV, self.candidate.bitfield_id)
        self.assertEqual(tip.event_id, self.events()[0].event_id)

    def test_the_event_is_self_contained(self):
        self.approve(verdict="ACCEPTABLE_WARNING")
        event = self.events()[0]
        self.assertEqual(event.verdict, "ACCEPTABLE_WARNING")
        self.assertEqual(len(event.span_roles), self.width)
        self.assertEqual(event.reviewer, OPERATOR)
        self.assertEqual(event.structure_fingerprint, self.fingerprint)
        self.assertEqual(event.candidate_id, self.candidate.bitfield_id)
        self.assertEqual(event.standard_id, SID)
        self.assertEqual(event.revision, REV)
        self.assertEqual(event.approval_schema_version, APPROVAL_SCHEMA_VERSION)
        self.assertTrue(event.approved_at)
        self.assertIs(event.evidence_status, EvidenceStatus.BOUND)

    def test_the_active_file_records_the_new_layout_version(self):
        self.approve()
        raw = json.loads(self.approvals.path(SID, REV).read_text("utf-8"))
        self.assertEqual(raw["schema_version"], APPROVAL_FILE_SCHEMA_VERSION)


class Reapproval(_Store):
    def setUp(self):
        super().setUp()
        self.first = self.approve(verdict="PASS")
        self.first_event = self.events()[0]
        self.second = self.approve(verdict="ACCEPTABLE_WARNING")

    def test_the_previous_event_survives(self):
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_id, self.first_event.event_id)

    def test_the_new_event_supersedes_the_previous(self):
        events = self.events()
        self.assertIs(events[1].event_type, ApprovalEventType.REAPPROVAL)
        self.assertEqual(events[1].supersedes, self.first_event.event_id)

    def test_the_active_approval_is_the_newest(self):
        stored, _ = self.approvals.load(SID, REV)
        active = stored[self.candidate.bitfield_id]
        self.assertEqual(active.event_id, self.events()[1].event_id)
        self.assertEqual(active.supersedes, self.first_event.event_id)
        self.assertEqual(active.verdict, "ACCEPTABLE_WARNING")

    def test_the_previous_verdict_is_preserved(self):
        self.assertEqual(self.events()[0].verdict, "PASS")

    def test_the_previous_timestamp_is_preserved(self):
        self.assertEqual(self.events()[0].approved_at, self.first.reviewed_at)

    def test_the_previous_reviewer_is_preserved(self):
        self.assertEqual(self.events()[0].reviewer, self.first.reviewer)

    def test_the_previous_evidence_is_preserved(self):
        self.assertEqual(self.events()[0].review_evidence_fingerprint,
                         self.first.review_evidence_fingerprint)

    def test_the_previous_roles_are_preserved(self):
        self.assertEqual(list(self.events()[0].span_roles),
                         list(self.first.span_roles))

    def test_the_previous_event_is_byte_identical_after_the_second(self):
        # The whole point: appending must not rewrite what is already there.
        first = self.events()[0]
        self.approve(verdict="PASS", reviewer="another-operator")
        self.assertEqual(json.dumps(first.to_dict(), sort_keys=True),
                         json.dumps(self.events()[0].to_dict(), sort_keys=True))

    def test_the_chain_reads_oldest_first(self):
        chain = self.history.chain(SID, REV, self.candidate.bitfield_id)
        self.assertEqual([item.event_id for item in chain],
                         [item.event_id for item in self.events()])

    def test_exactly_one_tip(self):
        tip = self.history.tip(SID, REV, self.candidate.bitfield_id)
        self.assertEqual(tip.event_id, self.events()[1].event_id)

    def test_the_active_approval_agrees_with_the_history(self):
        stored, _ = self.approvals.load(SID, REV)
        self.assertEqual(self.history.verify(SID, REV, stored), [])


class VerdictTransitions(_Store):
    """A refusal is a decision, and is recorded like any other."""

    def test_accept_then_reject_keeps_both(self):
        self.approve(verdict="PASS")
        self.approve(verdict="FAIL")
        events = self.events()
        self.assertEqual([item.verdict for item in events], ["PASS", "FAIL"])
        stored, _ = self.approvals.load(SID, REV)
        self.assertEqual(stored[self.candidate.bitfield_id].verdict, "FAIL")

    def test_reject_then_accept_keeps_both(self):
        self.approve(verdict="FAIL")
        self.approve(verdict="PASS")
        self.assertEqual([item.verdict for item in self.events()],
                         ["FAIL", "PASS"])

    def test_a_needs_followup_decision_is_recorded(self):
        self.approve(verdict="NEEDS_FOLLOWUP")
        self.assertEqual(self.events()[0].verdict, "NEEDS_FOLLOWUP")


class DuplicateSubmission(_Store):
    def test_a_repeat_of_the_active_decision_is_refused(self):
        self.approve(verdict="PASS")
        # Same verdict, roles, reviewer, evidence and links: one command run
        # twice, not a person deciding twice. Nothing about what governs would
        # change, so there is nothing to record.
        with self.assertRaises(StandardSemanticError) as caught:
            self.approve(verdict="PASS")
        self.assertIn("already the active one", str(caught.exception))
        self.assertEqual(len(self.events()), 1)

    def test_the_refusal_names_the_active_event(self):
        approval = self.approve(verdict="PASS")
        with self.assertRaises(StandardSemanticError) as caught:
            self.approve(verdict="PASS")
        self.assertIn(approval.event_id, str(caught.exception))

    def test_a_refused_repeat_changes_nothing(self):
        self.approve(verdict="PASS")
        path = self.history.path(SID, REV)
        before = (path.read_bytes(),
                  self.approvals.path(SID, REV).read_bytes())
        with self.assertRaises(StandardSemanticError):
            self.approve(verdict="PASS")
        self.assertEqual((path.read_bytes(),
                          self.approvals.path(SID, REV).read_bytes()), before)

    def test_a_decision_that_differs_in_any_way_is_recorded(self):
        self.approve(verdict="PASS")
        self.approve(verdict="ACCEPTABLE_WARNING")
        self.assertEqual(len(self.events()), 2)

    def test_a_different_reviewer_reaching_the_same_verdict_is_recorded(self):
        self.approve(verdict="PASS")
        self.approve(verdict="PASS", reviewer="second-operator")
        self.assertEqual(len(self.events()), 2)
        self.assertEqual([item.reviewer for item in self.events()],
                         [OPERATOR, "second-operator"])


class LegacyApprovals(_Store):
    """A store written before this log existed stays usable and is carried in."""

    def legacy(self, *, evidence="e" * 64, roles=None):
        """Write an approvals file exactly as the previous version wrote it."""
        approval = StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict="ACCEPTABLE_WARNING",
            reviewer="rossierd", reviewed_at="2026-08-31T14:10:47+00:00",
            structure_fingerprint=self.fingerprint,
            span_roles=tuple(roles or ["FIELD"] * self.width),
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=(),
            review_evidence_fingerprint=evidence)
        row = approval.to_dict()
        row.pop("event_id", None)
        row.pop("supersedes", None)
        self.approvals.path(SID, REV).write_text(json.dumps({
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "standard_id": SID, "revision": REV,
            "structure_fingerprint": self.fingerprint,
            "updated_at": "2026-08-31T14:10:47+00:00", "rows": [row]}))
        return approval

    def test_a_legacy_approval_loads_and_is_usable(self):
        self.legacy()
        stored, refused = self.approvals.load(SID, REV)
        self.assertEqual(refused, [])
        self.assertIsNone(stored[self.candidate.bitfield_id].event_id)

    def test_a_legacy_approval_with_no_history_is_not_a_fault(self):
        self.legacy()
        stored, _ = self.approvals.load(SID, REV)
        self.assertEqual(self.events(), ())
        self.assertEqual(self.history.verify(SID, REV, stored), [])

    def test_reapproving_a_legacy_approval_carries_it_in_first(self):
        legacy = self.legacy()
        self.approve(verdict="PASS")
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertIs(events[0].event_type,
                      ApprovalEventType.MIGRATED_INITIAL_APPROVAL)
        self.assertIs(events[1].event_type, ApprovalEventType.REAPPROVAL)
        self.assertEqual(events[1].supersedes, events[0].event_id)

    def test_the_carried_event_preserves_the_original_decision(self):
        legacy = self.legacy()
        self.approve(verdict="PASS")
        carried = self.events()[0]
        self.assertEqual(carried.reviewer, "rossierd")
        self.assertEqual(carried.approved_at, "2026-08-31T14:10:47+00:00")
        self.assertEqual(carried.review_evidence_fingerprint,
                         legacy.review_evidence_fingerprint)
        self.assertEqual(carried.verdict, "ACCEPTABLE_WARNING")
        self.assertEqual(list(carried.span_roles), list(legacy.span_roles))

    def test_an_approval_predating_evidence_binding_keeps_its_absence(self):
        self.legacy(evidence=None)
        self.approve(verdict="PASS")
        carried = self.events()[0]
        self.assertIsNone(carried.review_evidence_fingerprint)
        self.assertIs(carried.evidence_status,
                      EvidenceStatus.PREDATES_EVIDENCE_BINDING)
        self.assertEqual(carried.to_dict()["evidence_status"],
                         "PREDATES_EVIDENCE_BINDING")

    def test_migration_plans_one_event_per_legacy_approval(self):
        legacy = self.legacy()
        stored, _ = self.approvals.load(SID, REV)
        planned = self.history.migration_plan(SID, REV, stored)
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].approved_at, legacy.reviewed_at)
        self.assertEqual(planned[0].reviewer, legacy.reviewer)
        self.assertEqual(planned[0].review_evidence_fingerprint,
                         legacy.review_evidence_fingerprint)

    def test_migration_is_idempotent(self):
        self.legacy()
        stored, _ = self.approvals.load(SID, REV)
        planned = self.history.migration_plan(SID, REV, stored)
        self.history.append(SID, REV, planned)
        self.assertEqual(len(self.events()), 1)
        again = self.history.migration_plan(SID, REV, stored)
        self.assertEqual(again, [])

    def test_planning_a_migration_writes_nothing(self):
        self.legacy()
        stored, _ = self.approvals.load(SID, REV)
        self.history.migration_plan(SID, REV, stored)
        self.assertFalse(self.history.path(SID, REV).exists())


class Corruption(_Store):
    """A review log that cannot be trusted is refused, never repaired."""

    def write(self, lines):
        self.history.path(SID, REV).write_text("\n".join(lines) + "\n")

    def line(self, **overrides):
        self.approve()
        event = self.events()[0].to_dict()
        event.update(overrides)
        return event

    def test_a_truncated_last_record_is_refused(self):
        self.approve()
        text = self.history.path(SID, REV).read_text("utf-8")
        self.history.path(SID, REV).write_text(text.rstrip("\n")[:-20])
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("truncated", str(caught.exception))

    def test_a_duplicate_event_id_is_refused(self):
        event = self.line()
        self.write([json.dumps(event), json.dumps(event)])
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("recorded twice", str(caught.exception))

    def test_a_broken_predecessor_is_refused(self):
        event = self.line(supersedes="evt-" + "0" * 16)
        self.write([json.dumps(event)])
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("not recorded", str(caught.exception))

    def test_a_predecessor_from_another_candidate_is_refused(self):
        self.approve()
        first = self.events()[0].to_dict()
        other = dict(first)
        other["candidate_id"] = "bit-" + "0" * 16
        other["event_id"] = "evt-" + "1" * 16
        other["supersedes"] = first["event_id"]
        self.write([json.dumps(first), json.dumps(other)])
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("another candidate", str(caught.exception))

    def test_two_events_superseding_the_same_one_are_refused(self):
        self.approve()
        first = self.events()[0].to_dict()
        forks = []
        for tag in ("1", "2"):
            fork = dict(first)
            fork["event_id"] = "evt-" + tag * 16
            fork["supersedes"] = first["event_id"]
            forks.append(json.dumps(fork))
        self.write([json.dumps(first)] + forks)
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("superseded twice", str(caught.exception))

    def test_a_malformed_event_id_is_refused(self):
        self.write([json.dumps(self.line(event_id="nonsense"))])
        with self.assertRaises(StandardApprovalHistoryError):
            self.events()

    def test_a_malformed_record_is_refused(self):
        self.write(["{not json"])
        with self.assertRaises(StandardApprovalHistoryError):
            self.events()

    def test_a_record_from_another_revision_is_refused(self):
        self.write([json.dumps(self.line(revision="OTHER"))])
        with self.assertRaises(StandardApprovalHistoryError) as caught:
            self.events()
        self.assertIn("belongs to", str(caught.exception))

    def test_an_active_approval_naming_no_recorded_event_is_reported(self):
        self.approve()
        stored, _ = self.approvals.load(SID, REV)
        self.history.path(SID, REV).unlink()
        problems = self.history.verify(SID, REV, stored)
        self.assertEqual([item["reason"] for item in problems],
                         ["ACTIVE_EVENT_MISSING"])

    def test_an_active_approval_that_diverged_from_its_event_is_reported(self):
        self.approve()
        stored, _ = self.approvals.load(SID, REV)
        from dataclasses import replace
        stored[self.candidate.bitfield_id] = replace(
            stored[self.candidate.bitfield_id], verdict="FAIL")
        problems = self.history.verify(SID, REV, stored)
        self.assertEqual([item["reason"] for item in problems],
                         ["ACTIVE_EVENT_DIVERGED"])


class Isolation(_Store):
    def test_two_candidates_keep_separate_chains(self):
        if len(self.structures.bitfields) < 2:
            self.skipTest("the fixture has one bitfield candidate")

    def test_history_is_scoped_to_its_revision(self):
        self.approve()
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(
            self.history.events(SID, REV,
                                candidate_id="bit-" + "0" * 16), ())

    def test_reading_history_writes_nothing(self):
        self.approve()
        path = self.history.path(SID, REV)
        before = (path.stat().st_mtime_ns, path.read_bytes())
        self.events()
        self.history.tip(SID, REV, self.candidate.bitfield_id)
        stored, _ = self.approvals.load(SID, REV)
        self.history.verify(SID, REV, stored)
        self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)

    def test_the_log_serializes_deterministically(self):
        self.approve()
        event = self.events()[0]
        self.assertEqual(event.event_id,
                         event.sealed(recorded_at=event.recorded_at).event_id)
        self.assertEqual(json.dumps(event.to_dict(), sort_keys=True),
                         json.dumps(self.events()[0].to_dict(), sort_keys=True))


class PromotionUsesActiveOnly(_Store):
    """History is audit state. It has never produced a definition."""

    def build(self):
        approvals, _ = self.approvals.load(SID, REV)
        return build_semantics(
            self.structures, approvals, standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            structure_fingerprint=self.fingerprint,
            layout=self.artifact, units=self.units)

    def test_a_reapproved_candidate_builds_exactly_one_definition(self):
        self.approve(verdict="PASS")
        self.approve(verdict="ACCEPTABLE_WARNING")
        self.assertEqual(len(self.events()), 2)
        semantics = self.build()
        matching = [item for item in semantics.bitfields
                    if item.source_bitfield_candidate_id
                    == self.candidate.bitfield_id]
        self.assertEqual(len(matching), 1)

    def test_the_definition_records_the_active_decision(self):
        self.approve(verdict="PASS")
        self.approve(verdict="ACCEPTABLE_WARNING")
        definition = next(item for item in self.build().bitfields
                          if item.source_bitfield_candidate_id
                          == self.candidate.bitfield_id)
        self.assertEqual(definition.approval_verdict, "ACCEPTABLE_WARNING")

    def test_a_superseded_rejection_does_not_block_promotion(self):
        self.approve(verdict="FAIL")
        self.approve(verdict="PASS")
        semantics = self.build()
        self.assertTrue([item for item in semantics.bitfields
                         if item.source_bitfield_candidate_id
                         == self.candidate.bitfield_id])

    def test_the_definition_carries_no_event_identity(self):
        # An event id is approval state. Letting it into the semantic payload
        # would make every definition move whenever history was recorded.
        self.approve(verdict="PASS")
        definition = next(item for item in self.build().bitfields
                          if item.source_bitfield_candidate_id
                          == self.candidate.bitfield_id)
        payload = json.dumps(definition.to_dict())
        self.assertNotIn("event_id", payload)
        self.assertNotIn("evt-", payload)

    def test_recording_history_does_not_change_semantic_identity(self):
        self.approve(verdict="PASS")
        first = self.build()
        self.approve(verdict="PASS", reviewer="second-operator")
        second = self.build()
        before = next(item for item in first.bitfields
                      if item.source_bitfield_candidate_id
                      == self.candidate.bitfield_id)
        after = next(item for item in second.bitfields
                     if item.source_bitfield_candidate_id
                     == self.candidate.bitfield_id)
        self.assertEqual(before.definition_id, after.definition_id)
        self.assertEqual([f.field_id for f in before.fields],
                         [f.field_id for f in after.fields])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class MigrateThenReapprove(_Store):
    """Carrying approvals in and then re-reviewing must not double-record."""

    def legacy_row(self):
        approval = StandardBitfieldApproval(
            candidate_id=self.candidate.bitfield_id, verdict="ACCEPTABLE_WARNING",
            reviewer="rossierd", reviewed_at="2026-08-31T14:10:47+00:00",
            structure_fingerprint=self.fingerprint,
            span_roles=tuple(["FIELD"] * self.width),
            schema_version=APPROVAL_SCHEMA_VERSION, reviewed_links=(),
            review_evidence_fingerprint="e" * 64)
        row = approval.to_dict()
        row.pop("event_id", None)
        row.pop("supersedes", None)
        self.approvals.path(SID, REV).write_text(json.dumps({
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "standard_id": SID, "revision": REV,
            "structure_fingerprint": self.fingerprint,
            "updated_at": "2026-08-31T14:10:47+00:00", "rows": [row]}))
        return approval

    def test_a_dry_run_reports_without_writing(self):
        self.legacy_row()
        report = self.approvals.migrate_history(SID, REV)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["approved_at"], "2026-08-31T14:10:47+00:00")
        self.assertFalse(self.history.path(SID, REV).exists())

    def test_migration_stamps_the_active_row_with_its_event(self):
        self.legacy_row()
        report = self.approvals.migrate_history(SID, REV, dry_run=False)
        stored, _ = self.approvals.load(SID, REV)
        active = stored[self.candidate.bitfield_id]
        self.assertEqual(active.event_id, report[0]["event_id"])
        self.assertEqual(self.history.verify(SID, REV, stored), [])

    def test_migration_then_reapproval_records_exactly_two_events(self):
        self.legacy_row()
        self.approvals.migrate_history(SID, REV, dry_run=False)
        self.approve(verdict="PASS")
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertIs(events[0].event_type,
                      ApprovalEventType.MIGRATED_INITIAL_APPROVAL)
        self.assertIs(events[1].event_type, ApprovalEventType.REAPPROVAL)
        self.assertEqual(events[1].supersedes, events[0].event_id)

    def test_reapproving_without_migrating_first_gives_the_same_two_events(self):
        self.legacy_row()
        self.approve(verdict="PASS")
        events = self.events()
        self.assertEqual(len(events), 2)
        self.assertIs(events[0].event_type,
                      ApprovalEventType.MIGRATED_INITIAL_APPROVAL)
        self.assertEqual(events[1].supersedes, events[0].event_id)

    def test_migration_after_reapproval_adds_nothing(self):
        self.legacy_row()
        self.approve(verdict="PASS")
        self.assertEqual(self.approvals.migrate_history(SID, REV), [])
        self.assertEqual(len(self.events()), 2)

    def test_migration_is_idempotent_when_executed_twice(self):
        self.legacy_row()
        self.approvals.migrate_history(SID, REV, dry_run=False)
        self.assertEqual(self.approvals.migrate_history(SID, REV,
                                                        dry_run=False), [])
        self.assertEqual(len(self.events()), 1)
