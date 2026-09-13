"""Operator human-review workflow. Every fixture here is synthetic.

No licensed normative text appears in this file.
"""

import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_roles import AgentRole
from standard_commands import StandardCommandError, StandardOperator, handle_standard_command
from standard_ingest import ingest_pdf
from standard_review import (
    HUMAN_VERDICTS, REVIEW_SCHEMA_VERSION, StandardReview, StandardReviewError,
    UNREVIEWED, main, review_path, run_review,
)
from standard_schema import HumanValidationStatus
from standard_store import StandardStore, candidate_id_for
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from tests.standard_extraction_fixture import extraction_pdf_bytes
from tool_exposure import ToolExposurePolicy
from tool_registry import ToolRegistry


SID, REV = "REVIEW-STD", "R1"


class _ReviewFixture(unittest.TestCase):
    """Shared synthetic corpus and review file; carries no tests itself."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pdf = self.root / "review.pdf"
        pdf.write_bytes(extraction_pdf_bytes())
        self.store = StandardStore(self.root / "standards")
        self.manifest = ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                                   source_origin="TEST_FIXTURE")
        self.units = self.store.load_units(SID, REV)
        self.path = review_path(self.store, SID, REV)
        self.write_review()

    def tearDown(self):
        self.temp.cleanup()

    # -- fixtures ---------------------------------------------------------

    def rows(self, count=4):
        chosen = [unit for unit in self.units if unit.retrievable][:count]
        return [{
            "reason": "genuine_heading" if index % 2 == 0 else "columnar_row",
            "candidate_source_id": unit.source_id,
            "old_source_id": None,
            "page": unit.page,
            "old_section": None,
            "candidate_section": unit.section,
            "content_type": unit.content_type.value,
            "layout_kind": unit.layout_kind.value,
            "warnings": list(unit.warnings),
            "review_verdict": "PASS" if index else "ACCEPTABLE_WARNING",
            "review_note": "automated check",
            "review_status": UNREVIEWED,
            "inspect_path": "/etc/passwd",
        } for index, unit in enumerate(chosen)]

    def document(self, **overrides):
        value = {
            "schema_version": 1, "report_kind": "STD1E_OPERATOR_REVIEW",
            "standard_id": SID, "revision": REV,
            "corpus_manifest_sha256": self.manifest.corpus_manifest_sha256,
            "rows": self.rows(),
        }
        value.update(overrides)
        return value

    def write_review(self, **overrides):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.document(**overrides)))

    def load(self, **kwargs):
        return StandardReview.load(self.store, SID, REV, reviewer="operator",
                                   **kwargs)

    def drive(self, keys, review=None, positions=None):
        review = review or self.load()
        positions = positions if positions is not None else review.select()
        out = io.StringIO()
        recorded = run_review(review, positions,
                              stdin=io.StringIO("".join(f"{k}\n" for k in keys)),
                              stdout=out)
        return review, out.getvalue(), recorded


class OperatorReviewTests(_ReviewFixture):
    # -- loading and corpus safety ---------------------------------------

    def test_review_file_loads_with_its_rows(self):
        review = self.load()
        self.assertEqual(len(review.items()), 4)
        self.assertTrue(all(item.source_id.startswith("std-")
                            for item in review.items()))

    def test_review_file_must_match_the_active_corpus(self):
        self.write_review(corpus_manifest_sha256="b" * 64)
        with self.assertRaises(StandardReviewError) as raised:
            self.load()
        self.assertIn("different canonical corpus", str(raised.exception))

    def test_a_first_generation_file_is_accepted_only_for_its_own_corpus(self):
        document = self.document()
        document.pop("corpus_manifest_sha256")
        document["candidate_id"] = candidate_id_for(self.manifest.corpus_manifest_sha256)
        self.path.write_text(json.dumps(document))
        review = self.load()
        review.save()
        pinned = json.loads(self.path.read_text())
        self.assertEqual(pinned["corpus_manifest_sha256"],
                         self.manifest.corpus_manifest_sha256)
        document["candidate_id"] = "cand-" + "0" * 16
        self.path.write_text(json.dumps(document))
        with self.assertRaises(StandardReviewError):
            self.load()

    def test_a_corpus_replaced_mid_review_refuses_to_save(self):
        review = self.load()
        review.record(0, "PASS")
        review._document["corpus_manifest_sha256"] = "c" * 64
        with self.assertRaises(StandardReviewError) as raised:
            review.save()
        self.assertIn("changed during review", str(raised.exception))

    def test_malformed_review_files_are_rejected(self):
        for payload in ("[]", "{}", json.dumps({"report_kind": "OTHER"}),
                        json.dumps(self.document(rows=[])),
                        json.dumps(self.document(standard_id="OTHER")),
                        "not json"):
            self.path.write_text(payload)
            with self.assertRaises(StandardReviewError):
                self.load()

    def test_an_unknown_review_status_is_rejected(self):
        document = self.document()
        document["rows"][0]["review_status"] = "APPROVED-ISH"
        self.path.write_text(json.dumps(document))
        with self.assertRaises(StandardReviewError):
            self.load()

    # -- source resolution ------------------------------------------------

    def test_source_ids_resolve_through_the_store(self):
        review = self.load()
        for item in review.items():
            unit = self.store.resolve_source(SID, REV, item.source_id)
            self.assertEqual(unit.page, item.page)

    def test_injected_sources_and_paths_are_refused_or_ignored(self):
        for injected in ("../../etc/passwd", "std-" + "f" * 32, "", 17):
            document = self.document()
            document["rows"][0]["candidate_source_id"] = injected
            self.path.write_text(json.dumps(document))
            with self.assertRaises(StandardReviewError):
                self.load()
        # A path recorded in the file is never read, and never written back.
        self.write_review()
        review = self.load()
        review.record(0, "PASS")
        review.save()
        stored = json.loads(self.path.read_text())
        self.assertNotIn("inspect_path", stored["rows"][0])
        self.assertNotIn("/etc/passwd", self.path.read_text())

    def test_excerpts_are_bounded_and_context_is_small(self):
        review = self.load()
        excerpt = review.excerpt(0, limit=24)
        self.assertLessEqual(len(excerpt), 24)
        full = self.store.resolve_source(SID, REV, review.items()[0].source_id).text
        if len(" ".join(full.split())) > 24:
            self.assertTrue(excerpt.endswith("…"))
        self.assertLessEqual(len(review.context(0)), 2)
        for _, text in review.context(0, limit=20):
            self.assertLessEqual(len(text), 20)

    # -- verdicts ---------------------------------------------------------

    def test_every_human_verdict_persists(self):
        for offset, verdict in enumerate(HUMAN_VERDICTS):
            self.write_review()
            review = self.load()
            review.record(offset % 4, verdict)
            review.save()
            reloaded = self.load()
            item = reloaded.items()[offset % 4]
            self.assertEqual(item.human_verdict, verdict)
            self.assertEqual(item.review_status, verdict)
            self.assertEqual(item.reviewed_by, "operator")
            self.assertIsNotNone(item.reviewed_at)

    def test_the_automated_verdict_is_preserved_beside_the_human_one(self):
        review = self.load()
        before = review.items()[1].automated_verdict
        review.record(1, "FAIL")
        review.save()
        item = self.load().items()[1]
        self.assertEqual(item.automated_verdict, before)
        self.assertEqual(item.human_verdict, "FAIL")
        self.assertNotEqual(item.automated_verdict, item.human_verdict)
        stored = json.loads(self.path.read_text())["rows"][1]
        self.assertEqual(stored["automated_note"], "automated check")
        self.assertNotIn("review_verdict", stored)

    def test_an_unknown_verdict_is_refused(self):
        review = self.load()
        with self.assertRaises(StandardReviewError):
            review.record(0, "MAYBE")

    def test_skip_leaves_the_item_unreviewed(self):
        review, _, recorded = self.drive(["s", "s", "s", "s"])
        self.assertEqual(recorded, 0)
        self.assertEqual(self.load().summary()[UNREVIEWED], 4)

    def test_quit_saves_what_was_already_decided(self):
        review, _, recorded = self.drive(["p", "w", "q"])
        self.assertEqual(recorded, 2)
        summary = self.load().summary()
        self.assertEqual(summary["PASS"], 1)
        self.assertEqual(summary["ACCEPTABLE_WARNING"], 1)
        self.assertEqual(summary[UNREVIEWED], 2)

    def test_review_resumes_at_the_first_undecided_item(self):
        self.drive(["p", "p", "q"])
        review = self.load()
        positions = review.select()
        self.assertEqual(review.resume_at(positions), 2)
        review, output, _ = self.drive(["p", "p"], review=review, positions=positions)
        self.assertEqual(review.summary()[UNREVIEWED], 0)
        self.assertIn("3 / 4", output)
        self.assertNotIn("1 / 4", output)

    def test_an_earlier_item_can_be_revisited_and_changed(self):
        review, _, _ = self.drive(["p", "b", "f", "q"])
        items = review.items()
        self.assertEqual(items[0].human_verdict, "FAIL")
        review.clear(0)
        self.assertFalse(review.items()[0].reviewed)

    def test_filters_narrow_the_worklist_without_losing_rows(self):
        review = self.load()
        self.assertEqual(len(review.select(only_warning=True)), 1)
        self.assertEqual(len(review.select(stratum="columnar_row")), 2)
        self.assertEqual(len(review.select(only_unreviewed=True)), 4)
        review.record(0, "PASS")
        self.assertEqual(len(review.select(only_unreviewed=True)), 3)
        self.assertEqual(len(review.items()), 4)

    # -- persistence ------------------------------------------------------

    def test_the_review_file_is_replaced_atomically(self):
        review = self.load()
        review.record(0, "PASS")
        review.save()
        leftovers = [entry.name for entry in self.path.parent.iterdir()
                     if entry.name.startswith(".review-")]
        self.assertEqual(leftovers, [])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        stored = json.loads(self.path.read_text())
        self.assertEqual(stored["schema_version"], REVIEW_SCHEMA_VERSION)
        self.assertEqual(stored["human_review"]["reviewer"], "operator")

    def test_persistence_stores_verdicts_not_source_text(self):
        review = self.load()
        review.record(0, "PASS")
        review.save()
        raw = self.path.read_text()
        for unit in self.units:
            body = unit.text.strip()
            if len(body) > 24:
                self.assertNotIn(body, raw)

    # -- summary and approval gate ---------------------------------------

    def test_summary_counts_are_correct_and_completion_is_detected(self):
        review = self.load()
        self.assertEqual(review.summary(),
                         {"TOTAL": 4, UNREVIEWED: 4, "PASS": 0,
                          "ACCEPTABLE_WARNING": 0, "FAIL": 0, "NEEDS_FOLLOWUP": 0})
        self.assertFalse(review.is_complete())
        self.assertEqual(review.approval_state(), "REVIEW_INCOMPLETE")
        for position, verdict in enumerate(["PASS", "PASS", "ACCEPTABLE_WARNING",
                                            "PASS"]):
            review.record(position, verdict)
        self.assertEqual(review.summary()["PASS"], 3)
        self.assertTrue(review.is_complete())
        self.assertEqual(review.approval_state(), "ELIGIBLE_FOR_APPROVAL")

    def test_findings_block_the_approval_state(self):
        for blocking in ("FAIL", "NEEDS_FOLLOWUP"):
            self.write_review()
            review = self.load()
            for position in range(4):
                review.record(position, "PASS")
            review.record(2, blocking)
            self.assertTrue(review.is_complete())
            self.assertEqual(review.approval_state(), "APPROVAL_BLOCKED")
            self.assertEqual(review.blocking(), {blocking: 1})

    def test_completion_is_reported_without_approving_anything(self):
        review = self.load()
        for position in range(4):
            review.record(position, "PASS")
        _, output, _ = self.drive([], review=review, positions=review.select())
        self.assertIn("HUMAN REVIEW COMPLETE", output)
        self.assertIn("eligible for operator approval", output)
        self.assertEqual(
            self.store.load_manifest(SID, REV).human_validation_status,
            HumanValidationStatus.NOT_REVIEWED)


class ApprovalCommandTests(_ReviewFixture):
    def setUp(self):
        super().setUp()
        self.operator = StandardOperator(self.store)

    def approve(self):
        return handle_standard_command(
            f"/standard approve {SID} {REV} operator", self.operator)

    def decide(self, verdicts):
        review = self.load()
        for position, verdict in enumerate(verdicts):
            review.record(position, verdict)
        review.save()

    def test_approval_is_refused_while_rows_are_unreviewed(self):
        self.decide(["PASS", "PASS"])
        with self.assertRaises(StandardCommandError) as raised:
            self.approve()
        self.assertIn("not human-reviewed", str(raised.exception))
        self.assertEqual(self.store.load_manifest(SID, REV).human_validation_status,
                         HumanValidationStatus.NOT_REVIEWED)

    def test_approval_is_refused_when_findings_block_it(self):
        for blocking in ("FAIL", "NEEDS_FOLLOWUP"):
            self.write_review()
            self.decide(["PASS", "PASS", "PASS", blocking])
            with self.assertRaises(StandardCommandError) as raised:
                self.approve()
            self.assertIn("block approval", str(raised.exception))
            self.assertEqual(
                self.store.load_manifest(SID, REV).human_validation_status,
                HumanValidationStatus.NOT_REVIEWED)

    def test_approval_succeeds_with_only_passes_and_warnings(self):
        self.decide(["PASS", "ACCEPTABLE_WARNING", "PASS", "PASS"])
        rendered = self.approve()
        self.assertIn("NOT_REVIEWED -> APPROVED", rendered)
        manifest = self.store.load_manifest(SID, REV)
        self.assertEqual(manifest.human_validation_status,
                         HumanValidationStatus.APPROVED)
        # Approval records a verdict; it must not disturb corpus identity.
        self.assertEqual(manifest.corpus_manifest_sha256,
                         self.manifest.corpus_manifest_sha256)
        self.store.verify_corpus(SID, REV)
        audit = json.loads((self.store.revision_dir(SID, REV) / "evaluation"
                            / "approval.json").read_text())
        self.assertEqual(audit["reviewer"], "operator")
        self.assertEqual(audit["corpus_manifest_sha256"],
                         self.manifest.corpus_manifest_sha256)
        self.assertEqual(audit["evidence"]["rows_reviewed"], 4)

    def test_approval_is_refused_for_a_review_of_another_corpus(self):
        self.decide(["PASS", "PASS", "PASS", "PASS"])
        document = json.loads(self.path.read_text())
        document["corpus_manifest_sha256"] = "d" * 64
        self.path.write_text(json.dumps(document))
        with self.assertRaises(StandardCommandError) as raised:
            self.approve()
        self.assertIn("approval refused", str(raised.exception))
        self.assertEqual(self.store.load_manifest(SID, REV).human_validation_status,
                         HumanValidationStatus.NOT_REVIEWED)


class ReviewIsOperatorOnlyTests(_ReviewFixture):
    def test_review_and_approval_are_never_model_tools(self):
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        names = {spec.name for spec in registry.list_specs()}
        self.assertEqual(names, set(STANDARD_TOOL_NAMES))
        for forbidden in ("standard.review", "standard.approve",
                          "standard.validate", "standard.build_structure"):
            self.assertNotIn(forbidden, names)
        visible = ToolExposurePolicy().select(
            registry, AgentRole.MAIN, objective="review the standard",
            standard_bound=True)
        self.assertEqual(set(visible.names) & {"standard.review",
                                               "standard.approve"}, set())
        import standard_review
        self.assertFalse(hasattr(standard_review, "register"))
        self.assertNotIn("ToolSpec", dir(standard_review))

    def test_the_cli_reports_a_summary_without_changing_anything(self):
        out = io.StringIO()
        code = main(["--standard", SID, "--revision", REV,
                     "--root", str(self.store.root), "--summary"], stdout=out)
        self.assertEqual(code, 0)
        self.assertIn("UNREVIEWED", out.getvalue())
        self.assertEqual(self.load().summary()[UNREVIEWED], 4)

    def test_the_cli_refuses_an_unknown_stratum_and_a_missing_review(self):
        out = io.StringIO()
        self.assertEqual(main(["--standard", SID, "--revision", REV,
                               "--root", str(self.store.root),
                               "--stratum", "nope"], stdout=out), 2)
        self.assertIn("unknown stratum", out.getvalue())
        self.path.unlink()
        out = io.StringIO()
        self.assertEqual(main(["--standard", SID, "--revision", REV,
                               "--root", str(self.store.root)], stdout=out), 2)
        self.assertIn("review unavailable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
