"""STD1E: candidate corpora, generational storage and operator promotion.

Every corpus here is synthetic. No licensed normative text appears in this file.
"""

import json
import tempfile
import unittest
from pathlib import Path

from model_backend import ConversationMessage, TextBlock
from result_store import ResultStore
from session_store import (
    FileSessionStore, SessionCompatibilityError, SessionConfiguration,
    SessionSnapshot, new_session_id, restore_session,
)
from standard_commands import StandardOperator, handle_standard_command, StandardCommandError
from standard_crossrefs import rebuild_cross_reference_index
from standard_ingest import (
    EXTRACTOR_VERSION, build_manifest, extract_pdf_pages, ingest_candidate,
)
from standard_layout import StandardLayoutError, extract_layout, validate_layout
from standard_retrieval import rebuild_lexical_index
from standard_schema import StandardContentType
from standard_store import (
    INDEX_STATE_READY, INDEX_STATE_REBUILD_REQUIRED, StandardStore,
    StandardStoreError, candidate_id_for, generation_id_for,
)
from standard_tools import STANDARD_TOOL_NAMES, StandardToolService
from tests.standard_extraction_fixture import (
    extraction_pdf_bytes, legacy_canonical_units,
)
from tool_registry import ToolRegistry
from working_state import WorkingState


SID, REV = "SYNTH-STD", "R1"


class CandidatePromotionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / "structure.pdf"
        self.pdf.write_bytes(extraction_pdf_bytes())
        self.store = StandardStore(self.root / "standards")
        # The active corpus is what the previous extractor produced.
        pages, warnings = extract_pdf_pages(self.pdf)
        import hashlib
        sha = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        units = legacy_canonical_units(pages, standard_id=SID, revision=REV,
                                       pdf_sha256=sha)
        manifest = build_manifest(
            units, standard_id=SID, revision=REV, pdf_sha256=sha,
            filename=self.pdf.name, page_count=len(pages), warnings=warnings,
            source_origin="TEST_FIXTURE", retain_pdf=False,
            ingestion_timestamp="2026-01-01T00:00:00+00:00")
        object.__setattr__(manifest, "extractor_version", "poppler-layout-v1")
        object.__setattr__(manifest, "corpus_schema_version", 1)
        self.legacy = self.store.save_ingestion(manifest, units)
        rebuild_lexical_index(self.store, SID, REV, created_at="first")
        rebuild_cross_reference_index(self.store, SID, REV, created_at="first")
        self.operator = StandardOperator(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def make_candidate(self, *, layout=False):
        return ingest_candidate(self.store, self.pdf, standard_id=SID, revision=REV,
                                with_layout=layout,
                                ingestion_timestamp="2026-01-02T00:00:00+00:00")

    # -- isolation --------------------------------------------------------
    def test_candidate_ingestion_leaves_the_active_corpus_untouched(self):
        candidate_id, diagnostics = self.make_candidate()
        active = self.store.verify_corpus(SID, REV)
        self.assertEqual(active.corpus_manifest_sha256,
                         self.legacy.corpus_manifest_sha256)
        self.assertEqual(active.extractor_version, "poppler-layout-v1")
        self.assertNotEqual(diagnostics["candidate_corpus_sha256"],
                            active.corpus_manifest_sha256)
        self.assertEqual(self.store.list_candidates(SID, REV), (candidate_id,))

    def test_candidate_never_becomes_active_on_its_own(self):
        self.operator.use(SID, REV)
        before = self.operator.active_binding()
        self.make_candidate()
        after = self.operator.active_binding()
        self.assertEqual(before.corpus_manifest_sha256, after.corpus_manifest_sha256)
        self.assertEqual(self.store.index_state(SID, REV), INDEX_STATE_READY)

    def test_identical_re_extraction_is_deterministic(self):
        first, _ = self.make_candidate()
        second, _ = self.make_candidate()
        self.assertEqual(first, second)
        self.assertEqual(self.store.list_candidates(SID, REV), (first,))
        manifest, _ = self.store.load_candidate(SID, REV, first)
        self.assertEqual(candidate_id_for(manifest.corpus_manifest_sha256), first)

    def test_candidate_repairs_the_known_extraction_defects(self):
        _, diagnostics = self.make_candidate()
        old = diagnostics["comparison"]["old"]
        new = diagnostics["comparison"]["candidate"]
        self.assertLess(new["invalid_section_ids"], old["invalid_section_ids"])
        self.assertLess(new["units_on_invalid_sections"],
                        old["units_on_invalid_sections"])
        self.assertLessEqual(new["backwards_transitions"], old["backwards_transitions"])
        self.assertEqual(new["duplicate_section_anchors"], 0)
        self.assertGreater(new["page_furniture_units"], 0)
        self.assertEqual(old["page_furniture_units"], 0)
        self.assertGreater(new["columnar_units"], old["columnar_units"])
        self.assertLess(new["retrievable_units"], new["unit_count"])

    def test_source_id_changes_are_reported_not_hidden(self):
        _, diagnostics = self.make_candidate()
        ids = diagnostics["comparison"]["source_ids"]
        self.assertEqual(set(ids), {"retained", "removed", "added"})
        self.assertGreater(ids["removed"] + ids["added"], 0)

    # -- promotion integrity ---------------------------------------------
    def test_promotion_refuses_an_unknown_or_tampered_candidate(self):
        candidate_id, _ = self.make_candidate()
        with self.assertRaises(StandardStoreError):
            self.store.promote_candidate(SID, REV, "cand-" + "0" * 16)
        with self.assertRaises(StandardStoreError):
            self.store.promote_candidate(SID, REV, "../../etc")
        corpus = (self.store.revision_dir(SID, REV) / "candidates" / candidate_id
                  / "corpus")
        victim = sorted(corpus.iterdir())[0]
        payload = json.loads(victim.read_text())
        payload["page"] = payload["page"] + 1
        victim.write_text(json.dumps(payload))
        with self.assertRaises(StandardStoreError):
            self.store.promote_candidate(SID, REV, candidate_id)
        self.assertEqual(self.store.verify_corpus(SID, REV).corpus_manifest_sha256,
                         self.legacy.corpus_manifest_sha256)

    def test_promoting_one_document_leaves_another_bound(self):
        """The binding is shared by every session on the machine; promoting
        a second document dropped it, whichever document it named."""
        from tests.standard_fixture import synthetic_pdf_bytes

        other = self.root / "other.pdf"
        other.write_bytes(synthetic_pdf_bytes())
        handle_standard_command(
            f'/standard ingest "{other}" --id OTHER-STD --revision R9', self.operator)
        handle_standard_command("/standard use OTHER-STD R9", self.operator)
        candidate_id, _ = self.make_candidate()

        output = handle_standard_command(
            f"/standard promote-candidate {SID} {REV} {candidate_id}", self.operator)

        self.assertIn("Binding to OTHER-STD R9 left as it was", output)
        self.assertEqual(self.operator.active_binding().standard_id, "OTHER-STD")

    def test_promoting_the_bound_document_unbinds_it(self):
        handle_standard_command(f"/standard use {SID} {REV}", self.operator)
        candidate_id, _ = self.make_candidate()

        output = handle_standard_command(
            f"/standard promote-candidate {SID} {REV} {candidate_id}", self.operator)

        self.assertIn("Standard binding cleared", output)
        self.assertIsNone(self.operator.active_binding())

    def test_promotion_keeps_what_the_document_is(self):
        """A PUBLIC standard re-extracted and promoted stayed PUBLIC only by
        luck: the candidate took the licensed default and promotion adopted
        it, which refused embedding and would have left it out of a public
        image."""
        self.store.reclassify_origin(SID, REV, "PUBLIC")
        candidate_id, _ = self.make_candidate()
        self.assertEqual(self.store.load_candidate(SID, REV, candidate_id)[0]
                         .source_origin, "PUBLIC")
        self.store.promote_candidate(SID, REV, candidate_id)
        self.assertEqual(self.store.load_manifest(SID, REV).source_origin, "PUBLIC")

    def test_promotion_refuses_a_candidate_identical_to_the_active_corpus(self):
        candidate_id, _ = self.make_candidate()
        self.store.promote_candidate(SID, REV, candidate_id)
        again, _ = self.make_candidate()
        with self.assertRaises(StandardStoreError):
            self.store.promote_candidate(SID, REV, again)

    def test_promotion_retains_the_previous_generation(self):
        candidate_id, _ = self.make_candidate()
        promoted = self.store.promote_candidate(SID, REV, candidate_id)
        generation = generation_id_for(self.legacy.corpus_manifest_sha256)
        self.assertEqual(self.store.list_generations(SID, REV), (generation,))
        archive = self.store.revision_dir(SID, REV) / "generations" / generation
        kept = json.loads((archive / "manifest.json").read_text())
        self.assertEqual(kept["corpus_manifest_sha256"],
                         self.legacy.corpus_manifest_sha256)
        self.assertEqual(len(list((archive / "corpus").iterdir())),
                         self.legacy.canonical_unit_count)
        self.assertEqual(promoted.extractor_version, EXTRACTOR_VERSION)
        self.assertNotEqual(promoted.corpus_manifest_sha256,
                            self.legacy.corpus_manifest_sha256)

    def test_promotion_moves_the_candidate_and_leaves_no_stale_copy(self):
        candidate_id, _ = self.make_candidate(layout=True)
        self.store.promote_candidate(SID, REV, candidate_id)
        self.assertEqual(self.store.list_candidates(SID, REV), ())
        base = self.store.revision_dir(SID, REV)
        self.assertTrue((base / "diagnostics.json").is_file())
        self.assertTrue((base / "layout.json").is_file())

    def test_indexes_are_marked_rebuild_required_after_promotion(self):
        candidate_id, _ = self.make_candidate()
        self.assertEqual(self.store.index_state(SID, REV), INDEX_STATE_READY)
        promoted = self.store.promote_candidate(SID, REV, candidate_id)
        self.assertEqual(self.store.index_state(SID, REV),
                         INDEX_STATE_REBUILD_REQUIRED)
        with self.assertRaises((FileNotFoundError, StandardStoreError)):
            self.store.load_index(SID, REV)
        with self.assertRaises((FileNotFoundError, StandardStoreError)):
            self.store.binding(SID, REV)
        self.assertEqual(self.store.list_candidates(SID, REV), ())
        rebuilt = rebuild_lexical_index(self.store, SID, REV, created_at="second")
        self.assertEqual(rebuilt.source_corpus_sha256,
                         promoted.corpus_manifest_sha256)
        # Freshness is derived, so rebuilding clears the state by itself.
        self.assertEqual(self.store.index_state(SID, REV), INDEX_STATE_READY)

    def test_a_partial_promotion_fails_closed_and_stays_recoverable(self):
        candidate_id, _ = self.make_candidate()
        base = self.store.revision_dir(SID, REV)
        generation = generation_id_for(self.legacy.corpus_manifest_sha256)
        archive = base / "generations" / generation
        archive.mkdir(parents=True)
        (archive / "manifest.json").write_bytes(
            json.dumps(self.legacy.to_dict()).encode())
        import os
        os.rename(base / "corpus", archive / "corpus")          # interrupted here
        with self.assertRaises(StandardStoreError):
            self.store.verify_corpus(SID, REV)
        self.assertEqual(len(list((archive / "corpus").iterdir())),
                         self.legacy.canonical_unit_count)
        self.assertIn(candidate_id, self.store.list_candidates(SID, REV))

    # -- review history ---------------------------------------------------
    def test_promotion_keeps_the_review_with_the_generation_it_judged(self):
        """A review belongs to the corpus it judged, and survives that corpus."""
        from standard_review import (
            StandardReview, review_path, write_review_sample,
        )
        from standard_store import generation_id_for

        write_review_sample(self.store, SID, REV)
        review = StandardReview.load(self.store, SID, REV, reviewer="operator")
        review.record(0, "PASS")
        review.save()
        candidate_id, _ = self.make_candidate()
        self.store.promote_candidate(SID, REV, candidate_id)

        generation = (self.store.revision_dir(SID, REV) / "generations"
                      / generation_id_for(self.legacy.corpus_manifest_sha256))
        kept = json.loads((generation / "operator-review.json").read_text())
        self.assertEqual(kept["corpus_manifest_sha256"],
                         self.legacy.corpus_manifest_sha256)
        self.assertEqual(kept["rows"][0]["human_verdict"], "PASS")
        # The promoted corpus starts with no review of its own, rather than
        # inheriting one written against different units.
        self.assertFalse(review_path(self.store, SID, REV).exists())

    def test_a_review_of_a_superseded_generation_cannot_be_loaded(self):
        from standard_review import (
            StandardReview, StandardReviewError, review_path,
            write_review_sample,
        )

        write_review_sample(self.store, SID, REV)
        stale = json.loads(review_path(self.store, SID, REV).read_text())
        candidate_id, _ = self.make_candidate()
        self.store.promote_candidate(SID, REV, candidate_id)
        target = review_path(self.store, SID, REV)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(stale))
        with self.assertRaises(StandardReviewError):
            StandardReview.load(self.store, SID, REV, reviewer="operator")

    # -- session safety ---------------------------------------------------
    def test_an_old_session_fails_closed_after_promotion(self):
        self.operator.use(SID, REV)
        binding = self.operator.active_binding()
        sessions = FileSessionStore(self.root / "sessions")
        configuration = SessionConfiguration("/workspace", "project", "safe")
        session_id = new_session_id()
        sessions.save_snapshot(SessionSnapshot(
            session_id, "task_std1e_0001", 0, configuration,
            WorkingState.start("task_std1e_0001", "normative question"),
            (ConversationMessage("user", (TextBlock("question"),)),),
            standard_binding=binding.to_dict(),
            standard_source_ids_used=("std-" + "a" * 32,),
            standard_retrieval_cache_fingerprint=binding.index_fingerprint))
        candidate_id, _ = self.make_candidate()
        self.store.promote_candidate(SID, REV, candidate_id)
        rebuild_lexical_index(self.store, SID, REV, created_at="second")
        with self.assertRaises(SessionCompatibilityError) as raised:
            restore_session(sessions, session_id, configuration,
                            standard_store=self.store)
        self.assertIn("orpus", str(raised.exception))
        with self.assertRaises(StandardCommandError):
            self.operator.active_binding()

    # -- operator surface -------------------------------------------------
    def test_operator_commands_report_and_gate_promotion(self):
        self.operator.use(SID, REV)
        rendered = handle_standard_command(
            f"/standard ingest {self.pdf} --id {SID} --revision {REV} --candidate",
            self.operator)
        self.assertIn("Extracted candidate cand-", rendered)
        self.assertIn("the active corpus is unchanged", rendered)
        listed = handle_standard_command(
            f"/standard candidates {SID} {REV}", self.operator)
        self.assertIn("READY_FOR_OPERATOR_REVIEW", listed)
        self.assertIn("invalid section ids", listed)
        self.assertIn("/standard promote-candidate", listed)
        status = handle_standard_command("/standard status", self.operator)
        self.assertIn("Candidates: cand-", status)
        self.assertIn("Retrieval indexes: READY", status)
        candidate_id = self.store.list_candidates(SID, REV)[0]
        promoted = handle_standard_command(
            f"/standard promote-candidate {SID} {REV} {candidate_id}", self.operator)
        self.assertIn("Previous generation retained", promoted)
        self.assertIn("Retrieval indexes: REBUILD_REQUIRED", promoted)
        self.assertIsNone(self.operator.active_binding())

    def test_status_survives_a_binding_whose_corpus_was_replaced(self):
        self.operator.use(SID, REV)
        candidate_id, _ = self.make_candidate()
        # Promote behind the operator's back, leaving the persisted binding stale.
        self.store.promote_candidate(SID, REV, candidate_id)
        status = handle_standard_command("/standard status", self.operator)
        self.assertIn("Active standard: unavailable", status)
        self.assertIn("REBUILD_REQUIRED", status)
        self.assertIn("/standard rebuild", status)


class LayoutArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.temp.name) / "structure.pdf"
        self.pdf.write_bytes(extraction_pdf_bytes())
        self.artifact = extract_layout(self.pdf, pdf_sha256="d" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_layout_retains_word_geometry_without_semantics(self):
        validate_layout(self.artifact, pdf_sha256="d" * 64)
        self.assertEqual(self.artifact["page_count"], 7)
        self.assertGreater(self.artifact["word_count"], 0)
        words = [word for page in self.artifact["pages"] for block in page["blocks"]
                 for line in block["lines"] for word in line["words"]]
        self.assertTrue(all(len(word["bbox"]) == 4 for word in words))
        self.assertNotIn("cells", json.dumps(self.artifact)[:2000])

    def test_layout_rejects_a_foreign_pdf_or_a_broken_box(self):
        with self.assertRaises(StandardLayoutError):
            validate_layout(self.artifact, pdf_sha256="e" * 64)
        broken = json.loads(json.dumps(self.artifact))
        broken["pages"][0]["blocks"][0]["lines"][0]["words"][0]["bbox"] = [1, 2, 3]
        with self.assertRaises(StandardLayoutError):
            validate_layout(broken, pdf_sha256="d" * 64)
        inverted = json.loads(json.dumps(self.artifact))
        inverted["pages"][0]["blocks"][0]["bbox"] = [9.0, 9.0, 1.0, 1.0]
        with self.assertRaises(StandardLayoutError):
            validate_layout(inverted, pdf_sha256="d" * 64)
        with self.assertRaises(StandardLayoutError):
            validate_layout({"artifact_version": 99}, pdf_sha256="d" * 64)

    def test_layout_extraction_refuses_a_path_outside_a_regular_file(self):
        link = Path(self.temp.name) / "link.pdf"
        link.symlink_to(self.pdf)
        with self.assertRaises(StandardLayoutError):
            extract_layout(link, pdf_sha256="d" * 64)


class ToolSurfaceTests(unittest.TestCase):
    def test_std1e_registers_no_structure_or_validation_tool(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pdf = root / "structure.pdf"
            pdf.write_bytes(extraction_pdf_bytes())
            store = StandardStore(root / "standards")
            from standard_ingest import ingest_pdf
            ingest_pdf(store, pdf, standard_id=SID, revision=REV,
                       source_origin="TEST_FIXTURE")
            registry = ToolRegistry()
            StandardToolService(store).register(registry)
            names = {spec.name for spec in registry.list_specs()}
        self.assertEqual(names, set(STANDARD_TOOL_NAMES))
        self.assertIn("standard.get_structure", names)
        self.assertNotIn("standard.validate", names)
        self.assertNotIn("standard.promote_candidate", names)
        self.assertNotIn("standard.ingest", names)


if __name__ == "__main__":
    unittest.main()


class LayoutPageBoxTests(unittest.TestCase):
    """Word geometry has to sit inside the page it is reported on."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.temp.name) / "layout.pdf"
        self.pdf.write_bytes(extraction_pdf_bytes())
        self.artifact = extract_layout(self.pdf, pdf_sha256="d" * 64)

    def tearDown(self):
        self.temp.cleanup()

    def test_every_page_box_is_the_page_not_a_leftover_word(self):
        for page in self.artifact["pages"]:
            self.assertEqual(page["size"][:2], [0.0, 0.0], page["page"])
            self.assertGreater(page["size"][2], 0)
            self.assertGreater(page["size"][3], 0)

    def test_every_word_sits_inside_its_page(self):
        for page in self.artifact["pages"]:
            _, _, width, height = page["size"]
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        x0, y0, x1, y1 = word["bbox"]
                        self.assertGreaterEqual(x0, 0)
                        self.assertGreaterEqual(y0, 0)
                        self.assertLessEqual(x1, width + 1)
                        self.assertLessEqual(y1, height + 1)

    def test_the_word_count_matches_the_words_actually_stored(self):
        counted = sum(len(line["words"]) for page in self.artifact["pages"]
                      for block in page["blocks"] for line in block["lines"])
        self.assertEqual(self.artifact["word_count"], counted)
        self.assertGreater(counted, 0)

    def test_a_page_without_a_usable_box_is_rejected(self):
        broken = json.loads(json.dumps(self.artifact))
        broken["pages"][0]["size"] = [12.0, 5.0, 612.0, 792.0]
        with self.assertRaises(StandardLayoutError):
            validate_layout(broken, pdf_sha256="d" * 64)
