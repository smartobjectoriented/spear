"""STD2A: the private structure store and its operator review. Synthetic only."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_layout import extract_layout
from standard_store import StandardStore
from standard_structure import (
    StandardStructureError, extract_structures, structure_fingerprint,
)
from standard_structure_store import (
    MANIFEST_FILE, STRUCTURE_FILE, StandardStructureStore, build_manifest,
)
from standard_structure_review import (
    HUMAN_VERDICTS, UNREVIEWED, StandardStructureReview, run_review, sample_tables,
)
from tests.standard_geometry_fixture import geometry_pdf_bytes

SID, REV = "GEO", "R1"


class _Stored(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.pdf = root / "geometry.pdf"
        self.pdf.write_bytes(geometry_pdf_bytes())
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
        self.structures_store = StandardStructureStore(self.store)
        self.saved = self.structures_store.save(self.build(), self.structures)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **overrides):
        values = dict(
            standard_id=SID, revision=REV,
            corpus_fingerprint=self.manifest.corpus_manifest_sha256,
            layout_fingerprint=self.artifact["layout_fingerprint"],
            extractor_version=self.manifest.extractor_version,
            created_at="2026-01-01T00:00:00+00:00")
        values.update(overrides)
        return build_manifest(self.structures, **values)


class StructureStoreTests(_Stored):
    def test_structures_are_stored_privately_under_the_revision(self):
        directory = self.structures_store.directory(SID, REV)
        self.assertEqual(directory.parent, self.store.revision_dir(SID, REV))
        self.assertTrue((directory / MANIFEST_FILE).is_file())
        self.assertTrue((directory / STRUCTURE_FILE).is_file())
        self.assertEqual(self.structures_store.state(SID, REV), "READY")

    def test_the_manifest_pins_what_the_structures_were_built_from(self):
        manifest, payload = self.structures_store.load(SID, REV)
        self.assertEqual(manifest.corpus_fingerprint,
                         self.manifest.corpus_manifest_sha256)
        self.assertEqual(manifest.layout_fingerprint,
                         self.artifact["layout_fingerprint"])
        self.assertEqual(manifest.structure_fingerprint,
                         structure_fingerprint(self.structures))
        self.assertEqual(manifest.table_count, len(payload["tables"]))

    def test_a_manifest_that_does_not_describe_its_structures_is_refused(self):
        from dataclasses import replace
        wrong = replace(self.build(), structure_fingerprint="c" * 64)
        with self.assertRaises(StandardStructureError):
            self.structures_store.save(wrong, self.structures)

    def test_structures_built_against_another_corpus_are_refused(self):
        from dataclasses import replace
        foreign = replace(self.build(), corpus_fingerprint="d" * 64)
        with self.assertRaises(StandardStructureError):
            self.structures_store.save(foreign, self.structures)

    def test_a_stale_corpus_fails_closed_on_load(self):
        manifest_path = self.saved / MANIFEST_FILE
        raw = json.loads(manifest_path.read_text())
        raw["corpus_fingerprint"] = "e" * 64
        manifest_path.write_text(json.dumps(raw))
        with self.assertRaises(StandardStructureError) as raised:
            self.structures_store.load(SID, REV)
        self.assertIn("canonical corpus has changed", str(raised.exception))
        self.assertIn("UNAVAILABLE", self.structures_store.state(SID, REV))

    def test_a_stale_layout_fails_closed_on_load(self):
        manifest_path = self.saved / MANIFEST_FILE
        raw = json.loads(manifest_path.read_text())
        raw["layout_fingerprint"] = "f" * 64
        manifest_path.write_text(json.dumps(raw))
        with self.assertRaises(StandardStructureError) as raised:
            self.structures_store.load(SID, REV)
        self.assertIn("layout artifact has changed", str(raised.exception))

    def test_a_corrupt_manifest_or_structure_file_is_refused(self):
        (self.saved / MANIFEST_FILE).write_text("{not json")
        with self.assertRaises(StandardStructureError):
            self.structures_store.load_manifest(SID, REV)
        self.structures_store.save(self.build(), self.structures)
        (self.saved / STRUCTURE_FILE).write_text(json.dumps({"nope": 1}))
        with self.assertRaises(StandardStructureError):
            self.structures_store.load(SID, REV)

    def test_traversal_and_symlink_escape_are_refused(self):
        from standard_store import StandardStoreError
        for standard_id in ("../../etc", "..", "a/b"):
            with self.assertRaises((StandardStoreError, StandardStructureError)):
                self.structures_store.directory(standard_id, REV)
        directory = self.structures_store.directory(SID, REV)
        target = Path(self.temp.name) / "outside"
        target.mkdir()
        escaped = directory / "escape"
        escaped.symlink_to(target)
        with self.assertRaises(StandardStructureError):
            StandardStructureStore._atomic_write(escaped, b"{}")

    def test_the_structure_directory_is_private(self):
        directory = self.structures_store.directory(SID, REV)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual((directory / MANIFEST_FILE).stat().st_mode & 0o777, 0o600)


class StructureReviewTests(_Stored):
    def review(self, **kwargs):
        return StandardStructureReview(self.store, SID, REV, reviewer="operator",
                                       **kwargs)

    def drive(self, keys, review=None, ids=None):
        review = review or self.review()
        ids = ids if ids is not None else review.selection()
        out = io.StringIO()
        recorded = run_review(review, ids, stdout=out,
                              stdin=io.StringIO("".join(f"{k}\n" for k in keys)))
        return review, out.getvalue(), recorded

    def test_every_candidate_starts_unreviewed(self):
        review = self.review()
        summary = review.summary()
        self.assertEqual(summary[UNREVIEWED], summary["TOTAL"])
        self.assertGreater(summary["TOTAL"], 0)

    def test_each_verdict_persists_and_the_review_resumes(self):
        review, _, recorded = self.drive(["p", "w", "q"])
        self.assertEqual(recorded, 2)
        again = self.review()
        summary = again.summary()
        self.assertEqual(summary["PASS"], 1)
        self.assertEqual(summary["ACCEPTABLE_WARNING"], 1)
        self.assertEqual(again.resume_at(again.selection()), 2)

    def test_fail_and_follow_up_persist_too(self):
        review = self.review()
        ids = review.selection()
        review.record(ids[0], "FAIL")
        review.record(ids[1], "NEEDS_FOLLOWUP")
        review.save()
        summary = self.review().summary()
        self.assertEqual(summary["FAIL"], 1)
        self.assertEqual(summary["NEEDS_FOLLOWUP"], 1)

    def test_a_review_of_different_structures_is_refused(self):
        self.drive(["p", "q"])
        path = self.structures_store.directory(SID, REV) / "review.json"
        raw = json.loads(path.read_text())
        raw["structure_fingerprint"] = "a" * 64
        path.write_text(json.dumps(raw))
        with self.assertRaises(StandardStructureError):
            self.review()

    def test_a_grid_and_an_svg_render_without_leaving_the_store(self):
        review = self.review()
        table_id = review.selection()[0]
        self.assertIn("|", review.grid_text(table_id))
        target = review.svg(table_id)
        self.assertTrue(target.is_file())
        self.assertTrue(str(target).startswith(str(self.store.revision_dir(SID, REV))))
        self.assertTrue(target.read_text().startswith("<svg"))

    def test_the_sample_spreads_across_strata(self):
        _, payload = self.structures_store.load(SID, REV)
        tables = list(payload["tables"])
        for table in tables:
            table["_bitfield"] = False
        chosen = sample_tables(tables)
        self.assertTrue(chosen)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertLessEqual(len(chosen), len(tables))

    def test_review_does_not_alter_the_structures_or_the_corpus(self):
        before = structure_fingerprint(self.structures)
        self.drive(["p", "p", "q"])
        manifest, _ = self.structures_store.load(SID, REV)
        self.assertEqual(manifest.structure_fingerprint, before)
        self.assertEqual(self.store.verify_corpus(SID, REV).corpus_manifest_sha256,
                         self.manifest.corpus_manifest_sha256)
