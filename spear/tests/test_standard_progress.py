"""An ingestion long enough to need watching says where it is.

A 17 000-page manual is some 330 000 units, and every phase after extraction
runs over all of them; the command used to print nothing until it was done.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import standard_vector_index
from standard_commands import StandardOperator, handle_standard_command
from standard_ingest import ingest_pdf
from standard_progress import STEPS, TerminalProgress, report
from standard_store import StandardStore
from standard_vector_index import rebuild_vector_index
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_standard_hybrid_retrieval import (
    HYBRID_PAGES, FixtureSemanticEmbedder,
)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, label, done=None, total=None):
        self.calls.append((label, done, total))

    def labels(self):
        return list(dict.fromkeys(label for label, _, _ in self.calls))


class ReportTests(unittest.TestCase):

    def test_no_callback_is_free(self):
        report(None, "anything", 1, 10)         # must not raise

    def test_a_long_loop_is_throttled_but_ends_at_its_total(self):
        seen = Recorder()
        total = 330_000

        for done in range(1, total + 1):
            report(seen, "writing corpus", done, total)

        self.assertLessEqual(len(seen.calls), STEPS + 1)
        self.assertEqual(seen.calls[-1], ("writing corpus", total, total))

    def test_a_phase_without_a_count_is_announced(self):
        seen = Recorder()
        report(seen, "extracting text")
        self.assertEqual(seen.calls, [("extracting text", None, None)])


class TerminalProgressTests(unittest.TestCase):

    def test_one_line_per_phase_with_its_percentage(self):
        out = io.StringIO()
        progress = TerminalProgress(out)
        progress("extracting text")
        progress("structuring pages", 50, 200)
        progress("structuring pages", 200, 200)
        progress.finish()
        text = out.getvalue()

        self.assertIn("[1] extracting text", text)
        self.assertIn("[2] structuring pages   25%  (50/200)", text)
        self.assertIn("100%  (200/200)", text)
        self.assertEqual(text.count("\n"), 2)       # one per phase, closed

    def test_the_estimate_extrapolates_the_phase_rate(self):
        out = io.StringIO()
        progress = TerminalProgress(out)

        with mock.patch("standard_progress.time.monotonic", side_effect=[0, 10]):
            progress("writing corpus", 0, 100)
            progress("writing corpus", 25, 100)

        self.assertIn("~30s left", out.getvalue())

    def test_finish_resets_the_phase_count_for_the_next_command(self):
        out = io.StringIO()
        progress = TerminalProgress(out)
        progress("a"); progress.finish(); progress("b")
        self.assertIn("[1] b", out.getvalue())


class IngestReportsEveryPhaseTests(unittest.TestCase):

    def test_ingest_walks_through_the_phases_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            seen = Recorder()
            operator = StandardOperator(StandardStore(root / "standards"),
                                        progress=seen)
            handle_standard_command(
                f'/standard ingest "{pdf}" --id A-STD --revision R1', operator)

        labels = seen.labels()
        expected = ["extracting text", "structuring pages", "writing corpus",
                    "verifying corpus", "lexical index", "cross references"]
        self.assertEqual([label for label in labels if label in expected], expected)

        for label in ("structuring pages", "writing corpus", "lexical index",
                      "cross references"):
            last = [call for call in seen.calls if call[0] == label][-1]
            self.assertEqual(last[1], last[2], f"{label} did not reach 100%")


class ChunkedEmbeddingTests(unittest.TestCase):

    def test_chunking_does_not_change_the_index(self):
        """Progress must not cost identity: the same vectors, the same
        fingerprint, whether the texts went in one call or many."""

        def build(chunk):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); pdf = root / "hybrid.pdf"
                pdf.write_bytes(synthetic_pdf_bytes(HYBRID_PAGES))
                store = StandardStore(root / "standards")
                ingest_pdf(store, pdf, standard_id="SYNTH-STD", revision="R1",
                           source_origin="TEST_FIXTURE")
                seen = Recorder()

                with mock.patch.object(standard_vector_index, "EMBED_CHUNK", chunk):
                    manifest = rebuild_vector_index(
                        store, "SYNTH-STD", "R1", FixtureSemanticEmbedder(),
                        created_at="fixed", progress=seen)

                return manifest.vector_index_fingerprint, seen

        whole, _ = build(4096)
        chunked, seen = build(2)
        self.assertEqual(whole, chunked)

        embedding = [call for call in seen.calls if call[0].startswith("embedding")]
        self.assertGreater(len(embedding), 2)
        self.assertEqual(embedding[-1][1], embedding[-1][2])


if __name__ == "__main__":
    unittest.main()
