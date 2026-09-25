import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from standard_commands import (
    StandardCommandError, StandardOperator, handle_standard_command,
)
from standard_store import StandardStore
from tests.standard_fixture import synthetic_pdf_bytes


class StandardCommandTests(unittest.TestCase):
    def test_operator_ingest_list_use_status_rebuild_and_unbind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "licensed fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            operator = StandardOperator(StandardStore(root / "standards"))
            output = handle_standard_command(
                f'/standard ingest "{pdf}" --id TEST-STD --revision TEST-1', operator)
            self.assertIn("Raw PDF retained: no", output)
            self.assertFalse((operator.store.revision_dir(
                "TEST-STD", "TEST-1") / "source" / "original.pdf").exists())
            self.assertIn("TEST-STD TEST-1", handle_standard_command(
                "/standard list", operator))
            self.assertIn("Bound TEST-STD", handle_standard_command(
                "/standard use TEST-STD TEST-1", operator))
            with mock.patch.dict(os.environ, {"SPEAR_STANDARD_EMBED_MODEL": ""}):
                status = handle_standard_command("/standard status", operator)
            self.assertIn("Validation: NOT_REVIEWED", status)
            self.assertIn("Vector: NOT CONFIGURED", status)
            self.assertNotIn("FileNotFoundError", status)
            with mock.patch.dict(os.environ, {"SPEAR_STANDARD_EMBED_MODEL": "m",
                                              "SPEAR_STANDARD_EMBED_REMOTE": "gpu-host"}):
                status = handle_standard_command("/standard status", operator)
            # Licensed by default: rebuild would skip it, so it must not say to.
            self.assertIn("Vector: NOT BUILT (licensed", status)
            self.assertIn("--allow-offload would send its text to gpu-host", status)
            self.assertIn("Cross references: READY", status)
            self.assertIn("Retrieval fingerprint:", status)
            self.assertIn("Rebuilt lexical index", handle_standard_command(
                "/standard rebuild", operator))
            self.assertIn("cleared", handle_standard_command(
                "/standard unbind", operator))
            self.assertEqual(handle_standard_command("/standard status", operator),
                             "Active standard: none")

    def test_list_reports_the_parameters_each_document_was_ingested_with(self):
        """Seeing what an unbound document holds must not require binding it.

        `status` reports all of this for the BOUND document only, and the
        binding is shared by every session on the machine -- so "let me look
        at the other one" was a change of state for everybody.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            operator = StandardOperator(StandardStore(root / "standards"))
            handle_standard_command(
                f'/standard ingest "{pdf}" --id PUB-STD --revision R1 '
                f'--origin PUBLIC', operator)
            handle_standard_command(
                f'/standard ingest "{pdf}" --id LIC-STD --revision R2 '
                f'--origin LICENSED_STANDARD --retain-pdf', operator)

            listing = handle_standard_command("/standard list", operator)

            self.assertIn("PUB-STD R1", listing)
            self.assertIn("LIC-STD R2", listing)
            self.assertIn("PUBLIC", listing)
            self.assertIn("LICENSED_STANDARD", listing)
            self.assertIn("extraction only", listing)
            self.assertIn("the document itself is retained", listing)
            self.assertIn("validation", listing)
            self.assertIn("units", listing)

    def test_list_marks_which_one_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            operator = StandardOperator(StandardStore(root / "standards"))
            handle_standard_command(
                f'/standard ingest "{pdf}" --id A-STD --revision R1', operator)

            self.assertNotIn("bound", handle_standard_command(
                "/standard list", operator))

            handle_standard_command("/standard use A-STD R1", operator)

            self.assertIn("<- bound", handle_standard_command(
                "/standard list", operator))

    def test_an_empty_store_still_says_so(self):
        with tempfile.TemporaryDirectory() as directory:
            operator = StandardOperator(
                StandardStore(Path(directory) / "standards"))

            self.assertEqual(handle_standard_command("/standard list", operator),
                             "No standards ingested.")

    def test_strict_operator_grammar_rejects_injection_and_missing_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            operator = StandardOperator(StandardStore(Path(directory) / "standards"))
            for command in ("/standard", "/standard ingest x.pdf --id X",
                            "/standard use X", "/standard status extra",
                            "/standard ingest x.pdf --id X --revision 1 --shell=oops"):
                with self.assertRaises(StandardCommandError, msg=command):
                    handle_standard_command(command, operator)


class RetrievalFollowsTheDocumentTests(unittest.TestCase):
    """Retrieval settings were machine-wide, measured on one document; a
    second document bound on the same machine inherited them."""

    CLEAN = {"SPEAR_STANDARD_RETRIEVAL_MODE": "",
             "SPEAR_STANDARD_EVIDENCE_COMPLETION": ""}

    def operator_with_two(self, directory):
        root = Path(directory); pdf = root / "fixture.pdf"
        pdf.write_bytes(synthetic_pdf_bytes())
        operator = StandardOperator(StandardStore(root / "standards"))
        for name in ("A-STD", "B-STD"):
            handle_standard_command(
                f'/standard ingest "{pdf}" --id {name} --revision R1', operator)
        return operator

    def test_switching_documents_switches_how_they_are_searched(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, self.CLEAN):
            operator = self.operator_with_two(directory)
            handle_standard_command(
                "/standard retrieval A-STD R1 lexical --completion 2", operator)
            handle_standard_command(
                "/standard retrieval B-STD R1 hybrid", operator)

            handle_standard_command("/standard use A-STD R1", operator)
            self.assertIn("Retrieval: lexical, completion 2 (set for this document)",
                          handle_standard_command("/standard status", operator))

            handle_standard_command("/standard use B-STD R1", operator)
            self.assertIn("Retrieval: hybrid, completion 0 (set for this document)",
                          handle_standard_command("/standard status", operator))

    def test_changing_one_setting_keeps_the_other(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, self.CLEAN):
            operator = self.operator_with_two(directory)
            handle_standard_command("/standard use A-STD R1", operator)
            handle_standard_command(
                "/standard retrieval lexical --completion 2", operator)

            self.assertIn("vector, completion 2", handle_standard_command(
                "/standard retrieval vector", operator))
            self.assertIn("vector, completion 0", handle_standard_command(
                "/standard retrieval --completion 0", operator))

    def test_an_undeclared_document_gets_the_default(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, self.CLEAN):
            operator = self.operator_with_two(directory)
            self.assertIn("hybrid, completion 0 (default)", handle_standard_command(
                "/standard retrieval A-STD R1", operator))

    def test_the_session_variables_still_win_and_say_so(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, self.CLEAN):
            operator = self.operator_with_two(directory)
            handle_standard_command(
                "/standard retrieval A-STD R1 lexical --completion 2", operator)

            with mock.patch.dict(os.environ, {
                    "SPEAR_STANDARD_RETRIEVAL_MODE": "hybrid",
                    "SPEAR_STANDARD_EVIDENCE_COMPLETION": "0"}):
                summary = handle_standard_command(
                    "/standard retrieval A-STD R1", operator)

            self.assertIn("hybrid, completion 0", summary)
            self.assertIn("overridden by SPEAR_STANDARD_RETRIEVAL_MODE, "
                          "SPEAR_STANDARD_EVIDENCE_COMPLETION", summary)

    def test_the_retriever_reads_the_documents_completion_budget(self):
        from standard_retrieval import StandardRetrieval

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, self.CLEAN):
            operator = self.operator_with_two(directory)
            handle_standard_command(
                "/standard retrieval A-STD R1 lexical --completion 2", operator)
            retrieval = StandardRetrieval(operator.store)

            self.assertEqual(retrieval._completion_budget("A-STD", "R1"), 2)
            self.assertEqual(retrieval._completion_budget("B-STD", "R1"), 0)

    def test_bad_settings_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            operator = self.operator_with_two(directory)
            for command in ("/standard retrieval A-STD R1 semantic",
                            "/standard retrieval A-STD R1 lexical --completion -1",
                            "/standard retrieval A-STD R1 lexical --completion x",
                            "/standard retrieval NO-STD R1 lexical"):
                with self.assertRaises(StandardCommandError, msg=command):
                    handle_standard_command(command, operator)

            path = operator.store.revision_dir("A-STD", "R1") / "retrieval.json"
            path.write_text('{"mode": "semantic", "evidence_completion": 0}')
            self.assertIn("UNAVAILABLE", handle_standard_command(
                "/standard retrieval A-STD R1", operator))


class NoCorpusEmbeddingOnTheLocalCpuTests(unittest.TestCase):

    def test_a_local_cpu_embedder_is_skipped_not_run(self):
        import standard_commands

        class Local:                        # no `target`: runs on this machine
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pdf = root / "fixture.pdf"
            pdf.write_bytes(synthetic_pdf_bytes())
            operator = StandardOperator(StandardStore(root / "standards"))
            handle_standard_command(
                f'/standard ingest "{pdf}" --id A-STD --revision R1', operator)

            with mock.patch.object(standard_commands, "configured_embedder",
                                   return_value=Local()), \
                    mock.patch.object(standard_commands,
                                      "rebuild_vector_index") as rebuilt, \
                    mock.patch.dict(os.environ):
                os.environ.pop("SPEAR_STANDARD_EMBED_DEVICE", None)
                output = handle_standard_command(
                    "/standard rebuild A-STD R1", operator)

            rebuilt.assert_not_called()
            self.assertIn("not built", output)
            self.assertIn("local CPU", output)

            with mock.patch.object(standard_commands, "configured_embedder",
                                   return_value=Local()), \
                    mock.patch.object(standard_commands,
                                      "rebuild_vector_index") as rebuilt, \
                    mock.patch.dict(os.environ,
                                    {"SPEAR_STANDARD_EMBED_DEVICE": "cuda"}):
                handle_standard_command("/standard rebuild A-STD R1", operator)

            rebuilt.assert_called_once()


class CompletionTests(unittest.TestCase):
    """Tab after `/standard`: identifiers are long, exact, and written down
    only in the store."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name); pdf = root / "fixture.pdf"
        pdf.write_bytes(synthetic_pdf_bytes())
        self.store = StandardStore(root / "standards")
        operator = StandardOperator(self.store)
        for name, revision in (("ACME-STD-1", "R1"), ("ACME-STD-1", "R2"),
                               ("OTHER-9.2", "2017-R2024")):
            handle_standard_command(
                f'/standard ingest "{pdf}" --id {name} --revision {revision}',
                operator)

    def tearDown(self):
        self.temp.cleanup()

    def complete(self, words, text):
        from standard_commands import complete_standard
        return complete_standard(words, text, self.store)

    def test_actions(self):
        self.assertEqual(self.complete([], "u"), ["unbind", "use"])

    def test_identifiers_then_that_documents_revisions(self):
        self.assertEqual(self.complete(["use"], ""), ["ACME-STD-1", "OTHER-9.2"])
        self.assertEqual(self.complete(["use"], "OT"), ["OTHER-9.2"])
        self.assertEqual(self.complete(["use", "ACME-STD-1"], ""), ["R1", "R2"])
        self.assertEqual(self.complete(["use", "ACME-STD-1", "R1"], ""), [])

    def test_retrieval_takes_a_document_or_a_mode(self):
        self.assertIn("lexical", self.complete(["retrieval"], ""))
        self.assertIn("OTHER-9.2", self.complete(["retrieval"], ""))
        self.assertEqual(self.complete(["retrieval", "lexical"], ""),
                         ["--completion"])
        self.assertEqual(self.complete(["retrieval", "OTHER-9.2", "2017-R2024"], "h"),
                         ["hybrid"])

    def test_ingest_options_and_origins(self):
        self.assertEqual(self.complete(["ingest", "x.pdf"], "--o"), ["--origin"])
        self.assertEqual(self.complete(["ingest", "x.pdf", "--origin"], ""),
                         ["LICENSED_STANDARD", "PUBLIC"])
        self.assertEqual(self.complete(["ingest", "x.pdf", "--id"], ""), [])
