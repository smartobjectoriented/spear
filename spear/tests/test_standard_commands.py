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
            with mock.patch.dict(os.environ, {"SPEAR_STANDARD_EMBED_MODEL": "m"}):
                self.assertIn("Vector: NOT BUILT", handle_standard_command(
                    "/standard status", operator))
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
