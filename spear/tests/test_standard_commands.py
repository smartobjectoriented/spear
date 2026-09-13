import tempfile
import unittest
from pathlib import Path

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
            status = handle_standard_command("/standard status", operator)
            self.assertIn("Validation: NOT_REVIEWED", status)
            self.assertIn("Vector: UNAVAILABLE", status)
            self.assertIn("Cross references: READY", status)
            self.assertIn("Retrieval fingerprint:", status)
            self.assertIn("Rebuilt lexical index", handle_standard_command(
                "/standard rebuild", operator))
            self.assertIn("cleared", handle_standard_command(
                "/standard unbind", operator))
            self.assertEqual(handle_standard_command("/standard status", operator),
                             "Active standard: none")

    def test_strict_operator_grammar_rejects_injection_and_missing_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            operator = StandardOperator(StandardStore(Path(directory) / "standards"))
            for command in ("/standard", "/standard ingest x.pdf --id X",
                            "/standard use X", "/standard status extra",
                            "/standard ingest x.pdf --id X --revision 1 --shell=oops"):
                with self.assertRaises(StandardCommandError, msg=command):
                    handle_standard_command(command, operator)
