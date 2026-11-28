"""Standards bound beside the active one.

A question can need two documents: an architecture manual and the interrupt
controller specification it defers to, say. The active standard keeps the
verdict; a companion is searched, read and cited like it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from evidence import conformance_guard
from harness.tool_registry import ToolRegistry
from harness.tool_router import ToolExecutionContext, ToolRouter
from runtime.result_store import ResultStore
from runtime.tracing import NullTraceRecorder, TraceEmitter
from standard.standard_commands import (
    StandardCommandError, StandardOperator, handle_standard_command,
)
from standard.standard_ingest import ingest_pdf
from standard.standard_retrieval import rebuild_lexical_index
from standard.standard_store import StandardStore
from standard.standard_tools import StandardToolService
from tests.standard_fixture import synthetic_pdf_bytes

COMPANION_PAGES = (
    ((760, "Synthetic Controller Specification"),
     (738, "Revision C-1 - TEST_FIXTURE / SYNTHETIC"),
     (700, "4 Doorbells"),
     (670, "4.1 Doorbell Register"),
     (640, "Software shall write the doorbell register to raise a doorbell."),
     (600, "doorbell means a message sent from one core to another.")),
)


class _Store(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = StandardStore(root / "standards")

        for standard_id, revision, pdf_bytes in (
                ("TEST-STD", "TEST-1", synthetic_pdf_bytes()),
                ("CTRL-STD", "C-1", synthetic_pdf_bytes(pages=COMPANION_PAGES))):
            pdf = root / f"{standard_id}.pdf"
            pdf.write_bytes(pdf_bytes)
            ingest_pdf(self.store, pdf, standard_id=standard_id,
                       revision=revision, source_origin="TEST_FIXTURE")
            rebuild_lexical_index(self.store, standard_id, revision)

        self.operator = StandardOperator(self.store)
        self.root = root

    def tearDown(self):
        self.temp.cleanup()

    def command(self, text):
        return handle_standard_command(text, self.operator)


class Binding(_Store):
    def test_a_companion_needs_an_active_standard(self):
        with self.assertRaises(StandardCommandError):
            self.command("/standard add CTRL-STD C-1")

    def test_add_keeps_the_active_standard_and_records_the_companion(self):
        self.command("/standard use TEST-STD TEST-1")
        self.command("/standard add CTRL-STD C-1")

        self.assertEqual(self.operator.active_binding().standard_id, "TEST-STD")
        self.assertEqual([item.standard_id for item in self.operator.companion_bindings()],
                         ["CTRL-STD"])
        session = self.operator.session_binding()
        self.assertEqual(session["standard_id"], "TEST-STD")
        self.assertEqual([item["standard_id"] for item in session["companions"]],
                         ["CTRL-STD"])
        self.assertIn("Companion: CTRL-STD C-1", self.command("/standard status"))
        self.assertIn("<- companion", self.command("/standard list"))

    def test_the_active_standard_cannot_be_its_own_companion(self):
        self.command("/standard use TEST-STD TEST-1")

        with self.assertRaises(StandardCommandError):
            self.command("/standard add TEST-STD TEST-1")

    def test_rebinding_keeps_companions_except_the_new_active_one(self):
        self.command("/standard use TEST-STD TEST-1")
        self.command("/standard add CTRL-STD C-1")
        self.command("/standard use TEST-STD TEST-1")
        self.assertEqual(len(self.operator.companion_bindings()), 1)

        self.command("/standard use CTRL-STD C-1")
        self.assertEqual(self.operator.companion_bindings(), [])

    def test_drop_and_unbind_clear_companions(self):
        self.command("/standard use TEST-STD TEST-1")
        self.command("/standard add CTRL-STD C-1")
        self.command("/standard drop CTRL-STD")
        self.assertEqual(self.operator.companion_bindings(), [])

        with self.assertRaises(StandardCommandError):
            self.command("/standard drop CTRL-STD")

        self.command("/standard add CTRL-STD C-1")
        self.command("/standard unbind")
        self.assertIsNone(self.operator.session_binding())
        self.assertEqual(self.operator.companion_bindings(), [])


class Tools(_Store):
    def setUp(self):
        super().setUp()
        self.command("/standard use TEST-STD TEST-1")
        self.command("/standard add CTRL-STD C-1")
        registry = ToolRegistry()
        StandardToolService(self.store).register(registry)
        self.router = ToolRouter(registry)
        self.context = ToolExecutionContext(
            "task_12345678", TraceEmitter(NullTraceRecorder()), {},
            result_store=ResultStore(self.root / "results"),
            metadata={"standard_binding": self.operator.session_binding()})

    def execute(self, name, arguments):
        return self.router.execute(self.context, "call_1", name, arguments)

    def search(self, **arguments):
        result = self.execute("standard.search", arguments)
        self.assertTrue(result.success, result.text)

        return json.loads(result.text)

    def test_search_covers_every_bound_standard(self):
        payload = self.search(query="reserved value doorbell register")

        self.assertEqual({item["standard_id"] for item in payload["results"]},
                         {"TEST-STD", "CTRL-STD"})
        self.assertEqual(payload["standards_searched"], ["TEST-STD TEST-1", "CTRL-STD C-1"])

    def test_standard_id_narrows_the_search_to_one(self):
        payload = self.search(query="doorbell register", standard_id="CTRL-STD")

        self.assertTrue(payload["results"])
        self.assertEqual({item["standard_id"] for item in payload["results"]}, {"CTRL-STD"})

    def test_fetch_and_cite_find_the_standard_that_holds_the_source(self):
        found = self.search(query="doorbell register", standard_id="CTRL-STD")
        source_id = found["results"][0]["source_id"]

        fetched = self.execute("standard.fetch", {"source_id": source_id})
        self.assertTrue(fetched.success, fetched.text)
        self.assertEqual(json.loads(fetched.text)["unit"]["standard_id"], "CTRL-STD")
        cited = self.execute("standard.cite", {"source_id": source_id})
        self.assertTrue(cited.success, cited.text)

    def test_a_revision_alone_names_the_standard_that_carries_it(self):
        payload = self.search(query="doorbell register", revision="C-1")

        self.assertTrue(payload["results"])
        self.assertEqual({item["standard_id"] for item in payload["results"]}, {"CTRL-STD"})

    def test_a_revision_no_bound_standard_carries_is_denied(self):
        denied = self.execute("standard.search", {"query": "doorbell", "revision": "X-9"})

        self.assertFalse(denied.success)

    def test_a_standard_that_is_not_bound_is_denied(self):
        denied = self.execute("standard.search", {"query": "doorbell",
                                                  "standard_id": "NOT-BOUND"})

        self.assertFalse(denied.success)
        self.assertIn("active StandardBinding", denied.text)


class Names(unittest.TestCase):
    def test_a_companion_is_a_standard_the_guards_recognise(self):
        binding = {"standard_id": "ARM-DDI0487", "revision": "M.c",
                   "companions": [{"standard_id": "ARM-IHI0069", "revision": "H.b"}]}

        names = conformance_guard.standard_names(binding)

        self.assertIn("ddi0487", names)
        self.assertIn("ihi0069", names)


if __name__ == "__main__":
    unittest.main()
