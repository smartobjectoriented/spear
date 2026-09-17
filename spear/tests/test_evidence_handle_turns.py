"""A handle lives as long as the turn that retrieved it.

Exercised through the real registry against a synthetic ingested revision, so
what is measured is what a session does, not what a helper does in isolation.
"""

import json
import tempfile
import unittest
from pathlib import Path

from result_store import ResultStore
from standard_ingest import ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore
from standard_tools import STALE_EVIDENCE_HANDLE, StandardToolService
from tests.standard_fixture import synthetic_pdf_bytes
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext
from tracing import NullTraceRecorder, TraceEmitter

SID, REV = "TEST-STD", "TEST-1"


class HandleLifetime(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "fixture.pdf"
        pdf.write_bytes(synthetic_pdf_bytes())
        self.store = StandardStore(root / "standards")
        ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                   source_origin="TEST_FIXTURE")
        rebuild_lexical_index(self.store, SID, REV)
        self.registry = ToolRegistry()
        StandardToolService(self.store).register(self.registry)
        self.results = ResultStore(root / "results")
        self.binding = self.store.binding(SID, REV).to_dict()

    def tearDown(self):
        self.temp.cleanup()

    def turn(self, cache):
        """A context sharing one turn's cache, as the runtime builds them."""
        return ToolExecutionContext(
            "task_handles01", TraceEmitter(NullTraceRecorder()), cache,
            result_store=self.results,
            metadata={"standard_binding": self.binding})

    def call(self, name, cache, arguments):
        handler = self.registry.handler(name)
        result = handler(self.turn(cache), arguments)
        return json.loads(result.text)

    def a_handle(self, cache, query="reserved"):
        found = self.call("standard.search", cache, {"query": query})
        self.assertTrue(found.get("results"), "the fixture should retrieve")
        return found["results"][0]["source_id"]

    def test_a_handle_is_usable_for_the_rest_of_its_own_turn(self):
        # The runtime builds a fresh ToolExecutionContext per call and threads
        # ONE cache through the turn, so this is what several rounds of the
        # same turn look like.
        turn = {}
        handle = self.a_handle(turn)

        self.assertNotIn("error", self.call("standard.fetch", turn,
                                            {"source_id": handle}))
        self.assertNotIn("error", self.call("standard.cite", turn,
                                            {"source_id": handle}))
        self.assertNotIn("error", self.call("standard.fetch", turn,
                                            {"source_id": handle}))

    def test_the_next_turn_starts_owning_nothing(self):
        first = {}
        handle = self.a_handle(first)

        for tool in ("standard.fetch", "standard.cite"):
            with self.subTest(tool=tool):
                refused = self.call(tool, {}, {"source_id": handle})
                self.assertEqual(STALE_EVIDENCE_HANDLE, refused["error"])

    def test_the_next_turn_may_retrieve_the_same_evidence_again(self):
        handle = self.a_handle({})

        second = {}
        self.assertEqual(STALE_EVIDENCE_HANDLE,
                         self.call("standard.fetch", second,
                                   {"source_id": handle})["error"])

        # Retrieval is deterministic, so the same question returns the same
        # id. It is usable now because it was retrieved now.
        again = self.a_handle(second)
        self.assertEqual(handle, again)
        self.assertNotIn("error", self.call("standard.fetch", second,
                                            {"source_id": handle}))

    def test_what_a_fetch_shows_becomes_fetchable_in_that_turn(self):
        # A fetched unit carries its neighbours, and a handle the model can
        # see is a handle it may follow.
        turn = {}
        unit = self.call("standard.fetch", turn,
                         {"source_id": self.a_handle(turn)})
        nearby = {h for h in json.dumps(unit).split('"')
                  if h.startswith("std-") and len(h) == 36}

        for handle in sorted(nearby):
            with self.subTest(handle=handle[:12]):
                self.assertNotIn("error", self.call("standard.fetch", turn,
                                                    {"source_id": handle}))

    def test_a_handle_from_one_turn_does_not_leak_through_a_third(self):
        first = {}
        handle = self.a_handle(first)
        self.assertNotIn("error", self.call("standard.fetch", first,
                                            {"source_id": handle}))

        for _ in range(2):
            self.assertEqual(STALE_EVIDENCE_HANDLE,
                             self.call("standard.fetch", {},
                                       {"source_id": handle})["error"])


if __name__ == "__main__":
    unittest.main()
