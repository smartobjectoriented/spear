"""What every normative tool says when it cannot answer the question asked.

Two different things get called "no". A well-formed question with no answer is
evidence: nothing here matches. A question asked with the wrong kind of
identifier is not evidence at all -- it is a request that cannot be made that
way. A caller that cannot tell them apart either invents an answer or keeps
asking, and both cost it the budget it needed for the real question.

So each tool has to name which of the two it is, and say what to do next.
Nothing here quotes an evaluated standard: the fixture is synthetic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from result_store import ResultStore
from standard_ingest import ingest_pdf
from standard_retrieval import rebuild_lexical_index
from standard_store import StandardStore
from standard_tools import (
    STALE_EVIDENCE_HANDLE,
    INVALID_QUERY, INVALID_SOURCE_ID, SOURCE_NOT_FOUND, StandardToolService,
)
from tests.standard_fixture import synthetic_pdf_bytes
from tool_registry import ToolRegistry
from tool_router import ToolExecutionContext
from tracing import NullTraceRecorder, TraceEmitter

SID, REV = "TEST-STD", "TEST-1"
# The shape the corpus hands out, and one of each kind that does not belong here.
ABSENT_SOURCE = "std-" + "0" * 32
STRUCTURE_IDS = ("bit-" + "a" * 16, "bfd-" + "b" * 16, "fld-" + "c" * 16,
                 "pkg-" + "d" * 16, "vgr-" + "e" * 16)


class _Bound(unittest.TestCase):
    """One ingested revision, reachable through the real registry."""

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
        self.context = ToolExecutionContext(
            "task_refusal01", TraceEmitter(NullTraceRecorder()), {},
            result_store=ResultStore(root / "results"),
            metadata={"standard_binding": self.store.binding(SID, REV).to_dict()})

    def tearDown(self):
        self.temp.cleanup()

    def call(self, name, arguments):
        return self.registry.handler(name)(self.context, arguments)

    def refusal(self, name, arguments):
        result = self.call(name, arguments)
        payload = json.loads(result.text)
        self.assertIn("error", payload, f"{name} did not refuse: {payload}")
        return result, payload

    def a_real_source(self):
        found = json.loads(self.call("standard.search", {"query": "reserved"}).text)
        return found["results"][0]["source_id"]


class Fetch(_Bound):
    def test_a_real_source_id_still_fetches(self):
        source = self.a_real_source()
        payload = json.loads(self.call("standard.fetch",
                                       {"source_id": source}).text)
        self.assertEqual(payload["unit"]["source_id"], source)

    def test_a_structure_identifier_is_not_a_source_id(self):
        for asked in STRUCTURE_IDS:
            with self.subTest(asked=asked):
                _, payload = self.refusal("standard.fetch", {"source_id": asked})
                self.assertEqual(payload["error"], INVALID_SOURCE_ID)

    def test_a_structure_identifier_is_sent_to_the_tool_that_owns_it(self):
        _, payload = self.refusal("standard.fetch",
                                  {"source_id": STRUCTURE_IDS[0]})
        self.assertIn("standard.get_structure", json.dumps(payload))

    def test_a_human_readable_name_is_not_a_source_id(self):
        _, payload = self.refusal("standard.fetch", {"source_id": "Gain Field"})
        self.assertEqual(payload["error"], INVALID_SOURCE_ID)

    def test_a_source_this_turn_never_retrieved_is_refused_before_lookup(self):
        """Whether it exists is not the question being answered.

        A fetch is reading back something this turn retrieved. An id that came
        from somewhere else -- an older conversation, a memory, a guess -- is
        refused on that ground alone, and the corpus is not consulted to say
        whether it happens to name a real unit. Replying "no such unit" to one
        id and "stale" to another would answer, for anyone who asked twice,
        which ids exist.
        """
        _, payload = self.refusal("standard.fetch", {"source_id": ABSENT_SOURCE})
        self.assertEqual(payload["error"], STALE_EVIDENCE_HANDLE)
        self.assertIn("standard.search", json.dumps(payload["recovery"]))

    def test_the_recovery_names_the_tool_that_hands_out_source_ids(self):
        for asked in (STRUCTURE_IDS[0], ABSENT_SOURCE):
            with self.subTest(asked=asked):
                _, payload = self.refusal("standard.fetch", {"source_id": asked})
                self.assertIn("standard.search", json.dumps(payload["recovery"]))
                self.assertIn("std-", json.dumps(payload["recovery"]))

    def test_a_missing_source_id_is_refused_by_name(self):
        _, payload = self.refusal("standard.fetch", {})
        self.assertEqual(payload["error"], INVALID_SOURCE_ID)


class Cite(_Bound):
    def test_a_real_source_id_still_cites(self):
        source = self.a_real_source()
        payload = json.loads(self.call("standard.cite",
                                       {"source_id": source}).text)
        self.assertEqual(payload["source_id"], source)

    def test_a_structure_identifier_is_refused_actionably(self):
        _, payload = self.refusal("standard.cite",
                                  {"source_id": STRUCTURE_IDS[1]})
        self.assertEqual(payload["error"], INVALID_SOURCE_ID)
        self.assertIn("standard.search", json.dumps(payload["recovery"]))

    def test_an_unissued_source_stays_distinct_from_a_wrong_kind(self):
        """Two different mistakes keep two different answers.

        A well formed handle this turn never retrieved is refused as stale,
        without the corpus being asked whether it exists. An identifier from
        another namespace does not get that far: it is not the shape of a
        handle at all, and is refused on its form.
        """
        _, unissued = self.refusal("standard.cite", {"source_id": ABSENT_SOURCE})
        _, wrong = self.refusal("standard.cite", {"source_id": STRUCTURE_IDS[1]})
        self.assertEqual(unissued["error"], STALE_EVIDENCE_HANDLE)
        self.assertNotEqual(unissued["error"], wrong["error"])

    def test_no_citation_is_invented_for_a_refused_identifier(self):
        result, payload = self.refusal("standard.cite",
                                       {"source_id": STRUCTURE_IDS[1]})
        self.assertNotIn("rendered", payload)
        self.assertEqual(dict(result.metadata).get("standard_source_ids", ()), ())


class Search(_Bound):
    def test_a_query_with_no_match_is_a_result_not_a_refusal(self):
        payload = json.loads(self.call(
            "standard.search",
            {"query": "zzqqxx no such wording anywhere"}).text)
        self.assertEqual(payload["result_count"], 0)
        self.assertNotIn("error", payload)

    def test_an_empty_query_is_refused_by_name(self):
        _, payload = self.refusal("standard.search", {"query": "   "})
        self.assertEqual(payload["error"], INVALID_QUERY)

    def test_an_unusable_limit_is_refused_by_name(self):
        _, payload = self.refusal("standard.search",
                                  {"query": "reserved", "limit": "many"})
        self.assertEqual(payload["error"], INVALID_QUERY)

    def test_a_binding_argument_that_is_not_bound_is_still_denied(self):
        # An authorisation boundary, not a usage hint: it raises, and it says
        # what the contract is so the caller stops sending the argument.
        with self.assertRaises(PermissionError) as caught:
            self.call("standard.search", {"query": "reserved",
                                          "revision": "OTHER"})
        message = str(caught.exception).lower()
        self.assertIn("active standardbinding", message)
        self.assertIn("omit", message)


class EveryRefusal(_Bound):
    """Properties that must hold whichever tool refused."""

    def cases(self):
        return (("standard.fetch", {"source_id": STRUCTURE_IDS[0]}),
                ("standard.fetch", {"source_id": ABSENT_SOURCE}),
                ("standard.cite", {"source_id": "not-an-identifier"}),
                ("standard.search", {"query": ""}))
    # standard.get_structure carries the same properties over a fixture that
    # has an approved structure to refuse around: test_structure_identifier_
    # recovery.py.

    def test_a_refusal_never_resolves_to_something_close(self):
        source = self.a_real_source()
        near = source[:-1] + ("0" if source[-1] != "0" else "1")
        for tool in ("standard.fetch", "standard.cite"):
            with self.subTest(tool=tool):
                _, payload = self.refusal(tool, {"source_id": near})

                # Both dereference a handle, so both refuse one this turn
                # never retrieved, without looking it up. What neither does,
                # which is what this is really about, is hand back the
                # neighbouring id that does exist.
                self.assertEqual(payload["error"], STALE_EVIDENCE_HANDLE)
                self.assertNotIn(source, json.dumps(payload))

    def test_every_refusal_is_read_only(self):
        for name, arguments in self.cases():
            with self.subTest(tool=name, arguments=arguments):
                result = self.call(name, arguments)
                self.assertFalse(result.mutation)
                self.assertEqual(result.affected_paths, ())

    def test_every_refusal_names_its_reason_in_metadata(self):
        for name, arguments in self.cases():
            with self.subTest(tool=name, arguments=arguments):
                result = self.call(name, arguments)
                reason = dict(result.metadata).get("standard_refusal")
                self.assertEqual(reason, json.loads(result.text)["error"])

    def test_every_refusal_carries_a_recovery(self):
        for name, arguments in self.cases():
            with self.subTest(tool=name, arguments=arguments):
                payload = json.loads(self.call(name, arguments).text)
                self.assertTrue(payload.get("recovery"),
                                f"{name} refused without saying what to do")

    def test_no_refusal_disturbs_the_corpus(self):
        before = self.store.verify_corpus(SID, REV)
        for name, arguments in self.cases():
            self.call(name, arguments)
        self.assertEqual(self.store.verify_corpus(SID, REV), before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
