"""The battery counts every unit put in front of the model, not every tool
call the model made.

The policy opens a bound turn by reading the standard before the first round,
and it can inject one retrieval when a turn is about to conclude unevidenced.
Both are retrieval. Neither was counted, so a case answered from the opening
alone scored as having retrieved nothing while quoting, verbatim, the exact
provision it was meant to find -- six of eighteen cases in one run.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                       / "eval" / "modeluse"))

import harness


class EveryRetrievalCounts(unittest.TestCase):

    def trace(self):
        return {"sources_returned": []}

    def record(self, *, payload="{}", metadata=None):
        return {"result": payload, "metadata": metadata or {}}

    def test_a_tool_that_names_its_sources_is_counted(self):
        trace = self.trace()

        harness._record_sources(trace, self.record(
            metadata={"standard_source_ids": ("std-a", "std-b")}))

        self.assertEqual(trace["sources_returned"], ["std-a", "std-b"])

    def test_sources_inside_the_payload_are_counted(self):
        trace = self.trace()

        harness._record_sources(trace, self.record(
            payload='{"results": [{"source_id": "std-c"}]}'))

        self.assertEqual(trace["sources_returned"], ["std-c"])

    def test_the_same_unit_is_counted_once(self):
        """Two calls reaching the same provision is one piece of evidence."""
        trace = self.trace()
        twice = self.record(metadata={"standard_source_ids": ("std-a",)})

        harness._record_sources(trace, twice)
        harness._record_sources(trace, twice)

        self.assertEqual(trace["sources_returned"], ["std-a"])

    def test_an_unreadable_result_loses_nothing_already_counted(self):
        """A refusal comes back as text, not JSON. It must not discard the
        metadata the same call carried."""
        trace = self.trace()

        harness._record_sources(trace, self.record(
            payload="StandardStructureError: no structures",
            metadata={"standard_source_ids": ("std-d",)}))

        self.assertEqual(trace["sources_returned"], ["std-d"])

    def test_every_call_site_records(self):
        """The loop is not the only place a tool runs: the opening and the
        answer-boundary injection are retrieval the policy drove."""
        source = pathlib.Path(harness.__file__).read_text()

        self.assertEqual(source.count("_record_sources(trace, record)"),
                         1 + 3)      # the definition, then all three call sites


if __name__ == "__main__":
    unittest.main()
