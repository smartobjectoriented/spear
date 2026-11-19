"""The corpus index served over MCP: the protocol, not the retrieval."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import corpus_mcp  # noqa: E402


class FakeIndex:
    default = "alpha"

    def __init__(self):
        self.asked = []

    def describe(self):
        return "alpha: 3 chunks [default]"

    def search(self, query, corpus=None, top_k=corpus_mcp.TOP_K):
        self.asked.append((query, corpus, top_k))

        if corpus == "missing":
            raise corpus_mcp.ToolError("no indexed corpus named 'missing'")

        if query == "boom":
            raise RuntimeError("index unreadable")

        return f"# File: src/a.c\n\n{query}"


def exchange(*messages, index=None):
    """The responses to `messages`, sent as the wire carries them."""
    out = io.StringIO()
    lines = [m if isinstance(m, str) else json.dumps(m) for m in messages]
    corpus_mcp.serve(index or FakeIndex(), lines, out)

    return [json.loads(line) for line in out.getvalue().splitlines()]


def call(name, arguments, ident=7):
    return {"jsonrpc": "2.0", "id": ident, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}


class HandshakeTests(unittest.TestCase):
    def test_a_known_version_is_echoed_and_an_unknown_one_gets_ours(self):
        for asked, answered in (("2025-03-26", "2025-03-26"),
                                ("2099-01-01", corpus_mcp.PROTOCOL_VERSIONS[0])):
            reply, = exchange({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {"protocolVersion": asked}})

            self.assertEqual(reply["result"]["protocolVersion"], answered)
            self.assertEqual(reply["result"]["capabilities"], {"tools": {}})

    def test_a_notification_is_never_answered(self):
        self.assertEqual(exchange(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}}), [])

    def test_ping_and_an_unknown_method(self):
        ping, unknown = exchange(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "resources/list"})

        self.assertEqual(ping["result"], {})
        self.assertEqual(unknown["error"]["code"], corpus_mcp.METHOD_NOT_FOUND)

    def test_a_line_that_is_not_json_does_not_end_the_session(self):
        broken, ping = exchange("{not json", {"jsonrpc": "2.0", "id": 2,
                                              "method": "ping"})

        self.assertEqual(broken["error"]["code"], corpus_mcp.PARSE_ERROR)
        self.assertEqual(ping["id"], 2)


class ToolTests(unittest.TestCase):
    def test_every_tool_declares_an_object_schema(self):
        reply, = exchange({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = {tool["name"]: tool for tool in reply["result"]["tools"]}

        self.assertEqual(set(tools), {"list_corpora", "search_corpus"})

        for tool in tools.values():
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])

    def test_a_search_reaches_the_index_with_its_arguments(self):
        index = FakeIndex()
        reply, = exchange(call("search_corpus", {"query": "probe", "corpus": "beta",
                                                 "top_k": 99}), index=index)

        self.assertFalse(reply["result"]["isError"])
        self.assertIn("probe", reply["result"]["content"][0]["text"])
        self.assertEqual(index.asked, [("probe", "beta", corpus_mcp.TOP_K_MAX)])

    def test_the_default_corpus_is_the_index_s_business(self):
        index = FakeIndex()
        exchange(call("search_corpus", {"query": "probe"}), index=index)

        self.assertEqual(index.asked, [("probe", None, corpus_mcp.TOP_K)])

    def test_a_failure_is_a_result_the_model_can_read(self):
        for arguments, fragment in (({"query": "x", "corpus": "missing"}, "missing"),
                                    ({"query": "boom"}, "index unreadable"),
                                    ({"query": "  "}, "non-empty"),
                                    ({"query": "x", "top_k": "3"}, "integer"),
                                    ({}, "non-empty")):
            reply, = exchange(call("search_corpus", arguments))

            self.assertTrue(reply["result"]["isError"], arguments)
            self.assertIn(fragment, reply["result"]["content"][0]["text"])

    def test_an_unknown_tool_is_an_error_result(self):
        reply, = exchange(call("delete_corpus", {}))

        self.assertTrue(reply["result"]["isError"])

    def test_list_corpora(self):
        reply, = exchange(call("list_corpora", {}))

        self.assertIn("alpha", reply["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
