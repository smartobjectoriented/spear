"""The coding core's loop behaves as Hermes Agent's does (agent/loop.py).

A scripted model and a fake host: what the model reads, in which order, and
what the loop sends back, with no SPEAR machinery in between.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import loop
from agent.hermes import loop_constants as text
from agent.host import CommandOutcome
from model_backend import RawTurn

TOOLS = [{"type": "function", "function": {"name": name, "description": name,
                                           "parameters": {"type": "object"}}}
         for name in ("read_file", "search_files", "patch", "write_file", "terminal")]


def call(call_id, name, **arguments):
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


def turn(content="", *calls, finish=None):
    return RawTurn(content, tuple(calls), finish or ("tool_calls" if calls else "stop"))


class Model:
    def __init__(self, turns):
        self.turns, self.requests = list(turns), []

    def __call__(self, messages, tools, max_tokens):
        self.requests.append((json.loads(json.dumps(messages)), tools, max_tokens))
        return self.turns.pop(0)


class Host:
    def __init__(self, root, refuse=()):
        self.workspace_root, self.refuse, self.records = root, set(refuse), []

    def authorize(self, name, arguments):
        return "refused: not in scope" if name in self.refuse else None

    def resolve_read(self, path):
        return str(Path(self.workspace_root, path)), None

    resolve_write = resolve_workdir = resolve_read

    def write_file(self, path, content, *, action):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    def delete_file(self, path, reason):
        Path(path).unlink()

    def run_command(self, command, script, *, timeout, output_chars):
        import subprocess
        done = subprocess.run(["bash", "-c", script], cwd=self.workspace_root,
                              capture_output=True, text=True)
        return CommandOutcome("ok", done.stdout + done.stderr, done.returncode)

    def after_tool(self, record):
        self.records.append(record)


class Loop(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        Path(self.root, "a.txt").write_text("alpha\n")
        self.host = Host(self.root)

    def run_loop(self, turns, **kwargs):
        self.model = Model(turns)
        options = dict(model=self.model, host=self.host, system="SYS", history=[],
                       request="do it", tool_definitions=TOOLS, max_iterations=30,
                       max_tool_calls=60, context_window=524288)
        options.update(kwargs)
        return loop.run(**options)

    def sent(self, index=-1):
        return self.model.requests[index][0]

    def test_request_shape_and_answer(self):
        result = self.run_loop([turn("", call("c1", "read_file", path="a.txt")),
                                turn("Done.")])

        self.assertEqual(result.final, "Done.")
        self.assertEqual(self.model.requests[0][2], loop.MAX_TOKENS)
        tool_message = self.sent()[-1]
        self.assertEqual(set(tool_message), {"role", "name", "tool_call_id", "content"})
        self.assertIn("1|alpha", tool_message["content"])

    def test_no_harness_message_however_long_it_reads(self):
        turns = [turn("", call(f"c{i}", "terminal", command=f"echo {i}")) for i in range(25)]
        result = self.run_loop(turns + [turn("Done.")])

        users = [m for m in result.messages if m["role"] == "user"]
        self.assertEqual(len(users), 1)

    def test_arguments_are_resent_compact_and_sorted(self):
        self.run_loop([turn("", {"id": "c1", "name": "terminal",
                                 "arguments": '{"timeout": 5, "command": "true"}'}),
                       turn("Done.")])

        sent_call = self.sent()[2]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(sent_call, '{"command":"true","timeout":5}')

    def test_a_text_reply_cut_by_the_limit_is_continued(self):
        result = self.run_loop([turn("half", finish="length"), turn("whole")])

        self.assertEqual(self.sent()[-1]["content"], text._LENGTH_CONTINUATION_OUTPUT_LIMIT)
        self.assertEqual(result.final, "whole")

    def test_a_tool_call_cut_by_the_limit_is_asked_again_not_kept(self):
        result = self.run_loop([turn("", call("c1", "terminal", command="x"), finish="length"),
                                turn("", call("c2", "terminal", command="echo ok")),
                                turn("Done.")])

        ids = [c["id"] for m in result.messages for c in m.get("tool_calls", ())]
        self.assertEqual(ids, ["c2"])

    def test_dropped_tool_call_is_nudged(self):
        self.run_loop([turn("I will run it", finish="tool_calls"), turn("Done.")])

        self.assertEqual(self.sent()[-1]["content"], text._DROPPED_TOOLCALL_NUDGE_CONTENT)

    def test_invalid_json_is_retried_silently_then_answered(self):
        bad = {"id": "c1", "name": "terminal", "arguments": '{"command": '}
        result = self.run_loop([turn("", bad), turn("", bad), turn("", bad), turn("Done.")])

        self.assertEqual(len(self.model.requests), 4)
        tool = [m for m in result.messages if m["role"] == "tool"][0]
        self.assertTrue(tool["content"].startswith("Error: Invalid JSON arguments."))

    def test_empty_reply_after_tools_is_nudged_once(self):
        result = self.run_loop([turn("", call("c1", "terminal", command="true")),
                                turn(""), turn("Done.")])

        self.assertIn(text._EMPTY_TOOL_RESPONSE_NUDGE,
                      [m["content"] for m in result.messages if m["role"] == "user"])

    def test_trailing_intent_is_continued(self):
        result = self.run_loop([turn("Found it. Let me now patch it."), turn("Done.")])

        self.assertEqual(result.final, "Done.")
        self.assertIn(text._CODEX_ACK_CONTINUATION_NUDGE,
                      [m["content"] for m in result.messages if m["role"] == "user"])

    def test_identical_calls_in_one_batch_run_once(self):
        result = self.run_loop([turn("", call("c1", "terminal", command="echo a"),
                                     call("c2", "terminal", command="echo a")),
                                turn("Done.")])

        self.assertEqual(len(result.records), 1)

    def test_a_repeated_failure_carries_the_loop_warning(self):
        turns = [turn("", call(f"c{i}", "terminal", command="false")) for i in range(3)]
        result = self.run_loop(turns + [turn("Done.")])

        tools = [m["content"] for m in result.messages if m["role"] == "tool"]
        self.assertIn("Tool loop warning", tools[-1])

    def test_a_host_refusal_is_the_result(self):
        self.host.refuse.add("patch")
        result = self.run_loop([turn("", call("c1", "patch", path="a.txt",
                                              old_string="alpha", new_string="beta")),
                                turn("Done.")])

        self.assertEqual(Path(self.root, "a.txt").read_text(), "alpha\n")
        self.assertTrue(result.records[0].refused)
        self.assertIn("refused", [m for m in result.messages if m["role"] == "tool"][0]["content"])

    def test_an_unknown_tool_is_named_with_the_available_ones(self):
        result = self.run_loop([turn("", call("c1", "bash", command="ls")), turn("Done.")])

        tool = [m for m in result.messages if m["role"] == "tool"][0]
        self.assertIn("Tool 'bash' does not exist. Available tools:", tool["content"])

    def test_the_iteration_bound_asks_for_a_summary_without_tools(self):
        turns = [turn("", call(f"c{i}", "terminal", command="true")) for i in range(3)]
        result = self.run_loop(turns + [turn("Summary.")], max_iterations=3)

        self.assertEqual(result.stop, "iterations")
        self.assertEqual(self.model.requests[-1][1], [])
        self.assertEqual(result.final, "Summary.")

    def test_think_blocks_never_reach_the_answer(self):
        result = self.run_loop([turn("<think>hmm</think>Done.")])

        self.assertEqual(result.final, "Done.")


if __name__ == "__main__":
    unittest.main()
