"""A turn that repeats the same successful call to no effect is stopped at the
fourth identical result, and legitimate repetition is not."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import repeat_breaker
from runtime.agent_context import RuntimeTerminalReason
from runtime.agent_runtime import AgentRuntime
from tests.test_agent_loop import Host, call, turn
from tests.test_agent_runtime import MemoryRecorder, make_context
from tests.test_core_runtime import TOOLS, Backend, ToolDefinition, _Recording
from runtime.tracing import EventType, TraceEmitter

WRITE = {"path": "README", "content": "meta-local\n"}
WROTE = '{"bytes_written": 11, "dirs_created": true, "verified": true}'
WITH_WRITE = TOOLS + (ToolDefinition("write_file", "write_file", {"type": "object"}),)


class Counting(unittest.TestCase):
    def test_it_trips_at_the_fourth_identical_success_and_not_before(self):
        breaker = repeat_breaker.RepeatBreaker(limit=4)

        for _ in range(3):
            self.assertFalse(breaker.observe("write_file", WRITE, WROTE, True))

        self.assertTrue(breaker.observe("write_file", dict(reversed(WRITE.items())), WROTE, True))
        self.assertIn("4 times in a row", breaker.conclusion())

    def test_a_changed_result_resets_the_count(self):
        breaker = repeat_breaker.RepeatBreaker(limit=4)

        for i in range(10):
            breaker.observe("terminal", {"command": "make"}, f"built {i}", True)

        self.assertIsNone(breaker.tripped)

    def test_another_call_in_between_resets_the_count(self):
        breaker = repeat_breaker.RepeatBreaker(limit=4)

        for _ in range(6):
            breaker.observe("terminal", {"command": "make"}, "ok", True)
            breaker.observe("read_file", {"path": "log"}, "line", True)

        self.assertIsNone(breaker.tripped)

    def test_failed_and_refused_calls_are_not_counted(self):
        breaker = repeat_breaker.RepeatBreaker(limit=4)

        for _ in range(10):
            breaker.observe("terminal", {"command": "rm -rf x"}, "refused: rm", False)

        self.assertIsNone(breaker.tripped)

    def test_a_poller_may_repeat(self):
        breaker = repeat_breaker.RepeatBreaker(limit=4)

        for _ in range(10):
            breaker.observe("process_manage", {"action": "poll"}, "running", True)

        self.assertIsNone(breaker.tripped)


class ALoopIsStopped(unittest.TestCase):
    def run_turn(self, turns):
        root = tempfile.mkdtemp()
        backend = Backend(turns)
        self.recorder = MemoryRecorder()
        context = make_context(backend, rounds=500, actions=500,
                               trace=TraceEmitter(self.recorder))
        context.execution_core = "coding"
        context.tools = WITH_WRITE
        context.project_root = root
        host = Host(root)
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        return AgentRuntime().run(context), context, backend

    def test_the_same_successful_write_ends_the_turn(self):
        loop = [turn("", call(f"w{i}", "write_file", **WRITE)) for i in range(40)]
        result, _, backend = self.run_turn(loop + [turn("never reached")])

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        self.assertIn("safety stop", result.final_response)
        self.assertGreater(len(backend.raw), 35)
        events = [e for e in self.recorder.events if e.event_type == EventType.REPEATED_SUCCESS_STOPPED]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].metadata["reason"], "repeated_success_loop")

    def test_rewriting_after_a_real_change_finishes(self):
        turns = []
        for i in range(3):
            turns.append(turn("", call(f"w{i}", "write_file", path="README", content=f"v{i}\n")))
            turns.append(turn("", call(f"r{i}", "read_file", path="README")))
        turns.append(turn("Done."))
        result, _, _ = self.run_turn(turns)

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)

    def test_three_identical_writes_then_progress_finishes(self):
        turns = [turn("", call(f"w{i}", "write_file", **WRITE)) for i in range(3)]
        turns += [turn("", call("r", "read_file", path="README")), turn("Done.")]
        result, _, _ = self.run_turn(turns)

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)


if __name__ == "__main__":
    unittest.main()
