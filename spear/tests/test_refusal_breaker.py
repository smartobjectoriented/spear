"""A turn that keeps asking for what was refused is stopped well before its
round budget, and one that recovers is not."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import refusal_breaker
from runtime.agent_context import RuntimeTerminalReason
from runtime.agent_runtime import AgentRuntime
from tests.test_agent_loop import Host, call, turn
from tests.test_agent_runtime import MemoryRecorder, make_context
from tests.test_core_runtime import TOOLS, Backend, _Recording
from runtime.tracing import EventType, TraceEmitter

REFUSED = ("refused: command refused: '/srv/other/tree' is in another tree than this "
           "project (/srv/work).")


class Signature(unittest.TestCase):
    def test_formatting_does_not_make_a_new_operation(self):
        spellings = ["ls /srv/other/tree", "ls  /srv/other/tree/", "ls '/srv/other/tree'",
                     'ls "/srv/other/tree/"']
        keys = {refusal_breaker.signature("terminal", {"command": c}, REFUSED) for c in spellings}

        self.assertEqual(len(keys), 1)

    def test_a_different_target_is_a_different_operation(self):
        first = refusal_breaker.signature("terminal", {"command": "ls /srv/a"}, REFUSED)
        second = refusal_breaker.signature("terminal", {"command": "ls /srv/b"}, REFUSED)

        self.assertNotEqual(first, second)

    def test_it_trips_at_the_limit_and_not_before(self):
        breaker = refusal_breaker.RefusalBreaker(limit=5)

        for _ in range(4):
            self.assertFalse(breaker.observe("terminal", {"command": "ls /srv/x"}, REFUSED, True))

        self.assertTrue(breaker.observe("terminal", {"command": "ls /srv/x/"}, REFUSED, True))
        self.assertIn("refused 5 times", breaker.conclusion())

    def test_calls_that_ran_are_not_counted(self):
        breaker = refusal_breaker.RefusalBreaker(limit=2)

        for _ in range(5):
            breaker.observe("terminal", {"command": "ls"}, "a.sh", False)

        self.assertIsNone(breaker.tripped)


class _Refusing(_Recording):
    def authorize(self, name, arguments):
        if name == "terminal" and "other" in str(arguments.get("command")):
            return REFUSED
        return self._host.authorize(name, arguments)


class ALoopIsStopped(unittest.TestCase):
    def run_turn(self, turns):
        root = tempfile.mkdtemp()
        Path(root, "a.sh").write_text("#!/bin/sh\necho alpha\n")
        backend = Backend(turns)
        self.recorder = MemoryRecorder()
        context = make_context(backend, rounds=500, actions=500,
                               trace=TraceEmitter(self.recorder))
        context.execution_core = "coding"
        context.tools = TOOLS
        context.project_root = root
        host = Host(root)
        context.coding_host = lambda ctx, cache, record: _Refusing(host, record)
        return AgentRuntime().run(context), context, backend

    def test_the_same_refused_command_ends_the_turn(self):
        loop = [turn("", call(f"c{i}", "terminal", command="ls /srv/other/tree" + "/" * (i % 2)))
                for i in range(40)]
        result, context, backend = self.run_turn(loop + [turn("never reached")])

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.STALLED)
        self.assertIn("refused 5 times", result.final_response)
        self.assertGreater(len(backend.raw), 30)
        kinds = [event.event_type for event in self.recorder.events]
        self.assertIn(EventType.REPEATED_REFUSAL_STOPPED, kinds)

    def test_a_turn_that_changes_course_finishes(self):
        turns = [turn("", call(f"c{i}", "terminal", command="ls /srv/other/tree"))
                 for i in range(3)]
        turns += [turn("", call("r", "read_file", path="a.sh")), turn("Done.")]
        result, _, _ = self.run_turn(turns)

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)


if __name__ == "__main__":
    unittest.main()
