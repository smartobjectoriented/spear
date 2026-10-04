"""An implementation turn runs on the agent core, end to end through AgentRuntime."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_runtime import AgentRuntime, RuntimeTerminalReason
from model_backend import ToolDefinition
from tests.test_agent_loop import Host, call, turn
from tests.test_agent_runtime import ScriptedBackend, make_context, text_turn

TOOLS = tuple(ToolDefinition(name, name, {"type": "object"})
              for name in ("read_file", "patch", "terminal"))


class Backend(ScriptedBackend):
    def __init__(self, raw_turns):
        super().__init__([text_turn("legacy loop must not be used")])
        self.raw = list(raw_turns)
        self.model = "qwen3"

    def complete_messages(self, messages, tools, *, max_tokens, on_token=None):
        return self.raw.pop(0)


class CoreTurn(unittest.TestCase):
    def run_turn(self, raw_turns):
        self.root = tempfile.mkdtemp()
        Path(self.root, "a.sh").write_text("#!/bin/sh\necho alpha\n")
        Path(self.root, "a.sh").chmod(0o755)
        backend = Backend(raw_turns)
        context = make_context(backend, rounds=10, actions=10)
        context.execution_core = "coding"
        context.tools = TOOLS
        context.project_root = self.root
        host = Host(self.root)
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        return AgentRuntime().run(context), context

    def test_a_change_without_a_check_is_unverified_in_the_answer(self):
        result, context = self.run_turn([
            turn("", call("c1", "patch", path="a.sh", old_string="alpha", new_string="beta")),
            turn("Done. The change is complete and correct.")])

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)
        self.assertTrue(result.did_modify)
        self.assertTrue(result.final_response.startswith("**UNVERIFIED**"))
        self.assertIn("*(not verified)*", result.final_response)
        self.assertEqual(result.tool_log[0].split("\n")[1], "OK: a.sh updated")
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")

    def test_a_change_checked_by_running_it_is_verified(self):
        result, context = self.run_turn([
            turn("", call("c1", "patch", path="a.sh", old_string="alpha", new_string="beta")),
            turn("", call("c2", "terminal", command="./a.sh")),
            turn("Done. It prints beta.")])

        self.assertEqual(context.core_verdict.state, "VERIFIED", result.tool_log)
        self.assertEqual(result.final_response, "Done. It prints beta.")

    def test_a_long_patch_with_its_path_last_is_still_a_change(self):
        # The model's own key order: a long old_string first pushed "path"
        # past the tool log's 200-character head, and the turn was NO_CHANGE.
        long_line = "echo " + "alpha" * 60
        result, context = self.run_turn([
            turn("", call("c0", "terminal", command=f"printf '%s\\n' '{long_line}' >> a.sh")),
            turn("", call("c1", "patch", old_string=long_line,
                          new_string=long_line.replace("alpha", "beta"), path="a.sh")),
            turn("Done. The change is complete.")])

        self.assertNotIn('"path"', result.tool_log[1].partition("\n")[0])
        self.assertEqual(context.core_verdict.state, "UNVERIFIED")
        self.assertEqual(result.changed_paths, ("a.sh",))
        self.assertTrue(result.final_response.startswith("**UNVERIFIED**"))


class _Recording:
    """The fake host, with the runtime's record callback attached."""

    def __init__(self, host, record):
        self._host, self._record = host, record
        self.workspace_root = host.workspace_root

    def __getattr__(self, name):
        return getattr(self._host, name)

    def after_tool(self, item):
        self._host.after_tool(item)
        self._record(item)


if __name__ == "__main__":
    unittest.main()


class OneCodingLoop(unittest.TestCase):
    """An implementation turn reaches nothing of the legacy coding loop.

    Each legacy mechanism -- the conclusion and write demands, the build-gap
    nudge, the repeated-result forcing, the legacy model turn itself -- is a
    tripwire here. The turn must still run to its answer on the core.
    """

    TRIPWIRES = ("conclude_demand", "write_demand", "project_build_gap",
                 "_repeated_result")

    def test_no_legacy_mechanism_is_reached(self):
        from unittest.mock import patch
        import agent_runtime

        def tripped(name):
            def fail(*args, **kwargs):
                raise AssertionError(f"legacy mechanism reached: {name}")
            return fail

        patches = [patch.object(agent_runtime, name, tripped(name))
                   for name in self.TRIPWIRES if hasattr(agent_runtime, name)]
        patches += [patch.object(AgentRuntime, name, tripped(name))
                    for name in ("_complete_with_retry", "complete_model_turn")]

        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        result, context = CoreTurn.run_turn(self, [
            turn("", call("c1", "patch", path="a.sh", old_string="alpha", new_string="beta")),
            turn("", call("c2", "terminal", command="./a.sh")),
            turn("Done.")])

        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)
        self.assertEqual(context.core_verdict.state, "VERIFIED")
        self.assertEqual(len(patches), len(self.TRIPWIRES) + 2)


class ASmallWindowStillRuns(unittest.TestCase):
    """The core's output reservation fits the window SPEAR runs it in."""

    def reservations(self, window):
        seen = []

        class Recording(Backend):
            def complete_messages(self, messages, tools, *, max_tokens, on_token=None):
                seen.append(max_tokens)
                return super().complete_messages(messages, tools, max_tokens=max_tokens)

        root = tempfile.mkdtemp()
        Path(root, "a.sh").write_text("#!/bin/sh\necho alpha\n")
        backend = Recording([turn("", call("c1", "read_file", path="a.sh")), turn("Done.")])
        context = make_context(backend, rounds=10, actions=10)
        context.execution_core = "coding"
        context.tools = TOOLS
        context.project_root = root
        context.context_limit = window
        host = Host(root)
        context.coding_host = lambda ctx, cache, record: _Recording(host, record)
        result = AgentRuntime().run(context)

        return seen, result

    def test_the_default_window_gets_a_quarter_of_itself(self):
        seen, result = self.reservations(32_768)

        self.assertEqual(seen, [8_192, 8_192])
        self.assertEqual(result.terminal_reason, RuntimeTerminalReason.COMPLETED)

    def test_a_large_window_keeps_the_cores_reservation(self):
        from agent import loop

        seen, _ = self.reservations(524_288)

        self.assertEqual(seen, [loop.MAX_TOKENS, loop.MAX_TOKENS])
