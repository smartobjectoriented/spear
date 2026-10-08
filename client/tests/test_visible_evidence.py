"""Evidence is judged against what the model was actually sent.

Compaction rewrites the request, not the conversation: the conversation keeps
every tool result, and the guards that say "you already have this" consulted
the conversation. A turn compacted, re-read the recipe the summary had folded
away, and was refused it as "already given 2 times". The invariant: SPEAR
never refuses to show evidence because the model saw it before a compaction,
when that evidence is no longer in what the model is sent -- and without a
compaction, repetition is held exactly as before.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import agent_model_turn, agent_notes
from cli import tool_handlers, tool_routing
from runtime.agent_runtime import AgentRuntime
from runtime.compaction import CompactionPolicy
from models.model_backend import ConversationMessage, TextBlock, ToolResultBlock
from harness.tool_router import ToolResultEnvelope, ToolResultStatus

from tests.test_agent_runtime import (GroundedToolExecutor, ScriptedBackend,
                                      make_context, text_turn, tool_turn)

WALL = "the same recipe line\n" * 40
SOURCE = "src/link/handshake.c"


def envelope(call_id, text=WALL):
    return ToolResultEnvelope(call_id, f"a-{call_id}", "bash", True,
                              ToolResultStatus.OK, text, text, "command", 0.0,
                              len(text))


class IdenticalResults(unittest.TestCase):
    def test_a_without_compaction_the_third_is_refused_as_before(self):
        seen = {}
        counts = [agent_notes._repeated_result(seen, envelope(f"c{i}"))
                  for i in range(1, 5)]

        self.assertEqual(counts, [1, 2, 3, 3])
        self.assertIn("REFUSED",
                      agent_notes._with_repeat_note(envelope("c4"), counts[-1]).model_content)

    def test_b_after_compaction_drops_them_the_result_is_new_again(self):
        seen, dropped = {}, set()
        visible = lambda call_id: call_id not in dropped

        agent_notes._repeated_result(seen, envelope("c1"), visible)
        agent_notes._repeated_result(seen, envelope("c2"), visible)
        dropped.update({"c1", "c2"})

        self.assertEqual(agent_notes._repeated_result(seen, envelope("c3"), visible), 1)

    def test_c_it_stays_bounded_after_a_compaction(self):
        seen, dropped = {}, {"c1"}
        visible = lambda call_id: call_id not in dropped

        agent_notes._repeated_result(seen, envelope("c1"), visible)
        counts = [agent_notes._repeated_result(seen, envelope(f"c{i}"), visible)
                  for i in range(2, 6)]

        self.assertEqual(counts, [1, 2, 3, 3])


class WindowedReads(unittest.TestCase):
    """bash reads of the same lines: "already in evidence" while they are."""

    def context(self, call_id, dropped):
        return SimpleNamespace(
            cache=self.cache, metadata={"tool_call_id": call_id},
            evidence_available=lambda shown_by: shown_by not in dropped)

    def setUp(self):
        self.cache = {}

    def read(self, call_id, dropped=()):
        return tool_handlers._already_in_evidence(
            self.context(call_id, set(dropped)), f"sed -n '1,200p' {SOURCE}")

    def test_a_a_second_read_is_stubbed_while_the_first_is_visible(self):
        self.assertEqual(self.read("r1"), "")
        self.assertIn("ALREADY IN EVIDENCE", self.read("r2"))

    def test_b_a_read_whose_first_showing_was_compacted_runs_again(self):
        self.read("r1")

        self.assertEqual(self.read("r2", dropped={"r1"}), "")

    def test_c_the_new_showing_is_what_later_reads_are_judged_by(self):
        self.read("r1")
        self.read("r2", dropped={"r1"})

        self.assertIn("ALREADY IN EVIDENCE", self.read("r3", dropped={"r1"}))


class TheRuntimeRecordsWhatWasNotSent(unittest.TestCase):
    def test_a_compacted_result_is_reported_as_not_in_context(self):
        seen_dropped = []

        class Recording(GroundedToolExecutor):
            def __call__(self, context, tool_call_id, name, arguments, cache):
                seen_dropped.append(set(agent_context.dropped_tool_results))
                return super().__call__(context, tool_call_id, name, arguments, cache)

        outputs = [f"file {index} line\n" * 40 for index in range(8)]
        backend = ScriptedBackend(
            [tool_turn(f"k{index}", "bash", command=f"cat f{index}")
             for index in range(8)] + [text_turn("done")])
        agent_context = make_context(backend, rounds=12, actions=12,
                                     executor=Recording(outputs))
        agent_context.compaction_policy = CompactionPolicy(
            pressure_threshold=.01, minimum_compactable_tokens=1,
            recent_tail_groups=0)
        AgentRuntime().run(agent_context)

        # Folded into a summary, so not sent -- though the conversation
        # still holds the block.

        self.assertIn("conversation:2",
                      agent_context.compaction_artifact.compacted_item_ids)
        self.assertIn("k0", set().union(*seen_dropped))
        self.assertTrue(any(isinstance(block, ToolResultBlock)
                            and block.tool_call_id == "k0"
                            for message in agent_context.conversation
                            for block in message.content))
        self.assertFalse(tool_routing._evidence_in_context(agent_context, "k0"))

    def test_without_compaction_nothing_is_dropped(self):
        backend = ScriptedBackend([
            tool_turn("one", "bash", command="cat a"),
            text_turn("done"),
        ])
        agent_context = make_context(backend, rounds=4,
                                     executor=GroundedToolExecutor([WALL]))
        AgentRuntime().run(agent_context)

        self.assertEqual(agent_context.dropped_tool_results, frozenset())
        self.assertTrue(tool_routing._evidence_in_context(agent_context, "one"))

    def test_the_ids_are_read_from_result_blocks(self):
        messages = (
            ConversationMessage("user", (TextBlock("x"),)),
            ConversationMessage("user", (ToolResultBlock("k1", "out"),
                                         ToolResultBlock("k2", ""))),
        )

        self.assertEqual(agent_model_turn._tool_result_ids(messages), frozenset({"k1"}))


if __name__ == "__main__":
    unittest.main()
