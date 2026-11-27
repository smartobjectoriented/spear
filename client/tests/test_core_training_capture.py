"""The coding core's model calls reach the training recorder.

The core calls the model itself, in OpenAI-format messages, and from its
arrival every implementation turn was stored as an episode with no turns.
"""

import json
import unittest

from models.model_backend import RawTurn, StopReason, ToolResultBlock, ToolUseBlock
from runtime.agent_core_turn import _canonical_conversation, _model_turn

MESSAGES = [
    {"role": "system", "content": "you are the core"},
    {"role": "user", "content": "fix the counter"},
    {"role": "assistant", "content": "reading", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "read_file", "arguments": json.dumps({"path": "app.js"})}},
        {"id": "c2", "type": "function",
         "function": {"name": "search_files", "arguments": "{\"pattern\": \"x\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "const x = 1;"},
    {"role": "tool", "tool_call_id": "c2", "content": "app.js:1"},
]


class Conversion(unittest.TestCase):
    def test_the_wire_history_becomes_typed_blocks(self):
        system, conversation = _canonical_conversation(MESSAGES)

        self.assertEqual(system, "you are the core")
        self.assertEqual([item.role for item in conversation], ["user", "assistant", "user"])
        calls = [block for block in conversation[1].content if isinstance(block, ToolUseBlock)]
        self.assertEqual([(call.id, call.arguments) for call in calls],
                         [("c1", {"path": "app.js"}), ("c2", {"pattern": "x"})])
        self.assertEqual([block.tool_call_id for block in conversation[2].content
                          if isinstance(block, ToolResultBlock)], ["c1", "c2"])

    def test_a_raw_completion_becomes_a_model_turn(self):
        turn = _model_turn(RawTurn("", ({"id": "c3", "name": "edit_file",
                                         "arguments": "{\"path\": \"app.js\"}"},), "tool_calls"))

        self.assertEqual(turn.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(turn.tool_calls[0].arguments, {"path": "app.js"})
        self.assertEqual(_model_turn(RawTurn("done", (), "stop")).stop_reason, StopReason.END_TURN)
        self.assertEqual(_model_turn(RawTurn("x", ({"id": "c", "name": "n", "arguments": "{bad"},),
                                             None)).tool_calls[0].arguments, {"_raw": "{bad"})


if __name__ == "__main__":
    unittest.main()
