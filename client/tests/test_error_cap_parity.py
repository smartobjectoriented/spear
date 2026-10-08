"""An oversized error field is cut where, and as, Hermes Agent cuts it.

Hermes' registry bounds the ``error`` of every handler's result to 2048
characters (code points) plus "… [truncated]" as the result leaves the
dispatcher, and tool_error bounds its message the same way. A deterministic
paired run diverged on exactly this: a failed patch whose "Did you mean" hint
ran past 2048 characters reached Qwen whole from the port and cut from Hermes.
fixtures/hermes_error_cap.json holds what Hermes v0.21.0 (0cbc6e37) returned.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import dispatch, tools
from tests.test_agent_loop import Host

FIXTURE = json.loads((ROOT / "tests/fixtures/hermes_error_cap.json").read_text())
MESSAGES, RESULTS = FIXTURE["messages"], FIXTURE["results"]


class ErrorCap(unittest.TestCase):
    def test_tool_error(self):
        for key, message in MESSAGES.items():
            with self.subTest(case=key):
                self.assertEqual(tools.tool_error(message), RESULTS["tool_error"][key])

        self.assertEqual(tools.tool_error("w" * 3000, success=False),
                         RESULTS["tool_error_extra"])

    def test_json_error_field(self):
        for key, message in MESSAGES.items():
            with self.subTest(case=key):
                raw = json.dumps({"success": False, "error": message, "note": "kept"},
                                 ensure_ascii=False)
                self.assertEqual(tools.bound_json_error_result(raw), RESULTS["json_result"][key])

    def test_reserialised_like_hermes_and_only_when_cut(self):
        self.assertEqual(tools.bound_json_error_result(
            json.dumps({"success": False, "error": "é" * 3000})), RESULTS["json_result_ascii"])
        self.assertEqual(tools.bound_json_error_result(json.dumps({"output": "o" * 5000})),
                         RESULTS["json_non_error"])

    def test_a_failed_patch_through_the_dispatcher(self):
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, "big.txt")
            Path(path).write_text("".join(f"line {i:04d} {'=' * 200}\n" for i in range(400)))
            text, record = dispatch.execute(
                Host(root), tools.new_state(), "c1", "patch",
                {"mode": "replace", "path": path,
                 "old_string": "line 0200 " + "=" * 199 + "X\nline 0201", "new_string": "zzz"},
                ("patch",))
            self.assertEqual(text, RESULTS["dispatch_patch"].replace("<DIR>", root))
            self.assertEqual(record.result, text)       # the record holds what the model read
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
