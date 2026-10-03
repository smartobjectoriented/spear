"""Terminal output reaches the model exactly as Hermes Agent delivers it.

The expected strings in fixtures/hermes_terminal.json were returned by Hermes
v0.21.0 (0cbc6e37) for the same commands on the same bytes. The run goes
through SPEAR's real execution path -- the coding host, the command policy,
tool_runtime and the bwrap sandbox -- because that is where the defect was:
reading the command's pipes with text=True applied Python's universal-newline
translation, so every \\r and \\r\\n reached the model as \\n, and binary output
(an 0xff byte) raised UnicodeDecodeError instead of being shown.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_chat
from agent import dispatch, tools
from cancellation import NEVER_CANCELLED
from tool_runtime import ExecutionMode, decode_command_output

FIXTURE = json.loads((ROOT / "tests/fixtures/hermes_terminal.json").read_text())


class DecodeCommandOutput(unittest.TestCase):
    def test_line_endings_are_kept_as_emitted(self):
        for raw, text in ((b"a\nb\n", "a\nb\n"), (b"a\r\nb\r\n", "a\r\nb\r\n"),
                          (b"a\rb\r", "a\rb\r"), (b"x\r\ny\rz\nw", "x\r\ny\rz\nw")):
            with self.subTest(raw=raw):
                self.assertEqual(decode_command_output(raw), text)

    def test_invalid_utf8_is_replaced_not_raised(self):
        self.assertEqual(decode_command_output(b"\xff\xfeok"), "��ok")


@unittest.skipUnless(shutil.which("bwrap"), "needs the bubblewrap sandbox")
class TerminalParity(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=str(rag_chat.WORKSPACE.root))
        for name, data in FIXTURE["files"].items():
            Path(self.root, name).write_bytes(bytes.fromhex(data))
        context = SimpleNamespace(
            task_id="task_term", cancellation=NEVER_CANCELLED, work_phase=None,
            checkpoint_manager=None, checkpoint=None, role="main", read_only=False,
            advisory=False, standard_binding=None, execution_core="coding",
            conversation=[], dropped_tool_results=frozenset(),
            working_state=SimpleNamespace(objective="read files"))
        self.host = rag_chat.coding_host(context, {}, lambda record: None)
        self.state = tools.new_state()
        self.run_command(f"cd {self.root} && pwd")   # Hermes ran in that directory

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_command(self, command):
        with patch.object(rag_chat, "EXECUTION_MODE", ExecutionMode.AUTO), \
                redirect_stdout(io.StringIO()):
            text, _ = dispatch.execute(self.host, self.state, "c", "terminal",
                                       {"command": command}, ("terminal",))
        return text

    def test_output_matches_hermes(self):
        for key, command in FIXTURE["commands"].items():
            with self.subTest(case=key):
                self.assertEqual(self.run_command(command), FIXTURE["results"][key])


if __name__ == "__main__":
    unittest.main()
