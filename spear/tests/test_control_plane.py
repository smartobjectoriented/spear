"""The agent core runs inside SPEAR's control plane, with no way around it.

Real policy, real workspace resolution, real sandbox: every read, write,
deletion and command the core asks for crosses the same checks.
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
from tool_runtime import ExecutionMode

NAMES = ("read_file", "search_files", "patch", "write_file", "delete_file", "terminal")


class ControlPlane(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(dir=str(rag_chat.WORKSPACE.root))
        self.rel = os.path.relpath(self.root, rag_chat.WORKSPACE.root)
        Path(self.root, "a.txt").write_text("alpha\n")
        self.records = []
        self.state = tools.new_state()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def host(self, **overrides):
        context = SimpleNamespace(
            task_id="task_cp", cancellation=NEVER_CANCELLED, work_phase=None,
            checkpoint_manager=None, checkpoint=None, role="main", read_only=False,
            advisory=False, standard_binding=None, execution_core="coding",
            conversation=[], dropped_tool_results=frozenset(),
            working_state=SimpleNamespace(objective="edit a.txt"))
        for key, value in overrides.items():
            setattr(context, key, value)
        return rag_chat.coding_host(context, {}, self.records.append)

    def call(self, name, mode=ExecutionMode.AUTO, **arguments):
        with patch.object(rag_chat, "EXECUTION_MODE", mode), redirect_stdout(io.StringIO()):
            text, record = dispatch.execute(self.host_, self.state, f"c{len(self.records)}",
                                            name, arguments, NAMES)
        return text, record

    def use(self, **overrides):
        self.host_ = self.host(**overrides)

    def test_reads_and_writes_stay_in_the_workspace(self):
        self.use()

        self.assertTrue(self.call("read_file", path="/etc/hostname")[1].refused)
        _, record = self.call("patch", path="/etc/hostname", old_string="a", new_string="b")
        self.assertTrue(record.refused)
        self.assertTrue(self.call("write_file", path="/tmp/spear_outside_test.txt",
                                  content="x")[1].refused)
        self.assertFalse(Path("/tmp/spear_outside_test.txt").exists())

    def test_a_read_only_turn_cannot_write_by_any_tool(self):
        self.use(read_only=True)

        for name, arguments in (("patch", {"path": f"{self.rel}/a.txt", "old_string": "alpha",
                                           "new_string": "beta"}),
                                ("write_file", {"path": f"{self.rel}/b.txt", "content": "x"}),
                                ("delete_file", {"path": f"{self.rel}/a.txt", "reason": "x"})):
            with self.subTest(name=name):
                self.assertTrue(self.call(name, **arguments)[1].refused)

        self.assertEqual(Path(self.root, "a.txt").read_text(), "alpha\n")

    def test_a_safe_session_cannot_write(self):
        self.use()
        text, record = self.call("patch", mode=ExecutionMode.SAFE, path=f"{self.rel}/a.txt",
                                 old_string="alpha", new_string="beta")

        self.assertFalse(record.changed_paths)
        self.assertEqual(Path(self.root, "a.txt").read_text(), "alpha\n")

    def test_a_generated_file_is_refused(self):
        self.use()
        Path(self.root, "gen.c").write_text("/* DO NOT EDIT: generated */\nint a;\n")
        text, record = self.call("patch", path=f"{self.rel}/gen.c", old_string="int a;",
                                 new_string="int b;")

        self.assertFalse(record.changed_paths)
        self.assertIn("generated", json.loads(text)["error"])

    def test_a_permitted_patch_lands_and_is_recorded(self):
        self.use()
        text, record = self.call("patch", path=f"{self.rel}/a.txt", old_string="alpha",
                                 new_string="beta")

        self.assertTrue(json.loads(text)["success"])
        self.assertEqual(Path(self.root, "a.txt").read_text(), "beta\n")
        self.assertEqual(self.records[-1], record)

    def test_write_file_creates_parents(self):
        self.use()
        _, record = self.call("write_file", path=f"{self.rel}/x/y/z.txt", content="z\n")

        self.assertTrue(record.ok)
        self.assertEqual(Path(self.root, "x/y/z.txt").read_text(), "z\n")

    @unittest.skipUnless(shutil.which("bwrap"), "needs the bubblewrap sandbox")
    def test_terminal_runs_sandboxed_with_a_session(self):
        self.use()
        first, _ = self.call("terminal", command=f"cd {self.rel} && export Q=1 && pwd")
        second, record = self.call("terminal", command="echo $Q; ls")

        self.assertTrue(json.loads(first)["output"].endswith(self.rel))
        self.assertEqual(json.loads(second)["output"], "1\na.txt")
        self.assertEqual(record.exit_code, 0)

    @unittest.skipUnless(shutil.which("bwrap"), "needs the bubblewrap sandbox")
    def test_terminal_cannot_write_outside_the_workspace(self):
        self.use()
        text, record = self.call("terminal", command="cp a.txt /tmp/spear_escape_test")

        self.assertFalse(Path("/tmp/spear_escape_test").exists())
        self.assertNotEqual(record.exit_code, 0)

    @unittest.skipUnless(shutil.which("bwrap"), "needs the bubblewrap sandbox")
    def test_a_read_only_turn_cannot_write_through_the_terminal(self):
        self.use(read_only=True)
        self.call("terminal", command=f"echo x > {self.rel}/a.txt")

        self.assertEqual(Path(self.root, "a.txt").read_text(), "alpha\n")


if __name__ == "__main__":
    unittest.main()
