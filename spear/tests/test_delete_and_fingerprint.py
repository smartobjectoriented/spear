"""The two gaps the last live run exposed.

A file could not be deleted by any means, and nine consecutive looks at the
same three lines -- cat -A, od -c, hexdump -C, strings, tr -d -- counted as
nine different actions, so nothing noticed the model had stopped progressing.
"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import rag_chat
from progress_monitor import action_fingerprint
from tool_registry import native_tool_specs
from tool_runtime import ExecutionMode, Workspace


FILE = "/home/operator/soo/so3/doc/source/user_space.rst"


class ReadFingerprintTests(unittest.TestCase):
    def fingerprint(self, command):
        return action_fingerprint("bash", {"command": command})

    def test_the_same_file_read_five_ways_is_one_action(self):
        spellings = [f"sed -n '59,61p' {FILE}",
                     f"sed -n '58,63p' {FILE} | cat -A",
                     f"sed -n '59,61p' {FILE} | od -c",
                     f"sed -n '59,61p' {FILE} | hexdump -C",
                     f"cat {FILE} | tail -n +140"]
        self.assertEqual(1, len({self.fingerprint(item) for item in spellings}))

    def test_reading_a_different_file_is_a_different_action(self):
        self.assertNotEqual(self.fingerprint(f"cat {FILE}"),
                            self.fingerprint("cat doc/source/index.rst"))

    def test_searching_is_not_collapsed(self):
        """grep for another pattern is another question, not a repeat."""
        self.assertNotEqual(self.fingerprint(f"grep -n 'more' {FILE}"),
                            self.fingerprint(f"grep -n 'toctree' {FILE}"))

    def test_a_command_that_is_not_a_read_keeps_its_own_fingerprint(self):
        self.assertNotEqual(self.fingerprint("make html"),
                            self.fingerprint("make clean"))


class DeleteFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for attr, value in (("WORKSPACE", Workspace.from_path(root)),
                            ("PROJECT_ROOT", str(root)),
                            ("EXECUTION_MODE", ExecutionMode.AUTO),
                            ("BYPASS_PERMISSIONS", True)):
            self.addCleanup(setattr, rag_chat, attr, getattr(rag_chat, attr))
            setattr(rag_chat, attr, value)
        self.previous_cwd = os.getcwd()
        os.chdir(root)
        self.addCleanup(lambda: os.chdir(self.previous_cwd))
        self.target = root / "ls.rst"
        self.target.write_text(".. _ls:\n\nls\n##\n")

    def context(self):
        return SimpleNamespace(cache={}, role="main", cancellation=None,
                               checkpoint_manager=None, checkpoint=None,
                               action_id=None, trace=None, task_id="t")

    def test_the_tool_is_exposed_to_the_model(self):
        spec = next(item for item in native_tool_specs()
                    if item.name == "delete_file")
        self.assertTrue(spec.model_visible)
        self.assertIn("reason", spec.input_schema["required"])

    def test_a_file_is_removed(self):
        out = rag_chat._registered_delete_file(
            self.context(), {"path": "ls.rst", "reason": "content moved"})
        self.assertTrue(out.text.startswith("OK"), out.text)
        self.assertFalse(self.target.exists())

    def test_a_missing_file_and_a_directory_are_both_refused(self):
        for path in ("nope.rst", "."):
            with self.subTest(path=path):
                out = rag_chat._registered_delete_file(
                    self.context(), {"path": path, "reason": "x"})
                self.assertTrue(out.text.startswith("ERROR"), out.text)
        self.assertTrue(self.target.exists())


if __name__ == "__main__":
    unittest.main()
