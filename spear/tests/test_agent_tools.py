"""The coding core's tools behave as Hermes Agent's do (agent/tools.py)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import tools as coding_tools
from agent.host import CommandOutcome


class ReadFile(unittest.TestCase):
    """file_tools.read_file_tool, as Hermes returns it."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.path = self.root / "a.txt"
        self.path.write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")
        self.state = coding_tools.new_state()

    def read(self, offset=1, limit=500, name="a.txt"):
        return json.loads(coding_tools.read_file(self.state, name, str(self.root / name),
                                                 offset, limit))

    def test_numbered_window_with_continuation(self):
        payload = self.read(3, 2)

        self.assertEqual(payload["content"], "3|line 3\n4|line 4")
        self.assertTrue(payload["truncated"])
        self.assertIn("Use offset=5 to continue reading", payload["hint"])
        self.assertEqual(payload["total_lines"], 10)
        self.assertFalse(payload["is_binary"])

    def test_binary_is_described(self):
        (self.root / "x").write_bytes(b"\x7fELF\0\0\0")

        self.assertIn("Binary file (ELF executable", self.read(name="x")["error"])

    def test_past_the_end_says_so(self):
        self.assertIn("beyond the end", self.read(50)["hint"])

    def test_not_found_suggests_similar(self):
        payload = self.read(name="a.tx")

        self.assertEqual(payload["error"], "File not found: a.tx")
        self.assertIn("./a.txt", payload["similar_files"])

    def test_repeat_is_a_stub_then_blocked(self):
        self.read()

        self.assertEqual(self.read()["status"], "unchanged")
        self.assertIn("BLOCKED", self.read()["error"])

    def test_a_compaction_makes_the_read_new_again(self):
        self.read()
        coding_tools.reset_dedup(self.state)

        self.assertIn("content", self.read())

    def test_a_change_on_disk_makes_the_read_new_again(self):
        self.read()
        os.utime(self.path, (1, 1))

        self.assertIn("content", self.read())


class Search(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(f"{self.root}/a-1-b")
        Path(self.root, "a-1-b/x.c").write_text("int a;\nfoo-2-bar\nfoo\n")
        Path(self.root, "y.h").write_text("foo\n")
        self.state = coding_tools.new_state()

    def search(self, pattern, **kwargs):
        text = coding_tools.search(self.state, self.root, kwargs.pop("path", "."),
                                   self.root, pattern, **kwargs)
        return json.loads(text.split("\n\n[Hint:")[0]), text

    def test_content_matches_name_files_as_given(self):
        payload, _ = self.search("foo")

        self.assertEqual(payload["total_count"], 3)
        self.assertIn({"path": "./a-1-b/x.c", "line": 2, "content": "foo-2-bar"},
                      payload["matches"])

    def test_files_and_counts(self):
        self.assertEqual(self.search("*.h", target="files")[0]["files"], ["./y.h"])
        self.assertEqual(self.search("foo", output_mode="count")[0]["counts"],
                         {"./a-1-b/x.c": 2, "./y.h": 1})

    def test_zero_matches_probe_the_casing(self):
        payload, _ = self.search("FOO")

        self.assertIn("case-insensitive", payload["warning"])

    def test_five_or_more_matches_are_grouped(self):
        for i in range(6):
            Path(self.root, f"m{i}.txt").write_text("needle\n")

        self.assertIn("matches_text", self.search("needle")[0])

    def test_same_search_four_times_is_blocked(self):
        for _ in range(3):
            self.search("foo")

        self.assertIn("BLOCKED", self.search("foo")[0]["error"])


class Patch(unittest.TestCase):
    CONTENT = "def f():\n    return 1\n"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.path = self.root / "f.py"
        self.path.write_text(self.CONTENT)
        self.state = coding_tools.new_state()

    def write(self, content):
        self.path.write_text(content)

    def patch(self, old, new):
        text, changed = coding_tools.patch(self.state, "f.py", str(self.path), old, new,
                                           write=self.write)
        return json.loads(text), changed

    def test_replaces_and_returns_a_diff_and_lint(self):
        payload, changed = self.patch("return 1", "return 2")

        self.assertTrue(changed)
        self.assertIn("+    return 2", payload["diff"])
        self.assertEqual(payload["lint"]["status"], "ok")

    def test_a_syntax_error_is_reported_by_lint(self):
        payload, _ = self.patch("return 1", "return (")

        self.assertEqual(payload["lint"]["status"], "error")

    def test_no_match_is_an_error_with_a_hint(self):
        payload, changed = self.patch("nowhere at all", "x")

        self.assertFalse(changed)
        self.assertFalse(payload["success"])
        self.assertIn("_hint", payload)

    def test_the_third_failure_escalates(self):
        for _ in range(3):
            payload, _ = self.patch("nowhere at all", "x")

        self.assertIn("failure #3", payload["_hint"])

    def test_an_edit_already_present_is_a_no_op(self):
        self.write("x = compute_value(1)\n")
        payload, changed = self.patch("y = other_thing(2)", "x = compute_value(1)")

        self.assertTrue(payload["no_change"])
        self.assertFalse(changed)


class WriteFile(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.state = coding_tools.new_state()

    def write_to(self, target):
        def write(content):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            Path(target).write_text(content)

        return write

    def test_creates_parents(self):
        target = str(self.root / "a/b/c.txt")
        text, ok = coding_tools.write_file(self.state, "a/b/c.txt", target, "x\n",
                                           write=self.write_to(target))

        self.assertTrue(ok)
        self.assertTrue(json.loads(text)["dirs_created"])

    def test_invalid_json_is_refused_before_disk(self):
        target = str(self.root / "c.json")
        text, ok = coding_tools.write_file(self.state, "c.json", target, "{bad",
                                           write=self.write_to(target))

        self.assertFalse(ok)
        self.assertFalse(os.path.exists(target))

    def test_read_file_display_text_is_refused(self):
        target = str(self.root / "c.txt")
        text, ok = coding_tools.write_file(self.state, "c.txt", target, "1|a\n2|b\n",
                                           write=self.write_to(target))

        self.assertFalse(ok)
        self.assertIn("read_file display text", json.loads(text)["error"])


class TerminalResult(unittest.TestCase):
    def test_hermes_fields(self):
        payload = json.loads(coding_tools.terminal_result("make", "\x1b[31mok\x1b[0m\n", 0))

        self.assertEqual(payload, {"output": "ok", "exit_code": 0, "error": None})

    def test_exit_code_meaning(self):
        payload = json.loads(coding_tools.terminal_result("grep x f", "", 1))

        self.assertEqual(payload["exit_code_meaning"], "No matches found (not an error)")

    def test_workspace_block(self):
        root = tempfile.mkdtemp()
        import subprocess
        subprocess.run(["git", "init", "-q", root], check=True)
        block = coding_tools.workspace_block(root)

        self.assertIn("- Status: clean", block)


class TerminalSession(unittest.TestCase):
    """Hermes' terminal session, reproduced inside one sandboxed script."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.session = {"cwd": None, "env": ""}

    def run_command(self, command, timeout=5):
        script = coding_tools.session_script(command, self.session["cwd"],
                                             self.session["env"], timeout)
        done = subprocess.run(["bash", "-c", script], cwd=self.root,
                              capture_output=True, text=True)
        text, code, timed_out = coding_tools.terminal_from_outcome(
            command, CommandOutcome("ok", done.stdout + done.stderr, done.returncode),
            self.session, self.root, timeout)
        return json.loads(text), code, timed_out

    def test_cwd_and_exports_carry_over(self):
        first, _, _ = self.run_command("mkdir -p sub && cd sub && export G=hi && echo ok")
        second, _, _ = self.run_command("echo $G; pwd")

        self.assertEqual(first["output"], "ok")
        self.assertTrue(first["cwd"].endswith("/sub"))
        self.assertEqual(second["output"].split("\n")[0], "hi")
        self.assertNotIn("cwd", second)

    def test_stderr_is_merged_and_failure_keeps_state(self):
        result, code, _ = self.run_command("export K=1; echo out; echo err >&2; false")
        after, _, _ = self.run_command("echo $K")

        self.assertEqual((result["output"], code), ("out\nerr", 1))
        self.assertEqual(after["output"], "1")

    def test_timeout_keeps_partial_output_and_drops_state(self):
        result, code, timed_out = self.run_command("export T=1; echo partial; sleep 3", 1)
        after, _, _ = self.run_command("echo [$T]")

        self.assertTrue(timed_out)
        self.assertEqual(code, 124)
        self.assertEqual(result["output"], "partial\n\n[Command timed out after 1s]")
        self.assertIsNone(result["error"])
        self.assertIn("Exit 124", result["hint"])
        self.assertEqual(after["output"], "[]")

    def test_long_output_keeps_head_and_tail(self):
        result, _, _ = self.run_command("seq 1 30000")

        self.assertTrue(result["output"].startswith("1\n2\n"))
        self.assertTrue(result["output"].endswith("30000"))
        self.assertIn("[OUTPUT TRUNCATED - ", result["output"])

    def test_exit_code_meaning(self):
        result, _, _ = self.run_command("grep nothing /dev/null")

        self.assertEqual(result["exit_code_meaning"], "No matches found (not an error)")

    def test_timeout_above_the_foreground_maximum_is_refused(self):
        self.assertIsNone(coding_tools.terminal_timeout(601))
        self.assertEqual(coding_tools.terminal_timeout(None), 180)

    def test_a_refused_command_reads_as_blocked(self):
        text, code, _ = coding_tools.terminal_from_outcome(
            "rm -rf x", CommandOutcome("denied", "", -1, "refused: rm is not allowed"),
            self.session, self.root, 5)

        self.assertEqual(json.loads(text)["status"], "blocked")
        self.assertEqual(code, -1)


class SchemasAreHermesWithListedEdits(unittest.TestCase):
    def test_every_edit_is_recorded(self):
        data = json.loads((ROOT / "agent" / "tool_schemas.json").read_text())

        self.assertIn("Hermes Agent 0cbc6e37", data["_source"])
        self.assertTrue(all({"tool", "removed", "replaced_by"} <= set(edit)
                            for edit in data["_edits"]))
        self.assertNotIn("background", data["terminal"]["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
