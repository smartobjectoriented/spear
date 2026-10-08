"""The coding tools render exactly what Hermes Agent's do.

fixtures/hermes_file_tools.json holds the strings Hermes v0.21.0 (0cbc6e37)
itself returned to the model, on the files and in the call sequences rebuilt
here. The whole string is compared -- field presence, order, wording, line
numbering -- not its meaning: deterministic paired runs (same request, same
sampling seed) showed the port and Hermes diverging on exactly such details,
and from that point on the model sees two different conversations.
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

from agent import tools

FIXTURE = json.loads((ROOT / "tests/fixtures/hermes_file_tools.json").read_text())["results"]


def writer(path):
    """What the host's write port does: parent directories, then the file."""
    def write(content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return None
    return write


class HermesToolParity(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        d = Path(self.root)
        (d / "dir").mkdir()
        (d / "nl.txt").write_text("alpha\nbeta\ngamma\n")
        (d / "nonl.txt").write_text("alpha\nbeta\ngamma")
        (d / "empty.txt").write_text("")
        (d / "ten.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)))
        os.mkfifo(d / "pipe")
        (d / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00data")
        (d / "noext").write_bytes(b"ELF\x00\x00\x00\x01\x02\x03\x00\x00" * 20)
        (d / ".hidden").mkdir()
        (d / ".hidden/secret.cfg").write_text("needle_only_hidden = 1\n")
        (d / "main.c").write_text("int main(void)\n{\n\treturn 0;\n}\n")
        for i in range(1, 5):
            (d / f"edit{i}.txt").write_text("first line\nsecond line\nthird line\n")
        (d / "over_partial.txt").write_text("one\ntwo\nthree\nfour\n")
        (d / "over_stale.txt").write_text("one\ntwo\n")
        (d / "over_unread.txt").write_text("one\ntwo\n")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def expected(self, key):
        return FIXTURE[key].replace("<DIR>", self.root)

    def path(self, name):
        return os.path.join(self.root, name)

    def read(self, state, display, offset=1, limit=2000):
        resolved = display if os.path.isabs(display) else self.path(display)
        return tools.read_file(state, display, resolved, offset, limit)

    def patch(self, state, name):
        return tools.patch(state, self.path(name), self.path(name), "second line",
                           "2nd line", write=writer(self.path(name)))[0]

    def write(self, state, name, content):
        return tools.write_file(state, self.path(name), self.path(name), content,
                                write=writer(self.path(name)))[0]

    def make_stale(self, name):
        st = os.stat(self.path(name))
        os.utime(self.path(name), (st.st_atime, st.st_mtime + 10))

    # ------------------------------------------------------------ read_file

    def test_read_file(self):
        for key in [k for k in FIXTURE if k.startswith("read:")]:
            _, path, offset, limit = key.split(":")
            with self.subTest(key=key):
                self.assertEqual(self.read(tools.new_state(), path, int(offset), int(limit)),
                                 self.expected(key))

    # --------------------------------------------------------- search_files

    def test_zero_match_probe_runs_for_every_output_mode(self):
        for mode in ("content", "files_only", "count"):
            for case, pattern in (("hidden", "needle_only_hidden"),
                                  ("none", "nothing_matches_this_anywhere")):
                key = f"search:{case}:{mode}"
                with self.subTest(key=key):
                    got = tools.search(tools.new_state(), self.root, ".", self.root, pattern,
                                       target="content", output_mode=mode)
                    self.assertEqual(got, self.expected(key))

    # ---------------------------------------------------------------- patch

    def test_patch_after_a_full_read(self):
        state = tools.new_state()
        self.read(state, self.path("edit1.txt"))
        self.assertEqual(self.patch(state, "edit1.txt"), self.expected("patch:full_read"))

    def test_patch_after_a_partial_read_warns(self):
        state = tools.new_state()
        self.read(state, self.path("edit2.txt"), 2, 1)
        self.assertEqual(self.patch(state, "edit2.txt"), self.expected("patch:partial_read"))

    def test_patch_after_an_external_change_warns(self):
        state = tools.new_state()
        self.read(state, self.path("edit3.txt"))
        self.make_stale("edit3.txt")
        self.assertEqual(self.patch(state, "edit3.txt"), self.expected("patch:stale_read"))

    def test_patch_without_a_read(self):
        self.assertEqual(self.patch(tools.new_state(), "edit4.txt"),
                         self.expected("patch:no_read"))

    # ----------------------------------------------------------- write_file

    def test_write_creating_parent_directories(self):
        self.assertEqual(self.write(tools.new_state(), "new/sub/created.txt", "hello\n"),
                         self.expected("write:new_parent_dirs"))

    def test_write_after_a_partial_read_warns(self):
        state = tools.new_state()
        self.read(state, self.path("over_partial.txt"), 2, 2)
        self.assertEqual(self.write(state, "over_partial.txt", "replaced\n"),
                         self.expected("write:partial_read"))

    def test_write_after_an_external_change_warns(self):
        state = tools.new_state()
        self.read(state, self.path("over_stale.txt"))
        self.make_stale("over_stale.txt")
        self.assertEqual(self.write(state, "over_stale.txt", "replaced\n"),
                         self.expected("write:stale_read"))

    def test_write_over_an_unread_file(self):
        self.assertEqual(self.write(tools.new_state(), "over_unread.txt", "replaced\n"),
                         self.expected("write:unread_existing"))

    def test_write_into_an_existing_directory_still_reports_dirs_created(self):
        self.assertEqual(self.write(tools.new_state(), "dir/inside.txt", "x\n"),
                         self.expected("write:existing_dir"))

    def test_every_captured_case_has_a_test(self):
        covered = {"patch:full_read", "patch:partial_read", "patch:stale_read",
                   "patch:no_read", "write:new_parent_dirs", "write:partial_read",
                   "write:stale_read", "write:unread_existing", "write:existing_dir"}
        for key in FIXTURE:
            self.assertTrue(key.startswith(("read:", "search:")) or key in covered, key)


if __name__ == "__main__":
    unittest.main()
