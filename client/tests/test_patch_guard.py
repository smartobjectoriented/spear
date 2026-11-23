"""The patch tool never rewrites literal text it only approximately matched.

Three corruptions reproduced from FT-3A runs (ft4-design/REPORT.md §10): added
lines of a .patch file sent without their '+', a context block whose ' '
marker line was sent empty, and diff-prefixed text matched against plain C.
Each one used to be accepted by a fuzzy strategy and written as a successful
edit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import patch_guard
from agent import tools as coding_tools

# Built line by line: the ' ' context line is the point of one reproduction,
# and an editor stripping trailing blanks would silently turn it into ''.

PATCH = "\n".join([
    "--- a/drivers/foo.c",
    "+++ b/drivers/foo.c",
    "@@ -10,6 +10,8 @@ static int foo_probe(struct device *dev)",
    " {",
    " \tint ret;",
    " ",
    "+\tret = foo_enable_clock(dev);",
    "+\tif (ret)",
    "+\t\treturn ret;",
    " \treturn foo_register(dev);",
    " }",
    ""])

SOURCE = """static int foo_probe(struct device *dev)
{
\tint ret;

\tret = foo_enable_clock(dev);
\tif (ret)
\t\treturn ret;
\treturn foo_register(dev);
}
"""


class Reproductions(unittest.TestCase):
    """The three FT-3A corruptions are refused and leave the file as it was."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.state = coding_tools.new_state()

    def patch(self, name, content, old, new):
        path = self.root / name
        path.write_text(content)
        text, changed = coding_tools.patch(self.state, name, str(path), old, new,
                                           write=lambda text: path.write_text(text) and None)
        return json.loads(text), changed, path.read_text()

    def test_added_lines_without_their_plus_do_not_match_a_patch(self):
        payload, changed, after = self.patch(
            "0001-foo.patch", PATCH,
            "\tret = foo_enable_clock(dev);\n\tif (ret)\n\t\treturn ret;",
            "\tret = foo_enable_clock(dev);\n\tif (ret < 0)\n\t\treturn ret;")

        self.assertFalse(changed)
        self.assertFalse(payload["success"])
        self.assertIn("unified diff", payload["error"])
        self.assertEqual(after, PATCH)

    def test_a_context_line_sent_empty_does_not_match_a_patch(self):
        payload, changed, after = self.patch(
            "0001-foo.patch", PATCH,
            "\tint ret;\n\n+\tret = foo_enable_clock(dev);",
            "\tint ret = 0;\n\n+\tret = foo_enable_clock(dev);")

        self.assertFalse(changed)
        self.assertIn("unified diff", payload["error"])
        self.assertEqual(after, PATCH)

    def test_diff_prefixed_text_does_not_match_plain_c(self):
        payload, changed, after = self.patch(
            "foo.c", SOURCE,
            "+\tret = foo_enable_clock(dev);\n+\tif (ret)",
            "+\tret = foo_enable_clock(dev);\n+\tif (ret < 0)")

        self.assertFalse(changed)
        self.assertIn("leading '+' or '-' is text", payload["error"])
        self.assertEqual(after, SOURCE)


class StillWorks(unittest.TestCase):
    """What the guard must not take away."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.state = coding_tools.new_state()

    def patch(self, name, content, old, new, replace_all=False):
        path = self.root / name
        path.write_text(content)
        text, changed = coding_tools.patch(self.state, name, str(path), old, new,
                                           replace_all=replace_all, write=lambda text: path.write_text(text) and None)
        return json.loads(text), changed, path.read_text()

    def test_an_exact_edit_of_a_patch_file_is_applied(self):
        payload, changed, after = self.patch(
            "0001-foo.patch", PATCH, "+\tif (ret)\n", "+\tif (ret < 0)\n")

        self.assertTrue(changed, payload)
        self.assertIn("+\tif (ret < 0)\n", after)

    def test_an_indentation_mismatch_in_source_is_still_tolerated(self):
        payload, changed, after = self.patch(
            "foo.c", SOURCE, "    if (ret)\n        return ret;", "    if (ret < 0)\n        return ret;")

        self.assertTrue(changed, payload)
        self.assertIn("\tif (ret < 0)\n", after)

    def test_a_negative_literal_is_not_a_marker(self):
        payload, changed, after = self.patch(
            "f.py", "def f():\n    return -1\n", "  return -1", "  return -2")

        self.assertTrue(changed, payload)
        self.assertEqual("def f():\n    return -2\n", after)

    def test_an_ambiguous_old_string_changes_nothing(self):
        content = "x = 1\ny = 2\nx = 1\n"
        payload, changed, after = self.patch("f.py", content, "x = 1", "x = 3")

        self.assertFalse(changed)
        self.assertIn("matches", payload["error"])
        self.assertEqual(after, content)


class Detection(unittest.TestCase):
    def test_a_unified_diff_needs_both_a_hunk_and_a_file_header(self):
        self.assertTrue(patch_guard.is_unified_diff(PATCH))
        self.assertFalse(patch_guard.is_unified_diff(SOURCE))
        self.assertFalse(patch_guard.is_unified_diff("@@ -1 +1 @@ only a hunk\n"))
        self.assertFalse(patch_guard.is_unified_diff("SRC_URI += \"file://0001-x.patch\"\n"))


if __name__ == "__main__":
    unittest.main()
