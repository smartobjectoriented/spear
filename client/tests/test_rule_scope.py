"""A rule reaches a session only when its declared scope matches it.

Every rules.d file was injected into every session. One tree's build system
-- its platforms, its build commands, its patch-regeneration script -- had
been written into those files, and sessions on an unrelated tree received it
as standing instructions: every benchmark run there tried the other tree's
build commands. Rules now carry an optional scope header; without one a rule
is global, with one it is injected only where the scope matches, and a scope
SPEAR cannot read is injected nowhere.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import session_workspace, turn_context


class RuleScope(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.rules = base / "rules.d"
        self.rules.mkdir()
        self.tree_a = base / "work" / "alpha"
        self.tree_b = base / "work" / "beta"
        (self.tree_a / "sub" / "deep").mkdir(parents=True)
        self.tree_b.mkdir(parents=True)

        self.write("10-generic.md", "Verify a change before claiming it works.")
        self.write("20-alpha.md", "---\nscope: corpus alpha\n---\nRun alpha-build.sh.")
        self.write("30-alpha-sub.md",
                   f"---\nscope: path {self.tree_a / 'sub'}\n---\nSub-tree rule.")
        self.write("40-unreadable.md", "---\nowner: someone\n---\nTree-specific text.")
        self.write("50-gamma.md", "---\nscope: corpus gamma, delta\n---\nGamma rule.")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        (self.rules / name).write_text(text)

    def rules_for(self, project, root, corpora=()):
        with patch.multiple(turn_context, RULES_DIR=str(self.rules),
                            LEARNED_RULES_FILE=str(self.rules / "absent")), \
                patch.multiple(session_workspace, PROJECT=project,
                               PROJECT_ROOT=str(root), CORPUS_ROOT=str(root),
                               PROJECT_SPEC={"corpora": list(corpora)}):
            return turn_context.load_rules()

    def test_a_a_workspace_receives_its_own_project_rule(self):
        self.assertIn("alpha-build.sh", self.rules_for("alpha", self.tree_a))

    def test_b_another_workspace_does_not(self):
        text = self.rules_for("beta", self.tree_b)

        self.assertNotIn("alpha-build.sh", text)
        self.assertNotIn("Rule: alpha", text)

    def test_c_a_global_rule_applies_to_both(self):
        for project, root in (("alpha", self.tree_a), ("beta", self.tree_b)):
            with self.subTest(project=project):
                self.assertIn("Verify a change", self.rules_for(project, root))

    def test_d_a_subtree_rule_applies_only_under_its_subtree(self):
        self.assertIn("Sub-tree rule", self.rules_for("alpha", self.tree_a / "sub" / "deep"))
        self.assertNotIn("Sub-tree rule", self.rules_for("alpha", self.tree_a))
        self.assertNotIn("Sub-tree rule", self.rules_for("beta", self.tree_b))

    def test_e_a_scope_that_cannot_be_read_is_injected_nowhere(self):
        for project, root in (("alpha", self.tree_a), ("beta", self.tree_b)):
            with self.subTest(project=project):
                self.assertNotIn("Tree-specific text", self.rules_for(project, root))

    def test_e_a_federated_corpus_counts_as_the_sessions_own(self):
        """An umbrella session over gamma's parts is gamma's session."""
        self.assertIn("Gamma rule", self.rules_for("workspace:umbrella",
                                                   self.tree_b, corpora=("gamma",)))
        self.assertNotIn("Gamma rule", self.rules_for("beta", self.tree_b))

    def test_f_order_is_the_file_order_whatever_applies(self):
        text = self.rules_for("alpha", self.tree_a / "sub")
        titles = [line for line in text.splitlines() if line.startswith("## Rule:")]

        self.assertEqual(titles, ["## Rule: generic", "## Rule: alpha", "## Rule: alpha-sub"])

    def test_f_the_header_never_reaches_the_model(self):
        text = self.rules_for("alpha", self.tree_a)

        self.assertNotIn("scope:", text)
        self.assertNotIn("---", text)


class ScopeHeaderSyntax(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(turn_context.rule_scope("plain")[0], ("global",))
        self.assertEqual(turn_context.rule_scope("---\nscope: global\n---\nx")[0], ("global",))
        self.assertEqual(turn_context.rule_scope("---\nscope: corpus a, b\n---\nx")[0],
                         ("corpus", "a", "b"))
        self.assertEqual(turn_context.rule_scope("---\nscope: path /x/y\n---\nx")[0],
                         ("path", "/x/y"))

    def test_unreadable_forms_fail_closed(self):
        for header in ("scope: corpus", "scope: tree x", "scope: global extra",
                       "other: value"):
            with self.subTest(header=header):
                self.assertIsNone(turn_context.rule_scope(f"---\n{header}\n---\nx")[0])
                self.assertFalse(turn_context.rule_applies(
                    turn_context.rule_scope(f"---\n{header}\n---\nx")[0]))


class TheShippedRulesStayGeneric(unittest.TestCase):
    """G: the public tree ships no rule at all, scoped or not -- a
    deployment's rules live in its own directory (SPEAR_RULES_DIR)."""

    def test_the_public_rules_directory_holds_no_rule(self):
        shipped = ROOT / "rules.d"

        if shipped.is_dir():
            self.assertEqual(sorted(p.name for p in shipped.glob("*.md")), [])


if __name__ == "__main__":
    unittest.main()
