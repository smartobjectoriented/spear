"""Where the shipped content lives, and who decides.

rules.d/, skills/ and benches/ are content, not code. A deployment's own rules,
its learned skills and the bench that rates it are exactly the material that
does not belong in a public tree -- and pinning them to APP_DIR meant a copy
kept outside the checkout was simply unreachable, with no diagnostic: the rules
loader returns "" for a missing directory and the skill library returns [], so
a session ran with none of either and said nothing.

Each directory now answers to an environment variable. These tests fix both
halves of that contract: the default is still the in-tree directory, so a plain
checkout is unaffected, and no default may point outside the checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_chat
import skill_library

DIRS = (("SPEAR_RULES_DIR", "rules.d"),
        ("SPEAR_SKILLS_DIR", "skills"),
        ("SPEAR_BENCH_DIR", "benches"))


class TheDefaultIsTheCheckout(unittest.TestCase):
    """Nothing moves unless asked, and nothing points out of the tree."""

    def setUp(self):
        self.env = dict(os.environ)

        for var, _ in DIRS:
            os.environ.pop(var, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def test_an_unset_variable_gives_the_in_tree_directory(self):
        for var, name in DIRS:
            with self.subTest(var=var):
                self.assertEqual(rag_chat.resource_dir(var, name),
                                 f"{rag_chat.APP_DIR}/{name}")

    def test_an_empty_variable_is_not_a_request_for_the_filesystem_root(self):
        """`export SPEAR_RULES_DIR=` is a mistake, not an instruction."""
        for var, name in DIRS:
            with self.subTest(var=var):
                os.environ[var] = ""
                self.assertEqual(rag_chat.resource_dir(var, name),
                                 f"{rag_chat.APP_DIR}/{name}")

    def test_no_default_leaves_the_checkout(self):
        """A default that named a private directory would publish its path."""
        for var, name in DIRS:
            with self.subTest(var=var):
                resolved = Path(rag_chat.resource_dir(var, name)).resolve()
                self.assertEqual(resolved.parent,
                                 Path(rag_chat.APP_DIR).resolve())


class AnOverrideIsHonoured(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def test_the_variable_wins_over_the_default(self):
        for var, name in DIRS:
            with self.subTest(var=var):
                os.environ[var] = "/somewhere/else"
                self.assertEqual(rag_chat.resource_dir(var, name),
                                 "/somewhere/else")


class TheContentIsReadFromWhereItWasPointed(unittest.TestCase):
    """The variable is worth nothing if the loaders do not follow it."""

    def test_rules_are_read_from_the_configured_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "10-first.md").write_text("never rewrite a header\n")
            Path(tmp, "20-second.md").write_text("comments in english\n")
            Path(tmp, "notes.txt").write_text("ignored: not markdown\n")

            with _pointed_at(rag_chat, RULES_DIR=tmp,
                             LEARNED_RULES_FILE=os.path.join(tmp, "absent.md")):
                rules = rag_chat.load_rules()

        self.assertIn("never rewrite a header", rules)
        self.assertIn("comments in english", rules)
        self.assertNotIn("ignored", rules)
        # the NN- ordering prefix is stripped from the title, as in-tree
        self.assertIn("## Rule: first", rules)

    def test_skills_are_read_from_the_configured_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "fix-the-build.md").write_text(
                "---\nname: fix-the-build\ndescription: how\n---\n\nrun make\n")

            library = skill_library.load_library(tmp)

        self.assertEqual([skill.name for skill in library], ["fix-the-build"])

    def test_a_bare_bench_name_resolves_under_the_configured_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp, "acceptance.sh")
            bench.write_text("#!/bin/sh\nexit 0\n")

            with _pointed_at(rag_chat, BENCH_DIR=tmp,
                             project_bench=lambda: "acceptance.sh"):
                self.assertEqual(rag_chat.bench_command(), str(bench))

    def test_an_unresolvable_bench_name_is_passed_through_unchanged(self):
        """A project may declare a command rather than a shipped script."""
        with tempfile.TemporaryDirectory() as tmp:
            with _pointed_at(rag_chat, BENCH_DIR=tmp,
                             project_bench=lambda: "make check"):
                self.assertEqual(rag_chat.bench_command(), "make check")


class TheOverrideSurvivesImport(unittest.TestCase):
    """The constants are module-level, so the only honest proof is an import.

    Patching the attribute afterwards would pass even if the environment were
    never read -- which is precisely the bug this replaces.
    """

    def test_a_fresh_interpreter_picks_the_environment_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ,
                       SPEAR_RULES_DIR=os.path.join(tmp, "r"),
                       SPEAR_SKILLS_DIR=os.path.join(tmp, "s"),
                       SPEAR_BENCH_DIR=os.path.join(tmp, "b"),
                       SPEAR_STATE_DIR=os.path.join(tmp, "state"))
            out = subprocess.run(
                [sys.executable, "-c",
                 "import rag_chat as r; "
                 "print(r.RULES_DIR, r.SKILLS_DIR, r.BENCH_DIR, "
                 "r.SHIPPED_CORPUS_RULES)"],
                cwd=str(ROOT), env=env, capture_output=True, text=True)

            self.assertEqual(out.returncode, 0, out.stderr)
            rules, skills, bench, corpora = out.stdout.split()

            self.assertEqual(rules, os.path.join(tmp, "r"))
            self.assertEqual(skills, os.path.join(tmp, "s"))
            self.assertEqual(bench, os.path.join(tmp, "b"))
            self.assertEqual(corpora, os.path.join(tmp, "r", "corpora"))


class _pointed_at:
    """Temporarily rebind module attributes, restoring them afterwards."""

    def __init__(self, module, **attrs):
        self.module = module
        self.attrs = attrs
        self.saved = {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.saved[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self.module

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(self.module, name, value)
        return False


if __name__ == "__main__":
    unittest.main()
