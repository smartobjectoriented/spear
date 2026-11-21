"""Rules and skills exported for Claude Code: what is written, and for whom."""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import claude_export  # noqa: E402
from context import context_sources, skill_library  # noqa: E402

RULES = {
    "10-global.md": "---\nscope: global\npaths: doc/**, src/**\n---\nAlways build before you claim.",
    "15-family.md": "---\nscope: family build\n---\nThe framework rule.",
    "17-bare.md": "Nobody scoped this one.",
    "20-alpha.md": "---\nscope: corpus alpha, beta\n---\nAlpha builds with make.",
    "30-gamma.md": "---\nscope: corpus gamma\n---\nGamma only.",
    "40-path.md": "---\nscope: path /srv/tree\n---\nA tree of this machine.",
    "50-broken.md": "---\nowner: nobody\n---\nClaims a scope it does not state.",
    "notes.txt": "not a rule",
}


def skill(name, text):
    return skill_library.parse_skill(name, text)


class ExportRules(unittest.TestCase):
    def test_only_what_applies_to_the_corpus_is_written(self):
        files = claude_export.export_rules(RULES, {"alpha"})

        self.assertEqual(sorted(files), [".claude/rules/spear-alpha.md",
                                         ".claude/rules/spear-global.md"])
        self.assertEqual(files[".claude/rules/spear-alpha.md"],
                         "# alpha\n\nAlpha builds with make.\n")
        self.assertTrue(files[".claude/rules/spear-global.md"].startswith(
            '---\npaths:\n  - "doc/**"\n  - "src/**"\n---\n\n# global\n'))

    def test_no_corpus_leaves_what_applies_everywhere(self):
        files = claude_export.export_rules(RULES, set())

        self.assertEqual(sorted(files), [".claude/rules/spear-global.md"])

    def test_a_family_rule_follows_the_registry_entry(self):
        names = claude_export.corpus_names({"alpha": {}}, "alpha", ("build",))

        self.assertIn(".claude/rules/spear-family.md",
                      claude_export.export_rules(RULES, names))

    def test_a_federated_corpus_brings_its_rules(self):
        names = claude_export.corpus_names(
            {"top": {"corpora": ["gamma"]}, "gamma": {}}, "top")
        files = claude_export.export_rules(RULES, names)

        self.assertIn(".claude/rules/spear-gamma.md", files)
        self.assertNotIn(".claude/rules/spear-alpha.md", files)


class ExportSkills(unittest.TestCase):
    def test_scope_selects_and_the_description_is_the_summary(self):
        skills = [
            skill("add-thing", "---\nname: add-thing\nscope: [alpha]\n---\n"
                               "To add a thing: do \"this\".\n\n1. Step."),
            skill("generic", "---\nname: generic\nscope: [any]\n---\n## Procedure\n\nGo."),
            skill("other", "---\nname: other\nscope: [gamma]\n---\nNot here."),
        ]
        files = claude_export.export_skills(skills, {"alpha"})

        self.assertEqual(sorted(files), [".claude/skills/add-thing/SKILL.md",
                                         ".claude/skills/generic/SKILL.md"])

        text = files[".claude/skills/add-thing/SKILL.md"]

        self.assertTrue(text.startswith(
            '---\nname: add-thing\ndescription: "To add a thing: do \\"this\\"."\n---\n\n'))
        self.assertIn("1. Step.", text)
        self.assertNotIn("scope:", text)

    def test_a_family_scope_follows_the_registry_entry(self):
        skills = [skill("fam", "---\nname: fam\nscope: [family:build]\n---\nBuild.")]

        self.assertEqual(claude_export.export_skills(skills, {"alpha"}), {})

        names = claude_export.corpus_names({"alpha": {}}, "alpha", ("build",))

        self.assertIn(".claude/skills/fam/SKILL.md",
                      claude_export.export_skills(skills, names))

    def test_a_long_description_is_cut(self):
        files = claude_export.export_skills(
            [skill("long", "---\nname: long\nscope: [any]\n---\n" + "word " * 200)], set())
        line = files[".claude/skills/long/SKILL.md"].splitlines()[2]

        self.assertLessEqual(len(line), len('description: ""') + claude_export.DESCRIPTION_MAX)


class WhatATurnStartsFromBesideTheRules(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.tree = os.path.join(self.dir, "tree")
        self.shipped = os.path.join(self.dir, "corpora")
        os.makedirs(self.tree)
        os.makedirs(self.shipped)

    def write(self, path, text):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_the_map_is_the_trees_own_before_the_shipped_one(self):
        self.write(os.path.join(self.shipped, "alpha.md"), "Shipped map.")

        self.assertEqual(
            claude_export.corpus_map(self.tree, self.tree, "alpha", self.shipped),
            ("alpha, shipped", "Shipped map."))

        self.write(os.path.join(self.tree, claude_export.CORPUS_RULES_FILE), "Its own map.")

        self.assertEqual(
            claude_export.corpus_map(self.tree, self.tree, "alpha", self.shipped),
            ("tree", "Its own map."))

    def test_a_tree_without_a_map_gets_none(self):
        self.assertEqual(claude_export.corpus_map(self.tree, "", None, self.shipped), ("", ""))

    def test_the_map_file_is_the_one_a_turn_reads(self):
        from cli import turn_context

        self.assertEqual(claude_export.CORPUS_RULES_FILE, turn_context.CORPUS_RULES_FILE)

    def test_each_source_is_a_rule_file_holding_the_text_a_turn_is_shown(self):
        learned = os.path.join(self.dir, "rules-learned.md")
        self.write(learned, "Never push to main.")
        found = context_sources.learned_candidate(learned)

        files = claude_export.export_session((
            ("spear.learned.md", found),
            ("spear.corpus-map.md", context_sources.corpus_rule_candidates([]))))

        self.assertEqual(list(files), [".claude/rules/spear.learned.md"])
        self.assertEqual(files[".claude/rules/spear.learned.md"], found[0].text.strip() + "\n")

    def test_the_guide_names_what_arrives_and_the_tools(self):
        text = claude_export.knowledge_base("alpha")

        for word in ("`alpha`", "Retrieved Context", "search_corpus", "Workspace knowledge",
                     "spear-knowledge"):
            self.assertIn(word, text)

        self.assertIn("this tree", claude_export.knowledge_base(None))


class Archive(unittest.TestCase):
    def test_the_tar_holds_every_file(self):
        out = io.BytesIO()
        claude_export.write_tar({".claude/rules/spear-a.md": "é\n"}, out)
        out.seek(0)

        with tarfile.open(fileobj=out) as archive:
            member = archive.getmember(".claude/rules/spear-a.md")

            self.assertEqual(archive.extractfile(member).read().decode(), "é\n")


if __name__ == "__main__":
    unittest.main()
