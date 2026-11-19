"""Rules and skills exported for Claude Code: what is written, and for whom."""

from __future__ import annotations

import io
import sys
import tarfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import claude_export  # noqa: E402
from context import skill_library  # noqa: E402

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
