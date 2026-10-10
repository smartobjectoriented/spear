"""A project family is what the registry says, and nothing else.

A registered project that declares "families": ["alpha"] is given the rules,
skills and capabilities scoped to `family alpha`, beside those scoped to its
own name. Membership is never inferred, an unregistered tree has none, and a
registry name that would read as a family is refused.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cli import corpus_registry
from context import context_selection as sel
from context import context_sources, skill_library, workspace_context
from harness import capabilities


def workspace(name="tool", families=(), registered=True):
    spec = {"kind": "generic", "families": list(families)}

    return workspace_context.from_session(
        project=name, spec=spec, registered=registered, project_root="/tmp/x",
        corpus_root="/tmp/x")


def rule(scope):
    text = f"---\nscope: {scope}\n---\nKeep the patchset in step."

    return sel.Candidate("rule:r", sel.SourceType.PROJECT_RULE, "rules.d/r.md", text,
                         scope=context_sources.parse_rule(text)[0])


def skill(*names):
    return context_sources.skill_candidates([("s", "Procedure.", names)])[0]


def selected(ws, candidate):
    result = sel.DeterministicContextSelector().select(
        ws, sel.IMPLEMENTATION, [candidate], request="change it")

    return candidate.item_id in {item.item_id for item in result.selected}


class FamilyScopes(unittest.TestCase):
    def test_a_member_gets_the_family_rule(self):
        self.assertTrue(selected(workspace(families=("alpha",)), rule("family alpha")))

    def test_a_member_gets_the_family_skill(self):
        self.assertTrue(selected(workspace(families=("alpha",)), skill("family:alpha")))

    def test_a_registered_project_outside_the_family_does_not(self):
        for candidate in (rule("family alpha"), skill("family:alpha")):
            with self.subTest(candidate=candidate.item_id):
                self.assertFalse(selected(workspace(families=("beta",)), candidate))
                self.assertFalse(selected(workspace(), candidate))

    def test_an_unregistered_tree_inherits_no_family(self):
        ws = workspace(families=("alpha",), registered=False)

        self.assertEqual(ws.project_families, ())
        self.assertFalse(selected(ws, rule("family alpha")))
        self.assertFalse(selected(ws, skill("family:alpha")))

    def test_an_exact_project_scope_is_unchanged(self):
        member = workspace(name="tool", families=("alpha",))

        self.assertTrue(selected(member, rule("corpus tool")))
        self.assertTrue(selected(member, skill("tool")))
        self.assertFalse(selected(workspace(name="other", families=("alpha",)),
                                  rule("corpus tool")))

    def test_two_families_are_independent(self):
        both = workspace(families=("alpha", "beta"))
        alpha = workspace(families=("alpha",))

        self.assertTrue(selected(both, rule("family beta")))
        self.assertFalse(selected(alpha, rule("family beta")))
        self.assertTrue(selected(alpha, rule("family alpha, beta")))

    def test_a_family_word_is_not_a_project_name(self):
        """`corpus alpha` names a project called alpha, not the family."""
        self.assertFalse(selected(workspace(families=("alpha",)), rule("corpus alpha")))

    def test_the_skill_listing_agrees(self):
        parsed = skill_library.parse_skill("s", "---\nscope: [family:alpha]\n---\nBody.")

        self.assertTrue(skill_library.applies_to(parsed, project="tool", families=("alpha",)))
        self.assertFalse(skill_library.applies_to(parsed, project="tool"))


class FamilyCapabilities(unittest.TestCase):
    def test_a_capability_follows_the_same_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "capabilities.json")
            path.write_text(json.dumps({"providers": [
                {"id": "lookup", "command": ["true"], "scope": "family alpha"}]}))
            configs, problems = capabilities.load_configs(str(path))

        self.assertEqual(problems, [])
        candidate = sel.Candidate("capability:lookup", sel.SourceType.EXTERNAL_CAPABILITY,
                                  "capabilities.json:lookup", "", scope=configs[0].scope,
                                  tasks=configs[0].tasks, bucket="capabilities")

        self.assertTrue(selected(workspace(families=("alpha",)), candidate))
        self.assertFalse(selected(workspace(families=("beta",)), candidate))
        self.assertFalse(selected(workspace(families=("alpha",), registered=False), candidate))


class TheRegistryDecides(unittest.TestCase):
    def load(self, registry):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "projects.json")
            path.write_text(json.dumps(registry))
            corpus_registry._WARNED.clear()
            stderr = io.StringIO()

            with mock.patch.object(corpus_registry, "PROJECTS_FILE", str(path)), \
                    contextlib.redirect_stderr(stderr):
                return corpus_registry.load_projects(), stderr.getvalue()

    def test_a_name_in_the_family_namespace_is_refused(self):
        projects, warning = self.load({"family:alpha": {"path": "/tmp/a"},
                                       "tool": {"path": "/tmp/b", "families": ["alpha"]}})

        self.assertEqual(list(projects), ["tool"])
        self.assertIn("reserved for project families", warning)
        self.assertTrue(workspace_context.reserved_project_id("Family:Alpha"))

    def test_unusable_family_values_are_reported_and_ignored(self):
        projects, warning = self.load({"tool": {"path": "/tmp/b",
                                                "families": ["alpha", "a:b", 3]}})

        self.assertIn("unusable families", warning)
        self.assertEqual(workspace_context.project_families(projects["tool"]),
                         (("alpha",), ("a:b", "3")))

    def test_membership_is_only_what_the_entry_declares(self):
        self.assertEqual(workspace_context.project_families({"kind": "generic"}), ((), ()))
        self.assertEqual(workspace(name="alpha").project_families, ())
        self.assertNotIn("family:alpha", workspace(name="alpha").names)


if __name__ == "__main__":
    unittest.main()
