"""A turn is offered every skill its workspace's scope admits, and nothing
decides by resemblance to the request.

A similarity gate stood in front of the skills and missed the one a request
needed more often than it found it: the procedure for reading a failing
build's task log was not offered for a failing build because its text never
says the build system's name. Eligibility -- scope, project, task class --
is the whole decision now; the fixtures are synthetic.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import context_selection as cs
import rag_chat

SKILLS = {
    "c-readability": ("[any]", "Improve the readability of a C function."),
    "compile-failure": ("[any]", "Read the first error in the build output."),
    "layered-build": ("[firmware-a]", "Read the failing recipe's task log."),
    "needs-tool": ("[any]", "Run the frobnicator.", "requires: [no-such-command-here]"),
}


def library(directory, extra=()):
    for name, (scope, body, *more) in list(SKILLS.items()) + list(extra):
        header = "\n".join([f"name: {name}", f"scope: {scope}"] + list(more))
        Path(directory, f"{name}.md").write_text(f"---\n{header}\n---\n## Procedure\n\n{body}\n")


class Offering(unittest.TestCase):
    def setUp(self):
        self.skills = tempfile.mkdtemp()
        library(self.skills)
        self.root = tempfile.mkdtemp()

    def offered(self, request, *, project="", registered=False, scope="IMPLEMENTATION",
                binding=None, write=True):
        projects = {project: {"path": self.root}} if registered else {}

        with mock.patch.multiple(
                rag_chat, PROJECT=project or f"workspace:{self.root}", PROJECT_ROOT=self.root,
                CORPUS_ROOT=self.root, SKILLS_DIR=self.skills, RULES_DIR=tempfile.mkdtemp(),
                LEARNED_RULES_FILE="/nonexistent", CTX_LIMIT=200_000), \
                mock.patch.object(rag_chat, "load_projects", lambda: projects):
            _, selections, rendered = rag_chat.select_turn_context(
                user_input=request, turn_scope=scope, binding=binding, write=write,
                project_spec=projects.get(project, {}), project_commands=None,
                history_text="", memories="", skills=rag_chat.library_skills(), retrieval="",
                system_instructions="", system_source="", tool_guide="",
                working_directory="")

        return ({phase: sorted(item.item_id for item in selection.selected
                               if item.bucket == "skills")
                 for phase, selection in selections.items()}, rendered, selections)

    def test_every_eligible_skill_is_offered_whatever_the_request_says(self):
        for request in ("Clean up parse() and make it readable.",
                        "The image fails in do_compile.", "Rename cnt to count."):
            with self.subTest(request=request):
                offered, _, _ = self.offered(request)

                self.assertEqual(offered[cs.IMPLEMENTATION],
                                 ["skill:c-readability", "skill:compile-failure"])

    def test_a_project_skill_reaches_its_project_only(self):
        mine, _, _ = self.offered("Fix the build.", project="firmware-a", registered=True)
        other, _, _ = self.offered("Fix the build.", project="app-x", registered=True)

        self.assertIn("skill:layered-build", mine[cs.IMPLEMENTATION])
        self.assertNotIn("skill:layered-build", other[cs.IMPLEMENTATION])

    def test_generic_skills_reach_an_unregistered_tree_and_project_ones_do_not(self):
        offered, _, _ = self.offered("Fix the build.")

        self.assertEqual(offered[cs.IMPLEMENTATION],
                         ["skill:c-readability", "skill:compile-failure"])

    def test_a_skill_whose_command_is_missing_is_not_offered(self):
        self.assertNotIn("needs-tool", [name for name, _, _ in rag_chat.library_skills()])

    def test_normative_and_general_turns_are_offered_none(self):
        binding = {"standard_id": "SYNTH-1", "revision": "1"}

        for scope, bound in (("NORMATIVE", binding), ("GENERAL", None)):
            with self.subTest(scope=scope):
                offered, _, _ = self.offered("What does Rule 4-1 require?", scope=scope,
                                             binding=bound, write=False)

                self.assertTrue(all(not skills for skills in offered.values()), offered)

    def test_a_mixed_change_offers_skills_to_its_implementation_only(self):
        offered, _, _ = self.offered("Fix encode() in wire.c the way Rule 4-1 requires.",
                                     scope="MIXED", binding={"standard_id": "SYNTH-1",
                                                             "revision": "1"})

        self.assertEqual(offered[cs.MIXED_PREPASS], [])
        self.assertEqual(offered[cs.MIXED_IMPLEMENTATION],
                         ["skill:c-readability", "skill:compile-failure"])

    def test_the_model_is_told_to_follow_a_skill_only_when_it_fits(self):
        _, rendered, _ = self.offered("Rename cnt to count.")

        self.assertIn("Follow one only when it fits the task at hand.",
                      rendered[cs.IMPLEMENTATION]["skills"])

    def test_a_library_past_the_limit_says_what_it_left_out(self):
        extra = [(f"extra-{index:02d}", ("[any]", f"Procedure {index}."))
                 for index in range(rag_chat.MAX_SKILLS)]
        library(self.skills, extra)
        offered, _, selections = self.offered("Rename cnt to count.")
        left_out = [decision for decision in selections[cs.IMPLEMENTATION].decisions
                    if decision.reason.startswith("skill limit")]

        self.assertEqual(len(offered[cs.IMPLEMENTATION]), rag_chat.MAX_SKILLS)
        self.assertTrue(left_out)

    def test_offering_skills_needs_no_index(self):
        with mock.patch.object(rag_chat, "_db", side_effect=AssertionError("index opened")):
            self.assertTrue(rag_chat.library_skills())


if __name__ == "__main__":
    unittest.main()
