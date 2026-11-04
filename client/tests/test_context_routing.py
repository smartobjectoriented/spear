"""A request in a project of a declared family is given that family's context.

The task class was read from the request's words alone, and a request that
named no subject this vocabulary knows ran as GENERAL, which is given no
project rule, skill or metadata: "Enable CONFIG_SCHEDSTATS in the kernel
configuration" in an Infrabase tree was answered without the Infrabase rules.
The workspace already said what the request was about: a registered project
that declares a family. That is the binding now, as a bound standard is for
a question about the document; the rules and skills stay selected by their
declared scopes, with no model in the decision.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from context import answer_scope
from context import context_selection as sel
from context import context_sources, workspace_context

# The project patterns the gap hid, in words of the kind that named no subject.

KERNEL_CONFIGURATION = ("Turn on CONFIG_DEBUG_PREEMPT in the board-a kernel configuration, "
                        "built in, and keep it across rebuilds.")
UBOOT_PATCH = ("On board-a, make U-Boot print the line 'Booting from mmc 0' before it reads "
               "boot.scr from that device.")
ATTACHED_TREE_UPDATE = "Make the kernel recipe also exclude the Documentation directory from updiff."

PATTERNS = {"kernel configuration": KERNEL_CONFIGURATION, "U-Boot patch": UBOOT_PATCH,
            "attached-tree update": ATTACHED_TREE_UPDATE}

RULES = {
    "10-board-packages.md": "scope: corpus board-a\n", "15-infrabase.md": "scope: family infrabase\n",
    "20-conventions.md": "scope: global\ntasks: implementation, mixed\n",
    "30-doc-build.md": "scope: global\ntasks: implementation, mixed\npaths: doc/**\n",
    "40-opencn.md": "scope: corpus opencn\n",
}

SKILLS = [("debug-infrabase-build", "Procedure.", ("family:infrabase",)),
          ("add-board-package", "Procedure.", ("board-a",)),
          ("improve-c-code-quality", "Procedure.", ("any",)),
          ("customer-protocol", "Procedure.", ("customer-x",))]

INFRABASE_CONTEXT = {"rule:infrabase", "skill:debug-infrabase-build"}


def workspace(name, families=(), corpora=(), registered=True):
    spec = {"kind": "generic", "families": list(families), "corpora": list(corpora)}

    return workspace_context.from_session(project=name, spec=spec, registered=registered,
                                          project_root=f"/tmp/{name}", corpus_root=f"/tmp/{name}")


INFRABASE = workspace("infrabase", ("infrabase",), ("board-a", "board-b"))
TEACHING_OS = workspace("teaching-os", ("infrabase",), ("teaching-os-u-boot",))
OPENCN = workspace("opencn", ("infrabase",), ("opencn-usr",))
GENERIC = workspace("tool")
UNREGISTERED = workspace("scratch", ("infrabase",), registered=False)


class Routing(unittest.TestCase):
    def setUp(self):
        rules = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, rules)

        # Written in reverse, so the order the selection keeps is the files' names.

        for name, header in reversed(RULES.items()):
            with open(os.path.join(rules, name), "w") as handle:
                handle.write(f"---\n{header}---\nThe {name} rule.\n")

        self.rules = rules

    def candidates(self, request):
        return (context_sources.rule_candidates(self.rules)
                + context_sources.skill_candidates(SKILLS)
                + [sel.Candidate("request", sel.SourceType.USER_REQUEST, "user", request,
                                 mandatory=True)])

    def route(self, ws, request):
        """(phase, selected ids) as the client decides them for one turn."""
        scope = answer_scope.of(request, project_bound=ws.project_bound)
        phase = sel.primary_phase(scope, bound=False)
        result = sel.DeterministicContextSelector().select(ws, phase, self.candidates(request),
                                                           request=request)

        return phase, [item.item_id for item in result.selected]

    def test_the_patterns_named_no_subject_and_ran_without_project_context(self):
        for name, request in PATTERNS.items():
            with self.subTest(pattern=name):
                self.assertEqual(answer_scope.of(request), answer_scope.GENERAL)
                phase = sel.primary_phase(answer_scope.of(request), bound=False)
                result = sel.DeterministicContextSelector().select(
                    INFRABASE, phase, self.candidates(request), request=request)

                self.assertFalse(INFRABASE_CONTEXT & {item.item_id for item in result.selected})

    def test_an_infrabase_project_is_given_the_infrabase_context(self):
        for ws in (INFRABASE, TEACHING_OS, OPENCN):
            for name, request in PATTERNS.items():
                with self.subTest(workspace=ws.workspace_id, pattern=name):
                    phase, ids = self.route(ws, request)

                    self.assertEqual(phase, sel.IMPLEMENTATION)
                    self.assertLessEqual(INFRABASE_CONTEXT, set(ids))
                    self.assertIn("rule:conventions", ids)
                    self.assertIn("skill:improve-c-code-quality", ids)
                    self.assertNotIn("skill:customer-protocol", ids)

    def test_the_corpus_specific_context_follows_the_workspace(self):
        _, ids = self.route(INFRABASE, KERNEL_CONFIGURATION)

        self.assertIn("rule:board-packages", ids)
        self.assertIn("skill:add-board-package", ids)
        _, ids = self.route(TEACHING_OS, UBOOT_PATCH)

        self.assertNotIn("rule:board-packages", ids)
        self.assertNotIn("skill:add-board-package", ids)

    def test_opencn_is_not_confused_with_the_generic_infrabase_tree(self):
        _, opencn = self.route(OPENCN, KERNEL_CONFIGURATION)
        _, infrabase = self.route(INFRABASE, KERNEL_CONFIGURATION)

        self.assertIn("rule:opencn", opencn)
        self.assertNotIn("rule:board-packages", opencn)
        self.assertNotIn("skill:add-board-package", opencn)
        self.assertNotIn("rule:opencn", infrabase)

    def test_an_unrelated_workspace_is_given_no_infrabase_context(self):
        for ws in (GENERIC, UNREGISTERED):
            for name, request in PATTERNS.items():
                with self.subTest(workspace=ws.workspace_id, pattern=name):
                    phase, ids = self.route(ws, request)

                    self.assertEqual(phase, sel.GENERAL)
                    self.assertFalse(INFRABASE_CONTEXT & set(ids))

            with self.subTest(workspace=ws.workspace_id, request="a named change"):
                phase, ids = self.route(ws, "Fix the build of the kernel module.")

                self.assertEqual(phase, sel.IMPLEMENTATION)
                self.assertFalse(INFRABASE_CONTEXT & set(ids))

    def test_a_bound_standard_still_owns_a_request_that_names_no_subject(self):
        scope = answer_scope.of(KERNEL_CONFIGURATION, standard_bound=True, project_bound=True)

        self.assertEqual(scope, answer_scope.NORMATIVE)

    def test_a_request_that_names_its_subject_is_unchanged(self):
        for request, scope in (("Fix the build of the kernel module.", answer_scope.IMPLEMENTATION),
                               ("What does the standard require of the encoder?",
                                answer_scope.MIXED)):
            with self.subTest(request=request):
                self.assertEqual(answer_scope.of(request, project_bound=True), scope)
                self.assertEqual(answer_scope.of(request), scope)

    def test_the_ordering_is_stable(self):
        first = self.route(INFRABASE, UBOOT_PATCH)

        self.assertEqual(first[1], ["rule:board-packages", "rule:infrabase", "rule:conventions",
                                    "skill:debug-infrabase-build", "skill:add-board-package",
                                    "skill:improve-c-code-quality", "request"])

        for _ in range(3):
            self.assertEqual(self.route(INFRABASE, UBOOT_PATCH), first)

    def test_the_workspace_and_the_session_agree_on_the_binding(self):
        for ws, spec, registered in ((INFRABASE, {"families": ["infrabase"]}, True),
                                     (GENERIC, {"families": []}, True),
                                     (UNREGISTERED, {"families": ["infrabase"]}, False)):
            with self.subTest(workspace=ws.workspace_id):
                self.assertEqual(workspace_context.project_bound(registered=registered, spec=spec),
                                 ws.project_bound)


if __name__ == "__main__":
    unittest.main()
