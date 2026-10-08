"""A question is not a write, a named target is not the tree, and this
project is not the disk.

A real run was asked whether an artifact of ONE target could live in another
directory. The classifier read that as a request to change the code, the turn
was told it was not finished until a file changed, it edited the same line in
every analogous file of every other target, then searched the home directory
for a build script and found one in an unrelated repository.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from context import request_intent
from context import request_scope
from runtime.agent_notes import (
    _WRITE_REQUEST_RE, conclude_demand, is_write_request, wants_write,
)
from runtime.agent_verification import unverified_write_note
from models.model_backend import ConversationMessage, TextBlock
from runtime.agent_roles import AgentRole
from harness.tool_exposure import ToolExposurePolicy
from harness.tool_registry import ToolRegistry, native_tool_specs
from harness.tool_router import ToolExecutionContext, ToolRouter
from runtime.tracing import TraceEmitter


def registry():
    result = ToolRegistry()

    for spec in native_tool_specs():
        result.register(spec, None if spec.name == "bash"
                        else lambda *_: "OK: updated")

    return result


def intent(text, answered=False):
    return request_intent.mutation_intent(text, _WRITE_REQUEST_RE,
                                          answered=answered)


class Backend:
    """A classifier that always says WRITE -- the one the real run had."""

    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1

        return SimpleNamespace(text="WRITE", error=None)


def context_for(*messages):
    conversation = tuple(ConversationMessage(role, (TextBlock(text),))
                         for role, text in messages)

    return SimpleNamespace(backend=Backend(), judge_intent=True,
                           budget_manager=None, conversation=conversation,
                           observer=SimpleNamespace(notice=lambda *a: None))


class AdvisoryQuestionsAreReadOnly(unittest.TestCase):
    ADVISORY = (
        "Could we move artifact X to directory Y?",
        "Would it be possible to restructure this?",
        "How could we change this layout?",
        "Should we update the docs?",
        "What would be needed to split module A?",
        "What if we moved the parser into its own file?",
        "Would it make sense to rename it?",
        "Could we have the output in images/ instead of in out/alpha ?",
    )

    def test_advisory_shapes_are_advisory(self):
        for text in self.ADVISORY:
            with self.subTest(text=text):
                self.assertEqual(intent(text), "advisory")
                self.assertFalse(is_write_request(text))

    def test_the_model_cannot_raise_an_advisory_question(self):
        """The failure: the model said WRITE, and it was believed."""
        for text in self.ADVISORY:
            with self.subTest(text=text):
                context = context_for(("user", text))

                self.assertFalse(wants_write(context, text))
                self.assertEqual(context.backend.calls, 0)

    def test_edit_file_is_not_offered_and_bash_cannot_write(self):
        view = ToolExposurePolicy().select(
            registry(), AgentRole.MAIN,
            objective="Could we move artifact X to directory Y?")

        self.assertTrue(view.read_only)
        self.assertTrue(view.advisory)
        self.assertNotIn("edit_file", view.names)
        self.assertNotIn("write_file", view.names)

    def test_the_conclusion_asks_for_an_answer_not_an_edit(self):
        text = conclude_demand("Could we move artifact X to directory Y?",
                               is_write=False)

        self.assertNotIn("edit_file", text)
        self.assertIn("Do NOT edit", text)


class ExplicitRequestsStillWrite(unittest.TestCase):
    def test_an_imperative_is_a_write(self):
        self.assertEqual(intent("Please move artifact X to directory Y."), "write")

    def test_a_question_followed_by_the_request_is_a_write(self):
        self.assertEqual(
            intent("Could we move X to Y? Please make that change."), "write")

    def test_a_polite_imperative_is_a_write(self):
        self.assertEqual(intent("Could you fix the bug in main.c?"), "write")

    def test_its_tools_include_edit_file(self):
        view = ToolExposurePolicy().select(
            registry(), AgentRole.MAIN,
            objective="Please move artifact X to directory Y.")

        self.assertFalse(view.read_only)
        self.assertIn("edit_file", view.names)


class AcceptingAProposalIsAWrite(unittest.TestCase):
    def test_yes_do_it_after_an_answer(self):
        context = context_for(
            ("user", "Could we move X to Y?"),
            ("assistant", "Yes: change a.cfg and b.cfg as follows ..."),
            ("user", "yes, do it"))

        self.assertTrue(wants_write(context, "yes, do it"))

    def test_the_approach_just_described(self):
        self.assertTrue(is_write_request(
            "please implement the approach you just described"))

    def test_do_it_with_nothing_before_accepts_nothing(self):
        self.assertNotEqual(intent("do it"), "write")
        self.assertEqual(intent("go ahead", answered=True), "write")

    def test_the_question_itself_never_became_one(self):
        context = context_for(("user", "Could we move X to Y?"))

        self.assertFalse(wants_write(context, "Could we move X to Y?"))


def family(root: Path):
    """out/{alpha,beta,gamma} laid out alike, cfg/{alpha,beta,gamma}.cfg,
    and a shared recipe that all of them use."""
    for name in ("alpha", "beta", "gamma"):
        (root / "out" / name).mkdir(parents=True)
        (root / "out" / name / "post.sh").write_text("x")
        (root / "out" / name / "image.bin").write_text("x")
        (root / "cfg").mkdir(exist_ok=True)
        (root / "cfg" / f"{name}.cfg").write_text("x")
        (root / "cfg" / f"{name}_guest.cfg").write_text("x")

    (root / "common.recipe").write_text("x")
    (root / "src").mkdir()
    (root / "src" / "main.c").write_text("x")


class WritesStayOnTheNamedTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        family(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def scope(self, text, previous=""):
        return request_scope.RequestScope.of(text, previous=previous,
                                             root=str(self.root))

    def test_the_named_target_and_shared_files_are_writable(self):
        scope = self.scope("Please move the image of alpha to images/.")

        for path in ("cfg/alpha.cfg", "cfg/alpha_guest.cfg",
                     "out/alpha/post.sh", "common.recipe", "src/main.c"):
            with self.subTest(path=path):
                self.assertEqual(
                    request_scope.sibling_write_refusal(scope, path), "")

    def test_analogous_siblings_are_refused(self):
        scope = self.scope("Please move the image of alpha to images/.")

        for path in ("cfg/beta.cfg", "cfg/gamma_guest.cfg", "out/beta/post.sh"):
            with self.subTest(path=path):
                refusal = request_scope.sibling_write_refusal(scope, path)

                self.assertIn("alpha", refusal)
                self.assertIn("scope_reason", refusal)

    def test_all_platforms_opens_the_siblings(self):
        scope = self.scope("Apply this to all platforms.",
                           previous="Please move the image of alpha.")

        self.assertEqual(
            request_scope.sibling_write_refusal(scope, "cfg/beta.cfg"), "")

    def test_a_bare_acceptance_inherits_the_target(self):
        scope = self.scope("yes, do it", previous="Could we move alpha's image?")

        self.assertIn("alpha", scope.anchors)
        self.assertTrue(request_scope.sibling_write_refusal(scope, "cfg/beta.cfg"))

    def test_naming_two_members_spans_the_family(self):
        """Places named along the way ("from out/alpha to out/beta") are not
        a narrowing to either."""
        scope = self.scope("Move the file from cfg/alpha.cfg to cfg/beta.cfg.")

        self.assertEqual(
            request_scope.sibling_write_refusal(scope, "cfg/beta.cfg"), "")

    def test_layers_that_share_only_their_skeleton_are_not_a_family(self):
        """meta-<x> layers each hold classes/, conf/ and their own recipes-<x>/.
        A request that says "root filesystem" names no layer: the rootfs
        layer is where the change belongs, not a sibling of meta-filesystem."""
        build = self.root / "build"

        for name in ("filesystem", "rootfs", "bsp"):
            for child in ("classes", "conf", f"recipes-{name}"):
                (build / f"meta-{name}" / child).mkdir(parents=True)

        target = "build/meta-rootfs/recipes-rootfs/0003-virt64_defconfig.patch"
        scope = self.scope("Add the strace package to the virt64 root filesystem.")

        self.assertEqual(request_scope.sibling_write_refusal(scope, target), "")

    def test_target_directories_laid_out_alike_stay_a_family(self):
        for name in ("alpha", "beta", "gamma"):
            (self.root / "out" / name / f"{name}.dtb").write_text("x")

        scope = self.scope("Please move the image of alpha to images/.")

        self.assertIn("alpha", request_scope.sibling_write_refusal(scope, "out/beta/post.sh"))

    def test_the_router_refuses_and_a_stated_reason_passes(self):
        router = ToolRouter(registry())
        scope = self.scope("Please move the image of alpha to images/.")

        def edit(path, **extra):
            context = ToolExecutionContext(task_id="t", trace=TraceEmitter(),
                                           cache={}, scope=scope)
            envelope = router.execute(context, "id", "edit_file", {
                "path": path, "old_text": "x", "new_text": "y", **extra})

            return envelope, context

        refused, _ = edit("cfg/beta.cfg")
        allowed, _ = edit("cfg/alpha.cfg")
        reasoned, context = edit(
            "cfg/beta.cfg", scope_reason="the shared loader reads every cfg")

        self.assertEqual(refused.error_category, "outside_requested_scope")
        self.assertNotEqual(allowed.error_category, "outside_requested_scope")
        self.assertNotEqual(reasoned.error_category, "outside_requested_scope")
        self.assertEqual(context.cache["__scope_reasons__"][0]["path"],
                         "cfg/beta.cfg")

    def test_the_router_refuses_a_search_outside_the_project(self):
        router = ToolRouter(registry())
        context = ToolExecutionContext(
            task_id="t", trace=TraceEmitter(), cache={},
            scope=self.scope("build it"))
        envelope = router.execute(context, "id", "bash",
                                  {"command": "find / -name build.sh"})

        self.assertEqual(envelope.error_category, "outside_project")


class ShellWritesObeyTheSameScope(unittest.TestCase):
    """`--auto` must not turn cp, sed -i or python -c into a way round the
    refusal edit_file gives."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        family(self.root)
        self.scope = request_scope.RequestScope.of(
            "Please move the image of alpha to images/.", root=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, command):
        return bool(request_scope.shell_write_refusal(command, self.scope))

    def test_explicit_sibling_destinations_are_refused(self):
        for command in ("sed -i s/a/b/ cfg/beta.cfg",
                        "cp cfg/alpha.cfg cfg/beta.cfg",
                        "echo x > cfg/gamma.cfg",
                        "echo x >> out/beta/post.sh",
                        "tee cfg/beta_guest.cfg < cfg/alpha.cfg",
                        "rm out/gamma/image.bin",
                        "sed -i s/a/b/ cfg/*.cfg"):
            with self.subTest(command=command):
                self.assertTrue(self.refused(command))

    def test_run_time_destinations_cannot_reach_a_sibling(self):
        for command in ("for f in cfg/*.cfg; do sed -i s/a/b/ $f; done",
                        "find cfg -name '*.cfg' | xargs sed -i s/a/b/",
                        "find cfg -name '*.cfg' -exec sed -i s/a/b/ {} \\;",
                        "python3 -c \"open('cfg/beta.cfg','w').write('x')\"",
                        "perl -pi -e s/a/b/ cfg/gamma.cfg",
                        "sh -c 'echo x > cfg/beta.cfg'"):
            with self.subTest(command=command):
                self.assertTrue(self.refused(command))

    def test_the_target_shared_files_and_builds_pass(self):
        for command in ("sed -i s/a/b/ cfg/alpha.cfg",
                        "sed -i s/a/b/ cfg/alpha*.cfg",
                        "for f in cfg/alpha*.cfg; do sed -i s/a/b/ $f; done",
                        "touch common.recipe",
                        "echo x > src/main.c",
                        "grep -l x cfg/*.cfg",
                        "cat cfg/beta.cfg > /tmp/copy",
                        "find cfg -name '*.cfg' | xargs grep -l x",
                        "make -C src", "./scripts/build.sh all"):
            with self.subTest(command=command):
                self.assertFalse(self.refused(command))

    def test_all_platforms_opens_the_shell_too(self):
        scope = request_scope.RequestScope.of(
            "Apply it to all platforms.", root=str(self.root))

        self.assertEqual(request_scope.shell_write_refusal(
            "sed -i s/a/b/ cfg/*.cfg", scope), "")

    def test_the_router_refuses_it_under_the_scope_category(self):
        router = ToolRouter(registry())
        context = ToolExecutionContext(task_id="t", trace=TraceEmitter(),
                                       cache={}, scope=self.scope)
        envelope = router.execute(context, "id", "bash",
                                  {"command": "cp cfg/alpha.cfg cfg/beta.cfg"})

        self.assertEqual(envelope.error_category, "outside_requested_scope")


class DiscoveryStaysInTheProject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        home = Path(self.tmp.name)
        self.project = home / "work" / "project"
        self.sibling = home / "other" / "repo"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "scripts" / "build.sh").write_text("#!/bin/sh\n")
        (self.project / "README.md").write_text("see scripts/build.sh\n")
        (self.sibling / "scripts").mkdir(parents=True)
        (self.sibling / "build.sh").write_text("#!/bin/sh\n")
        self.home = home
        self.saved = os.environ.get("HOME")
        os.environ["HOME"] = str(home)
        self.scope = request_scope.RequestScope.of(
            "build it", root=str(self.project))

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self.saved

        self.tmp.cleanup()

    def test_searching_the_home_directory_is_refused(self):
        for command in (f"find {self.home} -name build.sh",
                        "find ~ -name build.sh",
                        f"grep -rn build {self.home}",
                        "locate build.sh"):
            with self.subTest(command=command):
                self.assertTrue(
                    request_scope.outside_project_refusal(command, self.scope))

    def test_a_sibling_repository_is_not_read(self):
        for command in (f"ls {self.sibling}",
                        f"head -50 {self.sibling}/build.sh",
                        f"cd {self.sibling} && ./build.sh"):
            with self.subTest(command=command):
                self.assertTrue(
                    request_scope.outside_project_refusal(command, self.scope))

    def test_the_project_and_system_files_are_readable(self):
        for command in (f"ls {self.project}/scripts",
                        "find . -name build.sh -not -path './.git/*'",
                        "cat /usr/include/stdio.h",
                        f"cd {self.project} && ./scripts/build.sh -l"):
            with self.subTest(command=command):
                self.assertEqual(
                    request_scope.outside_project_refusal(command, self.scope), "")

    def test_a_path_the_user_named_is_readable(self):
        scope = request_scope.RequestScope.of(
            f"compare with {self.sibling}/build.sh", root=str(self.project))

        self.assertEqual(request_scope.outside_project_refusal(
            f"cat {self.sibling}/build.sh", scope), "")

    def test_a_missing_command_points_at_the_projects_own(self):
        hint = request_scope.missing_command_hint(
            "./build.sh -a thing", "bash: ./build.sh: No such file or directory",
            127, self.scope)

        self.assertIn("scripts/build.sh", hint)
        self.assertIn("README.md", hint)
        self.assertNotIn(str(self.sibling), hint)

    def test_discovery_is_bounded_to_the_root(self):
        found = request_scope.discover_build_entrypoints(str(self.project))

        self.assertIn("scripts/build.sh", found)
        self.assertTrue(all(not item.startswith("/") for item in found))

    def test_a_program_that_ran_and_failed_gets_no_hint(self):
        self.assertEqual(request_scope.missing_command_hint(
            "make", "make: *** [all] Error 2", 2, self.scope), "")


EDIT = 'edit_file {"path": "src/main.c"}\nOK: updated'


class AFailedVerificationIsReportedAsUnverified(unittest.TestCase):
    def test_changed_and_the_build_failed(self):
        log = [EDIT, 'bash {"command": "make"}\nmain.c:3: error: x\n(exit 2)']
        note = unverified_write_note(log)

        self.assertIn("UNVERIFIED", note)
        self.assertIn("src/main.c", note)
        self.assertIn("did not pass", note)

    def test_changed_and_nothing_ran(self):
        note = unverified_write_note([EDIT])

        self.assertIn("UNVERIFIED", note)
        self.assertIn("nothing was run", note)

    def test_changed_and_verified_says_nothing(self):
        self.assertEqual(unverified_write_note(
            [EDIT, 'bash {"command": "make"}\nbuilt\n(exit 0)']), "")


if __name__ == "__main__":
    unittest.main()
