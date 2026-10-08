"""The harness's own project validation, as evidence for the verdict.

A failed run of the project's declared build is negative evidence about the
final tree. It used to be read through the absence of a display note -- the
note stayed quiet on a failure so another note could word it -- and the
verdict took that silence for a pass: a tree that no longer compiled was
reported VERIFIED.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evidence import completion
from evidence.completion import Evidence, ProjectValidationEvidence
from evidence.project_build import ProjectCommands

COMMANDS = ProjectCommands(build="cmake --build build", test="ctest --test-dir build",
                           source="cmake")


def edit(path):
    return Evidence("patch", (path,))


def run(command, ok=True, output=""):
    return Evidence("terminal", (), command, 0 if ok else 2, output=output)


def decide(*log, project=(), answer="Done.", commands=COMMANDS):
    return completion.decide(list(log), project_runs=project, project_commands=commands,
                             answer=answer)


def harness(status, command=COMMANDS.build, epoch=1, kind="build"):
    return ProjectValidationEvidence(status, kind, epoch, command)


class ProjectValidation(unittest.TestCase):
    def test_a_a_passing_build_verifies_a_build_claim(self):
        verdict = decide(edit("a.c"), project=[harness(completion.PASSED)],
                         answer="The build succeeds.")

        self.assertEqual(verdict.state, "VERIFIED")

    def test_b_a_failed_build_is_never_verified(self):
        for log in ((edit("a.c"),), (edit("a.c"), run("gcc -c a.c"))):
            with self.subTest(log=log):
                verdict = decide(*log, project=[harness(completion.FAILED)])

                self.assertEqual(verdict.state, "UNVERIFIED")
                self.assertIn("project's own build failed", verdict.reason)

    def test_b_the_legacy_run_shape_is_read_the_same_way(self):
        verdict = decide(edit("a.c"), project=[(COMMANDS.build, "failed", "a.c:1: error")])

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_c_a_passing_unrelated_command_does_not_outweigh_it(self):
        verdict = decide(edit("a.c"), run("make"), run("pytest"),
                         project=[harness(completion.PASSED),
                                  harness(completion.FAILED, COMMANDS.test, kind="test")])

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertIn("project's own test failed", verdict.reason)

    def test_d_a_pass_before_the_last_edit_is_stale(self):
        verdict = decide(edit("a.c"), edit("b.c"), project=[harness(completion.PASSED, epoch=1)])

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertIn("earlier state of the tree", verdict.reason)

    def test_d_a_failure_before_the_last_edit_no_longer_judges_it(self):
        verdict = decide(edit("a.c"), edit("b.c"), run("make"),
                         project=[harness(completion.FAILED, epoch=1)])

        self.assertEqual(verdict.state, "VERIFIED")

    def test_e_a_failure_behind_a_filter_stays_a_failure(self):
        piped = "cmake --build build | tail -5"
        failing = decide(edit("a.c"), project=[(piped, "passed", "a.c:3: error: x")],
                         commands=ProjectCommands(build=piped, source="cmake"))
        silent = decide(edit("a.c"), project=[(piped, "passed", "")],
                        commands=ProjectCommands(build=piped, source="cmake"))

        self.assertEqual(failing.state, "UNVERIFIED")
        self.assertIn("project's own build failed", failing.reason)
        self.assertEqual(silent.state, "UNVERIFIED")

    def test_f_no_harness_run_proves_nothing(self):
        for project in ((), [harness(completion.NOT_RUN)], [(COMMANDS.build, "not_run", "")]):
            with self.subTest(project=project):
                self.assertEqual(decide(edit("a.c"), project=project).state, "UNVERIFIED")

    def test_g_a_build_does_not_show_a_link_survives_a_clean(self):
        answer = "The link now survives a clean and rebuild."
        build_only = decide(edit("a.c"), project=[harness(completion.PASSED)], answer=answer)
        full = decide(edit("a.c"), run("make clean"), project=[harness(completion.PASSED)],
                      answer=answer)

        self.assertEqual(build_only.state, "UNVERIFIED")
        self.assertIn("clean, build and check", build_only.reason)
        self.assertEqual(full.state, "UNVERIFIED")


class ConfiguredAndProbed(unittest.TestCase):
    """Declared and probed commands verify alike when they genuinely run; a
    declared command is the project's verification for its kind."""

    DECLARED = ProjectCommands(build="make firmware", source="projects.json")
    PROBED = ProjectCommands(build="cmake -S . -B build/harness && cmake --build build/harness",
                             source="cmake")

    def verdict(self, commands, status, command=None):
        return decide(edit("a.c"), commands=commands,
                      project=[(command or commands.build, status, "")])

    def test_a_declared_build_that_passes_verifies(self):
        self.assertEqual(self.verdict(self.DECLARED, "passed").state, "VERIFIED")

    def test_b_a_declared_build_that_fails_is_not_rescued(self):
        verdict = decide(edit("a.c"), run("cmake --build build"), commands=self.DECLARED,
                         project=[(self.DECLARED.build, "failed", "error: x")])

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_c_a_probed_cmake_build_that_passes_verifies(self):
        self.assertEqual(self.verdict(self.PROBED, "passed").state, "VERIFIED")

    def test_d_a_help_only_makefile_is_no_build_to_verify_with(self):
        import tempfile

        from evidence import project_build

        root = tempfile.mkdtemp()
        Path(root, "Makefile").write_text("help:\n\t@echo targets\nhtml:\n\tsphinx-build . b\n")
        probed = project_build.commands(root)

        self.assertEqual(probed.verifies(), ())
        self.assertEqual(decide(edit("index.rst"), commands=probed).state, "UNVERIFIED")

    def test_e_a_backgrounded_probed_build_proves_nothing(self):
        background = ProjectCommands(build="make &", source="make")

        self.assertEqual(self.verdict(background, "passed").state, "UNVERIFIED")

    def test_the_evidence_says_where_the_command_came_from(self):
        for commands, origin in ((self.DECLARED, "configured"), (self.PROBED, "probed")):
            with self.subTest(origin=origin):
                found = completion.project_evidence([(commands.build, "passed", "")], commands, 1)

                self.assertEqual(found[0].origin, origin)


if __name__ == "__main__":
    unittest.main()
