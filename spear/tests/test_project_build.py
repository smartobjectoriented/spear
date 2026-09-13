"""A tree states how it is built; the harness should not rediscover it.

Every writing turn on the same project worked it out again, and badly: `make
clean` against a tree with no Makefile, then `ls`, then `cat CMakeLists.txt`,
then two guesses at a cmake line. Ten rounds of sixty, before the first edit.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import project_build
from agent_runtime import project_build_gap, project_build_runs

VITA_CMAKE = """cmake_minimum_required(VERSION 3.16)
project(v492c LANGUAGES C)
set(V492C_SIDE "target" CACHE STRING "which side")
if(V492C_SIDE STREQUAL "target")
    set(V492C_BUILD_TARGET_SIDE ON)
else()
    set(V492C_BUILD_HOST_SIDE ON)   # the host side needs no vendor SDK
endif()
"""


class Probe(unittest.TestCase):
    def tree(self, **files):
        root = Path(tempfile.mkdtemp())

        for name, body in files.items():
            (root / name).write_text(body)

        return str(root)

    def test_the_cmake_option_that_cost_ten_rounds_a_run(self):
        found = project_build.probe(self.tree(**{"CMakeLists.txt": VITA_CMAKE}))

        self.assertIn("-DV492C_SIDE=host", found.build)
        self.assertIn("cmake --build", found.build)
        self.assertEqual(found.source, "cmake")

    def test_a_plain_cmake_tree_needs_no_option(self):
        found = project_build.probe(self.tree(**{
            "CMakeLists.txt": "project(x LANGUAGES C)\n"}))

        self.assertNotIn("-D", found.build)

    def test_the_ordinary_shapes(self):
        for name, expected in (("Makefile", "make"), ("Cargo.toml", "cargo"),
                               ("go.mod", "go"), ("package.json", "npm"),
                               ("pyproject.toml", "python")):
            with self.subTest(name=name):
                found = project_build.probe(self.tree(**{name: "x"}))
                self.assertEqual(found.source, expected)

    def test_a_tree_that_says_nothing_claims_nothing(self):
        found = project_build.probe(self.tree(**{"README.md": "hello"}))

        self.assertEqual(found.verifies(), ())
        self.assertEqual(found.source, "none")

    def test_projects_json_wins_over_every_probe(self):
        root = self.tree(**{"CMakeLists.txt": VITA_CMAKE})
        found = project_build.commands(
            root, spec={"build_commands": ["ninja -C out"],
                        "test_commands": "ninja -C out test"})

        self.assertEqual(found.build, "ninja -C out")
        self.assertEqual(found.source, "projects.json")

    def test_the_answer_is_remembered(self):
        root = self.tree(**{"CMakeLists.txt": VITA_CMAKE})
        cache = tempfile.mkdtemp()
        first = project_build.commands(root, cache_dir=cache)

        (Path(root) / "CMakeLists.txt").unlink()
        second = project_build.commands(root, cache_dir=cache)

        self.assertEqual(second.build, first.build)
        self.assertEqual(project_build.commands(
            root, cache_dir=cache, refresh=True).source, "none")


class Running(unittest.TestCase):
    def test_a_failure_carries_its_error_lines(self):
        ok, output = project_build.run(
            "echo compiling; echo 'x.c:12: error: no' >&2; exit 2", ".")

        self.assertFalse(ok)
        self.assertIn("error: no", output)
        self.assertNotIn("compiling", output)

    def test_success_says_nothing(self):
        self.assertEqual(project_build.run("true", "."), (True, ""))

    def test_no_command_is_not_a_failure(self):
        self.assertEqual(project_build.run("", "."), (True, ""))

    def test_the_demand_names_the_command_and_forbids_wandering(self):
        demand = project_build.demand("cmake --build build", "x.c:12: error")

        self.assertIn("cmake --build build", demand)
        self.assertIn("x.c:12: error", demand)
        self.assertIn("do not start anything else", demand)

    def test_the_note_is_one_line_the_reader_cannot_miss(self):
        note = project_build.note("make", "ld: undefined reference\nmore")

        self.assertIn("does not build after this turn", note)
        self.assertIn("undefined reference", note)


class PlainTestsPackageTests(unittest.TestCase):
    """A tests/ package is a shape the harness can be ASKED to recognise.

    Not one it assumes: `tests/__init__.py` says where a tree's tests live,
    not that `unittest discover` is how its owner runs them. The benchmark
    fixtures are exactly that shape and have no owner to ask.
    """

    def test_the_tests_package_shape_is_inferred_only_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
            self.assertEqual(project_build.probe(str(root)).source, "none")

            commands = project_build.probe(str(root), infer_unittest=True)
            self.assertEqual(commands.test, "python3 -m unittest discover -s tests")
            self.assertEqual(commands.source, "unittest")
            self.assertEqual(commands.build, "")

    def test_discovery_needs_the_package_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            self.assertEqual(
                project_build.probe(str(root), infer_unittest=True).test, "")

    def test_a_packaged_project_still_says_how_it_is_tested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
            self.assertEqual(
                project_build.probe(str(root), infer_unittest=True).source, "python")

    def test_only_a_configured_command_may_stand_as_verification(self):
        """A probed `make` builds the tree; it does not certify the turn."""

        self.assertFalse(project_build.ProjectCommands("make", "make test", "make").configured)
        self.assertFalse(project_build.ProjectCommands("", "ctest", "cmake").configured)
        self.assertTrue(project_build.declared({"test_commands": ["ctest"]}).configured)
        self.assertTrue(project_build.ProjectCommands(
            "", "python3 -m unittest discover -s tests", "unittest").configured)


class ProjectVerificationGateTests(unittest.TestCase):
    LOG = ['edit_file {"path": "a.py"}\nOK: updated']

    def context(self, verifier, commands=None):
        class State:
            mutation_generation = 1

        class Context:
            project_commands = commands or project_build.ProjectCommands(
                "", "run-tests", "projects.json")
            project_root = "."
            project_verifier = staticmethod(verifier)
            working_state = State()

        return Context()

    def test_the_gate_uses_the_verifier_it_is_given(self):
        """The project's commands, run where the model's commands run."""

        calls = []
        context = self.context(
            lambda command: (calls.append(command), ("failed", "boom"))[1])
        self.assertEqual(project_build_gap(context, self.LOG), ("run-tests", "boom"))
        self.assertEqual(calls, ["run-tests"])

    def test_a_verification_that_could_not_run_is_not_a_pass(self):
        context = self.context(lambda command: ("not_run", "sandbox unavailable"))
        runs = project_build_runs(context, self.LOG)
        self.assertEqual(runs, (("run-tests", "not_run", "sandbox unavailable"),))
        self.assertEqual(project_build_gap(context, self.LOG),
                         ("run-tests", "sandbox unavailable"))

    def test_without_a_verifier_nothing_is_run_anywhere(self):
        class Context:
            project_commands = project_build.ProjectCommands("", "run-tests", "projects.json")
            project_root = "."
            working_state = None

        runs = project_build_runs(Context(), self.LOG)
        self.assertEqual([status for _, status, _ in runs], ["not_run"])

    def test_it_runs_once_per_generation_of_changes(self):
        calls = []
        context = self.context(
            lambda command: (calls.append(command), ("passed", ""))[1])

        for _ in range(4):
            project_build_gap(context, self.LOG)

        self.assertEqual(calls, ["run-tests"])

        # A new mutation invalidates the answer, and only that.

        context.working_state.mutation_generation = 2
        project_build_gap(context, self.LOG)
        self.assertEqual(calls, ["run-tests", "run-tests"])


if __name__ == "__main__":
    unittest.main()


class OnlyWhatThisEditBroke(unittest.TestCase):
    """A repair round needs the difference, not the wall of errors again."""

    FIRST = "a.c:1: error: X\na.c:2: error: Y"
    SECOND = FIRST + "\na.c:9: error: Z"

    def test_the_first_failure_shows_everything(self):
        shown, filtered = project_build.new_errors(self.FIRST, "")

        self.assertEqual(shown, self.FIRST)
        self.assertFalse(filtered)

    def test_a_later_failure_shows_only_the_new_lines(self):
        shown, filtered = project_build.new_errors(self.SECOND, self.FIRST)

        self.assertEqual(shown, "a.c:9: error: Z")
        self.assertTrue(filtered)

    def test_nothing_new_still_shows_what_remains(self):
        """"Nothing new broke" and "nothing is broken" are different facts."""
        shown, filtered = project_build.new_errors(self.FIRST, self.FIRST)

        self.assertEqual(shown, self.FIRST)
        self.assertFalse(filtered)

    def test_the_demand_says_which_of_the_two_it_is_showing(self):
        first = project_build.demand("make", self.FIRST)
        later = project_build.demand("make", self.SECOND, self.FIRST)

        self.assertIn("does not compile", first)
        self.assertIn("What your last edit broke", later)
        self.assertNotIn("a.c:1: error: X", later)
