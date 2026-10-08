"""What the chat client does around a turn: diffs, evidence, records and rules."""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import agent_notes, agent_verification

from harness.tool_primitives import ExecutionMode, ToolResult, shell_argv
from harness.sandbox import BubblewrapSandbox
from harness.workspace import Workspace


class ShowDiffTests(unittest.TestCase):
    """In auto mode the diff is the only review the user gets."""

    @classmethod
    def setUpClass(cls):
        from cli import terminal_ui
        cls.terminal_ui = terminal_ui

    def render(self, old, new):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.terminal_ui.show_diff(old, new)
        return re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())

    INCLUDES = ("#include <stdio.h>\n#include <dirent.h>\n#include <stdlib.h>\n"
                "#include <string.h>\n#include <unistd.h>\n#include <sys/stat.h>")

    def test_an_edit_below_the_unchanged_head_is_visible(self):
        """The failure this replaced: six identical lines, then the change.

        Showing the head of each side rendered both as the same six #includes,
        so an added include was invisible and the user read "OK: updated" under
        a diff showing nothing.
        """
        out = self.render(self.INCLUDES,
                          self.INCLUDES + '\n#include "libxml2/triostr.h"')
        self.assertIn('+ #include "libxml2/triostr.h"', out)
        self.assertNotIn("- #include <stdio.h>", out)

    def test_a_no_op_edit_says_so_instead_of_drawing_a_change(self):
        out = self.render(self.INCLUDES, self.INCLUDES)
        self.assertIn("no change", out)
        self.assertNotIn("+ #include", out)

    def test_a_modified_line_shows_both_sides(self):
        out = self.render("a\nb\nc", "a\nB\nc")
        self.assertIn("- b", out)
        self.assertIn("+ B", out)
        self.assertNotIn("a", out.replace("no change", ""))

    def test_a_large_change_is_truncated_and_says_how_much(self):
        out = self.render("\n".join(f"old{i}" for i in range(20)),
                          "\n".join(f"new{i}" for i in range(20)))
        self.assertIn("more changed lines", out)


class TurnEvidenceTests(unittest.TestCase):
    """The conclusion is grounded in the tool log, not in intentions."""

    @classmethod
    def setUpClass(cls):
        from cli import rag_chat
        cls.rag_chat = rag_chat

    def test_no_tool_activity_adds_nothing(self):
        self.assertEqual(agent_notes.turn_evidence([]), "")

    def test_a_turn_that_changed_no_file_says_so(self):
        """The fabricated-repair case: a claimed fix, an untouched file."""
        note = agent_notes.turn_evidence(['bash {"command": "cat x"}\nsome output'])
        self.assertIn("files changed: none", note)

    def test_failed_commands_are_counted(self):
        note = agent_notes.turn_evidence([
            'bash {"command": "make"}\nbuilt',
            'bash {"command": "make bad"}\nError 2\n(exit 2)',
        ])
        self.assertIn("2 command(s) run", note)
        self.assertIn("1 of them exited non-zero", note)

    def test_applied_edits_are_named_and_rejected_ones_are_not(self):
        note = agent_notes.turn_evidence([
            'edit_file {"path": "a.c"}\nOK: a.c updated',
            'edit_file {"path": "b.c"}\nERROR: file not found: b.c',
        ])
        self.assertIn("a.c", note)
        self.assertNotIn("b.c", note)


class UnverifiedChangeTests(unittest.TestCase):
    """A change nobody ran is a claim, not a result."""

    @classmethod
    def setUpClass(cls):
        from cli import rag_chat
        cls.rag_chat = rag_chat

    EDIT = 'edit_file {"path": "a.c"}\nOK: a.c updated'
    RUN = 'bash {"command": "make"}\nbuilt'

    def test_running_after_the_change_clears_it(self):
        self.assertFalse(agent_verification.unverified_change([self.EDIT, self.RUN]))

    def test_running_BEFORE_the_change_does_not(self):
        """Order is the whole point: a build before the edit tested the edit
        that was not yet made."""
        self.assertTrue(agent_verification.unverified_change([self.RUN, self.EDIT]))

    def test_a_second_change_reopens_it(self):
        self.assertTrue(agent_verification.unverified_change(
            [self.EDIT, self.RUN, self.EDIT]))

    def test_a_question_answered_from_reads_is_not_caught(self):
        # Nothing was changed, so there is nothing to verify and the gate must
        # stay out of the way of informational turns.
        self.assertFalse(agent_verification.unverified_change(
            ['bash {"command": "cat a.c"}\nsome text']))

    def test_a_rejected_edit_is_not_a_change(self):
        self.assertFalse(agent_verification.unverified_change(
            ['edit_file {"path": "a.c"}\nERROR: file not found: a.c']))

    def test_the_demand_names_the_files_and_asks_for_failing_cases(self):
        demand = agent_verification.verify_demand([self.EDIT])
        self.assertIn("a.c", demand)
        self.assertIn("could reasonably fail", demand)


class TrajectoryRecordingTests(unittest.TestCase):
    """Rated trajectories are the training data a fine-tune would need."""

    @classmethod
    def setUpClass(cls):
        from cli import project_checks, session_history
        cls.project_checks = project_checks
        cls.session_history = session_history

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trajectories.jsonl"
        self._saved = self.session_history.TRAJECTORY_FILE
        self.session_history.TRAJECTORY_FILE = str(self.path)

    def tearDown(self):
        self.session_history.TRAJECTORY_FILE = self._saved
        self.temp.cleanup()

    STEPS = [{"tool": "bash", "arguments": {"command": "make"}, "result": "built"}]

    def test_a_trajectory_keeps_the_tool_steps_not_just_the_prose(self):
        """A question/answer pair cannot teach behaviour.

        What needs training is running the change and judging the output, and
        neither is visible in the final prose.
        """
        self.session_history.save_trajectory("q", self.STEPS, "a", "pass", "bench")
        sample = json.loads(self.path.read_text().strip())
        self.assertEqual(sample["steps"], self.STEPS)
        self.assertEqual(sample["verdict"], "pass")
        self.assertEqual(sample["source"], "bench")

    def test_failures_are_recorded_too_and_labelled(self):
        # A dataset of successes alone cannot teach what to stop doing, and
        # filtering later is free while re-running a session is not.
        self.session_history.save_trajectory("q", self.STEPS, "a", "fail", "bench")
        self.assertEqual(json.loads(self.path.read_text())["verdict"], "fail")

    def test_samples_accumulate_one_per_line(self):
        self.session_history.save_trajectory("q1", self.STEPS, "a", "pass", "bench")
        count = self.session_history.save_trajectory("q2", self.STEPS, "a", "fail", "bench")
        self.assertEqual(count, 2)
        self.assertEqual(len(self.path.read_text().strip().split("\n")), 2)

    def test_no_declared_bench_gives_no_verdict_rather_than_a_pass(self):
        with patch.object(self.project_checks, "project_bench", return_value=None):
            self.assertIsNone(self.project_checks.run_project_bench())

    def test_a_turn_no_bench_judged_is_kept_as_unrated_not_dropped(self):
        """Recording is wider than judging, on purpose.

        Gating the record on a verdict meant a project without a bench
        recorded nothing: one project declares one, so the dataset held a
        single trajectory. "unrated" says the verdict is missing, which a
        trainer can filter on; dropping the turn says nothing happened.
        """
        self.session_history.save_trajectory("q", self.STEPS, "a", "unrated", "answer")
        sample = json.loads(self.path.read_text().strip())
        self.assertEqual((sample["verdict"], sample["source"]),
                         ("unrated", "answer"))
        self.assertEqual(sample["steps"], self.STEPS)

    def test_every_recorded_source_stays_distinguishable(self):
        for verdict, source in (("pass", "bench"), ("unrated", "change"),
                                ("unrated", "answer"), ("pass", "user")):
            self.session_history.save_trajectory("q", self.STEPS, "a", verdict, source)
        rows = [json.loads(l) for l in self.path.read_text().strip().split("\n")]
        self.assertEqual([(r["verdict"], r["source"]) for r in rows],
                         [("pass", "bench"), ("unrated", "change"),
                          ("unrated", "answer"), ("pass", "user")])


class BenchLocationTests(unittest.TestCase):
    """The bench belongs to the harness, not to the tree it judges."""

    @classmethod
    def setUpClass(cls):
        from cli import project_checks
        cls.project_checks = project_checks

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._saved = self.project_checks.BENCH_DIR
        self.project_checks.BENCH_DIR = self.temp.name

    def tearDown(self):
        self.project_checks.BENCH_DIR = self._saved
        self.temp.cleanup()

    def test_a_bare_name_resolves_inside_the_harness(self):
        script = Path(self.temp.name) / "proj.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        with patch.object(self.project_checks, "project_bench", return_value="proj.sh"):
            self.assertEqual(self.project_checks.bench_command(), str(script))

    def test_an_unknown_name_is_passed_through_as_a_command(self):
        # Escape hatch: a project that wants to own its bench still can.
        with patch.object(self.project_checks, "project_bench",
                          return_value="./scripts/mine.sh"):
            self.assertEqual(self.project_checks.bench_command(), "./scripts/mine.sh")

    def test_no_declared_bench_resolves_to_nothing(self):
        with patch.object(self.project_checks, "project_bench", return_value=None):
            self.assertIsNone(self.project_checks.bench_command())


class SessionTmpdirTests(unittest.TestCase):
    """One /tmp per session, so a check can span two commands."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        self.workspace = Workspace.from_path(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_file_written_in_tmp_survives_to_the_next_command(self):
        """The reason for the change: compile in one call, run in the next.

        With a per-command tmpfs the binary was gone before it could be run,
        forcing every verification into one unwieldy command line.
        """
        sandbox = BubblewrapSandbox()
        first = sandbox.run(self.workspace,
                            shell_argv("echo kept > /tmp/marker"))
        self.assertTrue(first.ok, first.to_legacy_text())
        second = sandbox.run(self.workspace,
                             shell_argv("cat /tmp/marker"))
        self.assertIn("kept", second.stdout)

    def test_two_sessions_do_not_share_their_tmp(self):
        BubblewrapSandbox().run(
            self.workspace, shell_argv("echo x > /tmp/leak"))
        other = BubblewrapSandbox().run(
            self.workspace,
            shell_argv("test -e /tmp/leak && echo SHARED || echo private"))
        self.assertIn("private", other.stdout)

    def test_the_host_tmp_is_never_written_to_directly(self):
        sandbox = BubblewrapSandbox()
        sandbox.run(self.workspace,
                    shell_argv("touch /tmp/spear-host-leak-marker"))
        self.assertFalse(Path("/tmp/spear-host-leak-marker").exists())

    @patch.dict(os.environ, {"SPEAR_SANDBOX_EPHEMERAL_TMP": "1"})
    def test_the_per_command_tmpfs_can_be_restored(self):
        argv = BubblewrapSandbox().build_argv(
            self.workspace, ["/bin/true"])
        index = argv.index("--tmpfs")
        self.assertIn("/tmp", [argv[i + 1] for i, t in enumerate(argv)
                               if t == "--tmpfs"])
        self.assertIsNotNone(index)


class SandboxDownStopsBlindEditsTests(unittest.TestCase):
    """A sandbox that is unavailable does not come back mid-turn.

    Observed in a container where bwrap could not mount /proc: eight commands,
    eight "sandbox unavailable" refusals, then nine edits to a source file made
    from retrieved context alone -- the model could not read the file it was
    changing, let alone compile it. The result was plausible-looking nonsense
    (SOL_SOCKET redefined as 0xffff, Linux headers added to an SO3 userspace
    program) that the user then had to revert.
    """

    @classmethod
    def setUpClass(cls):
        from cli import session_workspace, tool_handlers, tool_routing
        cls.session_workspace = session_workspace
        cls.tool_handlers = tool_handlers
        cls.tool_routing = tool_routing

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "f.c").write_text("int main(void) { return 0; }\n")
        self._ws = self.session_workspace.WORKSPACE
        self.session_workspace.WORKSPACE = Workspace.from_path(self.root)
        self._mode = self.session_workspace.EXECUTION_MODE
        self.session_workspace.EXECUTION_MODE = ExecutionMode.AUTO

    def tearDown(self):
        self.session_workspace.WORKSPACE = self._ws
        self.session_workspace.EXECUTION_MODE = self._mode
        self.temp.cleanup()

    def test_an_unavailable_sandbox_is_remembered_for_the_turn(self):
        # Patched at run_cmd_result and not at run_cmd: the router calls the
        # structured boundary directly, and run_cmd is now only the renderer
        # around it. The claim under test is unchanged -- a command that
        # reports the sandbox missing makes the turn remember it.
        cache = {}
        down = ToolResult(
            status="failed", summary="bubblewrap sandbox unavailable",
            stderr="bubblewrap sandbox unavailable", exit_code=1)
        with patch.object(self.tool_handlers, "run_cmd_result", return_value=down):
            self.tool_routing.execute_tool("bash", {"command": "ls"}, cache)
        self.assertTrue(cache.get(self.tool_handlers.SANDBOX_DOWN))

    def test_edits_are_refused_once_it_is_down(self):
        cache = {self.tool_handlers.SANDBOX_DOWN: True}
        before = (self.root / "f.c").read_text()
        for tool, args in (
            ("edit_file", {"path": "f.c", "old_text": "return 0",
                           "new_text": "return 1"}),
            ("write_file", {"path": "f.c", "content": "wiped"}),
            ("append_file", {"path": "f.c", "content": "// more"}),
        ):
            with self.subTest(tool=tool):
                result = self.tool_routing.execute_tool(tool, args, cache)
                self.assertIn("REFUSED", result)
                # The refusal is the point: the file must be untouched.
                self.assertEqual(before, (self.root / "f.c").read_text())

    def test_a_working_sandbox_leaves_edits_alone(self):
        cache = {}
        result = self.tool_routing.execute_tool(
            "edit_file", {"path": "f.c", "old_text": "return 0",
                          "new_text": "return 1"}, cache)
        self.assertNotIn("REFUSED", result)
        self.assertIn("return 1", (self.root / "f.c").read_text())


class LearnedRulesAreGlobalTests(unittest.TestCase):
    """`/recall` writes where every corpus will see it.

    `/remember` lands in memories-<corpus>.md, so a rule that holds everywhere
    -- "an existing copyright header is never rewritten" -- was invisible in
    every other tree. This is the global counterpart, and it lives under
    STATE_DIR rather than in rules.d/ because rules.d/ ships inside the image.
    """

    @classmethod
    def setUpClass(cls):
        from cli import turn_context
        cls.turn_context = turn_context

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self._file = self.turn_context.LEARNED_RULES_FILE
        self.turn_context.LEARNED_RULES_FILE = str(
            Path(self.temp.name) / "rules-learned.md")

        # A shipped rule of this test's own. rules.d/ ships empty -- a rule
        # describes one organisation's code, so none belongs to the platform
        # -- and the ordering below needs something to be ordered after.

        self._rules_dir = self.turn_context.RULES_DIR
        shipped = Path(self.temp.name) / "rules.d"
        shipped.mkdir()
        (shipped / "20-conventions.md").write_text(
            "## Conventions\n\nLeave a blank line after a comment block.\n",
            encoding="utf-8")
        self.turn_context.RULES_DIR = str(shipped)

    def tearDown(self):
        self.turn_context.LEARNED_RULES_FILE = self._file
        self.turn_context.RULES_DIR = self._rules_dir
        self.temp.cleanup()

    def test_a_recalled_rule_is_injected_after_the_shipped_ones(self):
        self.turn_context.save_learned_rule("never rewrite an existing header")
        self.turn_context.save_learned_rule("- prefer build.sh over make")
        body = Path(self.turn_context.LEARNED_RULES_FILE).read_text()
        self.assertEqual(len(body.strip().splitlines()), 2)
        # The leading dash the user typed is not doubled.
        self.assertNotIn("- - ", body)

        rules = self.turn_context.load_rules()
        self.assertIn("## Rule: learned", rules)
        self.assertIn("never rewrite an existing header", rules)
        # After the shipped ones, so a later line can qualify an earlier one.
        self.assertGreater(rules.index("## Rule: learned"),
                           rules.index("## Rule: conventions"))


class RepeatedCorpusSearchIsCachedTests(unittest.TestCase):
    """The corpus does not change mid-turn, so neither does the answer.

    Observed in a real session: the model asked the corpus the same question
    twice in a row and got the same file back, spending a round on it. bash has
    had a per-turn cache for exactly this; the retrieval tools did not.
    """

    @classmethod
    def setUpClass(cls):
        from cli import tool_handlers
        cls.tool_handlers = tool_handlers

    def _context(self, cache):
        return SimpleNamespace(cache=cache, action_id="a1", role="main",
                               cancellation=None)

    def test_the_second_identical_search_is_served_from_the_turn_cache(self):
        cache = {}
        calls = []

        def fake_search(query):
            calls.append(query)
            return f"# File: so3/usr/src/more.c\n{query}"

        with patch.object(self.tool_handlers, "search_corpus", fake_search):
            first = self.tool_handlers._registered_search_corpus(
                self._context(cache), {"query": "how does more read stdin"})
            second = self.tool_handlers._registered_search_corpus(
                self._context(cache), {"query": "how does more read stdin"})
            other = self.tool_handlers._registered_search_corpus(
                self._context(cache), {"query": "something else"})

        self.assertEqual(calls, ["how does more read stdin", "something else"],
                         "the repeated query must not reach the corpus twice")
        self.assertIn("more.c", first.text)
        self.assertIn("ALREADY SEARCHED", second.text)
        self.assertIn("more.c", second.text, "the cached answer is still given")
        self.assertNotIn("ALREADY SEARCHED", other.text)


class CorpusRulesTravelWithTheHarnessTests(unittest.TestCase):
    """An orientation map must reach a colleague who only clones spear.

    They used to live in the tree they describe. In so3 a blanket `.*` line in
    .gitignore swallowed the file, so a fresh clone gave the assistant no idea
    where anything lived -- and the map is the highest-leverage context there
    is: a stale path in it sent the model looking for a build script that had
    been deleted months earlier.
    """

    @classmethod
    def setUpClass(cls):
        from cli import session_workspace, turn_context
        cls.session_workspace = session_workspace
        cls.turn_context = turn_context

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = (self.session_workspace.PROJECT, self.session_workspace.PROJECT_ROOT,
                      self.session_workspace.CORPUS_ROOT, self.turn_context.SHIPPED_CORPUS_RULES)
        self.shipped = self.root / "shipped"
        self.shipped.mkdir()
        (self.shipped / "demo.md").write_text("SHIPPED MAP\n")
        self.turn_context.SHIPPED_CORPUS_RULES = str(self.shipped)
        self.session_workspace.PROJECT = "demo"
        self.tree = self.root / "tree"
        self.tree.mkdir()
        self.session_workspace.PROJECT_ROOT = self.session_workspace.CORPUS_ROOT = str(self.tree)

    def tearDown(self):
        (self.session_workspace.PROJECT, self.session_workspace.PROJECT_ROOT,
         self.session_workspace.CORPUS_ROOT, self.turn_context.SHIPPED_CORPUS_RULES) = self.saved
        self.temp.cleanup()

    def test_the_shipped_map_is_used_when_the_tree_has_none(self):
        self.assertIn("SHIPPED MAP", self.turn_context.load_corpus_rules())

    def test_a_tree_that_has_its_own_map_wins(self):
        """A tree someone else owns may carry one, and theirs beats ours."""
        (self.tree / ".edgem-rules.md").write_text("TREE MAP\n")
        rules = self.turn_context.load_corpus_rules()
        self.assertIn("TREE MAP", rules)
        self.assertNotIn("SHIPPED MAP", rules)

    def test_a_corpus_with_no_map_anywhere_invents_nothing(self):
        self.session_workspace.PROJECT = "unknown-corpus"
        self.assertEqual("", self.turn_context.load_corpus_rules())


if __name__ == "__main__":
    unittest.main()
