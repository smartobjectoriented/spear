"""Whitespace-only edits, and the loop that made them impossible.

Asked to add the blank line reStructuredText needs before a paragraph, the
model sent the same edit nine times. Each was rejected as "identical" because
the guard compared with whitespace normalised away, and `sed -i` was refused on
the other side with "use edit_file instead" — a closed loop with no exit, ended
by the operator pressing ctrl-c.
"""
import os
import tempfile
import unittest
from pathlib import Path

import rag_chat
from tool_runtime import (
    CommandClassification, CommandPolicy, ExecutionMode, Workspace,
)


TABLE = """   * - MicroPython
     - the MicroPython interpreter (ARM64)
User-space libraries live in ``usr/lib/``.
"""


class WhitespaceOnlyEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for attr, value in (("WORKSPACE", Workspace.from_path(root)),
                            ("PROJECT_ROOT", str(root)),
                            ("EXECUTION_MODE", ExecutionMode.AUTO),
                            ("BYPASS_PERMISSIONS", True)):
            self.addCleanup(setattr, rag_chat, attr, getattr(rag_chat, attr))
            setattr(rag_chat, attr, value)
        self.target = root / "user_space.rst"
        self.target.write_text(TABLE)
        self.previous_cwd = os.getcwd()
        os.chdir(root)
        self.addCleanup(lambda: os.chdir(self.previous_cwd))

    def test_inserting_a_blank_line_is_a_real_edit(self):
        out = rag_chat.edit_file(
            "user_space.rst",
            "     - the MicroPython interpreter (ARM64)\nUser-space libraries",
            "     - the MicroPython interpreter (ARM64)\n\nUser-space libraries")
        self.assertTrue(out.startswith("OK"), out)
        self.assertIn("(ARM64)\n\nUser-space", self.target.read_text())

    def test_an_indentation_only_change_is_a_real_edit(self):
        self.target.write_text("def f():\n  return 1\n")
        out = rag_chat.edit_file("user_space.rst", "  return 1", "    return 1")
        self.assertTrue(out.startswith("OK"), out)
        self.assertEqual("def f():\n    return 1\n", self.target.read_text())

    def test_a_truly_identical_edit_is_still_refused_and_says_how_to_insert(self):
        out = rag_chat.edit_file("user_space.rst", "User-space", "User-space")
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertIn("identical", out)
        # The refusal has to teach the insertion idiom, or the model retries
        # the same call: that is exactly what happened.
        self.assertIn("INSERT", out)
        self.assertIn("new_text", out)


class RefusedCommandGuidanceTests(unittest.TestCase):
    def test_in_place_sed_names_the_alternative_and_the_idiom(self):
        assessment = CommandPolicy().classify("sed -i '79i\\' doc/source/x.rst")
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("edit_file", assessment.reason)
        # "use edit_file" alone was the dead end: edit_file was refusing too.
        self.assertIn("old_text", assessment.reason)
        self.assertIn("new_text", assessment.reason)

    def test_a_multiline_in_place_sed_is_caught_on_the_shell_path_too(self):
        """The insert form carries shell syntax, so it took the other branch.

        `sed -i '5a\...'` with embedded newlines was classified shell_complex,
        never reached the in-place rule, ran, and failed with "couldn't open
        temporary file: Read-only file system" -- sed writes its temp beside
        the target and a read-only-classified command gets the tree read-only.
        The model concluded the documentation was not writable and gave up on
        a tree it could edit.
        """
        command = ("sed -i '5a\\\n\\\n.. toctree::\\\n   ls' "
                   "/home/operator/soo/so3/doc/source/user_space.rst")
        assessment = CommandPolicy().classify(command)
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("edit_file", assessment.reason)

    def test_reading_sed_in_a_pipeline_is_untouched(self):
        for command in ('sed -n "1,5p" f.txt | head', "grep x f.c | sed s/a/b/"):
            with self.subTest(command=command):
                self.assertNotEqual(
                    CommandPolicy().classify(command).classification,
                    CommandClassification.DANGEROUS)


if __name__ == "__main__":
    unittest.main()


class StaleReadCacheTests(unittest.TestCase):
    """A cached read must not survive the write that invalidates it.

    The model re-ran `sed -n '1,20p' user_space.rst` after each of three
    splices, was handed the pre-edit text every time, concluded each write had
    failed, and spliced again -- ending with three toctrees, a duplicated
    paragraph and a lost line.
    """
    def context(self):
        from types import SimpleNamespace
        return SimpleNamespace(cache={}, role="main", cancellation=None,
                               checkpoint_manager=None, checkpoint=None,
                               action_id=None, trace=None, task_id="t")

    def test_a_mutation_drops_the_cached_command_output(self):
        ctx = self.context()
        ctx.cache["cat f.rst"] = "old contents"
        ctx.cache[rag_chat.READ_PATHS] = {"f.rst"}
        ctx.cache[rag_chat.SANDBOX_DOWN] = True
        rag_chat._record_mutation(ctx, "f.rst")
        self.assertNotIn("cat f.rst", ctx.cache)
        # The sentinels are not command output and must survive: the sandbox
        # does not come back, and the read-path set is cumulative.
        self.assertTrue(ctx.cache[rag_chat.SANDBOX_DOWN])
        self.assertEqual({"f.rst"}, ctx.cache[rag_chat.READ_PATHS])

    def test_the_command_that_wrote_stays_cached(self):
        ctx = self.context()
        ctx.cache["cat f.rst"] = "old contents"
        rag_chat._forget_cached_reads(ctx, keep="printf x > f.rst")
        ctx.cache["printf x > f.rst"] = "(empty)"
        rag_chat._forget_cached_reads(ctx, keep="printf x > f.rst")
        self.assertIn("printf x > f.rst", ctx.cache)
        self.assertNotIn("cat f.rst", ctx.cache)


class QuotedShellArgumentTests(unittest.TestCase):
    """A quoted argument is data, not shell syntax.

    The splitter used re.split and ignored quoting, so a sed script rewriting
    ``basic file utilities (``ls`` supports ``-l``; ``rm`` supports ``-r`` /
    ``-f``)`` was torn apart on its own parentheses and semicolon, leaving a
    fragment that began with ``/``. The refusal -- "'/' is not allowlisted, so
    the pipeline that contains it cannot run" -- named a path the command never
    contained, and the model tried four more spellings of the same command.
    """
    SED = ("sed -i 's/     - basic file utilities (``ls`` supports ``-l``; "
           "``rm`` supports ``-r`` \\/ ``-f``)/     - basic file utilities "
           "(``rm`` supports ``-r`` \\/ ``-f``)/' /home/x/user_space.rst")

    def test_the_refusal_is_about_sed_not_about_a_slash(self):
        assessment = CommandPolicy().classify(self.SED)
        self.assertEqual(assessment.classification, CommandClassification.DANGEROUS)
        self.assertIn("edit_file", assessment.reason)
        self.assertNotIn("not allowlisted", assessment.reason)

    def test_operators_inside_quotes_do_not_create_stages(self):
        stages = CommandPolicy._shell_stages("grep 'foo(bar); baz|qux' main.c")
        self.assertEqual(["grep 'foo(bar); baz|qux' main.c"], stages)

    def test_real_operators_still_split(self):
        self.assertEqual(["ls ", "|", " head"],
                         CommandPolicy._shell_stages("ls | head"))
        self.assertEqual(["a ", "&&", " b"],
                         CommandPolicy._shell_stages("a && b"))
        self.assertEqual(["make 2", ">", "/dev/null"],
                         CommandPolicy._shell_stages("make 2>/dev/null"))

    def test_an_escaped_operator_outside_quotes_is_not_one(self):
        self.assertEqual(["echo a\\|b"],
                         CommandPolicy._shell_stages("echo a\\|b"))


class TerminalOutputTests(unittest.TestCase):
    """Tool output must survive the spinner repainting over it.

    The spinner thread repaints with "\r\033[K" -- carriage return, erase to
    end of line. A line another thread was midway through printing went with
    it. That is how an edit_file which HAD applied showed its diff and then no
    result at all, and the model abandoned the tool that had just worked.
    """
    def test_the_printers_and_the_spinner_share_one_lock(self):
        import inspect
        for function in (rag_chat.tool_result, rag_chat.tool_use):
            with self.subTest(function=function.__name__):
                self.assertIn("terminal_output", inspect.getsource(function))
        # The spinner no longer takes the lock itself: it paints through
        # StatusLine, which is now the only writer of that line and the only
        # place the lock has to be held. The invariant is unchanged -- one
        # lock, shared with the printers -- and this follows it to where it
        # lives rather than pinning the line of code it used to be on.
        self.assertIn("STATUS.show", inspect.getsource(rag_chat.Spinner._run))

        for method in (rag_chat.StatusLine.show, rag_chat.StatusLine.clear):
            with self.subTest(method=method.__name__):
                self.assertIn("TERMINAL_LOCK", inspect.getsource(method))

    def test_an_empty_result_says_so_instead_of_printing_a_bare_marker(self):
        import io
        from contextlib import redirect_stdout
        stream = io.StringIO()
        with redirect_stdout(stream):
            rag_chat.tool_result("")
        self.assertIn("(empty)", stream.getvalue())


class NoDeleteCapabilityTests(unittest.TestCase):
    """Deleting is not available, and the refusals must say so.

    Told to remove a chapter it had made redundant, a model tried `rm`,
    `bash -c "rm ..."`, `python3 -c "os.remove(...)"` and an empty write_file
    -- fifteen calls -- because every refusal pointed at edit_file/write_file
    as "the way to change files". It finally emptied the file with edit_file,
    leaving a zero-byte document that still warned "isn't included in any
    toctree".
    """
    def test_removal_commands_say_deletion_is_unavailable(self):
        for command in ("rm doc/source/ls.rst", "rmdir build",
                        "mv a.rst b.rst", "unlink x"):
            with self.subTest(command=command):
                reason = CommandPolicy().classify(command).reason
                self.assertIn("no tool can delete or rename", reason)
                self.assertIn("operator", reason)

    def test_other_refusals_keep_pointing_at_the_edit_tools(self):
        reason = CommandPolicy().classify('python3 -c "print(1)"').reason
        self.assertNotIn("no tool can delete", reason)
