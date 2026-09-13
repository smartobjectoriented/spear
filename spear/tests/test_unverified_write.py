"""A turn that changed files must show the change works, or say it did not.

Four C files edited, the last rounds spent configuring a build, and the round
budget cut the turn on "The changes have been made. Let me verify the build
works for the host side". It did not work: a declaration had been added
without the old one being removed, and the tree no longer compiled. Nothing in
the answer said so, and the nudge that exists for exactly this could not fire,
because running `ls` and `cmake -S . -B build` after an edit counted as having
run something.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_runtime import (unverified_change, unverified_write_note,
                           _verification_runs)

EDIT = 'edit_file {"path": "src/command/command_wire.h"}\nOK: updated'
EDIT2 = 'edit_file {"path": "src/command/cmdlink.c"}\nOK: updated'


def bash(command, output="", exit_code=None):
    tail = f"\n(exit {exit_code})" if exit_code is not None else ""
    return f'bash {{"command": "{command}"}}\n{output}{tail}'


# The tail of the run this exists for.
CUT_RUN = [
    EDIT, EDIT2,
    bash("make clean", "make: *** No rule to make target 'clean'.", 2),
    bash("ls -la /home/operator/work/customer/acme/proj492/",
         "total 68"),
    bash("cat CMakeLists.txt | head -50", "cmake_minimum_required(VERSION 3.16)"),
    bash("cmake -S . -B build/host", "-- Build files have been written to: ..."),
]


class LookingIsNotVerifying(unittest.TestCase):
    def test_the_run_that_started_this_is_unverified(self):
        note = unverified_write_note(CUT_RUN)

        self.assertIn("did not pass", note)
        self.assertIn("src/command/command_wire.h", note)
        # cmake -S -B configures; it does not build. The last thing to run
        # was that, and before it `make clean` failed: nothing built.
        self.assertIn("make clean", note)

    def test_ls_and_cat_do_not_count_as_verification(self):
        log = [EDIT, bash("ls -la"), bash("cat CMakeLists.txt"),
               bash("git status")]

        self.assertEqual(_verification_runs(log), [])
        # The mid-turn nudge is looser on purpose and stays quiet here: the
        # turn did run something. The end-of-turn note is what judges it.
        self.assertFalse(unverified_change(log))
        self.assertIn("no build, no test, no lint",
                      unverified_write_note(log))

    def test_a_passing_build_says_nothing(self):
        log = [EDIT, bash("cmake --build build/host", "[100%] Built target v492c")]

        self.assertEqual(unverified_write_note(log), "")
        self.assertFalse(unverified_change(log))

    def test_a_failing_build_is_named(self):
        log = [EDIT, bash("cmake --build build/host",
                          "error: conflicting types for 'cmdWireEncodeAck'", 2)]
        note = unverified_write_note(log)

        self.assertIn("did not pass", note)
        self.assertIn("cmake --build build/host", note)

    def test_a_compiler_called_directly_is_a_build(self):
        """`gcc -c` was invisible: a turn that compiled every edited file
        one by one had, as far as this gate could tell, verified nothing."""
        for command in ("gcc -c -I src src/command/command_wire.c -o /tmp/a.o",
                        "cc -c x.c", "clang -c x.c", "g++ -c x.cpp"):
            with self.subTest(command=command):
                log = [EDIT, bash(command)]
                self.assertEqual(unverified_write_note(log), "", command)

    def test_a_later_success_settles_an_earlier_failure(self):
        """`make clean` failed because the project has no clean target, and
        every edited file then compiled. Reporting the earlier failure called
        a working change broken."""
        log = [EDIT, EDIT2,
               bash("make clean", "No rule to make target 'clean'.", 2),
               bash("gcc -c -I src src/command/command_wire.c -o /tmp/a.o"),
               bash("gcc -c -I src src/command/cmdlink.c -o /tmp/b.o")]

        self.assertEqual(unverified_write_note(log), "")

    def test_a_failure_after_a_success_still_counts(self):
        log = [EDIT,
               bash("gcc -c -I src src/command/command_wire.c -o /tmp/a.o"),
               bash("./test_command", "ack round trip  FAIL", 1)]
        note = unverified_write_note(log)

        self.assertIn("last verification to run did not pass", note)
        self.assertIn("./test_command", note)

    def test_a_failing_test_run_is_named_too(self):
        log = [EDIT, bash("./test_command", "ack round trip  FAIL", 1)]

        self.assertIn("did not pass", unverified_write_note(log))

    def test_a_turn_that_wrote_nothing_is_left_alone(self):
        log = [bash("cat src/command/cmdlink.c"), bash("grep -n Ack src")]

        self.assertEqual(unverified_write_note(log), "")
        self.assertFalse(unverified_change(log))

    def test_only_what_ran_after_the_last_write_counts(self):
        """A build that passed, then another edit, then nothing."""
        log = [EDIT, bash("cmake --build build/host", "Built target"), EDIT2]

        self.assertEqual(_verification_runs(log), [])
        self.assertIn("no build, no test", unverified_write_note(log))

    def test_a_refused_write_is_not_a_change(self):
        log = ['edit_file {"path": "src/x.c"}\nERROR: old_text not found',
               bash("ls")]

        self.assertEqual(unverified_write_note(log), "")


if __name__ == "__main__":
    unittest.main()
