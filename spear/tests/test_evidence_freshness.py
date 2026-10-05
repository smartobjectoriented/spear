"""A check proves the tree it ran on, and no later one.

Phase 8 left one success claim unqualified: the turn's last full build had
failed behind `| tail -10`, read as a pass, and what ran after the final
change was a clean and a single bitbake task -- nothing that shows a link
surviving a clean and a build. The verdict now counts a check only in the
final source epoch, reads a piped exit status from the output it hides, and
asks of each claim the checks that would show it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import completion
from completion import Evidence


def edit(path, tool="patch"):
    return Evidence(tool, (path,))


def run(command, ok=True, output=""):
    return Evidence("terminal", (), command, 0 if ok else 2, output=output)


def refused(path):
    return Evidence("patch", (), refused=True)


def read(path):
    return Evidence("read_file", ())


def decide(*log, answer="Done.", **kwargs):
    return completion.decide(list(log), answer=answer, **kwargs)


class FinalStateOnly(unittest.TestCase):
    def test_a_edit_then_passing_build_verifies(self):
        verdict = decide(edit("a.c"), run("make"), answer="Done. It builds successfully.")

        self.assertEqual(verdict.state, "VERIFIED")

    def test_b_an_edit_after_the_build_makes_it_stale(self):
        verdict = decide(edit("a.c"), run("make"), edit("a.c"))

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertIn("not revalidated after the last source modification", verdict.reason)

    def test_c_another_source_file_after_the_build_is_stale_too(self):
        self.assertEqual(decide(edit("a.c"), run("make"), edit("b.c")).state, "UNVERIFIED")

    def test_d_a_build_writing_its_outputs_stays_valid(self):
        verdict = decide(edit("a.c"), run("make"),
                         run("cp out.bin build/tmp/deploy/out.bin"),
                         run("make > logs/build.log"),
                         generated=lambda path: (completion.generated_path(path)
                                                 or path.startswith("logs/")))

        self.assertEqual(verdict.state, "VERIFIED")

    def test_e_a_failed_build_then_an_edit_and_nothing_after(self):
        verdict = decide(edit("a.c"), run("make", ok=False), edit("a.c"))

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_f_a_refused_write_changes_nothing(self):
        self.assertEqual(decide(edit("a.c"), run("make"), refused("b.c")).state, "VERIFIED")

    def test_g_reading_and_searching_change_nothing(self):
        verdict = decide(edit("a.c"), run("make"), read("a.c"), Evidence("search_files", ()))

        self.assertEqual(verdict.state, "VERIFIED")

    def test_h_a_build_after_the_last_edit_verifies(self):
        verdict = decide(edit("a.c"), run("make"), edit("a.c"), run("make"))

        self.assertEqual(verdict.state, "VERIFIED")

    def test_i_a_clean_before_the_edit_shows_no_persistence(self):
        verdict = decide(run("bitbake rootfs-linux -c clean"), edit("a.bb"),
                         answer="Done. The link survives a clean and build.")

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_j_a_lifecycle_before_the_last_edit_is_stale(self):
        verdict = decide(edit("a.bb"), run("bitbake rootfs-linux -c clean"),
                         run("bitbake rootfs-linux"), run("ls -la images/initrd.cpio"),
                         edit("a.bb"),
                         answer="Done. The link survives a clean and build.")

        self.assertEqual(verdict.state, "UNVERIFIED")


class AWorkItemOfSeveralFiles(unittest.TestCase):
    def test_one_check_after_all_the_edits_covers_them_all(self):
        verdict = decide(edit("a.c"), edit("b.c"), edit("include/c.h", "write_file"),
                         run("make"), answer="Done. It builds successfully.")

        self.assertEqual(verdict.state, "VERIFIED")
        self.assertEqual(verdict.changed, ("a.c", "b.c", "include/c.h"))


class EachClaimNeedsItsOwnCheck(unittest.TestCase):
    PERSISTS = "Done. The symlink survives a clean and build."

    def test_clean_then_build_then_check_shows_persistence(self):
        verdict = decide(edit("a.bb"), run(". ./env.sh && bitbake rootfs-linux -c clean"),
                         run(". ./env.sh && bitbake rootfs-linux"),
                         run("readlink -f linux/rootfs/images/initrd.cpio"),
                         answer=self.PERSISTS)

        self.assertEqual(verdict.state, "VERIFIED")

    def test_a_clean_and_build_in_one_command_counts_as_both(self):
        verdict = decide(edit("a.bb"), run("./scripts/build.sh -c rootfs-linux"),
                         run("ls -la linux/rootfs/images"), answer=self.PERSISTS)

        self.assertEqual(verdict.state, "VERIFIED")

    def test_a_single_bitbake_task_is_not_the_build(self):
        verdict = decide(edit("a.bb"), run("bitbake rootfs-linux -c clean"),
                         run("bitbake rootfs-linux -c attach_infrabase"),
                         run("ls -la linux/rootfs/images"), answer=self.PERSISTS)

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertIn("survives a clean and rebuild", verdict.reason)

    def test_a_build_without_a_look_shows_no_persistence(self):
        verdict = decide(edit("a.bb"), run("./scripts/build.sh -c rootfs-linux"),
                         answer=self.PERSISTS)

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_a_look_that_finds_nothing_is_not_a_check(self):
        verdict = decide(edit("a.bb"), run("./scripts/build.sh -c rootfs-linux"),
                         run("ls linux/rootfs/images", output="ls: cannot access "
                             "'linux/rootfs/images': No such file or directory"),
                         answer=self.PERSISTS)

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_what_bitbake_and_build_sh_runs_are(self):
        cases = {"bitbake bsp-linux -c do_clean": "clean",
                 "bitbake bsp-linux -c do_prepare_initrd": "task",
                 "bitbake -g bsp-linux": None,
                 "./scripts/build.sh -h": None,
                 "./scripts/build.sh -c rootfs-linux": "clean+build",
                 "./scripts/build.sh -x bsp-linux": "build"}

        for command, kind in cases.items():
            with self.subTest(command=command):
                self.assertEqual(completion._kind(command.split()), kind)

    def test_a_clean_alone_verifies_no_change(self):
        self.assertEqual(decide(edit("a.bb"), run("bitbake rootfs-linux -c clean")).state,
                         "UNVERIFIED")

    def test_a_hedged_sentence_claims_nothing(self):
        verdict = decide(edit("a.bb"), run("bitbake rootfs-linux"),
                         answer="Done. It should survive a clean and build.")

        self.assertEqual(verdict.state, "VERIFIED")


class WhatAPipeHides(unittest.TestCase):
    def test_a_failure_behind_tail_is_a_failure(self):
        verdict = decide(edit("a.bb"), run(
            "./scripts/build.sh rootfs-linux 2>&1 | tail -10",
            output="ERROR: linux-6.12-r0 do_configure: Execution of ... failed"))

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_a_clean_run_behind_tail_still_passes(self):
        verdict = decide(edit("a.bb"), run("make 2>&1 | tail -5", output="built"))

        self.assertEqual(verdict.state, "VERIFIED")


class WritesTheShellMakes(unittest.TestCase):
    def test_a_shell_edit_after_the_build_is_stale(self):
        verdict = decide(edit("a.c"), run("make"), run("echo '/* x */' >> a.c"))

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertEqual(verdict.changed, ("a.c",))

    def test_an_edit_the_same_command_then_builds_is_validated(self):
        self.assertEqual(decide(edit("a.c"), run("sed -i s/a/b/ a.c && make")).state, "VERIFIED")

    def test_a_build_then_an_edit_in_one_command_is_stale(self):
        self.assertEqual(decide(edit("a.c"), run("make && cp a.c b.c")).state, "UNVERIFIED")

    def test_a_deletion_is_a_source_change(self):
        verdict = decide(edit("a.c"), run("make"), Evidence("delete_file", ("old.c",)))

        self.assertEqual(verdict.state, "UNVERIFIED")

    def test_a_shell_only_change_is_a_change(self):
        self.assertEqual(decide(run("cp template.c a.c")).state, "UNVERIFIED")

    def test_writes_inside_the_tree_by_absolute_path(self):
        verdict = decide(edit("a.c"), run("make"), run("echo x >> /w/repo/a.c"), root="/w/repo")

        self.assertEqual(verdict.state, "UNVERIFIED")
        self.assertEqual(decide(edit("a.c"), run("make"), run("echo x >> /tmp/x"),
                                root="/w/repo").state, "VERIFIED")


class TheAnswerSaysWhatTheVerdictSays(unittest.TestCase):
    def test_a_stale_claim_is_qualified_where_it_stands(self):
        verdict = decide(edit("a.c"), run("make"), edit("a.c"))
        answer = completion.qualify("Done. The build succeeds and the change works.", verdict)

        self.assertTrue(answer.startswith("**UNVERIFIED**"))
        self.assertIn("not revalidated after the last source modification", answer)
        self.assertIn("*(not verified)*", answer)


if __name__ == "__main__":
    unittest.main()
