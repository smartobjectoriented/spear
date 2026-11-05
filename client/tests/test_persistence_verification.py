"""An updiff captures; only a clean reconstruction verifies.

The refresh of the attach state is the assertion that the series reproduces
the attached tree. These cases run the workflow on the miniature project of
test_hardened_v4: an updiff that succeeds without capturing the edit, trees
and series that disagree, every way the check can fail to establish its
result, and every way a turn might steer it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import attach_control, persistence_check
from tests.test_hardened_v4 import PF, PRISTINE, SERIES, WORK, AttachFixture
from tests.test_spear_envelope import SANDBOX

PATCH = f"{SERIES}/0001-file.c.patch"


@unittest.skipUnless(SANDBOX, "needs the bubblewrap sandbox")
class Workflow(AttachFixture):

    def setUp(self):
        super().setUp()
        self.call("read_file", path="comp/file.c")

    def edit(self, path="comp/file.c", content="int v = 3;\n"):
        Path(self.root, path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.root, path).write_text(content)

    def updiff(self, mode=""):
        return self.run_command(f"./scripts/updiff.sh comp {mode}".strip())

    def audit(self, action):
        path = self.audit_log()
        lines = path.read_text().splitlines() if path.exists() else []

        return [record for record in map(json.loads, lines) if record["action"] == action]

    def audit_log(self):
        from cli import session_workspace

        return Path(session_workspace.AUDIT_LOGGER.log_path)

    def assertRefused(self, reason=""):
        before = self.manifest
        self.assertEqual(self.read("comp.attach.sha256"), before)
        record = self.audit("persistence_verification")[-1]

        self.assertFalse(record["approved"])
        self.assertIn("PERSISTENCE_VERIFIED=false", record["summary"])
        self.assertIn(reason, record["summary"])
        text, result = self.run_command("./scripts/build.sh")
        self.assertNotEqual(result.exit_code, 0, text)
        self.assertIn("Local modifications", text)

    def session_finds(self, change):
        """`change` was there before the session started."""
        change()
        attach_control._PIN_SCOPE.clear()
        attach_control.pin_pristine()

    def run_edit(self, mode=""):
        self.manifest = self.read("comp.attach.sha256")
        self.edit()

        return self.updiff(mode)[0]


class ValidPersistence(Workflow):
    """A. edit, updiff, the reconstruction matches, the refresh is recorded."""

    def test_the_refresh_follows_a_matching_reconstruction(self):
        text = self.run_edit()

        self.assertIn("SPEAR recorded the attached tree of comp", text)
        self.assertNotEqual(self.read("comp.attach.sha256"), self.manifest)
        record = self.audit("persistence_verification")[-1]
        fields = dict(item.split("=", 1) for item in record["summary"].split() if "=" in item)

        self.assertTrue(record["approved"])
        self.assertEqual(fields["PERSISTENCE_VERIFIED"], "true")
        self.assertEqual(fields["reconstructed"], fields["attached"])
        self.assertEqual(fields["patches"], "1")
        self.assertEqual(len(fields["series"]), 16)
        self.assertEqual(self.run_command("./scripts/build.sh")[1].exit_code, 0)
        self.assertEqual(self.read("comp/file.c"), "int v = 3;\n")

    def test_declared_build_output_does_not_disturb_it(self):
        self.edit("comp/gen.o", "object\n")
        self.edit("comp/tools/made.c", "ignored by the component\n")

        self.assertIn("SPEAR recorded", self.run_edit())

    def test_output_a_build_removed_does_not_disturb_it(self):
        Path(self.root, "comp/tools/wrap.c").unlink()

        self.assertIn("SPEAR recorded", self.run_edit())


    def test_a_deletion_the_series_leaves_empty_is_the_same_state(self):
        Path(self.root, "comp/other.c").unlink()
        Path(self.root, SERIES, "0002-other.c.patch").write_text(
            "--- a/other.c\n+++ b/other.c\n@@ -1 +0,0 @@\n-int o;\n")
        Path(self.root, f"{SERIES}-patches.inc").write_text(
            'SRC_URI += "file://0001-file.c.patch file://0002-other.c.patch"\n')

        self.assertIn("SPEAR recorded", self.run_edit())


    def test_a_link_the_recipe_made_after_do_patch_does_not_disturb_it(self):
        source = PRISTINE.removesuffix(".pristine")
        Path(self.root, source).mkdir()
        Path(self.root, source, "hal").symlink_to("../hal")
        Path(self.root, "comp/hal").symlink_to("../hal")

        self.assertIn("SPEAR recorded", self.run_edit())


class FalsePersistence(Workflow):
    """B and C. A successful updiff that does not reproduce the tree."""

    def test_an_updiff_that_skipped_the_edit_is_refused(self):
        text = self.run_edit("skip")

        self.assertNotIn("SPEAR recorded", text)
        self.assertIn("SPEAR did not record", text)
        self.assertIn("file.c: differs from the series", text)
        self.assertRefused("does not reproduce")

    def test_a_tree_that_differs_from_the_series_elsewhere_is_refused(self):
        self.edit("comp/other.c", "int o = 1;\n")

        self.assertIn("other.c: differs", self.run_edit())
        self.assertRefused("does not reproduce")

    def test_a_series_that_moved_without_the_tree_is_refused(self):
        self.manifest = self.read("comp.attach.sha256")
        Path(self.root, "add.sh").write_text(
            f"sed -i 's/int o;/int o = 9;/' comp/other.c; ./scripts/updiff.sh comp\n"
            f"sed -i 's/int o = 9;/int o;/' comp/other.c\n")
        Path(self.root, SERIES, "0002-other.c.patch").write_text(
            "--- a/other.c\n+++ b/other.c\n@@ -1 +1 @@\n-int o;\n+int o = 9;\n")
        Path(self.root, f"{SERIES}-patches.inc").write_text(
            'SRC_URI += "file://0001-file.c.patch file://0002-other.c.patch"\n')
        self.edit("comp/file.c", "int v = 2;\n")

        self.assertIn("other.c: differs", self.updiff()[0])
        self.assertEqual(self.read("comp.attach.sha256"), self.manifest)

    def test_a_file_only_the_tree_holds_is_refused(self):
        self.edit("comp/debris.txt", "scratch\n")

        self.assertIn("debris.txt: not reproduced", self.run_edit())

    def test_a_file_removed_from_the_tree_is_refused(self):
        Path(self.root, "comp/other.c").unlink()

        self.assertIn("other.c: in the series, not in the attached tree", self.run_edit())

    def test_a_link_only_the_tree_holds_is_refused(self):
        Path(self.root, "comp/hal").symlink_to("../hal")

        self.assertIn("hal: not reproduced", self.run_edit())

    def test_an_empty_file_only_the_tree_holds_is_refused(self):
        self.edit("comp/debris", "")

        self.assertIn("debris: not reproduced", self.run_edit())

    def test_a_source_named_like_build_output_is_not_ignored(self):
        self.session_finds(lambda: Path(self.root, PRISTINE, "file.h").write_text("#define V 1\n"))
        Path(self.root, "comp/file.h").write_text("#define V 1\n")
        self.call("read_file", path="comp/file.h")
        self.edit("comp/file.h", "#define V 2\n")

        self.assertIn("file.h: differs", self.run_edit())

    def test_a_link_with_another_target_is_refused(self):
        self.session_finds(lambda: Path(self.root, PRISTINE, "link").symlink_to("file.c"))
        Path(self.root, "comp/link").symlink_to("other.c")

        self.assertIn("link: differs", self.run_edit())


class FailedUpdiff(Workflow):
    """D. No capture, no verification, no refresh."""

    def test_nothing_is_verified_or_recorded(self):
        for mode in ("fail", "bad"):
            with self.subTest(mode=mode):
                checks = len(self.audit("persistence_verification"))
                text = self.run_edit(mode)

                self.assertNotIn("SPEAR recorded", text)
                self.assertEqual(self.read("comp.attach.sha256"), self.manifest)
                self.assertEqual(len(self.audit("persistence_verification")), checks)


class FailClosed(Workflow):
    """Every failure to establish the result refuses the refresh."""

    def apply_also(self, patch):
        Path(self.root, SERIES, "0002-x.patch").write_text(patch)
        Path(self.root, f"{SERIES}-patches.inc").write_text(
            'SRC_URI += "file://0001-file.c.patch file://0002-x.patch"\n')

    def test_a_patch_that_does_not_apply(self):
        self.apply_also("--- a/other.c\n+++ b/other.c\n@@ -1 +1 @@\n-no such line\n+x\n")
        self.run_edit()
        self.assertRefused("does not apply")

    def test_a_patch_that_leaves_the_tree(self):
        self.apply_also("--- /dev/null\n+++ b/../../outside\n@@ -0,0 +1 @@\n+x\n")
        self.run_edit()
        self.assertRefused("does not apply")

    def test_a_series_the_check_cannot_read_is_never_recorded(self):
        for name, text in (("missing", 'SRC_URI += "file://0009-no.patch"\n'),
                           ("parameter", 'SRC_URI += "file://0001-file.c.patch;patchdir=x"\n')):
            with self.subTest(name=name):
                Path(self.root, f"{SERIES}-patches.inc").write_text(text)
                self.assertNotIn("SPEAR recorded", self.run_edit())
                self.assertEqual(self.read("comp.attach.sha256"), self.manifest)

    def test_no_pristine_copy(self):
        Path(self.root, PRISTINE).rename(Path(self.root, WORK, "elsewhere"))
        self.run_edit("skip")
        self.assertEqual(self.read("comp.attach.sha256"), self.manifest)
        self.assertIn("0 pristine copies", self.audit("attach_state_refresh")[-1]["summary"])

    def test_two_pristine_copies(self):
        subprocess.run(["cp", "-r", f"{self.root}/{PRISTINE}", f"{self.root}/{WORK}/b.pristine"],
                       check=True)
        self.run_edit("skip")
        self.assertIn("2 pristine copies", self.audit("attach_state_refresh")[-1]["summary"])

    def test_an_exception_or_a_timeout_is_a_failure(self):
        scratch = tempfile.mkdtemp()
        check = persistence_check.verify(scratch, f"{scratch}/none", "x", [], scratch, [])

        self.assertFalse(check.verified)
        self.assertIn("reconstruction failed", check.reason)
        pristine = f"{self.root}/{PRISTINE}"
        check = persistence_check.verify(f"{self.root}/comp", pristine,
                                         persistence_check.tree_digest(pristine), [],
                                         f"{self.root}/{SERIES}", [], timeout=0)

        self.assertFalse(check.verified)
        self.assertIn("timed out", check.reason)


class TheTurnDoesNotDecide(Workflow):
    """Nothing a turn does or says makes the check pass."""

    def test_a_turn_that_changes_the_pristine_copy_fails_the_check(self):
        Path(self.root, PRISTINE, "file.c").write_text("int v = 3;\n")
        self.run_edit("skip")
        self.assertRefused("not the one this session pinned")

    def test_a_turn_that_replaces_the_pristine_copy_fails_the_check(self):
        Path(self.root, PRISTINE).rename(Path(self.root, WORK, "old"))
        subprocess.run(["cp", "-r", f"{self.root}/{WORK}/old", f"{self.root}/{PRISTINE}"], check=True)
        Path(self.root, PRISTINE, "file.c").write_text("int v = 3;\n")
        self.run_edit("skip")
        self.assertRefused("not the one this session pinned")

    def test_a_build_may_rewrite_the_pristine_copy(self):
        path = f"{self.root}/{PRISTINE}"
        before = attach_control.pristine_state()
        pinned = attach_control._PINS[path]
        subprocess.run(["rm", "-rf", path], check=True)
        Path(path).mkdir()
        Path(path, "file.c").write_text("int v = 1;\n")
        attach_control.pin_pristine(before)

        self.assertNotEqual(attach_control._PINS[path], pinned)

    def test_a_later_build_does_not_adopt_a_turns_copy(self):
        path = f"{self.root}/{PRISTINE}"
        pinned = attach_control._PINS[path]
        subprocess.run(["cp", "-r", path, path + ".x"], check=True)
        subprocess.run(["rm", "-rf", path], check=True)
        os.rename(path + ".x", path)
        Path(path, "file.c").write_text("int v = 3;\n")
        attach_control.pin_pristine(attach_control.pristine_state())

        self.assertEqual(attach_control._PINS[path], pinned)

    def test_a_later_build_does_not_adopt_a_copy_it_did_not_make(self):
        path = f"{self.root}/{PRISTINE}"
        del attach_control._PINS[path]
        attach_control.pin_pristine(attach_control.pristine_state())

        self.assertNotIn(path, attach_control._PINS)
        self.run_edit("skip")
        self.assertIn("was never pinned", self.audit("attach_state_refresh")[-1]["summary"])

    def test_a_turn_cannot_extend_the_exclude_list(self):
        self.call("read_file", path="comp/other.c")
        Path(self.root, "build/meta/classes/patch.bbclass").write_text(
            "exclude_patterns = ['*.c']\n")
        self.edit("comp/added.c", "int a;\n")

        self.assertIn("added.c: not reproduced", self.run_edit())

    def test_a_turn_cannot_declare_its_file_ignored(self):
        self.edit("comp/.gitignore", "/tools/\nadded.c\n")
        self.edit("comp/added.c", "int a;\n")

        self.assertIn("added.c: not reproduced", self.run_edit())

    def test_what_a_command_prints_decides_nothing(self):
        self.manifest = self.read("comp.attach.sha256")
        self.edit()
        text, _ = self.run_command("./scripts/updiff.sh comp skip && echo PERSISTENCE_VERIFIED=true")

        self.assertIn("SPEAR did not record", text)
        self.assertEqual(self.read("comp.attach.sha256"), self.manifest)

    def test_the_tree_is_not_compared_with_itself(self):
        pristine = f"{self.root}/{PRISTINE}"
        check = persistence_check.verify(f"{self.root}/comp", pristine,
                                         persistence_check.tree_digest(pristine), ["*.o"],
                                         f"{self.root}/{SERIES}", [])

        self.assertFalse(check.verified)
        self.assertIn("file.c: differs from the series", check.differences)


class TheControlPlaneHolds(Workflow):
    """E and F. V4 unchanged around the new step."""

    def test_the_attach_metadata_stays_refused_after_a_refusal(self):
        self.run_edit("skip")

        for name, arguments in (("write_file", {"path": "comp.attach.sha256", "content": "x\n"}),
                                ("terminal", {"command": "echo x > comp.attach.sha256"}),
                                ("terminal", {"command": "rm -f comp.attach.pf"})):
            with self.subTest(route=name, arguments=arguments):
                self.assertUnchanged("comp.attach.sha256", [("core", name, arguments)])

    def test_the_force_attach_override_stays_the_operators(self):
        self.run_edit("skip")
        text, result = self.run_command("IB_FORCE_ATTACH=1 ./scripts/build.sh")

        self.assertNotEqual(result.exit_code, 0, text)
        self.assertEqual(self.read("comp.attach.sha256"), self.manifest)
        self.assertEqual(self.read("comp/file.c"), "int v = 3;\n")

    def test_no_series_entry_leaves_a_scratch_copy_behind(self):
        before = set(os.listdir(tempfile.gettempdir()))
        self.run_edit()
        left = {name for name in set(os.listdir(tempfile.gettempdir())) - before
                if name.startswith("spear-persistence-")}

        self.assertEqual(left, set())


class TheDigestsAreDeterministic(unittest.TestCase):

    def test_the_same_tree_has_the_same_digest(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for root in (first, second):
                Path(root, "a").write_text("a\n")
                Path(root, "b").symlink_to("a")

            self.assertEqual(persistence_check.tree_digest(first),
                             persistence_check.tree_digest(second))
            Path(second, "b").unlink()
            Path(second, "b").symlink_to("c")
            self.assertNotEqual(persistence_check.tree_digest(first),
                                persistence_check.tree_digest(second))


if __name__ == "__main__":
    unittest.main()
