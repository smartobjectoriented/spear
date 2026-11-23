"""Which files no tool may change, judged on the file and not the spelling."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import target_policy


class TargetPolicy(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(f"{self.root}/inc/generated")
        os.makedirs(f"{self.root}/u-boot.back")
        Path(self.root, "gen.h").write_text("/* DO NOT EDIT */\nint x;\n")
        Path(self.root, "plain.c").write_text("int y;\n")
        os.symlink("gen.h", f"{self.root}/alias.h")

    def refused(self, relative):
        return bool(target_policy.refusal(os.path.join(self.root, relative)))

    def test_generated_and_snapshot_trees_and_headers_are_protected(self):
        for path in ("inc/generated", "inc/generated/x.h", "u-boot.back", "u-boot.back/b.c",
                     "gen.h", "build/tmp/work/x", "alias.h", "inc/../gen.h"):
            with self.subTest(path=path):
                self.assertTrue(self.refused(path))

    def test_a_component_name_does_not_make_a_snapshot(self):
        for path in ("u-boot/boot/image-board.c", "build/meta-qemu/recipes-qemu/qemu/qemu_8.2.2.bb",
                     "atf/plat/x.c", "linux/linux/arch/arm64/configs/virt64_defconfig"):
            with self.subTest(path=path):
                self.assertFalse(self.refused(path))

    def test_a_dot_zero_tree_is_a_snapshot_only_beside_its_live_tree(self):
        os.makedirs(f"{self.root}/rootfs.0")
        os.makedirs(f"{self.root}/rootfs")
        os.makedirs(f"{self.root}/zynq-rev1.0")

        self.assertTrue(self.refused("rootfs.0/etc/inittab"))
        self.assertFalse(self.refused("zynq-rev1.0/board.c"))

    def test_a_pristine_copy_is_a_snapshot_wherever_it_sits(self):
        self.assertTrue(self.refused("build/work/linux-6.12/linux.pristine/Makefile"))

    def test_ordinary_files_are_not(self):
        for path in ("plain.c", "new.c", "more.c.0", "lib/libfoo.so.1.0", "src/regenerated.c"):
            with self.subTest(path=path):
                self.assertFalse(self.refused(path))


class GitMetadata(unittest.TestCase):
    """A repository's own state is refused, however the path reaches it."""

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        git = ["git", "-C", self.root, "-c", "user.name=t", "-c", "user.email=t@t"]
        Path(self.root, "a.txt").write_text("a\n")
        Path(self.root, ".gitignore").write_text("*.o\n")
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "base"], check=True)

        # A submodule-shaped checkout whose git directory is not called .git.

        os.makedirs(f"{self.root}/store")
        subprocess.run(["git", "init", "-q", "--separate-git-dir", f"{self.root}/store/sub",
                        f"{self.root}/sub"], check=True)
        Path(self.root, "sub/s.c").write_text("int s;\n")
        os.symlink(".git/config", f"{self.root}/alias")
        self.object = next(str(item.relative_to(self.root))
                           for item in Path(self.root, ".git/objects").glob("??/*"))

    def refusal(self, relative):
        return target_policy.refusal(os.path.join(self.root, relative), relative)

    def test_git_state_is_refused_as_written_and_as_reached(self):
        for path in (".git", ".git/index", ".git/HEAD", ".git/config", self.object,
                     ".git/hooks/pre-commit", "sub/.git", "store/sub/HEAD",
                     "store/sub/refs/heads/x", "alias", "sub/../.git/HEAD"):
            with self.subTest(path=path):
                self.assertIn("is git metadata", self.refusal(path))

    def test_the_files_beside_it_are_not(self):
        for path in ("a.txt", ".gitignore", ".github/workflows/ci.yml", "sub/s.c",
                     "store/notes.txt", "new.c"):
            with self.subTest(path=path):
                self.assertEqual(self.refusal(path), "")

    def test_a_gitdir_is_found_from_its_dot_git_file(self):
        self.assertEqual(target_policy.gitdir_named_by(f"{self.root}/sub/.git"),
                         f"{self.root}/store/sub")
        self.assertEqual(target_policy.gitdir_named_by(f"{self.root}/a.txt"), "")


# The opening of Linux's init/Kconfig: "auto-generated" in the help text of
# its first option, which made every edit of the file refused.

LINUX_INIT_KCONFIG = """# SPDX-License-Identifier: GPL-2.0-only
config CC_VERSION_TEXT
	string
	default "$(CC_VERSION_TEXT)"
	help
	  This is used in unclear ways:

	  - Re-run Kconfig when the compiler is updated
	    The 'default' property references the environment variable,
	    CC_VERSION_TEXT so it is recorded in include/config/auto.conf.cmd.
	    When the compiler is updated, Kconfig will be invoked.

	  - Ensure full rebuild when the compiler is updated
	    include/linux/compiler-version.h contains this option in the comment
	    line so fixdep adds include/config/CC_VERSION_TEXT into the
	    auto-generated dependency.
"""

SAVED_CONFIG = "#\n# Automatically generated file; DO NOT EDIT.\n# Buildroot 2024.02 Configuration\n#\nBR2_aarch64=y\n"


class GeneratedMarkers(unittest.TestCase):
    """A marker counts in a comment at the top of the file, and nowhere else."""

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())

    def file(self, relative, text):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(text)
        return path

    def test_a_marker_in_a_leading_comment_counts(self):
        for head in ("/* DO NOT EDIT: generated by gen.sh */\nint x;\n",
                     "// Code generated by stringer. DO NOT EDIT.\npackage x\n",
                     "#\n# Automatically generated file; DO NOT EDIT.\nCONFIG_A=y\n",
                     " * This file is auto-generated\n */\n",
                     "<!-- @generated -->\n<html/>\n", "; autogenerated, DO NOT MODIFY\n",
                     '"""Auto-generated by protoc."""\n'):
            with self.subTest(head=head):
                self.assertTrue(target_policy.says_generated(head))

    def test_the_words_in_prose_or_code_do_not(self):
        for head in (LINUX_INIT_KCONFIG, 'print("DO NOT EDIT this by hand")\n',
                     "int main(void) { return 0; } /* x */\nchar *s = \"auto-generated\";\n",
                     "\n" * 10 + "# DO NOT EDIT\n"):
            with self.subTest(head=head[:40]):
                self.assertFalse(target_policy.says_generated(head))

    def test_linux_init_kconfig_is_editable(self):
        self.assertEqual(target_policy.refusal(self.file("init/Kconfig", LINUX_INIT_KCONFIG)), "")

    def test_a_saved_configuration_is_a_source(self):
        for relative in ("configs/rpi4_64_defconfig", "arch/arm64/configs/defconfig",
                         "board/defconfig"):
            with self.subTest(path=relative):
                self.assertEqual(target_policy.refusal(self.file(relative, SAVED_CONFIG)), "")

        self.assertIn("generated", target_policy.refusal(self.file(".config", SAVED_CONFIG)))
        self.assertIn("generated", target_policy.refusal(
            self.file("out/generated/x_defconfig", SAVED_CONFIG)))
        self.assertIn("generated", target_policy.refusal(
            self.file("configs/defconfig.h", SAVED_CONFIG)))

    def test_a_refused_file_is_remembered_for_the_session(self):
        path = self.file("gen.h", "/* DO NOT EDIT */\n")
        target_policy.refusal(path)

        self.assertIn(path, target_policy.refused_paths())


if __name__ == "__main__":
    unittest.main()
