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


if __name__ == "__main__":
    unittest.main()
