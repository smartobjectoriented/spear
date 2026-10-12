"""Which files no tool may change, judged on the file and not the spelling."""

from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
