"""scripts/spear-image and env.sh: the operator's way in to the image scripts.

A fake `docker` on PATH records what it is asked, so nothing is built.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEAR_IMAGE = REPO / "scripts" / "spear-image"

FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$DOCKER_LOG"
case "$1 $2" in
    "image inspect")
        [ "$3" = spear:1.0-private ] || exit 1
        case "$*" in
            *redistributable*) echo false ;;
            *profile*) echo private ;;
        esac ;;
esac
exit 0
"""


class SpearImageTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        bin_dir = Path(self.temp.name) / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text(FAKE_DOCKER)
        (bin_dir / "docker").chmod(0o755)
        self.log = Path(self.temp.name) / "docker.log"
        self.log.write_text("")
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        DOCKER_LOG=str(self.log))

    def tearDown(self):
        self.temp.cleanup()

    def run_image(self, *args):
        return subprocess.run([str(SPEAR_IMAGE), *args], env=self.env,
                              capture_output=True, text=True, timeout=30)

    def test_help_names_every_action(self):
        out = self.run_image("--help").stdout

        for action in ("build", "list", "inspect", "run", "save", "load",
                       "push", "erase"):
            self.assertIn(f"spear-image {action}", out)

    def test_build_needs_a_profile(self):
        result = self.run_image("build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public or private", result.stderr)

    def test_an_unknown_name_is_refused_not_guessed(self):
        result = self.run_image("erase", "engagement")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("neither a profile", result.stderr)

    def test_a_profile_names_the_tag_build_sh_gives(self):
        result = self.run_image("inspect", "private")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("spear:1.0-private", result.stdout)
        self.assertIn("redistributable  false", result.stdout)

    def test_erase_removes_the_image_and_keeps_the_cache_unless_asked(self):
        self.run_image("erase", "private")
        log = self.log.read_text()
        self.assertIn("docker rmi -f spear:1.0-private", log)
        self.assertNotIn("builder prune", log)

        self.run_image("erase", "private", "--cache")
        self.assertIn("docker builder prune -f", self.log.read_text())

    def test_a_missing_image_is_reported(self):
        result = self.run_image("run", "public")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no such image: spear:1.0-public", result.stderr)


class EnvShTests(unittest.TestCase):

    def shell(self, script, cwd):
        return subprocess.run(["sh", "-c", script], cwd=cwd,
                              capture_output=True, text=True, timeout=30)

    def test_it_puts_scripts_on_path_once(self):
        result = self.shell('. ./env.sh; . ./env.sh; echo "$PATH"; '
                            'echo "$SPEAR_ROOT_DIR"', REPO)
        path, root = result.stdout.splitlines()
        self.assertEqual(path.split(":").count(f"{REPO}/scripts"), 1)
        self.assertNotIn(f"{REPO}/scripts/docker", path.split(":"))
        self.assertEqual(root, str(REPO))

    def test_it_refuses_to_be_sourced_from_elsewhere(self):
        result = self.shell(f". {REPO}/env.sh", "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root", result.stderr)


if __name__ == "__main__":
    unittest.main()
