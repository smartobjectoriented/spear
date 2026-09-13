"""One registry deserves one environment.

`spear-corpus` and the in-chat `/corpus` are one implementation on purpose --
a second one had already drifted. But the two entry points did not run in the
same environment: `spear-chat.sh` sources the machine's untracked `machine.env`
and `spear-corpus.sh` did not, so the CLI ran with `SPEAR_STATE_DIR` defaulting
to the app directory while the chat used the configured one. Importing the
module creates that directory, so the CLI quietly grew a second state tree
beside the code; and `SPEAR_CORPUS_ROOT`, which decides what a *relative*
corpus path resolves against, was missed the same way -- on the one command
whose whole job is resolving corpus paths.

These tests drive the launcher with a stub interpreter, so they assert what the
shell actually exports rather than what the file appears to say.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
ROOT = APP.parent

LAUNCHERS = ("spear-chat.sh", "spear-corpus.sh")

# The stub stands in for the venv interpreter: the launcher execs it, and it
# reports the environment it was handed instead of importing the harness.
STUB = """#!/bin/bash
for v in SPEAR_STATE_DIR SPEAR_CORPUS_ROOT SPEAR_RULES_DIR; do
    echo "$v=${!v-<unset>}"
done
exit 0
"""


class Deployment:
    """A throwaway app directory holding one launcher and a stub interpreter."""

    def __init__(self, tmp, launcher, machine_env=None):
        self.dir = Path(tmp)
        (self.dir / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy(APP / launcher, self.dir / launcher)
        stub = self.dir / "bin" / "python3"
        stub.write_text(STUB)
        stub.chmod(0o755)

        if machine_env is not None:
            (self.dir / "machine.env").write_text(machine_env)

        self.launcher = self.dir / launcher

    def run(self, *args):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SPEAR_")}
        done = subprocess.run([str(self.launcher), *args], env=env,
                              capture_output=True, text=True, timeout=30)
        reported = dict(line.split("=", 1)
                        for line in done.stdout.splitlines() if "=" in line)
        return done, reported


class MachineEnvReachesTheCorpusLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp"))

    def test_configured_values_reach_the_interpreter(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = Deployment(tmp, "spear-corpus.sh", machine_env=(
                'export SPEAR_STATE_DIR="/var/tmp/spear-state"\n'
                'export SPEAR_CORPUS_ROOT="/var/tmp/corpora"\n'))
            done, reported = app.run("list")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(reported["SPEAR_STATE_DIR"], "/var/tmp/spear-state")
        self.assertEqual(reported["SPEAR_CORPUS_ROOT"], "/var/tmp/corpora")

    def test_an_absent_machine_env_preserves_the_defaults(self):
        """No file is the normal case for a fresh clone; it must still run."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = Deployment(tmp, "spear-corpus.sh")      # no machine.env
            done, reported = app.run("list")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(reported["SPEAR_STATE_DIR"], "<unset>")
        self.assertEqual(reported["SPEAR_CORPUS_ROOT"], "<unset>")
        self.assertEqual(reported["SPEAR_RULES_DIR"], "<unset>")

    def test_an_unreadable_machine_env_is_not_fatal(self):
        """`set -e` plus a conditional source is a documented foot-gun."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = Deployment(tmp, "spear-corpus.sh", machine_env="")
            (app.dir / "machine.env").chmod(0o000)

            try:
                done, _ = app.run("list")
            finally:
                (app.dir / "machine.env").chmod(0o644)

        self.assertEqual(done.returncode, 0, done.stderr)


class TheLaunchersAgree(unittest.TestCase):
    def test_every_launcher_reads_the_machine_environment(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertIn(
                    '[ -r "$SCRIPT_DIR/machine.env" ] && . "$SCRIPT_DIR/machine.env"',
                    (APP / name).read_text(),
                    f"{name} does not read machine.env; the entry points would "
                    f"then disagree about where state lives")

    def test_the_chat_and_the_corpus_cli_report_the_same_environment(self):
        import tempfile

        machine_env = ('export SPEAR_STATE_DIR="/var/tmp/agreed"\n'
                       'export SPEAR_CORPUS_ROOT="/var/tmp/agreed-corpora"\n')
        seen = {}

        for name in LAUNCHERS:
            with tempfile.TemporaryDirectory() as tmp:
                app = Deployment(tmp, name, machine_env=machine_env)
                # spear-chat.sh re-execs into a user scope unless told not to
                env_guard = dict(os.environ, SPEAR_IN_USER_SCOPE="1")
                done = subprocess.run(
                    [str(app.launcher), "--help"],
                    env={k: v for k, v in env_guard.items()
                         if not k.startswith("SPEAR_")
                         or k == "SPEAR_IN_USER_SCOPE"},
                    capture_output=True, text=True, timeout=30)
                seen[name] = {
                    line.split("=", 1)[0]: line.split("=", 1)[1]
                    for line in done.stdout.splitlines()
                    if line.startswith("SPEAR_")}

        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                # not two empty dicts comparing equal: each launcher must have
                # reached the interpreter and handed it the configured value
                self.assertEqual(seen[name].get("SPEAR_STATE_DIR"),
                                 "/var/tmp/agreed")

        self.assertEqual(seen["spear-chat.sh"], seen["spear-corpus.sh"])


class ThePublicCheckoutIsSelfContained(unittest.TestCase):
    """Sourcing a machine file must not make the repository need one."""

    def test_no_launcher_is_tracked_with_a_machine_file(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                                 capture_output=True, text=True, check=True)
        self.assertNotIn("spear/machine.env", tracked.stdout.split())

    def test_the_machine_file_is_ignored_rather_than_merely_absent(self):
        done = subprocess.run(["git", "check-ignore", "-q",
                               "spear/machine.env"], cwd=ROOT)
        self.assertEqual(done.returncode, 0,
                         "spear/machine.env is not gitignored, so a local "
                         "deployment's paths could be committed by accident")

    # Assembled from fragments, like the public-boundary scanner's own needles:
    # a test that spells a forbidden path out loud puts it in the tree it is
    # guarding. Neither half is a complete needle on its own.
    OUTSIDE = ("/opt/llm/spear-" + "private",   # the private holding area
               "/ho" + "me/")                   # anyone's home directory

    def test_no_launcher_names_a_path_outside_the_repository(self):
        """Every path a launcher names is derived from $SCRIPT_DIR."""
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                for line in (APP / name).read_text().splitlines():
                    code = line.split("#", 1)[0]

                    for bad in self.OUTSIDE:
                        self.assertNotIn(bad, code, f"{name}: {line.strip()}")

    def test_the_check_would_notice(self):
        """Prove the detector detects, on a line built the same way."""
        planted = 'SPEAR_RULES_DIR=' + self.OUTSIDE[0] + '/rules.d'

        self.assertTrue(any(bad in planted for bad in self.OUTSIDE))


if __name__ == "__main__":
    unittest.main()
