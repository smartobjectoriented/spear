"""terminal(workdir=...) behaves as Hermes Agent's.

fixtures/hermes_workdir.json holds what Hermes v0.21.0 (0cbc6e37) returned
for each scenario. Two Phase-6 runs of the core stayed stuck after deleting
the directory their session was in: every command failed to enter it, and
the explicit workdir that gets Hermes out was silently dropped.

Hermes' semantics, reproduced here:

  - an explicit workdir is entered for that command; it overrides the
    session's recorded cwd but never replaces it, so the next command
    without one goes back to the recorded cwd -- still failing if that
    directory is gone (Hermes does the same);
  - a relative workdir is taken from where the last command finished, or
    its nearest surviving ancestor;
  - a workdir with shell metacharacters is blocked before anything runs.

SPEAR adds one rule of its own: the host resolves the workdir, and a
directory outside the workspace is refused (Hermes would enter it). That is
a policy difference, tested as such, not a parity case.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import dispatch, tools
from tests.test_wrapper_parity import BashHost

FIXTURE = json.loads((ROOT / "tests/fixtures/hermes_workdir.json").read_text())
POLICY = {"F_outside"}


class ContainedHost(BashHost):
    """BashHost with the workspace containment SPEAR's host applies."""

    def resolve_workdir(self, path):
        full = os.path.realpath(path)
        root = os.path.realpath(self.workspace_root)

        if full != root and not full.startswith(root + "/"):
            return None, "refused: outside the workspace"

        return full, None


class WorkdirParity(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp())
        os.makedirs(os.path.join(self.root, "sub", "sub2"))
        os.makedirs(os.path.join(self.root, "gone"))
        self.host = ContainedHost(self.root)
        self.state = tools.new_state()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_step(self, command, workdir):
        arguments = {"command": command}

        if workdir is not None:
            arguments["workdir"] = workdir.replace("<DIR>", self.root)

        return dispatch.execute(self.host, self.state, "c", "terminal", arguments,
                                ("terminal",))[0]

    def test_every_scenario_matches_hermes(self):
        for name, steps in FIXTURE["scenarios"].items():
            if name in POLICY:
                continue

            with self.subTest(scenario=name):
                self.setUp()
                try:
                    for step, (command, workdir) in enumerate(steps):
                        self.assertEqual(self.run_step(command, workdir),
                                         FIXTURE["results"][name][step].replace("<DIR>", self.root),
                                         f"step {step}: {command!r} workdir={workdir!r}")
                finally:
                    self.tearDown()

    def test_a_workdir_outside_the_workspace_is_refused_by_policy(self):
        self.assertEqual(json.loads(self.run_step("pwd", "/")),
                         {"output": "", "exit_code": -1,
                          "error": "refused: outside the workspace", "status": "blocked"})
        self.assertEqual(json.loads(self.run_step("pwd", None))["output"], self.root)

    def test_recovery_from_a_deleted_session_cwd(self):
        self.run_step("cd gone", None)
        self.run_step("rmdir ../gone", None)

        self.assertEqual(json.loads(self.run_step("pwd", None))["exit_code"], 126)
        self.assertEqual(json.loads(self.run_step("pwd", "<DIR>/sub"))["output"],
                         f"{self.root}/sub")
        self.assertEqual(json.loads(self.run_step("cd <DIR> && pwd".replace("<DIR>", self.root),
                                                  "<DIR>/sub")),
                         {"output": self.root, "exit_code": 0, "error": None, "cwd": self.root})


class WorkdirContract(unittest.TestCase):
    def test_the_production_terminal_schema_offers_workdir(self):
        from tool_registry import coding_schemas

        self.assertIn("workdir", coding_schemas()["terminal"]["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
