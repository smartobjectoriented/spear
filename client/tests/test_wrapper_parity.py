"""Commands run, and file tools answer, exactly as in Hermes Agent.

fixtures/hermes_terminal_wrapper.json holds what Hermes v0.21.0 (0cbc6e37)
returned. Three deterministic paired runs diverged on exactly these:

  - bash diagnostics: Hermes runs `bash -c <wrapper>` with the command eval'd
    on line 5, so errors read "/usr/bin/bash: line 5: ..."; the port ran a
    script file, and bash named the file and another line;
  - not-found suggestions: Hermes lists the directory with `ls -1`, in the
    locale's collation order, not Python's code-point order;
  - patch diffs: labelled with the resolved absolute path, not the path as
    the model typed it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import dispatch, tools
from agent.host import CommandOutcome

FIXTURE = json.loads((ROOT / "tests/fixtures/hermes_terminal_wrapper.json").read_text())


class BashHost:
    """Runs the core's session script with `bash -c`, as the pair sandbox does."""

    collation_locale = "en_US.UTF-8"

    def __init__(self, root):
        self.workspace_root = root

    def authorize(self, name, arguments):
        return None

    def resolve_read(self, path):
        return os.path.realpath(os.path.join(self.workspace_root, path)), None

    resolve_write = resolve_workdir = resolve_read

    def write_file(self, path, content, *, action):
        Path(path).write_text(content)

    def delete_file(self, path, reason):
        os.unlink(path)

    def run_command(self, command, script, *, timeout, output_chars):
        env = {k: v for k, v in os.environ.items() if k not in ("AI_AGENT", "HERMES_AGENT")}
        env.update(LANG="en_US.UTF-8")
        env.pop("LC_ALL", None)
        done = subprocess.run(["bash", "-c", script], cwd=self.workspace_root,
                              capture_output=True, env=env)
        return CommandOutcome("ok", (done.stdout + done.stderr).decode("utf-8", "replace"),
                              done.returncode)

    def after_tool(self, record):
        pass


class WrapperParity(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "sub/deep"))
        for name in FIXTURE["files"]:
            if name not in ("deep", "edit.txt"):
                Path(self.root, "sub", name).write_text("x\n")
        for rel in ("sub/edit.txt", "sub/deep/edit.txt", "sub/deep/other.txt"):
            Path(self.root, rel).write_text("first line\nsecond line\nthird line\n")
        self.host = BashHost(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def expected(self, value):
        return value.replace("<DIR>", self.root)

    def run_tool(self, state, name, **arguments):
        return dispatch.execute(self.host, state, "c", name, arguments, (name,))[0]

    def test_terminal_diagnostics_as_the_first_command(self):
        for key, command in FIXTURE["terminal_cases"].items():
            if key == "H_env":
                continue                       # the locale part is environment, below
            with self.subTest(case=key):
                self.assertEqual(self.run_tool(tools.new_state(), "terminal", command=command),
                                 self.expected(FIXTURE["terminal_first"][key]))

    def test_terminal_diagnostics_later_in_a_session(self):
        state = tools.new_state()
        self.run_tool(state, "terminal", command="true")
        for key, command in FIXTURE["terminal_cases"].items():
            if key in ("H_env", "G_cwd_change"):
                continue
            with self.subTest(case=key):
                self.assertEqual(self.run_tool(state, "terminal", command=command),
                                 self.expected(FIXTURE["terminal_session"][key]))
        self.assertEqual(self.run_tool(state, "terminal", command="cd sub && pwd"),
                         self.expected(FIXTURE["terminal_session"]["G_cwd_change"]))
        self.assertEqual(self.run_tool(state, "terminal", command="nosuchcmd_in_sub"),
                         self.expected(FIXTURE["terminal_session"]["I_after_cd_error"]))

    def test_agent_markers_are_exported(self):
        output = json.loads(self.run_tool(tools.new_state(), "terminal",
                                          command=FIXTURE["terminal_cases"]["H_env"]))["output"]
        hermes = json.loads(FIXTURE["terminal_first"]["H_env"])["output"]
        self.assertEqual(output.split("|")[:2], hermes.split("|")[:2])    # hermes-agent|true

    def test_not_found_suggestions_follow_ls_collation(self):
        for key, name in (("bsp_virt64", "bsp-virt64.inc"), ("bsp_rpi", "bsp_rpi.inc"),
                          ("bsp_inc", "bsp.inc"), ("bsp_upper", "BSP_rpi4.inc")):
            with self.subTest(case=key):
                path = os.path.join(self.root, "sub", name)
                self.assertEqual(tools.read_file(tools.new_state(), path, path,
                                                 collation="en_US.UTF-8"),
                                 self.expected(FIXTURE["similar"][key]))

    def test_patch_names_the_resolved_path(self):
        for key, path in (("relative", "sub/edit.txt"),
                          ("nested_absolute", os.path.join(self.root, "sub/deep/edit.txt")),
                          ("equivalent_form", "./sub/deep/../deep/other.txt")):
            with self.subTest(case=key):
                self.assertEqual(self.run_tool(tools.new_state(), "patch", path=path,
                                               old_string="second line", new_string="2nd line"),
                                 self.expected(FIXTURE["patch"][key]))


if __name__ == "__main__":
    unittest.main()
