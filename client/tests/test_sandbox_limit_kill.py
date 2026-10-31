"""A command the sandbox's limits kill says so, instead of failing silently.

A bitbake kernel build exceeded the 2 GiB memory cap: the OOM killer took
the whole sandbox, the session wrapper's output went with it, and the model
read exit -1, no output, "bubblewrap sandbox command failed" -- ten times.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import tools as coding_tools
from agent.host import CommandOutcome
from harness.resource_control import DEFAULT_CGROUP_LIMITS, CgroupLimits
from harness.sandbox import BubblewrapSandbox


def completed(code):
    return subprocess.CompletedProcess(["bwrap"], code, b"", b"")


class LimitKill(unittest.TestCase):
    def setUp(self):
        self.sandbox = BubblewrapSandbox.__new__(BubblewrapSandbox)
        self.sandbox.max_output_chars = 10_000

    def test_a_sigkill_under_active_limits_names_the_limits(self):
        for code in (-signal.SIGKILL, 128 + signal.SIGKILL):
            result = self.sandbox._result_from_completed(completed(code), DEFAULT_CGROUP_LIMITS)

            self.assertEqual(result.status, "failed")
            self.assertIn("killed", result.summary)
            self.assertIn("MemoryMax=6442450944", result.summary)
            self.assertIn("TasksMax=256", result.summary)

    def test_an_ordinary_failure_is_unchanged(self):
        result = self.sandbox._result_from_completed(completed(2), DEFAULT_CGROUP_LIMITS)

        self.assertEqual(result.summary, "bubblewrap sandbox command failed")

    def test_without_a_contract_a_kill_is_not_blamed_on_limits(self):
        result = self.sandbox._result_from_completed(completed(-signal.SIGKILL), CgroupLimits())

        self.assertEqual(result.summary, "bubblewrap sandbox command failed")

    def test_the_model_reads_the_reason(self):
        killed = self.sandbox._result_from_completed(completed(-signal.SIGKILL), DEFAULT_CGROUP_LIMITS)
        text, code, _ = coding_tools.terminal_from_outcome(
            "build.sh linux", CommandOutcome("ok", "", killed.exit_code, killed.summary),
            {"cwd": "/w", "env": ""}, "/w", 600)

        self.assertEqual(code, -1)
        self.assertIn("resource limits", json.loads(text)["error"])


class OutputBeforeTheKill(unittest.TestCase):
    """The session /tmp is a host directory: the wrapper's output file
    outlives the sandbox the cgroup killed, and its tail is returned."""

    def setUp(self):
        self.sandbox = BubblewrapSandbox.__new__(BubblewrapSandbox)
        self.sandbox.max_output_chars = 10_000
        self.sandbox._session_tmpdir = tempfile.mkdtemp()

    def write(self, name, text, age=0):
        path = Path(self.sandbox._session_tmpdir, name, "out")
        path.parent.mkdir()
        path.write_text(text)
        if age:
            past = time.time() - age
            os.utime(path, (past, past))

    def test_the_output_written_before_the_kill_is_returned(self):
        started = time.time()
        self.write("tmp.new", "SPEAR_BUILD_OUTPUT_SENTINEL before the kill\n")
        result = self.sandbox._result_from_completed(completed(-signal.SIGKILL),
                                                     DEFAULT_CGROUP_LIMITS, started)

        self.assertIn("SPEAR_BUILD_OUTPUT_SENTINEL before the kill", result.stdout)
        self.assertIn("output up to the kill follows", result.summary)

    def test_an_earlier_command_output_is_not_taken(self):
        self.write("tmp.old", "from an earlier command\n", age=600)
        result = self.sandbox._result_from_completed(completed(-signal.SIGKILL),
                                                     DEFAULT_CGROUP_LIMITS, time.time())

        self.assertNotIn("earlier command", result.stdout)
        self.assertIn("no output was kept", result.summary)


class TheBuildFits(unittest.TestCase):
    def test_the_memory_cap_covers_the_measured_kernel_build(self):
        self.assertGreaterEqual(DEFAULT_CGROUP_LIMITS.memory_max_bytes, 2 * int(2.9 * 1024 ** 3))
        self.assertGreater(DEFAULT_CGROUP_LIMITS.tasks_max, 111)


if __name__ == "__main__":
    unittest.main()
