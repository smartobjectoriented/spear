"""A fresh session starts without the corpus's stored conversation and leaves
it as it was; everything else a session is given still applies."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import (
    chat_settings, session_history, session_workspace, startup_banner,
)

STORED = [{"role": "user", "content": "rename cnt to count in parse.c"},
          {"role": "assistant", "content": "Done."}]


class FreshSession(unittest.TestCase):
    def setUp(self):
        self.file = os.path.join(tempfile.mkdtemp(), "history-adhoc-test.json")
        Path(self.file).write_text(json.dumps(STORED))
        patcher = mock.patch.object(session_workspace, "HISTORY_FILE", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_normal_session_resumes_the_stored_conversation(self):
        with mock.patch.dict(os.environ, {"SPEAR_FRESH": ""}):
            self.assertEqual(session_history.load_history(), STORED)

    def test_a_fresh_session_loads_nothing_and_keeps_the_store(self):
        with mock.patch.dict(os.environ, {"SPEAR_FRESH": "1"}):
            self.assertEqual(session_history.load_history(), [])
            session_history.save_history([{"role": "user", "content": "a new question"}])

        self.assertEqual(json.loads(Path(self.file).read_text()), STORED)

    def test_the_flag_sets_the_switch_and_leaves_argv(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPEAR_FRESH", None)
            remaining = chat_settings.apply_env_options(["--auto", "--fresh", "--single-root"])

            self.assertEqual(remaining, ["--auto", "--single-root"])
            self.assertTrue(session_history.fresh_session())

    def test_the_help_names_it(self):
        self.assertIn("--fresh", startup_banner._env_option_lines())


if __name__ == "__main__":
    unittest.main()
