"""Selecting a second standards store, and rolling back by not selecting it.

An extraction change cannot be measured by editing the store it is being
compared against. The mechanism is therefore a pointer, not a migration: one
variable names the store root, every reader honours it, and unsetting it puts
the production store back with nothing to undo.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import state_paths


class TheStandardsRootIsSelectable(unittest.TestCase):

    def test_unset_it_sits_under_the_state_directory(self):
        with mock.patch.dict(os.environ, {"SPEAR_STATE_DIR": "/tmp/s"},
                             clear=True):
            self.assertEqual(state_paths.standards_root(), Path("/tmp/s/standards"))

    def test_set_it_is_the_store_root_itself(self):
        with mock.patch.dict(os.environ,
                             {"SPEAR_STATE_DIR": "/tmp/s",
                              "SPEAR_STANDARDS_ROOT": "/tmp/candidate"},
                             clear=True):
            self.assertEqual(state_paths.standards_root(), Path("/tmp/candidate"))

    def test_it_moves_only_the_store(self):
        """Everything else a session accumulates stays where it was: an
        evaluation must not divert the audit log or the sessions with it."""
        with mock.patch.dict(os.environ,
                             {"SPEAR_STATE_DIR": "/tmp/s",
                              "SPEAR_STANDARDS_ROOT": "/tmp/candidate"},
                             clear=True):
            self.assertEqual(state_paths.state_dir(), Path("/tmp/s"))

    def test_an_empty_value_is_not_a_selection(self):
        """An exported-but-empty variable is how a shell rolls back, and it
        must not resolve the store to the current directory."""
        with mock.patch.dict(os.environ,
                             {"SPEAR_STATE_DIR": "/tmp/s",
                              "SPEAR_STANDARDS_ROOT": ""}, clear=True):
            self.assertEqual(state_paths.standards_root(), Path("/tmp/s/standards"))


if __name__ == "__main__":
    unittest.main()
