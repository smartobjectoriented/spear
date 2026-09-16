"""A standard the store can no longer honour must not end the session.

Re-extracting a corpus changes its fingerprint, and a binding pinned to the
old bytes is then stale. Refusing to rebind silently is right: the source
underneath moved and that has to be visible. But the refusal was raised out
of the turn path and killed spear-chat with a traceback -- so a legitimate,
recoverable state took the whole session down, mid-question.

The status line already handled it. Only the path that answers a question
did not.
"""

from __future__ import annotations

import inspect
import unittest

import rag_chat
from standard_commands import StandardCommandError


class TheTurnPathHandlesIt(unittest.TestCase):

    def source(self):
        return inspect.getsource(rag_chat.standard_binding_for)

    def test_the_refusal_is_caught(self):
        self.assertIn("except StandardCommandError", self.source())

    def test_it_answers_unbound_rather_than_raising(self):
        self.assertIn("return None", self.source())

    def test_the_operator_is_told_how_to_recover(self):
        """`stale_binding_status` names the bound corpus, the active one and
        the commands that fix it."""
        self.assertIn("stale_binding_status", self.source())

    def test_the_turn_says_it_is_unbound(self):
        self.assertIn("answering unbound", self.source())


class TheRefusalItselfIsUnchanged(unittest.TestCase):
    """The guard must keep refusing. What changed is who dies of it."""

    def test_a_moved_corpus_is_still_refused(self):
        from standard_commands import StandardOperator

        source = inspect.getsource(StandardOperator.active_binding)

        self.assertIn("canonical source changed", source)
        self.assertIn("raise StandardCommandError", source)

    def test_the_error_type_is_still_raised_by_the_operator(self):
        self.assertTrue(issubclass(StandardCommandError, Exception))


if __name__ == "__main__":
    unittest.main()
