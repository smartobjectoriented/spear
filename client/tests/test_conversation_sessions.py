"""Named conversations: one history file each, one process each, and a switch
that leaves nothing of the previous conversation's standard engagement."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import (
    chat_settings, conversation_sessions, operator_input, rag_chat,
    session_workspace, standard_session, startup_banner,
)
from cli.conversation_sessions import (
    ConversationError, ConversationLock, history_file, list_conversations,
    validate_name,
)


class Names(unittest.TestCase):
    def test_valid_names_pass_and_none_is_the_default(self):
        for name in ("ipi", "ft-2026.11", "A_b-9"):
            self.assertEqual(validate_name(name), name)

        self.assertIsNone(validate_name(None))

    def test_paths_and_oddities_are_refused(self):
        for name in ("", "a/b", "../x", "..", ".", "a b", "x" * 41, "é"):
            with self.assertRaises(ConversationError):
                validate_name(name)


class HistoryFiles(unittest.TestCase):
    def test_the_default_keeps_its_file_and_a_name_sits_beside_it(self):
        default = "/state/history-adhoc-1a2b3c4d.json"

        self.assertEqual(history_file(default, None), default)
        self.assertEqual(history_file(default, "ipi"),
                         "/state/history-adhoc-1a2b3c4d-ipi.json")

    def test_listing_finds_the_named_ones_and_not_the_locks(self):
        directory = tempfile.mkdtemp()
        default = os.path.join(directory, "history-adhoc-1a2b3c4d.json")

        for name in ("history-adhoc-1a2b3c4d-ipi.json",
                     "history-adhoc-1a2b3c4d-ipi.json.lock",
                     "history-adhoc-99999999.json"):
            Path(directory, name).write_text("[]")

        self.assertEqual([name for name, _ in list_conversations(default)],
                         [None, "ipi"])


class Locks(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "history-adhoc-t.json")

    def _hold_in_another_process(self):
        holder = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(f"""
                import fcntl, sys, time
                handle = open({self.path + '.lock'!r}, "a+")
                fcntl.flock(handle, fcntl.LOCK_EX)
                print("held", flush=True)
                sys.stdin.read()
            """)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.stdin.close)
        self.addCleanup(holder.stdout.close)
        self.assertEqual(holder.stdout.readline().strip(), "held")

    def test_a_second_process_is_refused_the_conversation(self):
        self._hold_in_another_process()

        self.assertTrue(ConversationLock.held_elsewhere(self.path))

        with self.assertRaises(ConversationError):
            ConversationLock().acquire(self.path)

    def test_an_open_empty_conversation_is_listed(self):
        # The holder locks the default's lock file; renamed, the lock follows
        # the file, which is what a named conversation held before its first
        # save looks like.

        self._hold_in_another_process()
        os.rename(self.path + ".lock", history_file(self.path, "busy") + ".lock")

        self.assertIn("busy", [name for name, _ in list_conversations(self.path)])

    def test_a_switch_releases_the_previous_lock(self):
        lock = ConversationLock()
        other = self.path.replace(".json", "-b.json")
        lock.acquire(self.path)
        lock.acquire(other)

        self.assertEqual(lock.path, other + ".lock")

        second = ConversationLock()
        second.acquire(self.path)
        second.release()
        lock.release()

    def test_rm_refuses_the_current_and_the_default(self):
        default = self.path

        with self.assertRaises(ConversationError):
            conversation_sessions.remove(default, "a", "a")

        with self.assertRaises(ConversationError):
            conversation_sessions.remove(default, None, "a")

        named = history_file(default, "b")
        Path(named).write_text("[]")
        conversation_sessions.remove(default, "b", None)

        self.assertFalse(os.path.exists(named))


class SwitchCommand(unittest.TestCase):
    def setUp(self):
        directory = tempfile.mkdtemp()
        self.default = os.path.join(directory, "history-adhoc-t.json")
        Path(self.default).write_text(json.dumps([{"role": "user", "content": "a"}]))

        for name, value in (("DEFAULT_HISTORY_FILE", self.default),
                            ("HISTORY_FILE", self.default),
                            ("CONVERSATION", None)):
            patcher = mock.patch.object(session_workspace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.lock = ConversationLock()
        self.addCleanup(self.lock.release)
        self.lock.acquire(self.default)

    def _run(self, argument, history):
        with redirect_stdout(io.StringIO()), \
                mock.patch.dict(os.environ, {"SPEAR_FRESH": ""}):
            return rag_chat.session_command(argument, history, self.lock)

    def test_new_switch_and_back_keep_each_history(self):
        history = self._run(" new ipi", [{"role": "user", "content": "a"}])

        self.assertEqual(history, [])
        self.assertEqual(session_workspace.CONVERSATION, "ipi")
        self.assertTrue(os.path.exists(history_file(self.default, "ipi")))

        history = self._run(" switch default", [{"role": "user", "content": "b"}])

        self.assertEqual(history, [{"role": "user", "content": "a"}])
        self.assertEqual(json.loads(Path(history_file(self.default, "ipi"))
                                    .read_text()),
                         [{"role": "user", "content": "b"}])

    def test_the_default_is_reachable_before_its_first_save(self):
        os.remove(self.default)
        self._run(" new ipi", [])
        self._run(" switch default", [])

        self.assertIsNone(session_workspace.CONVERSATION)

    def test_a_switch_resets_the_standard_engagement(self):
        standard_session.STANDARD_ENGAGED_BEFORE = True
        standard_session.STANDARD_PRIOR_ANSWER = "previous"
        self._run(" new other", [])

        self.assertFalse(standard_session.STANDARD_ENGAGED_BEFORE)
        self.assertEqual(standard_session.STANDARD_PRIOR_ANSWER, "")


class Surfaces(unittest.TestCase):
    def test_the_flag_and_the_command_are_listed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPEAR_SESSION", None)
            remaining = chat_settings.apply_env_options(["--session", "ipi", "--auto"])

            self.assertEqual(remaining, ["--auto"])
            self.assertEqual(os.environ["SPEAR_SESSION"], "ipi")

        self.assertIn("--session", startup_banner._env_option_lines())
        self.assertIn("/session", operator_input.SESSION_COMMANDS)


if __name__ == "__main__":
    unittest.main()
