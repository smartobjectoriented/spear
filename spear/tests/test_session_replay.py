import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import session_replay
from model_backend import ModelToolCall, ModelTurn, StopReason
from session_replay import RecordingBackend, ReplayBackend


class FakeBackend:
    """A provider that answers from a script and counts what it was asked."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0
        self.model = "fake-model"

    def discover_model_name(self):
        return self.model

    def complete(self, **kwargs):
        self.calls += 1

        return self.turns.pop(0)


def tool_turn(call_id, name, **arguments):
    return ModelTurn("", (ModelToolCall(call_id, name, arguments),),
                     StopReason.TOOL_USE)


class RecordAndReplay(unittest.TestCase):
    """A live session recorded once, replayed for free afterwards."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "run.jsonl")

    def record(self, turns):
        inner = FakeBackend(turns)
        backend = RecordingBackend(inner, self.path)

        for _ in turns:
            backend.complete(system="s", conversation=(), tools=(),
                             use_tools=True)

        return inner

    def test_a_recorded_session_replays_turn_for_turn(self):
        turns = [
            tool_turn("c1", "bash", command="pwd"),
            tool_turn("c2", "edit_file", path="a.c", old_text="x", new_text="y"),
            ModelTurn("done", (), StopReason.END_TURN),
        ]
        self.record(turns)
        replay = ReplayBackend(self.path)
        seen = [replay.complete(system="s", conversation=(), tools=(),
                                use_tools=True) for _ in range(3)]

        self.assertEqual([turn.text for turn in seen], ["", "", "done"])
        self.assertEqual([call.name for turn in seen for call in turn.tool_calls],
                         ["bash", "edit_file"])
        self.assertEqual(seen[1].tool_calls[0].arguments,
                         {"path": "a.c", "old_text": "x", "new_text": "y"})
        self.assertEqual([turn.stop_reason for turn in seen],
                         [StopReason.TOOL_USE, StopReason.TOOL_USE,
                          StopReason.END_TURN])

    def test_replaying_contacts_no_provider(self):
        """The point of the exercise: the slow half is not run at all."""
        self.record([ModelTurn("hello", (), StopReason.END_TURN)])
        replay = ReplayBackend(self.path)
        replay.complete(system="s", conversation=(), tools=(), use_tools=True)

        self.assertEqual(replay.discover_model_name(), "fake-model")

    def test_recording_is_written_as_the_session_goes(self):
        """An interrupted session still leaves what it got to."""
        inner = FakeBackend([ModelTurn("one", (), StopReason.END_TURN),
                             ModelTurn("two", (), StopReason.END_TURN)])
        backend = RecordingBackend(inner, self.path)
        backend.complete(system="s", conversation=(), tools=(), use_tools=True)
        written = [json.loads(line) for line in
                   Path(self.path).read_text().splitlines() if line.strip()]

        self.assertEqual(len(written), 2, "header plus the one turn so far")
        self.assertEqual(written[1]["text"], "one")

    def test_the_recording_running_out_ends_the_turn_rather_than_failing(self):
        """A harness change that adds a round is the case this must survive.

        A new redirect asks for a round the recorded session never had. That
        is the finding, not a crash: the turn ends carrying a marker the
        caller can see.
        """
        self.record([ModelTurn("only one", (), StopReason.END_TURN)])
        replay = ReplayBackend(self.path)
        replay.complete(system="s", conversation=(), tools=(), use_tools=True)
        extra = replay.complete(system="s", conversation=(), tools=(),
                                use_tools=True)

        self.assertTrue(replay.exhausted)
        self.assertEqual(extra.stop_reason, StopReason.END_TURN)
        self.assertEqual(extra.tool_calls, ())
        self.assertIn("recording ends here", extra.text)

    def test_replay_records_what_the_harness_offered_each_round(self):
        """What the harness DID is the thing under test, so it is observable."""
        self.record([tool_turn("c1", "bash", command="pwd"),
                     ModelTurn("done", (), StopReason.END_TURN)])
        replay = ReplayBackend(self.path)
        replay.complete(system="s", conversation=(), tools=(), use_tools=True)
        replay.complete(system="s", conversation=(), tools=(), use_tools=False)

        self.assertEqual([call["use_tools"] for call in replay.calls],
                         [True, False])


class Wiring(unittest.TestCase):
    """Which backend the environment selects."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = str(Path(self.dir.name) / "run.jsonl")
        RecordingBackend(FakeBackend([]), self.path)

    def test_no_environment_leaves_the_backend_alone(self):
        inner = FakeBackend([])

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(session_replay.RECORD_ENV, None)
            os.environ.pop(session_replay.REPLAY_ENV, None)

            self.assertIs(session_replay.wrap(inner), inner)

    def test_replay_wins_over_record(self):
        """Asked for both, recording a replay would overwrite the recording."""
        with patch.dict(os.environ, {session_replay.REPLAY_ENV: self.path,
                                     session_replay.RECORD_ENV: self.path}):
            self.assertIsInstance(session_replay.wrap(FakeBackend([])),
                                  ReplayBackend)

    def test_recording_wraps_without_replacing(self):
        inner = FakeBackend([ModelTurn("x", (), StopReason.END_TURN)])
        other = str(Path(self.dir.name) / "other.jsonl")

        with patch.dict(os.environ, {session_replay.RECORD_ENV: other}):
            os.environ.pop(session_replay.REPLAY_ENV, None)
            wrapped = session_replay.wrap(inner)

        self.assertIsInstance(wrapped, RecordingBackend)
        self.assertIs(wrapped.inner, inner)


class TheOperatorFlags(unittest.TestCase):
    """Recording and replay were reachable only by someone who already knew
    the environment variables existed."""

    def parse(self, argv):
        import os
        import sys
        from unittest.mock import patch

        sys.argv = ["rag_chat", "--safe"]
        import rag_chat

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(session_replay.RECORD_ENV, None)
            os.environ.pop(session_replay.REPLAY_ENV, None)
            rest = rag_chat.apply_env_options(list(argv))

            return (os.environ.get(session_replay.RECORD_ENV),
                    os.environ.get(session_replay.REPLAY_ENV), rest)

    def test_record_and_replay_are_command_line_options(self):
        record, replay, rest = self.parse(
            ["--record", "/tmp/a.jsonl", "--replay", "/tmp/b.jsonl", "--auto"])

        self.assertEqual(record, "/tmp/a.jsonl")
        self.assertEqual(replay, "/tmp/b.jsonl")
        self.assertEqual(rest, ["--auto"], "the flags are consumed")

    def test_they_are_named_in_the_help(self):
        import sys

        sys.argv = ["rag_chat", "--safe"]
        import rag_chat

        for flag in ("--record", "--replay"):
            with self.subTest(flag=flag):
                self.assertIn(flag, rag_chat.HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
