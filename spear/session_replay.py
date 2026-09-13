"""Record what the model said once, and hand it back for free ever after.

Most of what breaks in a long turn is not the model. It is the harness around
it: when the nudge fires, whether a refused round counts, whether a redirect
reaches the model, whether a guard leaves anything standing. All of that is
deterministic, and all of it was diagnosed one afternoon by watching a
ten-minute live run twelve times over -- two and three quarter hours to learn
nine facts that a fixed transcript would have given in seconds.

So the model's side of a session is recorded once and replayed afterwards. The
harness runs for real: real tools, real files, real gates, real guards. Only
the provider is a recording, which is the one part a harness change is not
supposed to alter.

    SPEAR_RECORD_TURNS=/path/run.jsonl   spear-chat ...   # once, live
    SPEAR_REPLAY_TURNS=/path/run.jsonl   spear-chat ...   # as often as needed

What this is honest about: replay answers "what does the harness do with this
transcript", not "what would the model do now". A change that alters what the
model is ASKED still needs a live run to judge -- the replay will faithfully
give back the old answers to the new questions. It is a fast filter in front
of the slow test, never a replacement for it.

When the harness inserts rounds the recording does not have -- which is
exactly what a new redirect does -- the recording runs out. That is reported
as an ordinary end of turn carrying a marker, not as an error: a redirect that
the model never answers is itself the thing under test.
"""

from __future__ import annotations

import json
import os
import threading

from model_backend import ModelToolCall, ModelTurn, StopReason

RECORD_ENV = "SPEAR_RECORD_TURNS"
REPLAY_ENV = "SPEAR_REPLAY_TURNS"

# What a replayed turn says once the recording is spent. Deliberately plain
# text with no tool call: the harness must end the turn, not loop on it.
EXHAUSTED_TEXT = ("[replay] the recording ends here; this turn asked for more "
                  "rounds than the recorded session had")


def _turn_to_dict(turn) -> dict:
    return {
        "text": turn.text,
        "tool_calls": [{"id": call.id, "name": call.name,
                        "arguments": dict(call.arguments)}
                       for call in turn.tool_calls],
        "stop_reason": str(getattr(turn.stop_reason, "value", turn.stop_reason)),
        "usage": dict(turn.usage) if turn.usage else None,
        "error": turn.error,
    }


def _turn_from_dict(raw) -> ModelTurn:
    calls = tuple(
        ModelToolCall(item["id"], item["name"], dict(item.get("arguments") or {}))
        for item in raw.get("tool_calls") or ()
    )

    try:
        stop = StopReason(raw.get("stop_reason") or "end_turn")
    except ValueError:
        stop = StopReason.OTHER

    return ModelTurn(raw.get("text") or "", calls, stop,
                     usage=raw.get("usage"), error=raw.get("error"))


class RecordingBackend:
    """A real backend that also writes down every turn it produces.

    Appended per turn rather than written at the end: a session that is
    interrupted -- and the long ones are -- still leaves everything it got
    up to that point, which is usually the part worth replaying.
    """

    def __init__(self, inner, path):
        self.inner = inner
        self.path = path
        self._lock = threading.Lock()

        directory = os.path.dirname(os.path.abspath(path))

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "kind": "header",
                "model": getattr(inner, "model", ""),
            }) + "\n")

    def discover_model_name(self):
        return self.inner.discover_model_name()

    def complete(self, **kwargs):
        turn = self.inner.complete(**kwargs)

        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_turn_to_dict(turn),
                                            ensure_ascii=False) + "\n")
            except OSError:
                # A recording that cannot be written must not cost the user
                # the session it was recording.
                pass

        return turn


class ReplayBackend:
    """The recorded turns, in order, at no cost.

    `use_tools` from the harness is not consulted: the recording holds what
    the model actually did, and a replayed tool call arriving in a round
    where the harness has closed the tool window is a real finding about the
    harness, not a fault of the replay.
    """

    def __init__(self, path):
        self.path = path
        self.model = ""
        self.turns = []
        self.index = 0
        self.calls = []

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                raw = json.loads(line)

                if raw.get("kind") == "header":
                    self.model = raw.get("model") or ""
                    continue

                self.turns.append(_turn_from_dict(raw))

    def discover_model_name(self):
        return self.model or "replay"

    @property
    def exhausted(self):
        return self.index >= len(self.turns)

    def complete(self, *, system="", conversation=(), tools=(), use_tools=True,
                 on_token=None):
        self.calls.append({"use_tools": use_tools,
                           "messages": len(list(conversation))})

        if self.exhausted:
            return ModelTurn(EXHAUSTED_TEXT, (), StopReason.END_TURN)

        turn = self.turns[self.index]
        self.index += 1

        # The spinner is fed so a replayed run looks like a run; it costs
        # nothing and keeps the CLI's own code on one path.
        if on_token and turn.text:
            on_token()

        return turn


def wrap(backend):
    """The backend this process should actually use, per the environment.

    Replay wins over record: asked for both, the intent is to replay, and
    recording a replay would overwrite the recording with itself.
    """
    replay = os.environ.get(REPLAY_ENV, "").strip()

    if replay:
        return ReplayBackend(replay)

    record = os.environ.get(RECORD_ENV, "").strip()

    if record:
        return RecordingBackend(backend, record)

    return backend
