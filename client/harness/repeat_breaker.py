"""Stop a turn that keeps making the same successful call to no effect.

The Hermes streak guard only appends a "do not repeat" note: in an FT-3A run
the model wrote the same one-line file 486 times in a row, each call
succeeding with a byte-identical result, and ignored the note until the round
budget ended the turn. Nobody is at the keyboard during a core turn.

Four consecutive calls with the same tool, the same canonical arguments and a
byte-identical successful result are the same evidence four times: the turn
is stopped, truthfully, as `repeated_success_loop`. A failed or refused call
never counts here (refusal_breaker owns those); any other call, a changed
result, or a poller resets the count.
"""

from __future__ import annotations

import hashlib
import json
import os

from agent.hermes.tool_guardrails import is_stall_guard_repeatable

LIMIT = int(os.environ.get("SPEAR_SUCCESS_REPEATS", "4"))
REASON = "repeated_success_loop"


def _key(name: str, arguments, result: str) -> tuple[str, str, str]:
    canonical = json.dumps(arguments if isinstance(arguments, dict) else {},
                           sort_keys=True, default=str)
    return name, canonical, hashlib.sha256(str(result).encode()).hexdigest()


class RepeatBreaker:
    """Counts consecutive identical successful calls; trips at `limit`."""

    def __init__(self, limit: int = LIMIT):
        self.limit = max(2, int(limit))
        self.last: tuple[str, str, str] | None = None
        self.count = 0
        self.tripped: tuple[str, str] | None = None

    def observe(self, name: str, arguments, result: str, ok: bool) -> bool:
        if self.tripped is not None:
            return True

        if not ok or is_stall_guard_repeatable(name):
            self.last, self.count = None, 0
            return False

        key = _key(name, arguments, result)
        self.count = self.count + 1 if key == self.last else 1
        self.last = key

        if self.count >= self.limit:
            self.tripped = (name, key[1])

        return self.tripped is not None

    def conclusion(self) -> str:
        name, arguments = self.tripped
        return (f"I stopped: the same {name} call ({arguments[:160]}) succeeded "
                f"{self.count} times in a row with an identical result, so repeating it "
                f"changed nothing. This is a safety stop, not a claim that the task is "
                f"done; the changes made before this point, if any, are listed above.")
