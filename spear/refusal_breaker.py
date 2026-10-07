"""Stop a turn that keeps asking for what was already refused.

A refusal is deterministic: the same operation, against the same boundary,
is refused the same way every time. A turn in the Phase 13 benchmark issued
one refused `ls` of a directory outside its workspace 482 times, until the
round budget ran out. Three repetitions are not yet a loop -- seven runs
that repeated a refused call three times still finished correctly -- so the
limit is five, the smallest that stopped none of them.

Two calls are the same operation when they name the same tool, target and
refusal once spacing, quoting, a leading ./ and trailing slashes are set
aside, and the refusal's paths and numbers are masked.
"""

from __future__ import annotations

import os
import re
from collections import Counter

LIMIT = int(os.environ.get("SPEAR_REFUSAL_REPEATS", "5"))

_PATH = re.compile(r"(?:/[\w.@+-]+)+/?")
_NUMBER = re.compile(r"\d+")


def _target(name: str, arguments) -> str:
    arguments = arguments if isinstance(arguments, dict) else {}

    if name == "terminal":
        text = str(arguments.get("command") or "")
    else:
        text = str(arguments.get("path") or arguments.get("pattern") or "")

    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"(?<![\w.])\./", "", text)
    text = re.sub(r"/+(?=\s|$)", "", text)

    return " ".join(text.split())


def _reason(result: str) -> str:
    first = str(result or "").strip().splitlines()[0] if str(result or "").strip() else ""

    return _NUMBER.sub("#", _PATH.sub("<path>", first))[:160]


def signature(name: str, arguments, result: str) -> tuple[str, str, str]:
    return name, _target(name, arguments), _reason(result)


class RefusalBreaker:
    """Counts refusals by operation; trips at `limit` of the same one."""

    def __init__(self, limit: int = LIMIT):
        self.limit = max(1, int(limit))
        self.counts: Counter = Counter()
        self.tripped: tuple[str, str, str] | None = None

    def observe(self, name: str, arguments, result: str, refused: bool) -> bool:
        if not refused or self.tripped is not None:
            return self.tripped is not None

        key = signature(name, arguments, result)
        self.counts[key] += 1

        if self.counts[key] >= self.limit:
            self.tripped = key

        return self.tripped is not None

    def conclusion(self) -> str:
        name, target, reason = self.tripped
        return (f"I stopped: the same {name} operation ({target or 'no target'}) was "
                f"refused {self.counts[self.tripped]} times, and repeating it cannot change "
                f"that. The refusal was: {reason}. The changes made before this point, if "
                f"any, are listed above; the remaining step needs a different approach or "
                f"an operator decision.")
