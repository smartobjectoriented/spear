"""Deterministic action fingerprinting and conservative stall detection."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Mapping


def _normalise_command(command: str) -> str:
    try:
        return shlex.join(shlex.split(command))
    except ValueError:
        return " ".join(command.split())


# Programs whose whole purpose is to show a file. Reading the same file again
# with a different one of these is the same action, not a new one.

_INSPECTION_BINARIES = frozenset({
    "cat", "head", "tail", "od", "hexdump", "xxd", "strings", "nl", "less",
    "more", "bat", "wc", "stat", "file", "sed", "cut", "tr", "fold", "rev",
    "column",
})


def _inspected_path(command: str) -> str | None:
    """The file a read command is looking at, if that is all it is doing.

    Fingerprinting the command text made nine consecutive looks at the same
    three lines -- cat -A, od -c, hexdump -C, strings, tr -d -- count as nine
    different actions, so nothing noticed the model had stopped making
    progress. What repeats there is the FILE, not the spelling.

    grep and find are deliberately absent: searching the same file for a
    different pattern is a different question, and collapsing those would call
    real work a stall.
    """

    try:
        argv = shlex.split(command)
    except ValueError:
        return None

    if not argv or argv[0].rsplit("/", 1)[-1] not in _INSPECTION_BINARIES:
        return None

    for argument in argv[1:]:
        if argument.startswith("-"):
            continue

        if "/" in argument or "." in argument:
            return argument

    return None


# A numeric sed address: "1,200p", "592p", "200,$p".
_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+|\$))?p$")

# Lines per block of evidence. Coarse on purpose: 59..61 and 58..63 are the
# same look, 1..200 and 200..400 are not.
_BLOCK = 100


def read_evidence(command: str) -> tuple[str, ...]:
    """The blocks of a file a windowed read puts in front of the model.

    Collapsing every look at a file into one action caught nine looks at the
    same three lines, and stays: cat -A, od -c and hexdump -C of one file are
    one action. It also caught a model reading an 897-line file in four
    200-line windows -- sed -n '1,200p', '200,400p', '400,600p', '600,897p'
    -- and called that a stall on the fourth window, one round before the
    edit the task had asked for. Same action, yes; but each window was
    content the task had not seen, and unseen content is what progress IS
    here. So a windowed read yields its blocks as evidence, and the monitor
    counts a repeat only when nothing new was shown.
    """
    path = _inspected_path(command)

    if path is None:
        return ()

    try:
        argv = shlex.split(command)
    except ValueError:
        return ()

    binary = argv[0].rsplit("/", 1)[-1]
    start = end = None

    if binary == "sed":
        for item in argv[1:]:
            found = _SED_RANGE.match(item)

            if found:
                start = int(found.group(1))
                end = (start if found.group(2) is None
                       else None if found.group(2) == "$"
                       else int(found.group(2)))
                break
    elif binary in ("head", "tail"):
        count = None

        for index, item in enumerate(argv[1:], 1):
            if item == "-n" and index < len(argv) - 1:
                count = argv[index + 1]
            elif re.fullmatch(r"-n?[+-]?\d+", item):
                count = item.lstrip("-n")

        if count is not None and re.fullmatch(r"[+-]?\d+", count):
            if binary == "head":
                start, end = 1, abs(int(count))
            elif count.startswith("+"):
                start, end = int(count[1:]), None
            else:
                # The last N lines: a window with no line numbers to block
                # on, distinct from the head and from any numbered range.
                return (f"{path}#tail{abs(int(count))}",)

    if start is None:
        return ()

    last = end if end is not None else start + _BLOCK

    return tuple(f"{path}#L{block * _BLOCK}"
                 for block in range((start - 1) // _BLOCK,
                                    (max(last, start) - 1) // _BLOCK + 1))


def action_fingerprint(tool_name: str, arguments: Mapping[str, object]) -> str:
    """An identity for an action, so the same action twice is recognisable."""

    data = dict(arguments)

    if tool_name == "bash" and isinstance(data.get("command"), str):
        looked_at = _inspected_path(data["command"])

        if looked_at is not None:
            return f"bash:read:{looked_at}"

        data["command"] = _normalise_command(data["command"])

    return tool_name + ":" + json.dumps(data, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ProgressObservation:
    fingerprint: str
    repeated: bool
    consecutive_without_progress: int
    stalled: bool


@dataclass
class ProgressMonitor:
    stall_threshold: int = 4
    _last_fingerprint: str | None = field(default=None, init=False, repr=False)
    _without_progress: int = field(default=0, init=False, repr=False)
    _evidence: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stall_threshold < 2:
            raise ValueError("stall_threshold must be at least 2")

    def observe_action(
        self, tool_name: str, arguments: Mapping[str, object], *,
        evidence: tuple[str, ...] = (), mutation: bool = False,
        verification: bool = False, fruitless: bool = False,
    ) -> ProgressObservation:
        """Record one action and say whether the task is getting anywhere."""

        # Progress is anything the task did not already have: a change, a
        # verification, or a path it had not read before. Repeating an action
        # that produced any of those is not a repeat worth counting.

        fingerprint = action_fingerprint(tool_name, arguments)

        # A windowed read of lines not yet shown is progress of its own.

        if tool_name == "bash" and isinstance(arguments.get("command"), str):
            evidence = tuple(evidence) + read_evidence(arguments["command"])

        new_evidence = any(item and item not in self._evidence for item in evidence)

        self._evidence.update(item for item in evidence if item)

        progress = mutation or verification or new_evidence
        repeated = fingerprint == self._last_fingerprint and not progress

        # A repeat with nothing gained accumulates; a different action that
        # gained something resets it. What was missing is the case between
        # the two: a DIFFERENT action returning a result the turn had
        # already been given. Twelve greps differing only in a --include
        # glob, each returning the same matches, reset the counter twelve
        # times and the turn ran to a hundred and twenty-seven calls. The
        # caller knows -- it compares results -- so it says so.

        if progress:
            self._without_progress = 0
        elif repeated or fruitless:
            self._without_progress += 1
        else:
            self._without_progress = 0

        self._last_fingerprint = fingerprint

        return ProgressObservation(
            fingerprint, repeated, self._without_progress,
            self._without_progress >= self.stall_threshold - 1,
        )

    def record_progress(self, evidence: str) -> None:
        if evidence:
            self._evidence.add(evidence)

        self._without_progress = 0

    @property
    def consecutive_without_progress(self) -> int:
        return self._without_progress

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stall_threshold": self.stall_threshold,
            "last_fingerprint": self._last_fingerprint,
            "without_progress": self._without_progress,
            "evidence": sorted(self._evidence),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ProgressMonitor":
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            raise ValueError("unsupported progress monitor state")

        monitor = cls(int(raw.get("stall_threshold", 4)))
        last = raw.get("last_fingerprint")

        if last is not None and not isinstance(last, str):
            raise ValueError("invalid last action fingerprint")

        count = raw.get("without_progress", 0)
        evidence = raw.get("evidence", ())

        if not isinstance(count, int) or count < 0:
            raise ValueError("invalid progress count")

        if not isinstance(evidence, (list, tuple)) or not all(
            isinstance(item, str) for item in evidence
        ):
            raise ValueError("invalid progress evidence")

        monitor._last_fingerprint = last
        monitor._without_progress = count
        monitor._evidence = set(evidence)

        return monitor
