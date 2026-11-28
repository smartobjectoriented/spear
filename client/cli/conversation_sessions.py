"""Named conversations on one corpus, and the lock that keeps each to one process.

A corpus used to hold a single conversation: two spear-chat processes on the
same tree read and rewrote the same history file, and the last to save won.
A conversation now has a name; the unnamed one keeps the file it always had,
so nothing changes for whoever never names one.
"""

import fcntl
import glob
import json
import os
import re
from datetime import datetime


NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,40}")

USAGE = ("usage: /session [list] | /session new <name> | "
         "/session switch <name> | /session rm <name>")


class ConversationError(Exception):
    """A conversation that cannot be named, opened or removed as asked."""


def validate_name(name):
    """The name itself, or ConversationError. None names the default one."""
    if name is None:
        return None

    if not NAME_PATTERN.fullmatch(name) or name in (".", ".."):
        raise ConversationError(
            f"invalid session name {name!r}: 1 to 40 of A-Z a-z 0-9 _ . -")

    return name


def history_file(default_file, name):
    """Where conversation `name` keeps its history, beside the default one."""
    if name is None:
        return default_file

    return f"{default_file[:-len('.json')]}-{validate_name(name)}.json"


def describe(name):
    return name if name is not None else "(default)"


class ConversationLock:
    """An exclusive, advisory lock on one conversation, held while it is open.

    flock and not a pid file: the kernel drops it when the process dies, so a
    crashed session never leaves its conversation unusable.
    """

    def __init__(self):
        self.path = None
        self._handle = None

    def acquire(self, history_path):
        """Take the lock of `history_path`, or raise ConversationError."""
        path = history_path + ".lock"

        if path == self.path:
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handle = open(path, "a+")

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise ConversationError(
                "this session is open in another spear-chat; "
                "pick another with --session <name> or /session switch <name>"
            ) from None

        self.release()
        self.path, self._handle = path, handle

    def release(self):
        if self._handle is not None:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()

        self.path, self._handle = None, None

    @staticmethod
    def held_elsewhere(history_path):
        """Whether another process holds `history_path` (a probe, not a take)."""
        path = history_path + ".lock"

        if not os.path.exists(path):
            return False

        with open(path, "a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True

            fcntl.flock(handle, fcntl.LOCK_UN)

        return False


def list_conversations(default_file):
    """(name, history path) for every conversation of this corpus, default first."""
    found = [(None, default_file)]
    prefix = default_file[:-len(".json")] + "-"

    names = set()

    for suffix in (".json", ".json.lock"):
        for path in glob.glob(glob.escape(prefix) + "*" + suffix):
            name = path[len(prefix):-len(suffix)]

            if NAME_PATTERN.fullmatch(name) and (
                    suffix == ".json" or ConversationLock.held_elsewhere(
                        path[:-len(".lock")])):
                names.add(name)

    return found + [(name, f"{prefix}{name}.json") for name in sorted(names)]


def summary_lines(default_file, current, lock):
    """The /session listing: name, messages, last change, who holds it."""
    lines = []

    for name, path in list_conversations(default_file):
        try:
            with open(path) as handle:
                count = len(json.load(handle))
            changed = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                "%Y-%m-%d %H:%M")
        except (OSError, ValueError):
            count, changed = 0, "-"

        mark = "*" if name == current else " "
        state = ("  (open elsewhere)" if name != current
                 and ConversationLock.held_elsewhere(path) else "")

        lines.append(f"  {mark} {describe(name):<24} {count:4d} messages  "
                     f"{changed}{state}")

    return lines


def remove(default_file, name, current):
    """Delete conversation `name`'s history; never the open or the default one."""
    validate_name(name)

    if name is None or name == current:
        raise ConversationError("cannot remove the session in use")

    path = history_file(default_file, name)

    if not os.path.exists(path):
        raise ConversationError(f"no session named {name}")

    if ConversationLock.held_elsewhere(path):
        raise ConversationError(f"session {name} is open in another spear-chat")

    os.remove(path)

    try:
        os.remove(path + ".lock")
    except FileNotFoundError:
        pass
