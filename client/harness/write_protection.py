"""What a command granted workspace write may still not change.

The sandbox binds every declared tree read-write for such a command, and the
command policy cannot know what an arbitrary program will write: a patch fed
on stdin, a script, a Makefile rule. The paths found here are bound back
read-only on top of that bind, so the kernel refuses the write whatever
program attempts it.

Discovery reads directories only, never file contents, and is cached per
root for the session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from harness import target_policy


@dataclass(frozen=True)
class ProtectedTrees:
    """What one root holds that no command may write."""

    #: `.git` directories and files, and the git directories `.git` files name.
    git: tuple[str, ...] = ()


# A build's own tree under build/ is its output: the checkouts bitbake
# unpacks there carry their own .git and are its to write, and the tree is
# far too large to walk before every command.

_BUILD_OUTPUT_PREFIXES = ("tmp", "downloads", "sstate-cache")

_CACHE: dict[str, tuple[int, ProtectedTrees]] = {}


def _walk(root: str) -> ProtectedTrees:
    git: list[str] = []
    pending = [root]

    while pending:
        directory = pending.pop()

        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue

        for entry in entries:
            if entry.name == ".git":
                git.append(entry.path)

                if entry.is_file(follow_symlinks=False):
                    named = target_policy.gitdir_named_by(entry.path)

                    if named:
                        git.append(named)

                continue

            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue

            if (os.path.basename(directory) == "build"
                    and entry.name.startswith(_BUILD_OUTPUT_PREFIXES)):
                continue

            pending.append(entry.path)

    return ProtectedTrees(git=tuple(sorted(set(git))))


def trees(root: str) -> ProtectedTrees:
    """The protected trees under `root`, walked once and then cached.

    Walked again when the root itself changes, or after forget(): a command
    that may create a repository (git clone) is followed by one.
    """
    root = str(root)

    try:
        stamp = os.stat(root).st_mtime_ns
    except OSError:
        return ProtectedTrees()

    cached = _CACHE.get(root)

    if cached is not None and cached[0] == stamp:
        return cached[1]

    found = _walk(root)
    _CACHE[root] = (stamp, found)

    return found


def forget() -> None:
    """Drop the cache: the next command walks every root again."""
    _CACHE.clear()


def within(path: str, roots) -> str:
    """The root among `roots` that holds `path`, or ""."""
    best = ""

    for root in roots:
        root = str(root)

        if (path == root or path.startswith(root.rstrip("/") + "/")) and len(root) > len(best):
            best = root

    return best
