"""Which files no mutation may touch, whatever tool attempts it.

A refusal attaches to the target, not to the tool. Generated output and
snapshot copies were protected only from write_file and
patch: delete_file then a fresh write_file, a shell redirection, cp, tee,
truncate or ln -sf reached the same file. Every mutation path asks here,
about the path as written and about the file it actually reaches.

A repository's own state is a target too: no tool edits git metadata, and
the sandbox binds it read-only under every command that may write
(write_protection.py).
"""

from __future__ import annotations

import os
import re

_GENERATED_PATH = re.compile(r"/generated/|/build/tmp/")

# A directory named as a copy is one wherever it sits. ".0" names a version as
# often as a copy (zynq-rev1.0, linux-6.0), so it counts only beside the live
# tree it copies (rootfs.0 next to rootfs). A component's name never decides:
# an attached u-boot/ or a recipes-qemu/qemu/ layer is the project's source.

_SNAPSHOT_SUFFIXES = (".back", ".pristine")
_COPY_SUFFIX = ".0"
_GENERATED_MARK = re.compile(
    r"DO NOT (?:MODIFY|EDIT)|auto(?:matically)?[ -]?generated|@generated", re.IGNORECASE)

# A generator says so in a comment at the top of what it writes. The same
# words in prose are not that: Linux's init/Kconfig mentions "auto-generated"
# in the help text of its first option.

_COMMENT_LINE = re.compile(r"\s*(?:#|//|/\*|\*|;|--|%|<!--|\"\"\"|\'\'\'|dnl\b)")
_MARKER_LINES = 10

# A saved Kconfig configuration is the source a .config is made from, even
# though Linux, U-Boot, Buildroot and BusyBox write it with the same
# "Automatically generated file; DO NOT EDIT." header.

_SAVED_CONFIG = re.compile(r"(?:^|_)defconfig$")

# Control-plane metadata: what a guard reads to decide whether a tree can be
# trusted. A tree's fingerprint sits beside the directory it describes and is
# named after it (a linux.attach.sha256 next to linux/). Edited or
# deleted by hand, it makes the guard vouch for a tree it never checked; only
# the operation that owns it, a build, writes it.

_FINGERPRINT_SUFFIXES = frozenset({"sha1", "sha256", "sha512", "md5", "fingerprint",
                                   "stamp", "manifest"})

_GENERATED = " is a generated file — change its source"

# Every file a refusal named in this session. The sandbox binds them back
# read-only under a command that is not a build (write_protection.py).

_REFUSED: set[str] = set()


def says_generated(head: str) -> bool:
    """Whether a file's opening says the file is generated.

    A unified diff carries the lines of the file it patches: a marker among
    them belongs to that file, not to the patch. A Buildroot defconfig patch
    adds "Automatically generated file; DO NOT EDIT" and is edited by hand.
    """
    if re.search(r"^\+\+\+ ", head, re.M) and re.search(r"^@@ ", head, re.M):
        head = "\n".join(line for line in head.splitlines()
                         if line.startswith(("--- ", "+++ "))
                         or not line.startswith(("+", "-", " ")))

    return any(_COMMENT_LINE.match(line) and _GENERATED_MARK.search(line)
               for line in head.splitlines()[:_MARKER_LINES])


def is_snapshot(path: str) -> bool:
    """Whether `path` lies in a snapshot copy, never written by a turn."""
    path = str(path)

    if "/build/tmp/" in path:
        return True

    parts = path.split("/")
    directories = parts if path.endswith("/") else parts[:-1]

    for index, part in enumerate(directories):
        if part.endswith(_SNAPSHOT_SUFFIXES):
            return True

        live = part[:-len(_COPY_SUFFIX)]

        if part.endswith(_COPY_SUFFIX) and live and os.path.isdir("/".join(parts[:index] + [live]) or "/"):
            return True

    return False


def fingerprinted_tree(path: str) -> str:
    """The directory `path` is the fingerprint of, or "".

    `name.sha256` or `name.<tag>.sha256` beside a directory `name`: every
    dotted prefix of the file name is tried, so `linux-6.12.attach.sha256`
    finds `linux-6.12/`.
    """
    path = str(path).rstrip("/")
    directory, name = os.path.split(path)
    parts = name.split(".")

    if len(parts) < 2 or parts[-1] not in _FINGERPRINT_SUFFIXES:
        return ""

    for end in range(1, len(parts)):
        tree = ".".join(parts[:end])

        if tree and os.path.isdir(os.path.join(directory or ".", tree)):
            return os.path.join(directory, tree)

    return ""


def gitdir_named_by(dotgit: str) -> str:
    """The directory a `.git` FILE names (a worktree's or a submodule's), or ""."""
    try:
        with open(dotgit, "r", encoding="utf-8", errors="replace") as handle:
            line = handle.readline(4096)
    except OSError:
        return ""

    if not line.startswith("gitdir:"):
        return ""

    return os.path.realpath(os.path.join(os.path.dirname(dotgit), line[7:].strip()))


def _is_gitdir(directory: str) -> bool:
    return (os.path.isfile(os.path.join(directory, "HEAD"))
            and (os.path.isdir(os.path.join(directory, "objects"))
                 or os.path.isfile(os.path.join(directory, "commondir"))))


def is_git_metadata(path: str) -> bool:
    """Whether `path` is part of a repository's own state.

    A `.git` component, file or directory, is. So is anything inside a git
    directory with another name: the one a `.git` file points to
    (--separate-git-dir, a worktree's) or a bare repository, recognised by
    its HEAD beside its objects or its commondir.
    """
    path = os.path.abspath(str(path))

    if ".git" in path.split("/"):
        return True

    directory = path

    while directory != "/":
        if _is_gitdir(directory):
            return True

        directory = os.path.dirname(directory)

    return False


def _head(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(600)
    except OSError:
        return ""


def refusal(path: str, label: str | None = None, *, follow: bool = True,
            deleting: bool = False) -> str:
    """Why `path` may not be mutated, or "". `label` names it in the answer.

    `follow=False` judges a symbolic link as itself, for deleting it: what
    it points at is not what changes. `deleting`: generated output the
    session created itself may be removed as cleanup (provenance.py); git
    state, snapshot copies and control-plane metadata may not, whoever
    made them.
    """
    if not path:
        return ""

    answer = _judge(str(path), label or str(path), follow)

    if deleting and answer.endswith(_GENERATED):
        from harness import provenance

        if not provenance.removal_refusal(str(path)):
            return ""

    if answer and follow:
        _REFUSED.add(os.path.realpath(path))

    return answer


def refused_paths() -> frozenset[str]:
    """The files a refusal named in this session, as they resolve."""
    return frozenset(_REFUSED)


def forget_refused() -> None:
    """Start a new session's record of refused files."""
    _REFUSED.clear()


def _judge(path: str, label: str, follow: bool) -> str:
    candidates = [str(path)]
    real = (os.path.realpath(path) if follow else
            os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(path))),
                         os.path.basename(path)))

    if real != str(path):
        candidates.append(real)

    # A directory is judged as the tree it is: `mkdir generated` or a write
    # aimed at `.../generated` itself is a change inside it.

    candidates += [item.rstrip("/") + "/" for item in list(candidates) if os.path.isdir(item)]

    for candidate in candidates:
        if is_git_metadata(candidate):
            return f"{label} is git metadata — change it through git, not by editing it"

    for candidate in candidates:
        tree = fingerprinted_tree(candidate)

        if tree:
            return (f"{label} is control-plane metadata: the guard of "
                    f"{os.path.basename(tree)}/ reads it, and only the build that owns it "
                    f"may change it")

    for candidate in candidates:
        if _GENERATED_PATH.search(candidate):
            return f"{label}{_GENERATED}"

        if is_snapshot(candidate):
            return f"{label} is a snapshot copy"

    if (follow and os.path.isfile(real) and not _SAVED_CONFIG.search(os.path.basename(real))
            and says_generated(_head(real))):
        return f"{label}{_GENERATED}"

    return ""
