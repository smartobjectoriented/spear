"""Which files no mutation may touch, whatever tool attempts it.

A refusal attaches to the target, not to the tool. Generated output and
snapshot or third-party copies were protected only from write_file and
patch: delete_file then a fresh write_file, a shell redirection, cp, tee,
truncate or ln -sf reached the same file. Every mutation path asks here,
about the path as written and about the file it actually reaches.
"""

from __future__ import annotations

import os
import re

_GENERATED_PATH = re.compile(r"/generated/|/build/tmp/")
_SNAPSHOT_PATH = re.compile(r"/(?:u-boot|atf|qemu)(?:\.back)?/|\.back/|\.pristine/|\.0/|"
                            r"/build/tmp/")
_GENERATED_MARK = re.compile(
    r"DO NOT (?:MODIFY|EDIT)|auto(?:matically)?[ -]?generated|@generated", re.IGNORECASE)


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

    return bool(_GENERATED_MARK.search(head))


def is_snapshot(path: str) -> bool:
    """Third-party and snapshot trees, never written by a turn."""
    return bool(_SNAPSHOT_PATH.search(str(path)))


def _head(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(600)
    except OSError:
        return ""


def refusal(path: str, label: str | None = None) -> str:
    """Why `path` may not be mutated, or "". `label` names it in the answer."""
    if not path:
        return ""

    label = label or path
    candidates = [str(path)]
    real = os.path.realpath(path)

    if real != str(path):
        candidates.append(real)

    for candidate in candidates:
        if _GENERATED_PATH.search(candidate):
            return f"{label} is a generated file — change its source"

        if is_snapshot(candidate):
            return f"{label} is a snapshot or third-party copy"

    if os.path.isfile(real) and says_generated(_head(real)):
        return f"{label} is a generated file — change its source"

    return ""
