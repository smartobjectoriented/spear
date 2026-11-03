"""Which objects of the workspace this session created.

Deleting what the session itself made is cleanup; deleting what was there
before is a change to the project. The policy refused both alike, so a
model that had built into a scratch directory could neither remove it nor
keep it out of the patch series it regenerated, and left the change
unpersisted rather than leak the scratch files.

The inventory of the declared trees is taken when a session binds its
workspace: every path present then, and every (device, inode). An object
is SESSION_CREATED when neither its path nor its inode was in it: a file
recreated at a path that existed, or a hard link to a file that existed,
stays what it was. A tree too large to inventory is never anyone's to
delete.
"""

from __future__ import annotations

import os

from harness import target_policy

# A walk of a whole build tree is a few hundred thousand entries; past this
# the root is left out of the inventory and nothing in it counts as created.

_MAX_ENTRIES = 2_000_000
_MAX_REMOVED = 200_000

_paths: set[int] = set()
_inodes: set[tuple[int, int]] = set()
# The roots last asked for, so that a rebind of the same workspace keeps its
# inventory, and the ones whose walk completed: only those hold created objects.
_asked: tuple[str, ...] = ()
_complete: tuple[str, ...] = ()


def _key(path: str) -> int:
    return hash(path)


def _where(path: str) -> str:
    """`path` with its directories resolved and its last component as named:
    a link is judged as the link, as rm and delete_file remove the link."""
    path = os.path.abspath(path).rstrip("/") or "/"

    return os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))


def _walk(root: str):
    """(path keys, inodes) of everything under `root`, or None past the bound."""
    paths, inodes = {_key(root)}, set()
    pending = [root]
    count = 0

    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue

        for entry in entries:
            count += 1

            if count > _MAX_ENTRIES:
                return None

            paths.add(_key(entry.path))

            try:
                status = entry.stat(follow_symlinks=False)
            except OSError:
                continue

            inodes.add((status.st_dev, status.st_ino))

            if entry.is_dir(follow_symlinks=False):
                pending.append(entry.path)

    return paths, inodes


def capture(roots) -> None:
    """Take the start inventory of `roots`, once per set of roots."""
    global _asked, _complete

    roots = tuple(sorted(os.path.realpath(str(root)) for root in roots))

    if roots == _asked:
        return

    _paths.clear()
    _inodes.clear()
    complete = []

    for root in roots:
        found = _walk(root)

        if found is not None:
            _paths.update(found[0])
            _inodes.update(found[1])
            complete.append(root)

    _asked, _complete = roots, tuple(complete)


def forget() -> None:
    """Start over: no inventory, nothing counts as created."""
    global _asked, _complete

    _paths.clear()
    _inodes.clear()
    _asked, _complete = (), ()


def known(path: str) -> bool:
    """Whether `path` lies in a tree whose start state was inventoried."""
    return bool(_inside(_where(path)))


def _inside(path: str) -> str:
    return next((root for root in _complete
                 if path == root or path.startswith(root + "/")), "")


def created(path: str) -> bool:
    """Whether the object at `path` was created in this session."""
    where = _where(path)
    root = _inside(where)

    if not root or where == root:
        return False

    try:
        status = os.lstat(where)
    except OSError:
        return False

    return _key(where) not in _paths and (status.st_dev, status.st_ino) not in _inodes


def start_state_file(path: str) -> bool:
    """Whether `path` is a file of the start state: inventoried, present,
    and not this session's. Its changes go through the checkpointed tools."""
    where = _where(path)

    if not _inside(where) or not os.path.lexists(where) or os.path.isdir(where):
        return False

    return not created(where)


def _protected(path: str) -> str:
    """What no provenance makes deletable: git state, a snapshot copy, a
    guard's fingerprint."""
    if target_policy.is_git_metadata(path):
        return "git metadata"

    if target_policy.is_snapshot(path + ("/" if os.path.isdir(path) and not os.path.islink(path) else "")):
        return "a snapshot copy"

    if target_policy.fingerprinted_tree(path):
        return "control-plane metadata"

    return ""


def removal_refusal(path: str, label: str | None = None) -> str:
    """Why the session may not delete `path` as cleanup, or "".

    A file or link: created in this session, and not protected. A
    directory: the same for itself and for every object under it, links
    judged as links and refused when they point at anything the session did
    not create.
    """
    label = label or path
    where = _where(path)

    if not os.path.lexists(where):
        return f"{label} does not exist"

    if not _inside(where):
        return f"{label} is outside the trees whose start state is known"

    found = _protected(where)

    if found:
        return f"{label} is {found}"

    if not created(where):
        return f"{label} existed when the session started, so it is not the session's to delete"

    if os.path.islink(where) or not os.path.isdir(where):
        return _link_refusal(where, label)

    count = 0

    for directory, subdirectories, files in os.walk(where, followlinks=False):
        for name in subdirectories + files:
            count += 1
            item = os.path.join(directory, name)
            relative = os.path.join(label, os.path.relpath(item, where))

            if count > _MAX_REMOVED:
                return f"{label} holds more than {_MAX_REMOVED} objects: delete it in parts"

            found = _protected(item)

            if found:
                return f"{relative} is {found}"

            if not created(item):
                return f"{relative} existed when the session started, so {label} is not the session's to delete"

            refusal = _link_refusal(item, relative) if os.path.islink(item) else ""

            if refusal:
                return refusal

    return ""


def _link_refusal(path: str, label: str) -> str:
    if not os.path.islink(path):
        return ""

    target = os.path.realpath(path)

    if os.path.lexists(target) and not created(target):
        return f"{label} is a link to {target}, which the session did not create"

    return ""
