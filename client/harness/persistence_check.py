"""Whether a patch series reproduces an attached tree.

An updiff that succeeds has only captured what it chose to capture: it skips
a file with a chain of patches, a binary, an excluded name. Whether the
series persists the tree is a separate fact, established here from outside
the turn: the recipe's pristine copy and its series, in a disposable
directory, with the patch tool as do_patch drives it, compared against the
attached tree. The working tree contributes nothing to the reconstruction.

Three things are never decided by the attached tree or the turn:

- the base: the pristine copy is pinned by digest when the session first sees
  it and when a build rewrites it, so a turn that changes it fails the check;
- the build-output rules: patch.bbclass's own exclude list as the session
  found it, and the ignore rules of the pristine copy, so a turn cannot
  declare its own change to be output;
- the verdict: nothing here reads what a command or the model reported.

Only a file on one side alone can be build output. A file both trees hold
must be identical whatever its name, because the exclude list also stops an
updiff from capturing a real source named like an artefact.

A recipe's do_handle_symlinks adds links after do_patch, declared in the
recipe, not in the series. A link the attached tree alone holds is the recipe's
when the patched source beside the pristine copy holds the same link.

An updiff writes a deletion with `diff -N`, as a patch that empties the file,
and the patch tool then leaves it empty. So a file the reconstruction holds
empty and the attached tree does not hold is the same state. Not the reverse:
an empty file only the attached tree holds is not reproduced.

Every failure to establish the result, of any kind, is a failure of the check.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

TIMEOUT_SECONDS = 300
PATCH = ("patch", "-p1", "--forward", "--batch", "--quiet", "--no-backup-if-mismatch")
CLASS = "build/meta/classes/patch.bbclass"

# What the attach leaves at the top of a tree and no series describes: the
# quilt staging do_patch made, and a fetched repository's own metadata.

_STAGING = (".git", ".pc", "patches")


@dataclass(frozen=True)
class Verification:
    """The result of one check; `verified` is PERSISTENCE_VERIFIED."""

    verified: bool
    reason: str = ""
    differences: tuple[str, ...] = ()
    patches: int = 0
    base: str = ""
    series: str = ""
    reconstructed: str = ""
    attached: str = ""
    seconds: float = 0.0

    def summary(self) -> str:
        fields = [f"PERSISTENCE_VERIFIED={'true' if self.verified else 'false'}",
                  f"patches={self.patches}", f"differences={len(self.differences)}",
                  f"seconds={self.seconds:.1f}", f"base={self.base[:16]}",
                  f"series={self.series[:16]}", f"reconstructed={self.reconstructed[:16]}",
                  f"attached={self.attached[:16]}"]

        return " ".join(fields + ([f"reason={self.reason}"] if self.reason else []))


class _Failed(Exception):
    pass


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _objects(root: str) -> dict[str, tuple[str, str]]:
    """{relative path: (kind, path or link target)}, the staging left out."""
    found = {}

    for directory, subdirectories, names in os.walk(root):
        if directory == root:
            subdirectories[:] = [name for name in subdirectories if name not in _STAGING]

        for name in names + [item for item in subdirectories
                             if os.path.islink(os.path.join(directory, item))]:
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, root)

            if os.path.islink(path):
                found[relative] = ("link", os.readlink(path))
            elif os.path.isfile(path):
                found[relative] = ("file", path)
            else:
                found[relative] = ("other", "")

    return found


def _identity(item: tuple[str, str]) -> str:
    kind, value = item

    return f"file:{_file_digest(value)}" if kind == "file" else f"{kind}:{value}"


def _digest_of(identities: dict[str, str]) -> str:
    digest = hashlib.sha256()

    for relative in sorted(identities, key=lambda path: path.encode("utf-8", "surrogateescape")):
        digest.update(f"{relative}\0{identities[relative]}\n".encode("utf-8", "surrogateescape"))

    return digest.hexdigest()


def tree_digest(root: str) -> str:
    return _digest_of({relative: _identity(item) for relative, item in _objects(root).items()})


def series_digest(patch_dir: str, listed: list[str]) -> str:
    return _digest_of({name: _file_digest(os.path.join(patch_dir, name)) for name in listed})


def exclude_patterns(project: str) -> list[str]:
    """do_diffcompose's exclude list, read from the project's own class."""
    with open(os.path.join(project, CLASS), encoding="utf-8") as handle:
        source = handle.read()

    block = source[source.index("exclude_patterns = ["):]
    block = block[block.index("["):block.index("]") + 1]

    return list(ast.literal_eval(re.sub(r"#.*", "", block)))


def _ignored_by_base(base: str, paths: list[str], deadline: float) -> set[str]:
    """The paths the pristine copy's own .gitignore files ignore. A throwaway
    repository, so that no project or user rule takes part."""
    if not paths:
        return set()

    with tempfile.TemporaryDirectory(prefix="spear-ignore-") as directory:
        subprocess.run(["git", "init", "-q", directory], capture_output=True, check=True,
                       timeout=_left(deadline))
        done = subprocess.run(
            ["git", "-c", "core.excludesFile=/dev/null", "--git-dir",
             os.path.join(directory, ".git"), "--work-tree", base, "check-ignore", "--no-index",
             "-z", "--stdin"],
            input="\0".join(paths).encode("utf-8", "surrogateescape"), capture_output=True,
            timeout=_left(deadline))

    if done.returncode not in (0, 1):
        raise _Failed("the ignore rules could not be read")

    return {name.decode("utf-8", "surrogateescape") for name in done.stdout.split(b"\0") if name}


def _left(deadline: float) -> float:
    left = deadline - time.monotonic()

    if left <= 0:
        raise _Failed("reconstruction timed out")

    return left


def _apply(patch_dir: str, listed: list[str], top: str, deadline: float) -> None:
    for name in listed:
        with open(os.path.join(patch_dir, name), "rb") as handle:
            done = subprocess.run((*PATCH, "-d", top), stdin=handle, capture_output=True,
                                  timeout=_left(deadline))

        if done.returncode != 0:
            raise _Failed(f"{name} does not apply")


def _deleted(item: tuple[str, str]) -> bool:
    return item[0] == "file" and os.path.getsize(item[1]) == 0


def _recipe_link(base: str, path: str, item: tuple[str, str]) -> bool:
    source = os.path.join(base.removesuffix(".pristine"), path)

    return item[0] == "link" and os.path.islink(source) and os.readlink(source) == item[1]


def differences(attached: str, rebuilt: str, base: str, patterns: list[str],
                deadline: float) -> tuple[list[str], str, str]:
    """(what differs, digest of the attached state, digest of the rebuilt one)."""
    mine, theirs = _objects(attached), _objects(rebuilt)
    alone = sorted(set(mine) ^ set(theirs))
    named = {path for path in alone
             if any(fnmatch.fnmatch(part, pattern) for part in path.split("/")
                    for pattern in patterns)}
    output = named | _ignored_by_base(base, [path for path in alone if path not in named], deadline)
    found, left, right = [], {}, {}

    for path in alone:
        if path in output or (path in theirs and _deleted(theirs[path])):
            continue

        if path in mine and _recipe_link(base, path, mine[path]):
            continue

        if path in mine:
            found.append(f"{path}: not reproduced by the series")
            left[path] = _identity(mine[path])
        else:
            found.append(f"{path}: in the series, not in the attached tree")
            right[path] = _identity(theirs[path])

    for path in sorted(set(mine) & set(theirs)):
        _left(deadline)
        left[path], right[path] = _identity(mine[path]), _identity(theirs[path])

        if left[path] != right[path]:
            found.append(f"{path}: differs from the series")

    return found, _digest_of(left), _digest_of(right)


def verify(tree: str, base: str, pinned: str, patterns: list[str], patch_dir: str,
           listed: list[str], *, timeout: float = TIMEOUT_SECONDS) -> Verification:
    """Reconstruct `tree` from `base` and the series, and compare. Never raises."""
    started = time.monotonic()
    deadline = started + timeout
    scratch = ""
    facts: dict[str, object] = {"patches": len(listed)}

    try:
        scratch = tempfile.mkdtemp(prefix="spear-persistence-")
        top = os.path.join(scratch, "tree")
        facts["series"] = series_digest(patch_dir, listed)
        subprocess.run(["cp", "-a", "--reflink=auto", base, top], check=True, capture_output=True,
                       timeout=_left(deadline))
        facts["base"] = tree_digest(top)

        if facts["base"] != pinned:
            raise _Failed("the pristine copy is not the one this session pinned")

        _apply(patch_dir, listed, top, deadline)
        found, facts["attached"], facts["reconstructed"] = differences(
            tree, top, base, patterns, deadline)
        facts["differences"] = tuple(found)

        if found:
            raise _Failed("the series does not reproduce the attached tree")

        return Verification(True, seconds=time.monotonic() - started, **facts)
    except _Failed as failure:
        reason = str(failure)
    except subprocess.TimeoutExpired:
        reason = "reconstruction timed out"
    except Exception as failure:
        reason = f"reconstruction failed: {type(failure).__name__}: {failure}"
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    return Verification(False, reason, seconds=time.monotonic() - started, **facts)
