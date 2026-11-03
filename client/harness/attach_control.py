"""The attach mechanism's own state, and what may change it.

A layered build attaches a component's source to a tree (linux/linux, u-boot/)
and keeps beside it what its guard reads: `<tree>.attach.sha256`, the files
it wrote, and `<tree>.attach.pf`, the recipe that owns the tree. The whole
`<tree>.attach.*` family is control-plane metadata (target_policy): the
model changes none of it, by any route. A command that does is undone here,
whatever wrote it -- a tool, git apply, a wrapped script.

Two trusted effects maintain it instead:

- a build (bitbake, a build driver) attaches and records, as it always did;
- after a successful updiff, SPEAR records the attached tree as the state
  the patch series now reproduces. The build's updiff folds an edit into the
  series but leaves the manifest naming the pre-edit tree, so the guard
  refused every later rebuild as "not captured in the patch set", and the
  only exits were the override or a hand-edited manifest.

A build may also remove what it owns: an object of the start state it
removes is allowed when the project itself marks it generated (under
build/tmp or generated/, ignored by the project's version control, or, in
an attached tree, absent from the files the attach wrote). Removing a
source, the attach state or anything else is a violation, reported; the
attach state is put back.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time

from harness import provenance, target_policy

_MANIFEST = ".attach.sha256"
_OWNER = ".attach.pf"
_PRUNED = (".pc", "patches")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def is_attach_metadata(path: str) -> bool:
    return bool(target_policy.attach_metadata_tree(path))


# The family's files, as they stand: in every directory that held one at the
# start, the only places the attach writes them.

def _directories() -> set[str]:
    return {os.path.dirname(path) for path in provenance.start_paths()
            if ".attach." in os.path.basename(path)}


def snapshot() -> dict[str, bytes | None]:
    found: dict[str, bytes | None] = {}

    for directory in _directories():
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue

        for entry in entries:
            if not is_attach_metadata(entry.path):
                continue

            try:
                with open(entry.path, "rb") as handle:
                    found[entry.path] = handle.read()
            except OSError:
                found[entry.path] = None

    return found


def enforce(before: dict[str, bytes | None], *, build: bool) -> list[str]:
    """Undo what a command did to the attach state; the paths it changed.

    A build may write the state (it attaches); no command may remove it, and
    nothing else may write it.
    """
    after = snapshot()
    changed = []

    for path, content in before.items():
        if path not in after:
            changed.append(path)
            _put(path, content)
        elif after[path] != content and not build:
            changed.append(path)
            _put(path, content)

    for path in after.keys() - before.keys():
        if not build:
            changed.append(path)

            try:
                os.remove(path)
            except OSError:
                pass

    return changed


def _put(path: str, content: bytes | None) -> None:
    if content is None:
        return

    try:
        with open(path, "wb") as handle:
            handle.write(content)
    except OSError:
        pass


# What a build removed from the start state, and whether it was its own.

def present_start_objects() -> list[str]:
    return [path for path in provenance.start_paths() if os.path.lexists(path)]


def removed_by(before: list[str]) -> list[str]:
    return [path for path in before if not os.path.lexists(path)]


def build_owned(path: str) -> bool:
    """Whether the project marks `path` as output a build may remove."""
    if re.search(r"/build/tmp/|/generated/", path):
        return True

    tree = _attached_tree(path)

    # In an attached tree the attach's own manifest names the sources: what
    # it did not write, a build made. A re-attach moves the tree to .back.

    if tree:
        relative = os.path.relpath(path, tree)

        if os.path.lexists(os.path.join(tree + ".back", relative)):
            return True

        listed = _manifest_files(tree)

        return listed is not None and relative not in listed

    return _ignored(path)


def _attached_tree(path: str) -> str:
    directory = os.path.dirname(path)

    while directory and directory != os.path.dirname(directory):
        if os.path.isfile(directory + _MANIFEST):
            return directory

        directory = os.path.dirname(directory)

    return ""


_LISTED: dict[str, set[str] | None] = {}


def _manifest_files(tree: str) -> set[str] | None:
    if tree not in _LISTED:
        try:
            with open(tree + _MANIFEST, encoding="utf-8", errors="replace") as handle:
                _LISTED[tree] = {line.split("  ", 1)[1].rstrip("\n").removeprefix("./")
                                 for line in handle if "  " in line}
        except OSError:
            _LISTED[tree] = None

    return _LISTED[tree]


def _ignored(path: str) -> bool:
    directory = os.path.dirname(path)

    while directory and directory != os.path.dirname(directory):
        if os.path.exists(os.path.join(directory, ".git")):
            done = subprocess.run(["git", "-C", directory, "check-ignore", "-q", "--no-index", "--",
                                   os.path.relpath(path, directory)],
                                  capture_output=True, timeout=30)

            return done.returncode == 0

        directory = os.path.dirname(directory)

    return False


def violations(removed: list[str]) -> list[str]:
    """The removed objects no build owned; attach state is reported apart."""
    _LISTED.clear()

    return [path for path in removed
            if not is_attach_metadata(path) and not build_owned(path)]


# The trusted refresh after an updiff.

def updiff_recipes(stages) -> list[str]:
    """The recipes a command runs updiff on: `updiff.sh X`, `bitbake X -c updiff`."""
    found = []

    for argv in stages:
        name = os.path.basename(argv[0])

        if name == "updiff.sh" and len(argv) > 1:
            found.append(argv[1])
        elif name == "bitbake" and any(word in ("updiff", "do_updiff") for word in argv):
            index = argv.index("-c") if "-c" in argv else -1
            found += [word for position, word in enumerate(argv[1:], 1)
                      if not word.startswith("-") and position != index + 1]

    return found


def refresh(recipe: str, started: float) -> tuple[str, str]:
    """(manifest written, "") after a successful updiff of `recipe`, or ("", why not)."""
    owners = []

    for directory in _directories():
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue

        for entry in entries:
            if entry.name.endswith(_OWNER) and os.path.isdir(entry.path[:-len(_OWNER)]):
                try:
                    with open(entry.path, encoding="utf-8") as handle:
                        pf = handle.read().strip()
                except OSError:
                    continue

                if pf == recipe or pf.startswith(recipe + "-"):
                    owners.append((entry.path[:-len(_OWNER)], pf))

    if len(owners) != 1:
        return "", f"{len(owners)} attached trees belong to {recipe}"

    tree, pf = owners[0]
    root = _project_root(tree)

    if not root:
        return "", f"no project build directory above {tree}"

    log = os.path.join(root, "build/tmp/work", pf, "temp/log.do_updiff")

    try:
        if os.path.getmtime(log) < started:
            return "", "updiff did not run in this command"

        with open(log, encoding="utf-8", errors="replace") as handle:
            if re.search(r"^ERROR:", handle.read(), re.MULTILINE):
                return "", "updiff reported an error"
    except OSError:
        return "", "no updiff log"

    problem = _series_problem(root, pf)

    if problem:
        return "", problem

    _write_manifest(tree)

    return tree + _MANIFEST, ""


def _project_root(tree: str) -> str:
    directory = os.path.dirname(tree)

    while directory and directory != os.path.dirname(directory):
        if os.path.isdir(os.path.join(directory, "build/tmp/work")):
            return directory

        directory = os.path.dirname(directory)

    return ""


def _series_problem(root: str, pf: str) -> str:
    """Why the patch series of `pf` is not one the attach can apply, or ""."""
    import glob

    patchsets = glob.glob(os.path.join(root, "build", "*", "recipes-*", "*", "files", f"0001-{pf}"))

    if len(patchsets) != 1:
        return f"{len(patchsets)} patch sets for {pf}"

    series = patchsets[0] + "-patches.inc"

    try:
        with open(series, encoding="utf-8", errors="replace") as handle:
            listed = re.findall(r"file://([^\s\\\"']+\.patch)", handle.read())
    except OSError:
        return f"no series file for {pf}"

    for name in listed:
        path = os.path.join(patchsets[0], name)

        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                problem = _patch_problem(handle.read())
        except OSError:
            return f"{name} is listed but missing"

        if problem:
            return f"{name}: {problem}"

    return ""


def _patch_problem(text: str) -> str:
    """A hunk whose lines disagree with its header, or no hunk at all."""
    lines = text.split("\n")
    hunks = 0
    index = 0

    while index < len(lines):
        found = _HUNK.match(lines[index])
        index += 1

        if not found:
            continue

        hunks += 1
        old = int(found.group(2) if found.group(2) is not None else 1)
        new = int(found.group(4) if found.group(4) is not None else 1)

        while (old > 0 or new > 0) and index < len(lines):
            line = lines[index]
            index += 1

            if line.startswith("\\"):
                continue

            if line.startswith(" ") or line == "":
                old, new = old - 1, new - 1
            elif line.startswith("-"):
                old -= 1
            elif line.startswith("+"):
                new -= 1
            else:
                return f"hunk {hunks} ends early"

        if old or new:
            return f"hunk {hunks} disagrees with its header"

    return "" if hunks else "no hunk"


def _write_manifest(tree: str) -> None:
    """The manifest the attach task writes: every regular file under the
    tree, .pc and patches pruned, sorted bytewise, `sha256  ./path`."""
    files = []

    for directory, subdirectories, names in os.walk(tree):
        subdirectories[:] = [name for name in subdirectories if name not in _PRUNED]

        for name in names:
            path = os.path.join(directory, name)

            if name not in _PRUNED and os.path.isfile(path) and not os.path.islink(path):
                files.append("./" + os.path.relpath(path, tree))

    lines = []

    for relative in sorted(files, key=lambda item: item.encode("utf-8", "surrogateescape")):
        digest = hashlib.sha256()

        with open(os.path.join(tree, relative), "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)

        lines.append(f"{digest.hexdigest()}  {relative}\n")

    with open(tree + _MANIFEST, "w", encoding="utf-8", errors="surrogateescape") as handle:
        handle.writelines(lines)


def now() -> float:
    return time.time()
