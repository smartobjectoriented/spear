"""The attach mechanism's own state, and what may change it.

A layered build attaches a component's source to a tree (linux/linux, u-boot/)
and keeps beside it what its guard reads: `<tree>.attach.sha256`, the files
it wrote, and `<tree>.attach.pf`, the recipe that owns the tree. The whole
`<tree>.attach.*` family is control-plane metadata (target_policy): the
model changes none of it, by any route. A command that does is undone here,
whatever wrote it -- a tool, git apply, a wrapped script.

Two trusted effects maintain it instead:

- a build (bitbake, a build driver) attaches and records, as it always did;
- after an updiff, SPEAR records the attached tree as the state the patch
  series reproduces, once a clean reconstruction has shown that it does
  (persistence_check.py). The build's updiff folds an edit into the series
  but leaves the manifest naming the pre-edit tree, so the guard refused
  every later rebuild as "not captured in the patch set", and the only exits
  were the override or a hand-edited manifest. An updiff that succeeded has
  only captured something: until it is verified the guard stays armed.

A build may also remove what it owns: an object of the start state it
removes is allowed when the project itself marks it generated (under
build/tmp or generated/, ignored by the project's version control, or, in
an attached tree, absent from the files the attach wrote). Removing a
source, the attach state or anything else is a violation, reported; the
attach state is put back.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
import time

from harness import persistence_check, provenance, target_policy

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

    # In an attached tree the attach's own manifest names what it wrote:
    # what it did not write, a build made. What it wrote the component's own
    # ignore rules may still call generated (U-Boot ships tools/lib/*.c
    # wrappers and ignores them as build output). A re-attach moves the
    # tree to .back.

    if tree:
        relative = os.path.relpath(path, tree)

        if os.path.lexists(os.path.join(tree + ".back", relative)):
            return True

        listed = _manifest_files(tree)

        if listed is not None and relative not in listed:
            return True

        return _component_ignored(tree, relative)

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


_EMPTY_GIT: list[str] = []


def _component_ignored(tree: str, relative: str) -> bool:
    """Whether the attached tree's own .gitignore files ignore `relative`,
    read alone: the project's rules, which ignore the whole tree, are not."""
    import tempfile

    if not _EMPTY_GIT:
        directory = tempfile.mkdtemp(prefix="spear-attach-git-")
        subprocess.run(["git", "init", "-q", directory], capture_output=True, check=True)
        _EMPTY_GIT.append(os.path.join(directory, ".git"))

    done = subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "--git-dir", _EMPTY_GIT[0],
                           "--work-tree", tree, "check-ignore", "-q", "--no-index", "--", relative],
                          capture_output=True, timeout=30)

    return done.returncode == 0


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


# What the persistence check starts from, pinned before a turn can reach it:
# each pristine copy, again when a build rewrites one, and the project's
# build-output rules, once.

_PINS: dict[str, str] = {}
_RULES: dict[str, list[str]] = {}
_PIN_SCOPE: list[set[str]] = []


def _pristine_copies() -> list[str]:
    found = []

    for tree, pf in _owners():
        root = _project_root(tree)

        if root:
            found += glob.glob(os.path.join(root, "build/tmp/work", pf, "*.pristine"))

    return found


def _rules_root(tree: str) -> str:
    """The project above `tree`, by its class: a start state may have built
    nothing yet, so there is no build/tmp/work to find it by."""
    directory = os.path.dirname(tree)

    while directory and directory != os.path.dirname(directory):
        if os.path.isfile(os.path.join(directory, persistence_check.CLASS)):
            return directory

        directory = os.path.dirname(directory)

    return ""


def _signature(path: str) -> tuple[int, int]:
    status = os.stat(path)

    return status.st_ino, status.st_ctime_ns


def pristine_state() -> dict[str, tuple[int, int]]:
    """Each pristine copy as it stands, to tell what a build then changed."""
    found = {}

    for path in _pristine_copies():
        try:
            found[path] = _signature(path)
        except OSError:
            pass

    return found


def pin_pristine(before: dict[str, tuple[int, int]] | None = None) -> None:
    """Pin what a session finds when it first looks; given the state before a
    build, pin what that build changed. A copy anything else changed or made
    keeps its pin or has none, and fails the check."""
    directories = _directories()
    first = not _PIN_SCOPE or _PIN_SCOPE[0] != directories

    if before is None and not first:
        return

    _PIN_SCOPE[:] = [directories]

    if first:
        _PINS.clear()
        _RULES.clear()

        for root in {_rules_root(tree) for tree, _ in _owners()} - {""}:
            try:
                _RULES[root] = persistence_check.exclude_patterns(root)
            except (OSError, ValueError, SyntaxError):
                pass

    for path, signature in pristine_state().items():
        if first or before.get(path) != signature:
            try:
                _PINS[path] = persistence_check.tree_digest(path)
            except OSError:
                _PINS.pop(path, None)


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


def _owners() -> list[tuple[str, str]]:
    """(attached tree, recipe PF) of every tree with an owner beside it."""
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
                        owners.append((entry.path[:-len(_OWNER)], handle.read().strip()))
                except OSError:
                    continue

    return owners


def refresh(recipe: str, started: float) -> tuple[str, str, persistence_check.Verification | None]:
    """After an updiff of `recipe`: (manifest written, why not, the check).

    UPDIFF_CAPTURED is what the log says; PERSISTENCE_VERIFIED is the check.
    The manifest is written only on the second, and the check runs only on the
    first: without a capture the third item is None.
    """
    owners = [(tree, pf) for tree, pf in _owners()
              if pf == recipe or pf.startswith(recipe + "-")]

    if len(owners) != 1:
        return "", f"{len(owners)} attached trees belong to {recipe}", None

    tree, pf = owners[0]
    root = _project_root(tree)

    if not root:
        return "", f"no project build directory above {tree}", None

    log = os.path.join(root, "build/tmp/work", pf, "temp/log.do_updiff")

    try:
        if os.path.getmtime(log) < started:
            return "", "updiff did not run in this command", None

        with open(log, encoding="utf-8", errors="replace") as handle:
            if re.search(r"^ERROR:", handle.read(), re.MULTILINE):
                return "", "updiff reported an error", None
    except OSError:
        return "", "no updiff log", None

    problem = _series_problem(root, pf)

    if problem:
        return "", problem, None

    check = _verify(root, tree, pf)

    if not check.verified:
        return "", check.reason, check

    _write_manifest(tree)

    return tree + _MANIFEST, "", check


def _verify(root: str, tree: str, pf: str) -> persistence_check.Verification:
    provenance.ready()

    def failed(reason: str) -> persistence_check.Verification:
        return persistence_check.Verification(False, reason)

    if not os.path.isfile(tree + _MANIFEST):
        return failed(f"no attach manifest beside {os.path.basename(tree)}")

    copies = glob.glob(os.path.join(root, "build/tmp/work", pf, "*.pristine"))

    if len(copies) != 1:
        return failed(f"{len(copies)} pristine copies for {pf}")

    if copies[0] not in _PINS:
        return failed(f"the pristine copy of {pf} was never pinned")

    if root not in _RULES:
        return failed("the project's build-output rules were not readable")

    patch_dir, listed = _series(root, pf)

    return persistence_check.verify(tree, copies[0], _PINS[copies[0]], _RULES[root],
                                    patch_dir, listed)


def _project_root(tree: str) -> str:
    directory = os.path.dirname(tree)

    while directory and directory != os.path.dirname(directory):
        if os.path.isdir(os.path.join(directory, "build/tmp/work")):
            return directory

        directory = os.path.dirname(directory)

    return ""


def _series(root: str, pf: str) -> tuple[str, list[str]]:
    """The patch directory of `pf` and the patches its series lists, in order."""
    patch_dir = glob.glob(os.path.join(root, "build", "*", "recipes-*", "*", "files", f"0001-{pf}"))[0]

    with open(patch_dir + "-patches.inc", encoding="utf-8", errors="replace") as handle:
        return patch_dir, re.findall(r"file://([^\s\\\"']+\.patch)", handle.read())


def _series_problem(root: str, pf: str) -> str:
    """Why the patch series of `pf` is not one the attach can apply, or ""."""
    patchsets = glob.glob(os.path.join(root, "build", "*", "recipes-*", "*", "files", f"0001-{pf}"))

    if len(patchsets) != 1:
        return f"{len(patchsets)} patch sets for {pf}"

    series = patchsets[0] + "-patches.inc"

    try:
        with open(series, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return f"no series file for {pf}"

    # A parameter (;striplevel=, ;patchdir=) changes how a patch applies, and
    # the check applies every patch alike.

    if re.search(r"\.patch;", text):
        return f"the series of {pf} carries patch parameters"

    for name in re.findall(r"file://([^\s\\\"']+\.patch)", text):
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
