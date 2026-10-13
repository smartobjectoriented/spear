"""What a command granted workspace write may still not change.

The sandbox binds every declared tree read-write for such a command, and the
command policy cannot know what an arbitrary program will write: a patch fed
on stdin, a script, a Makefile rule. The paths found here are bound back
read-only on top of that bind, so the kernel refuses the write whatever
program attempts it.

Two kinds of path are found. The trees, by a walk that reads directories
only and is cached per root for the session. And the protected files a
command names, in its own text or in the files it feeds or runs: a marker
says a file is generated only inside the file, and no walk reads contents.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from harness import target_policy


@dataclass(frozen=True)
class ProtectedTrees:
    """What one root holds that no command may write."""

    #: `.git` directories and files, and the git directories `.git` files name.
    git: tuple[str, ...] = ()
    #: Generated directories and snapshot copies (build/tmp among them).
    outputs: tuple[str, ...] = ()


# A build's own trees are its output, and not walked: the checkouts BitBake
# unpacks under build/tmp and the packages Buildroot extracts under the build/
# beside its .config carry their own .git, which a rebuild deletes. They are
# also far too large to walk before every command.

_BUILD_OUTPUT_PREFIXES = ("tmp", "downloads", "sstate-cache")

_CACHE: dict[str, tuple[int, ProtectedTrees]] = {}


def _walk(root: str) -> ProtectedTrees:
    git: list[str] = []
    outputs: list[str] = []

    # Each directory with whether it lies in an output tree already found:
    # what is inside one is bound with it.

    pending = [(root, False)]

    while pending:
        directory, inside = pending.pop()

        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue

        configured = any(entry.name == ".config" for entry in entries)

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

            output = inside or entry.name == "generated" or target_policy.is_snapshot(
                entry.path + "/")

            if output and not inside:
                outputs.append(entry.path)

            if entry.name == "generated":
                continue

            if ((os.path.basename(directory) == "build"
                 and entry.name.startswith(_BUILD_OUTPUT_PREFIXES))
                    or (entry.name == "build" and configured)):
                continue

            pending.append((entry.path, output))

    return ProtectedTrees(git=tuple(sorted(set(git))), outputs=tuple(sorted(outputs)))


def trees(root: str) -> ProtectedTrees:
    """The protected trees under `root`, walked once and then cached.

    Walked again when the root itself changes, or after forget(): a command
    that may create a repository (git clone) or a generated tree (a build)
    is followed by one.
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


def _entered(directory: str, word: str, roots, aliases) -> str:
    """The directory `cd word` enters from `directory`, canonical, or "".

    "" when it cannot be followed or may not be: a word the shell expands,
    a directory outside the roots, one inside git's own state.
    """
    if not word or word == "-" or re.search(r"[$`~*?\[\]{}]", word):
        return ""

    path = os.path.realpath(_candidates(word, [directory], aliases)[0])

    if (not os.path.isdir(path) or not within(path, roots)
            or target_policy.is_git_metadata(os.path.join(path, "x"))):
        return ""

    return path


def _repository(directory: str, roots) -> tuple[str, ...]:
    """The git directories of the repository holding `directory`.

    Its `.git` directory; or the git directory its `.git` file names, with
    the common directory a linked worktree keeps its refs and objects in.
    A `.git` that is a link is not followed.
    """
    top = within(directory, roots)
    current = directory

    while True:
        dotgit = os.path.join(current, ".git")

        if os.path.islink(dotgit):
            return ()

        if os.path.isdir(dotgit):
            return (os.path.realpath(dotgit),)

        if os.path.isfile(dotgit):
            named = target_policy.gitdir_named_by(dotgit)

            if not named:
                return ()

            try:
                with open(os.path.join(named, "commondir"), "r", encoding="utf-8") as handle:
                    common = handle.readline(4096).strip()
            except OSError:
                common = ""

            return (named,) + ((os.path.realpath(os.path.join(named, common)),) if common else ())

        if current == top or current == os.path.dirname(current):
            return ()

        current = os.path.dirname(current)


def git_dirs(steps, roots, *, cwd: str | None = None, aliases=None) -> tuple[str, ...]:
    """The git directories the git commands of `steps` keep their records in.

    `steps` is CommandPolicy.git_writes: each git command's repository is
    the one holding the directory it runs in, followed through the cd and
    -C steps before it. Nothing when a step cannot be followed, and nothing
    outside the roots, which the sandbox does not bind at all.
    """
    roots = [os.path.realpath(str(root)) for root in roots]
    aliases = dict(aliases or {})

    if not roots:
        return ()

    current = (os.path.realpath(_candidates(cwd, (), aliases)[0])
               if cwd and cwd.startswith("/") else roots[0])

    if not within(current, roots):
        return ()

    found: list[str] = []

    for kind, words in steps:
        directory = current

        for word in words:
            directory = _entered(directory, word, roots, aliases)

            if not directory:
                return ()

        if kind == "cd":
            current = directory
            continue

        found += [path for path in _repository(directory, roots)
                  if within(path, roots) and os.path.isdir(path)]

    return tuple(dict.fromkeys(found))


def within(path: str, roots) -> str:
    """The root among `roots` that holds `path`, or ""."""
    best = ""

    for root in roots:
        root = str(root)

        if (path == root or path.startswith(root.rstrip("/") + "/")) and len(root) > len(best):
            best = root

    return best


# Words that may name a file: split on whitespace, quotes and shell syntax,
# and on `=` so that `T=x`, `of=x` and `F=x; ... $F` name x.

_WORD_SPLIT = re.compile(r"[\s'\"`;|&()<>=,:{}\[\]$]+")
_MAX_WORDS = 2000
_MAX_SCANNED = 200
_SCAN_LIMIT = 1 << 20
_TEST_RUNNERS = frozenset({"pytest", "unittest"})


def _words(text: str) -> list[str]:
    found = []

    for word in _WORD_SPLIT.split(text):
        if not word or len(word) > 4096 or not re.search(r"[\w.]", word):
            continue

        found.append(word)

        # A diff names its files under a/ and b/.

        if word.startswith(("a/", "b/")):
            found.append(word[2:])

        # A dotted module (`python3 -m unittest tests.test_x`) names a file.

        if "/" not in word and word.count(".") and re.fullmatch(r"[\w.]+", word):
            found.append(word.replace(".", "/") + ".py")

    return found[:_MAX_WORDS]


def _candidates(word: str, bases, aliases) -> list[str]:
    if word.startswith("/"):
        for inside, host in aliases.items():
            if word == inside or word.startswith(inside.rstrip("/") + "/"):
                return [host + word[len(inside):]]

        return [word]

    return [os.path.join(base, word) for base in bases]


def _test_files(directory: str) -> list[str]:
    """The test modules a runner discovers under `directory`, bounded."""
    found: list[str] = []
    visited = 0

    for current, subdirectories, files in os.walk(directory):
        visited += 1
        subdirectories[:] = [name for name in subdirectories
                             if not name.startswith(".") and name not in ("build", "node_modules")]
        found += [os.path.join(current, name) for name in files
                  if name.endswith(".py") and (name.startswith("test") or name.endswith("_test.py"))]

        if visited > 2000 or len(found) > _MAX_SCANNED:
            break

    return found[:_MAX_SCANNED]


def command_files(command: str, roots, *, cwd: str | None = None,
                  build: bool = False, aliases=None) -> tuple[str, ...]:
    """Protected files `command` may reach, as host paths under `roots`.

    The ones its words name (a directory named among them stands for the
    file of each word's basename inside it, as `cp x dir/` writes), and the
    ones named inside the tests a runner discovers. Unless the command
    builds, also the ones named inside the files it names -- the diff it
    feeds to patch, the script it runs -- and every file refused this
    session. A build is spared those: what it regenerates is exactly what a
    refusal sent the model to rebuild, and its scripts name its outputs.

    `aliases` maps a path inside the sandbox to the host path it is bound
    from: the session's /tmp, a tree mounted at /workspace.
    """
    roots = [str(root) for root in roots]
    aliases = dict(aliases or {})

    if cwd and cwd.startswith("/"):
        cwd = _candidates(cwd, (), aliases)[0]

    bases = [cwd] if cwd else []
    words = _words(command)

    # `cd dir` moves where a later word is read from.

    bases += [base for base in (os.path.join(cwd or roots[0], word)
                                for previous, word in zip(words, words[1:])
                                if previous == "cd")
              if os.path.isdir(base)]
    bases += roots
    files: dict[str, None] = {}
    directories: dict[str, None] = {}

    def resolve(word, extra=()):
        for candidate in _candidates(word, list(extra) + bases, aliases):
            if os.path.isfile(candidate):
                files[os.path.realpath(candidate)] = None
                return
            if os.path.isdir(candidate):
                directories[os.path.realpath(candidate)] = None
                return

    for word in words:
        resolve(word)

    named = list(files)

    for directory in list(directories)[:50]:
        for word in words:
            candidate = os.path.join(directory, os.path.basename(word))

            if os.path.isfile(candidate):
                files[os.path.realpath(candidate)] = None

    # The tests a runner imports are code it runs, build or not.

    readable = roots + list(aliases.values())
    tests = any(word in _TEST_RUNNERS for word in words)
    scanned = [path for path in named if within(path, readable)
               and (not build or (tests and path.endswith(".py")))
               and os.path.getsize(path) < _SCAN_LIMIT]

    if tests:
        for directory in list(directories)[:20] or [cwd or roots[0]]:
            scanned += _test_files(directory)

    for path in scanned[:_MAX_SCANNED]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(_SCAN_LIMIT)
        except OSError:
            continue

        for word in _words(text):
            resolve(word, (os.path.dirname(path),))

    if not build:
        files.update(dict.fromkeys(target_policy.refused_paths()))

    return tuple(path for path in files
                 if within(path, roots) and target_policy.refusal(path))
