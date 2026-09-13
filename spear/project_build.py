"""How this tree is built and tested, discovered once and remembered.

Every writing turn on the same project rediscovered it, and badly: `make
clean` against a tree with no Makefile, then `ls`, then `cat CMakeLists.txt`,
then two guesses at a cmake invocation. Ten rounds of a sixty-round budget,
every time, before the first edit.

It is not a per-turn question. A tree is built one way, and the answer belongs
to the corpus rather than to the turn that happened to work it out. So it is
probed once from the files that are actually there, cached beside the corpus,
and handed to the turn -- and to the verification gate that runs after a
write, which is the reason this exists at all: a change that has not been
built is not a change that works, and the harness cannot say so if it does not
know how to build.

Nothing here is specific to one project. The probes are the ordinary shapes:
a CMake tree, a Makefile, cargo, go, npm, a Python package. A project that
wants something else says so in projects.json (`build_commands`,
`test_commands`), which wins over every probe below.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass

# Where the answer is kept. Beside the corpus state, not in the tree: the
# tree belongs to the customer and a harness cache is not theirs to carry.
CACHE_NAME = "build-commands.json"

SCHEMA_VERSION = 1

# Where a command has to come from before the harness will treat running it as
# a verification: the project said so in projects.json, or the caller asked
# for the tests/ shape to be inferred and knows what it is asking for.

_CONFIGURED_SOURCES = frozenset({"projects.json", "unittest"})


@dataclass(frozen=True)
class ProjectCommands:
    """What to run to build this tree, and what to run to test it."""

    build: str = ""
    test: str = ""
    source: str = "none"

    def verifies(self):
        """The commands a writing turn must survive, in order."""
        return tuple(item for item in (self.build, self.test) if item)

    @property
    def configured(self):
        """True when this tree SAID how it is verified, rather than looking like it.

        Only a configured command is allowed to stand as the run's own
        verification. A probed `make` is a good guess about how to build a
        tree and a bad basis for telling a turn it has been verified: the
        guess can be wrong, and the operator never asked for it to count.
        """
        return self.source in _CONFIGURED_SOURCES

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "build": self.build,
                "test": self.test, "source": self.source}

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            return None

        return cls(str(raw.get("build") or ""), str(raw.get("test") or ""),
                   str(raw.get("source") or "cache"))


def _cmake_side(root):
    """A CMake option this tree needs before it will configure.

    The VITA 49.2 converter has two sides and the target one wants a vendor
    SDK that is not on a workstation; configuring without -DV492C_SIDE=host
    stops on "libusb-1.0 not found" and the turn concludes the project cannot
    be built. Read from the file rather than assumed: an option with a
    "host"-ish value is the one that lets a build machine build.
    """
    path = os.path.join(root, "CMakeLists.txt")

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(40000)
    except OSError:
        return ""

    found = re.search(r'set\(\s*(\w+_SIDE)\s+"?(\w+)"?', text) or re.search(
        r'if\(\s*(\w+_SIDE)\s+STREQUAL\s+"(\w+)"', text)

    if not found:
        return ""

    name = found.group(1)

    return f" -D{name}=host" if re.search(rf'{name}.*host|host.*{name}', text,
                                          re.I | re.S) else ""


def probe(root, *, infer_unittest=False):
    """The build and test commands this tree actually supports.

    ``infer_unittest`` is off by default and deliberately narrow: a `tests/`
    package says where a tree's tests live, not that `unittest discover` is
    how its owner runs them. The benchmark fixtures are exactly that shape and
    have no owner to ask, so they turn it on; a real tree says so itself.
    """
    def here(*names):
        return all(os.path.exists(os.path.join(root, name)) for name in names)

    if here("CMakeLists.txt"):
        option = _cmake_side(root)

        return ProjectCommands(
            f"cmake -S . -B build/harness{option} && cmake --build build/harness",
            "ctest --test-dir build/harness --output-on-failure",
            "cmake")

    if here("Makefile") or here("makefile"):
        return ProjectCommands("make", "make test", "make")

    if here("Cargo.toml"):
        return ProjectCommands("cargo build", "cargo test", "cargo")

    if here("go.mod"):
        return ProjectCommands("go build ./...", "go test ./...", "go")

    if here("package.json"):
        return ProjectCommands("npm run build --if-present", "npm test", "npm")

    if here("pyproject.toml") or here("setup.py"):
        return ProjectCommands("", "python -m pytest -q", "python")

    # A plain tests package. Last, because a packaged project says how it is
    # tested and this only says where its tests are. `__init__.py` is required
    # rather than incidental: without it `unittest discover` collects nothing
    # and reports "NO TESTS RAN", which reads like a passing run.

    if infer_unittest and here(os.path.join("tests", "__init__.py")):
        return ProjectCommands("", "python3 -m unittest discover -s tests",
                               "unittest")

    return ProjectCommands()


def declared(spec):
    """What projects.json says, which wins over every probe."""
    spec = spec or {}

    def first(name):
        value = spec.get(name)

        if isinstance(value, str):
            return value

        if isinstance(value, (list, tuple)) and value:
            return str(value[0])

        return ""

    build, test = first("build_commands"), first("test_commands")

    return (ProjectCommands(build, test, "projects.json")
            if build or test else None)


def commands(root, *, spec=None, cache_dir=None, refresh=False,
             infer_unittest=False):
    """This tree's build and test commands: declared, cached, or probed once."""
    stated = declared(spec)

    if stated is not None:
        return stated

    path = os.path.join(cache_dir, CACHE_NAME) if cache_dir else ""

    if path and not refresh:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                found = ProjectCommands.from_dict(json.load(handle))

            if found is not None:
                return found
        except (OSError, ValueError):
            pass

    found = probe(root, infer_unittest=infer_unittest)

    if path and found.source != "none":
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(found.to_dict(), handle, indent=2, sort_keys=True)
        except OSError:
            pass

    return found


def summarize(text):
    """The last words of a failed command: its errors, or its final lines."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    errors = [line for line in lines
              if re.search(r"\berror\b|\bfailed\b|\bFAIL\b", line, re.I)]

    return "\n".join((errors or lines)[-6:])[:1200]


def run(command, root, timeout=900):
    """One verification command, on the host. Returns (ok, what went wrong).

    The fallback for callers with no sandbox. Anything that has one passes a
    ``verifier`` instead: the command is the project's, but the code it
    exercises was just written by the model, and that belongs where every
    other command the harness runs belongs.
    """
    if not command:
        return True, ""

    try:
        done = subprocess.run(["bash", "-lc", command], cwd=root, timeout=timeout,
                              capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if done.returncode == 0:
        return True, ""

    return False, summarize((done.stdout or "") + (done.stderr or ""))


def kind(command):
    """Whether this command builds or tests, for a caller outside this module."""

    return _kind(command)


def _kind(command):
    """Whether this command builds or tests, in the words the reader uses."""
    return "tests" if re.search(r"\b(ctest|pytest|test|check)\b", command,
                                re.I) else "build"


def new_errors(output, previous):
    """The failure lines this attempt added, or all of them the first time.

    A repair round is answered with the whole compiler output, and the
    output is mostly what was already broken: on a live run a turn was
    handed the same wall of errors three times running and spent each round
    re-reading the file instead of editing it. What a repair needs is the
    difference -- the lines that were not there before its edit.

    Nothing is hidden: when the difference is empty the full output is
    returned, because "nothing new broke" and "nothing is broken" are not
    the same thing and the turn still has to see what remains.
    """
    if not previous:
        return output, False

    seen = {line.strip() for line in previous.splitlines() if line.strip()}
    fresh = [line for line in (output or "").splitlines()
             if line.strip() and line.strip() not in seen]

    return ("\n".join(fresh), True) if fresh else (output, False)


def demand(command, output, previous=""):
    """What to hand back to a turn whose change does not survive the project.

    Building and testing are different failures and the message says which:
    told "the tree no longer builds" about eight failing assertions, a turn
    went looking for a compile error that was not there.
    """
    shown, filtered = new_errors(output, previous)
    detail = f"\n\n{shown}" if shown else ""
    what = ("does not compile" if _kind(command) == "build"
            else "compiles, but the project's tests no longer pass")
    lead = ("What your last edit broke that was not broken before:"
            if filtered else f"The tree {what}. `{command}` failed:")

    return (
        f"{lead}{detail}\n\n"
        f"Fix it now, in the code you changed. Read the failure, make the "
        f"smallest edit that answers it, and do not start anything else "
        f"until this command passes. If a test fails for a reason that has "
        f"nothing to do with your change, say which and why -- do not edit "
        f"the test to make it pass."
    )


def note(command, output):
    """What to append when a turn ends with the project still broken."""
    what = ("does not build" if _kind(command) == "build"
            else "does not pass its own tests")
    head = f"\n\n⚠ The project {what} after this turn: `{command}` fails."

    return f"{head}\n{output.splitlines()[0][:200]}" if output else head


def shlex_ok(command):
    """A command the sandbox can be asked to run at all."""
    try:
        return bool(shlex.split(command))
    except ValueError:
        return False
