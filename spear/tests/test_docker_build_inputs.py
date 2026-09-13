"""A clean clone must be able to run the documented build.

The container quick start is the first thing a reader is told to do, and it
had three dependencies a fresh checkout cannot satisfy: a hard guard on a
notes corpus that is one deployment's own writing, a COPY of a benches/
directory that is not in the repository, and a COPY of the ChromaDB index,
which is built on the host and gitignored. Each failed the build outright.

Every such input is now an OPTIONAL NAMED CONTEXT: build.sh resolves it to the
in-tree directory, to its SPEAR_*_DIR, or to an empty one, and the image
carries less rather than not being built. These tests hold that line — a new
COPY from a path a clone does not have would otherwise reintroduce the same
failure silently, since it only shows up on a machine that is not this one.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOCKERFILE = ROOT / "docker" / "Dockerfile"
BUILD_SH = ROOT / "docker" / "build.sh"

# The default build context, set by build.sh's last line.
DEFAULT_CONTEXT = ROOT / "spear"

COPY = re.compile(r"^COPY\s+(?P<rest>.*)$", re.MULTILINE)
FROM = re.compile(r"--from=(?P<name>[A-Za-z0-9_-]+)")
FLAG = re.compile(r"^--[a-z-]+(=\S*)?$")


def tracked():
    """Paths git actually carries, which is what a clone gets."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return {Path(p) for p in out.stdout.split("\0") if p}


def copies():
    """(named context or None, source, destination) for every COPY."""
    for match in COPY.finditer(DOCKERFILE.read_text()):
        words = match.group("rest").split()
        context = None

        while words and FLAG.match(words[0]):
            found = FROM.search(words[0])

            if found:
                context = found.group("name")

            words.pop(0)

        *sources, destination = words

        for source in sources:
            yield context, source, destination


class EveryMandatoryInputIsInTheRepository(unittest.TestCase):
    """What a COPY takes unconditionally, a clone must have."""

    def setUp(self):
        self.tracked = tracked()

    def _assert_available(self, root, source, where):
        pattern = source.rstrip("/")

        if any(ch in pattern for ch in "*?["):
            hits = list(root.glob(pattern))
            self.assertTrue(hits, f"{where}: {source} matches nothing in {root}")
            return

        path = root / pattern
        relative = path.relative_to(ROOT)
        carried = (relative in self.tracked
                   or any(str(p).startswith(f"{relative}/")
                          for p in self.tracked))
        self.assertTrue(carried,
                        f"{where}: {source} is not in the repository — a clean "
                        f"clone cannot build. Make it an optional context.")

    def test_the_default_context_only_takes_tracked_paths(self):
        for context, source, _ in copies():
            if context is not None:
                continue

            with self.subTest(source=source):
                self._assert_available(DEFAULT_CONTEXT, source,
                                       "default context")

    def test_the_repo_context_only_takes_tracked_paths(self):
        for context, source, _ in copies():
            if context != "repo":
                continue

            with self.subTest(source=source):
                self._assert_available(ROOT, source, "repo context")


class EveryOptionalInputIsDeclaredOptional(unittest.TestCase):
    """A named context that build.sh does not resolve is a build failure."""

    def setUp(self):
        self.script = BUILD_SH.read_text()

    def test_every_named_context_is_resolved_by_the_builder(self):
        declared = set(re.findall(r'^\s*"(\w+):SPEAR_\w+:', self.script,
                                  re.MULTILINE))
        declared.add("repo")           # the repository itself, always present

        used = {context for context, _, _ in copies() if context}

        self.assertEqual(used - declared, set(),
                         "a COPY names a build context nothing provides")

    def test_each_optional_context_falls_back_to_an_empty_directory(self):
        self.assertIn('dir="$EMPTY"', self.script)
        self.assertIn('EMPTY="$(mktemp -d)"', self.script)

    def test_each_optional_context_is_overridable(self):
        overrides = set(re.findall(r'^\s*"\w+:(SPEAR_\w+):', self.script,
                                   re.MULTILINE))
        self.assertEqual(
            overrides,
            {"SPEAR_INDEX_DIR", "SPEAR_RULES_DIR", "SPEAR_SKILLS_DIR",
             "SPEAR_BENCH_DIR", "SPEAR_NOTES_DIR"})


class NothingPrivateIsRequired(unittest.TestCase):
    def test_the_builder_refuses_only_on_the_harness_itself(self):
        """One precondition, and it is about the code, not about content."""
        guards = re.findall(r"^\[ -[df] \"([^\"]+)\" \].*exit 1",
                            BUILD_SH.read_text(), re.MULTILINE)
        self.assertEqual(guards, ['$APP/rag_chat.py'])

    def test_no_copy_reaches_outside_the_repository(self):
        for context, source, _ in copies():
            with self.subTest(source=source):
                self.assertFalse(source.startswith("/"),
                                 "an absolute source binds the image to one "
                                 "machine's filesystem")
                self.assertNotIn("..", source)


if __name__ == "__main__":
    unittest.main()
