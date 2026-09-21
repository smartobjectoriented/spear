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

# Produced by build.sh before it calls docker, and gitignored: a committed copy
# would publish one machine's corpus graph and be stale besides. The COPY may
# take it, but only because the builder guarantees it exists.
GENERATED = {"docker/projects.docker.json"}

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

        if pattern in GENERATED:
            self.assertIn(pattern, BUILD_SH.read_text(),
                          f"{pattern} is declared generated but the builder "
                          f"never mentions it")
            return

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


class NoHostRegistryIsPublished(unittest.TestCase):
    """The image's registry is generated, never committed.

    The first public snapshot shipped this machine's real one: twenty-four
    corpora with their names, their federations and the collection hash of
    each -- a readable map of one developer's disk, in a repository whose
    whole point is that it carries no deployment.
    """

    def test_the_generated_registry_is_not_tracked(self):
        for path in GENERATED:
            with self.subTest(path=path):
                self.assertNotIn(Path(path), tracked(),
                                 f"{path} is build output; it must be "
                                 f"gitignored, not committed")

    def test_it_is_ignored_rather_than_merely_absent(self):
        """Absent is one build away from committed by accident."""
        for path in GENERATED:
            with self.subTest(path=path):
                done = subprocess.run(["git", "check-ignore", "-q", path],
                                      cwd=ROOT)
                self.assertEqual(done.returncode, 0,
                                 f"{path} is not in .gitignore")

    def test_exactly_one_registry_is_carried_by_hand(self):
        """One canonical example, not two to keep in step."""
        registries = sorted(p for p in tracked()
                            if p.name.endswith(".json")
                            and "projects" in p.name)
        self.assertEqual(registries, [Path("spear/projects.example.json")])

    def test_the_example_names_no_real_tree(self):
        """A template full of placeholders, not a sanitised real registry."""
        import json

        example = json.loads(
            (ROOT / "spear" / "projects.example.json").read_text())

        for name, spec in example.items():
            if not isinstance(spec, dict):
                continue

            path = spec["path"]

            with self.subTest(corpus=name):
                self.assertFalse(
                    path.startswith("/") and not path.startswith("/path/to/"),
                    f"{name}: {path} looks like a real absolute path")


class EveryOptionalInputIsDeclaredOptional(unittest.TestCase):
    """A named context that build.sh does not resolve is a build failure."""

    def setUp(self):
        self.script = BUILD_SH.read_text()

    def declared_contexts(self):
        """Every context build.sh provides, by either of the two mechanisms.

        The OPTIONAL table is one of them: a directory that may or may not
        exist, falling back to an empty one. The other is a context build.sh
        STAGES first, because what goes in is a selection rather than a
        directory -- the normative store is filtered per document, and the
        baked trees are a named subset of the registry. Both end as
        --build-context; only the first is a table entry.
        """
        declared = set(re.findall(r'^\s*"(\w+):SPEAR_\w+:', self.script,
                                  re.MULTILINE))
        declared |= set(re.findall(r'--build-context "(\w+)=', self.script))
        declared.add("repo")           # the repository itself, always present

        return declared

    def test_every_named_context_is_resolved_by_the_builder(self):
        used = {context for context, _, _ in copies() if context}

        self.assertEqual(used - self.declared_contexts(), set(),
                         "a COPY names a build context nothing provides")

    def test_the_staged_contexts_are_staged_before_they_are_passed(self):
        """A --build-context pointing at a directory nobody filled is an
        empty one, and an empty standards context is a container that opens
        with no normative store and says nothing about it."""
        for context, stager in (("standards", "stage-standards.py"),
                                ("baked", "stage-corpora.py")):
            with self.subTest(context=context):
                self.assertIn(stager, self.script)
                self.assertLess(self.script.index(stager),
                                self.script.index(f'--build-context "{context}='),
                                f"{context} is passed before {stager} fills it")

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
    def test_the_builder_refuses_on_nothing_a_clone_lacks(self):
        """Every precondition is about code or build output, never content.

        Matched across lines: a guard whose `exit 1` sits in a brace block on
        the next line is still a guard, and an expression anchored to one line
        would report a clean bill on a build.sh that refuses again.
        """
        script = BUILD_SH.read_text()
        guards = re.findall(r"\[ -[df] \"([^\"]+)\" \][^\n]*\|\|[^{]*(?:\{[^}]*)?exit 1",
                            script)
        self.assertEqual(sorted(guards),
                         ['$APP/rag_chat.py', '$REGISTRY'])

        # and $REGISTRY is the generated file, not something a clone must have
        self.assertIn('REGISTRY="$REPO/docker/projects.docker.json"', script)

    def test_no_copy_reaches_outside_the_repository(self):
        for context, source, _ in copies():
            with self.subTest(source=source):
                self.assertFalse(source.startswith("/"),
                                 "an absolute source binds the image to one "
                                 "machine's filesystem")
                self.assertNotIn("..", source)


if __name__ == "__main__":
    unittest.main()
