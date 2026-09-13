"""What a corpus does is in its registry entry, not in its type.

`kind` used to decide six things at once: which Chroma collection the session
read, which indexer rebuilt it, whether a missing index was built on sight,
which system prompt the session ran on, and -- through that prompt -- whether
the normative evidence contract was present at all, plus how history and
memory files were named.

None of that was visible in the registry. An entry said `"kind": "edgem1"`
and the behaviour lived in six branches elsewhere, so a corpus could not ask
for the curated indexer without also taking a collection name, an autoindex
policy and a domain prompt it might not want; and a reader of the registry
could not tell what any entry would do.

Each behaviour now has a key of its own. `kind` is a label: it scopes skills
and prints in the listing, and decides nothing.

These tests do not assert that any particular corpus exists. They assert the
mechanism, on registries they build themselves.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_chat


class Registry(unittest.TestCase):
    """A registry of our own, and the module pointed at it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)

        for name in ("PROJECTS_FILE", "PROJECT", "PROJECT_SPEC", "PROJECT_KIND",
                     "PROJECT_ROOT", "CORPUS_ROOT", "COLLECTION_NAME"):
            self.addCleanup(setattr, rag_chat, name, getattr(rag_chat, name))

        rag_chat.PROJECTS_FILE = os.path.join(self.root, "projects.json")
        os.makedirs(f"{self.root}/tree", exist_ok=True)

    def register(self, **spec):
        spec.setdefault("path", f"{self.root}/tree")
        with open(rag_chat.PROJECTS_FILE, "w") as f:
            json.dump({"c": spec}, f)

        rag_chat.PROJECT = "c"
        rag_chat.PROJECT_SPEC = dict(spec, name="c")
        rag_chat.PROJECT_KIND = spec.get("kind", "generic")
        rag_chat.CORPUS_ROOT = spec["path"]
        rag_chat.PROJECT_ROOT = spec["path"]
        rag_chat.COLLECTION_NAME = rag_chat.collection_name_for(spec)

        return spec


class TheIndexerIsDeclared(Registry):
    def test_generic_by_default(self):
        self.register(kind="generic")

        self.assertEqual(rag_chat.corpus_indexer(), "generic")
        self.assertTrue(rag_chat.reindex_command()[1].endswith("index_dir.py"))

    def test_the_curated_walk_is_asked_for_by_name(self):
        self.register(kind="generic", indexer="buildsystem")

        self.assertTrue(rag_chat.reindex_command()[1].endswith("index_corpus.py"))

    def test_the_kind_does_not_choose_it(self):
        """The whole point: a label must not move the machinery."""
        self.register(kind="buildsystem")

        self.assertEqual(rag_chat.corpus_indexer(), "generic")
        self.assertTrue(rag_chat.reindex_command()[1].endswith("index_dir.py"))

    def test_only_the_generic_indexer_is_told_the_collection(self):
        """The curated walk derives its own name; telling it another would
        index into a collection and query a different one."""
        self.register(kind="generic")
        self.assertIn("--collection", rag_chat.reindex_command())

        self.register(indexer="buildsystem")
        self.assertNotIn("--collection", rag_chat.reindex_command())


class AutoindexIsDeclared(Registry):
    def test_off_by_default(self):
        """A generic tree can be enormous; indexing one because a session
        opened it is a surprise."""
        self.register(kind="generic")

        self.assertFalse(rag_chat.corpus_autoindexes())

    def test_on_when_declared(self):
        self.register(kind="generic", autoindex=True)

        self.assertTrue(rag_chat.corpus_autoindexes())

    def test_the_kind_does_not_choose_it(self):
        self.register(kind="buildsystem")

        self.assertFalse(rag_chat.corpus_autoindexes())


class ThePromptIsDeclared(Registry):
    def test_no_prompt_file_means_the_generic_prompt(self):
        self.register(kind="buildsystem")

        self.assertIsNone(rag_chat.corpus_property("prompt_file"))

    def test_a_corpus_names_its_own(self):
        self.register(kind="generic", prompt_file="system-prompt.md")

        self.assertEqual(rag_chat.corpus_property("prompt_file"),
                         "system-prompt.md")

    def test_the_selection_is_not_a_kind_branch(self):
        """Read from main(), which is where the choice is made."""
        import inspect

        source = inspect.getsource(rag_chat.main)

        self.assertIn('corpus_property("prompt_file")', source)
        self.assertNotIn('PROJECT_KIND != "edgem1"', source)
        self.assertNotIn('PROJECT_KIND ==', source)


class TheCollectionIsPinnedOrDerived(Registry):
    def test_derived_from_the_tree_by_default(self):
        import hashlib

        spec = self.register(kind="generic")
        tag = hashlib.md5(os.path.realpath(spec["path"]).encode()).hexdigest()[:8]

        self.assertEqual(rag_chat.collection_name_for(spec), f"adhoc_{tag}")

    def test_an_existing_index_is_named(self):
        """How a corpus keeps an index built before its path -- or its kind --
        was what it is now. No rename, no rebuild."""
        spec = self.register(kind="buildsystem", collection="legacy_name")

        self.assertEqual(rag_chat.collection_name_for(spec), "legacy_name")

    def test_the_kind_does_not_derive_a_name(self):
        spec = self.register(kind="buildsystem")

        self.assertTrue(rag_chat.collection_name_for(spec).startswith("adhoc_"))


class NoBehaviourHidesBehindTheKind(unittest.TestCase):
    def test_production_code_has_no_kind_equality_branch(self):
        """Not `edgem1` specifically: ANY behavioural test of `kind`."""
        import re

        # The whole tracked tree, not just the top-level modules: the first
        # version of this guard globbed ROOT/*.py and missed a second copy of
        # the branch in docker/gen-registry.py, where it could have made the
        # container disagree with the platform about a corpus's collection.
        import subprocess

        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT.parent,
                                 capture_output=True, text=True).stdout.split()
        offenders = {}

        for name in tracked:
            if not name.endswith(".py") or "/tests/" in name:
                continue
            path = ROOT.parent / name
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            # Scoped to the CORPUS registry: "kind" is a common word, and
            # session_replay uses it for a record type that has nothing to do
            # with corpora. A guard that fires on that is a guard people edit
            # until it stops firing.
            hits = re.findall(
                r'(?:PROJECT_KIND|spec\w*\.get\(["\']kind["\'][^)]*\)|'
                r'spec\w*\[["\']kind["\']\])\s*[=!]=\s*["\'][a-z0-9_]+["\']',
                body)
            if hits:
                offenders[name] = hits

        self.assertEqual(offenders, {})

    def test_the_registry_documents_the_keys(self):
        body = (ROOT / "rag_chat.py").read_text(encoding="utf-8")

        for key in ("collection", "indexer", "autoindex", "prompt_file"):
            with self.subTest(key=key):
                self.assertIn(key, body)

    def test_the_cli_stores_what_it_was_given(self):
        """A round trip, because the first version of these options put their
        VALUES in the positionals: `--kind buildsystem` registered a corpus
        whose kind was the literal string "--indexer". The splitter already
        knew that an option with a value cannot be filtered by
        startswith("-"); it knew it about one option.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tree = os.path.join(tmp, "tree")
            os.makedirs(tree)
            previous = rag_chat.PROJECTS_FILE
            rag_chat.PROJECTS_FILE = os.path.join(tmp, "projects.json")
            self.addCleanup(setattr, rag_chat, "PROJECTS_FILE", previous)
            open(rag_chat.PROJECTS_FILE, "w").write("{}")

            for spelling in (["--kind", "buildsystem", "--indexer", "buildsystem",
                              "--autoindex", "--prompt-file", "p.md"],
                             ["--kind=buildsystem", "--indexer=buildsystem",
                              "--autoindex", "--prompt-file=p.md"]):
                with self.subTest(spelling=spelling[0]):
                    rag_chat.handle_corpus_command(["rm", "c"])
                    rag_chat.handle_corpus_command(["add", "c", tree] + spelling)
                    spec = json.load(open(rag_chat.PROJECTS_FILE))["c"]

                    self.assertEqual(spec["kind"], "buildsystem")
                    self.assertEqual(spec["indexer"], "buildsystem")
                    self.assertEqual(spec["prompt_file"], "p.md")
                    self.assertIs(spec["autoindex"], True)
                    self.assertEqual(spec["path"], tree)

    def test_a_value_taking_option_never_lands_in_the_positionals(self):
        """The regression that produced kind="--indexer"."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tree = os.path.join(tmp, "tree")
            os.makedirs(tree)
            previous = rag_chat.PROJECTS_FILE
            rag_chat.PROJECTS_FILE = os.path.join(tmp, "projects.json")
            self.addCleanup(setattr, rag_chat, "PROJECTS_FILE", previous)
            open(rag_chat.PROJECTS_FILE, "w").write("{}")

            rag_chat.handle_corpus_command(
                ["add", "c", tree, "--kind", "buildsystem"])
            spec = json.load(open(rag_chat.PROJECTS_FILE))["c"]

            self.assertEqual(spec["path"], tree, "the value became the path")
            self.assertNotIn("buildsystem", spec["path"])

    def test_the_cli_offers_each_behaviour_separately(self):
        """One flag that switched four things at once is how they became
        invisible."""
        self.assertNotIn("--edgem1", rag_chat.CORPUS_USAGE)

        body = (ROOT / "rag_chat.py").read_text(encoding="utf-8")
        for flag in ("--kind", "--indexer", "--autoindex", "--prompt-file"):
            with self.subTest(flag=flag):
                self.assertIn(flag, body)


if __name__ == "__main__":
    unittest.main()
