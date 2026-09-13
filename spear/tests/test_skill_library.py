import os
import tempfile
import unittest
from datetime import datetime

import skill_library
from skill_library import (
    ANY_SCOPE, Skill, SkillError, applies_to, load_library,
    missing_requirements, parse_skill, reconcile, render_skill, save,
)


FRONTMATTER = """---
name: debug-buildsystem-build
description: Read the failing bitbake task log.
scope: [buildsystem, verdin]
requires: [bitbake]
version: 3
created: 2026-06-23T13:26:00
updated: 2026-09-07T15:00:00
---

## Procedure

1. Read the log.
"""


class FakeCollection:
    """Just enough Chroma to reconcile against."""

    def __init__(self, records=()):
        self.records = {identifier: dict(metadata)
                        for identifier, metadata in records}
        self.documents = {}
        self.deleted = []

    def get(self, include=None):
        return {"ids": list(self.records),
                "metadatas": [self.records[key] for key in self.records]}

    def upsert(self, *, ids, documents, metadatas):
        for identifier, document, metadata in zip(ids, documents, metadatas):
            self.records[identifier] = dict(metadata)
            self.documents[identifier] = document

    def delete(self, *, ids):
        self.deleted.extend(ids)

        for identifier in ids:
            self.records.pop(identifier, None)


class ParsingTests(unittest.TestCase):
    def test_a_full_frontmatter_is_read_and_the_body_starts_below_it(self):
        skill = parse_skill("ignored", FRONTMATTER)
        self.assertEqual(skill.name, "debug-buildsystem-build")
        self.assertEqual(skill.description, "Read the failing bitbake task log.")
        self.assertEqual(skill.scope, ("buildsystem", "verdin"))
        self.assertEqual(skill.requires, ("bitbake",))
        self.assertEqual(skill.version, 3)
        self.assertEqual(skill.created, "2026-06-23T13:26:00")
        self.assertTrue(skill.body.startswith("## Procedure"))
        self.assertNotIn("---", skill.body)

    def test_a_file_without_frontmatter_stays_a_valid_skill(self):
        """The six skills already in the library predate this format."""

        skill = parse_skill("improve-c-code-quality", "## Procedure\n\n1. Read.\n")
        self.assertEqual(skill.name, "improve-c-code-quality")
        self.assertEqual(skill.scope, (ANY_SCOPE,))
        self.assertEqual(skill.requires, ())
        self.assertEqual(skill.version, 1)
        self.assertFalse(skill.declared)
        self.assertEqual(skill.body, "## Procedure\n\n1. Read.")

    def test_an_unreadable_block_is_body_rather_than_a_failure(self):
        text = "---\nrequires:\n  - bitbake\n---\n\nbody\n"
        skill = parse_skill("nested", text)
        self.assertFalse(skill.declared)
        self.assertIn("requires:", skill.body)

    def test_the_document_is_unchanged_when_no_description_is_declared(self):
        # Skills already embedded must keep matching without a re-index.
        skill = parse_skill("s", "## Procedure\n\n1. Read.\n")
        self.assertEqual(skill.document, "# Skill: s\n\n## Procedure\n\n1. Read.")

    def test_a_declared_description_joins_the_embedded_document(self):
        skill = parse_skill("x", FRONTMATTER)
        self.assertIn("Read the failing bitbake task log.", skill.document)
        self.assertNotIn("version:", skill.document)

    def test_the_summary_falls_back_to_the_first_line_of_the_body(self):
        self.assertEqual(parse_skill("s", "# Title\n\nrest\n").summary, "Title")
        self.assertEqual(parse_skill("s", FRONTMATTER).summary,
                         "Read the failing bitbake task log.")

    def test_render_round_trips_every_field_it_reads(self):
        skill = parse_skill("x", FRONTMATTER)
        self.assertEqual(parse_skill("x", render_skill(skill)), skill)

    def test_an_unknown_key_survives_the_round_trip(self):
        skill = parse_skill("x", "---\nname: x\nauthor: someone\n---\n\nbody\n")
        self.assertEqual(skill.extra, (("author", "someone"),))
        self.assertIn("author: someone", render_skill(skill))


class ApplicabilityTests(unittest.TestCase):
    def skill(self, **fields):
        return Skill(name="s", body="b", **fields)

    def test_the_default_scope_withholds_nothing(self):
        self.assertTrue(applies_to(self.skill(), project="lvgl", kind="generic"))

    def test_a_narrowed_scope_matches_the_corpus_or_its_kind(self):
        scoped = self.skill(scope=("buildsystem",))
        self.assertTrue(applies_to(scoped, project="verdin", kind="buildsystem"))
        self.assertFalse(applies_to(scoped, project="lvgl", kind="generic"))
        named = self.skill(scope=("lvgl",))
        self.assertTrue(applies_to(named, project="lvgl", kind="generic"))

    def test_an_empty_project_never_matches_an_empty_scope_entry(self):
        self.assertFalse(applies_to(self.skill(scope=("",)), project="", kind=""))

    def test_a_missing_command_withholds_the_skill_and_is_named(self):
        needs = self.skill(requires=("bitbake", "pdftotext"))
        absent = {"pdftotext": "/usr/bin/pdftotext"}.get
        self.assertEqual(missing_requirements(needs, which=absent), ("bitbake",))
        self.assertFalse(applies_to(needs, project="verdin", kind="buildsystem",
                                    which=absent))
        self.assertTrue(applies_to(needs, project="verdin", kind="buildsystem",
                                   which=lambda command: "/usr/bin/x"))


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = holder.name

    def write(self, name, text):
        with open(os.path.join(self.directory, f"{name}.md"), "w") as handle:
            handle.write(text)

    def test_a_hand_written_file_reaches_the_index(self):
        self.write("hand-written", "## Procedure\n\n1. Read.\n")
        collection = FakeCollection()
        moved = reconcile(load_library(self.directory), collection, project="so3")
        self.assertEqual(moved, {"indexed": 1, "removed": 0, "added": 1})
        self.assertIn("hand-written", collection.documents)
        self.assertEqual(collection.records["hand-written"]["project"], "so3")

    def test_an_unchanged_library_embeds_nothing(self):
        self.write("stable", "## Procedure\n\n1. Read.\n")
        collection = FakeCollection()
        library = load_library(self.directory)
        reconcile(library, collection)
        collection.documents.clear()
        moved = reconcile(library, collection)
        self.assertEqual(moved, {"indexed": 0, "removed": 0, "added": 0})
        self.assertEqual(collection.documents, {})

    def test_an_edited_file_is_re_embedded(self):
        self.write("edited", "## Procedure\n\n1. Read.\n")
        collection = FakeCollection()
        reconcile(load_library(self.directory), collection)
        self.write("edited", "## Procedure\n\n1. Read the other log.\n")
        moved = reconcile(load_library(self.directory), collection)
        self.assertEqual(moved["indexed"], 1)
        self.assertIn("other log", collection.documents["edited"])

    def test_a_deleted_file_leaves_the_index(self):
        collection = FakeCollection([("gone", {"digest": "x"})])
        moved = reconcile([], collection)
        self.assertEqual(moved["removed"], 1)
        self.assertEqual(collection.deleted, ["gone"])
        self.assertNotIn("gone", collection.records)

    def test_a_non_markdown_file_is_not_a_skill(self):
        self.write("real", "body\n")

        with open(os.path.join(self.directory, "notes.txt"), "w") as handle:
            handle.write("not a skill")

        self.assertEqual([skill.name for skill in load_library(self.directory)],
                         ["real"])

    def test_a_missing_directory_is_an_empty_library(self):
        self.assertEqual(load_library(os.path.join(self.directory, "nope")), [])


class SaveTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = holder.name

    def read(self, name):
        with open(os.path.join(self.directory, f"{name}.md")) as handle:
            return handle.read()

    def test_a_new_skill_is_written_with_frontmatter(self):
        skill = save(self.directory, "Debug The Build!", "1. Read the log.",
                     description="Read the failing log.",
                     now=datetime(2026, 9, 7, 15, 0, 0))
        self.assertEqual(skill.name, "debug-the-build")
        self.assertEqual(skill.version, 1)
        text = self.read("debug-the-build")
        self.assertTrue(text.startswith("---\n"))
        self.assertIn("created: 2026-09-07T15:00:00", text)
        self.assertIn("1. Read the log.", text)

    def test_an_overwrite_bumps_the_version_and_keeps_the_first_stamp(self):
        save(self.directory, "s", "first", now=datetime(2026, 9, 1, 9, 0, 0))
        skill = save(self.directory, "s", "second",
                     now=datetime(2026, 9, 7, 15, 0, 0))
        self.assertEqual(skill.version, 2)
        self.assertEqual(skill.created, "2026-09-01T09:00:00")
        self.assertEqual(skill.updated, "2026-09-07T15:00:00")
        self.assertEqual(skill.body, "second")

    def test_a_rewrite_does_not_widen_a_scope_an_operator_narrowed(self):
        save(self.directory, "s", "first", scope=("buildsystem",), requires=("bitbake",))
        skill = save(self.directory, "s", "rewritten by the model")
        self.assertEqual(skill.scope, ("buildsystem",))
        self.assertEqual(skill.requires, ("bitbake",))

    def test_an_empty_name_or_body_is_refused(self):
        for name, content in (("", "body"), ("s", "  "), ("!!!", "body")):
            with self.subTest(name=name):
                with self.assertRaises(SkillError):
                    save(self.directory, name, content)

    def test_the_saved_file_reloads_as_the_same_skill(self):
        saved = save(self.directory, "s", "1. Read.", description="Purpose.")
        self.assertEqual(load_library(self.directory), [saved])


if __name__ == "__main__":
    unittest.main()
