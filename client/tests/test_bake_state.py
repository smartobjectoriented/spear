"""A user's state, written into an image: registry, index, knowledge, skills
and learned rules."""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from cli import bake_state
from context import workspace_knowledge as wk


class Registry(unittest.TestCase):
    def bake(self, user, shipped):
        return bake_state.bake_registry(user, shipped, "/home/me",
                                        lambda spec: "named_" + os.path.basename(spec["path"]))

    def test_a_path_under_the_home_becomes_relative_and_keeps_its_collection(self):
        merged, added = self.bake({"tht": {"path": "/home/me/work/tht", "kind": "generic"}}, {})

        self.assertEqual(added, ["tht"])
        self.assertEqual(merged["tht"], {"path": "work/tht", "kind": "generic",
                                         "collection": "named_tht"})

    def test_a_path_elsewhere_is_kept(self):
        merged, _ = self.bake({"x": {"path": "/srv/x"}}, {})

        self.assertEqual(merged["x"]["path"], "/srv/x")

    def test_a_shipped_entry_is_not_replaced(self):
        shipped = {"tht": {"path": "a/b", "collection": "c"}}
        merged, added = self.bake({"tht": {"path": "/home/me/other"}}, shipped)

        self.assertEqual((merged, added), (shipped, []))


class Index(unittest.TestCase):
    def test_the_named_collections_are_copied_with_their_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            own = chromadb.PersistentClient(path=f"{tmp}/own")
            shipped = chromadb.PersistentClient(path=f"{tmp}/shipped")
            source = own.create_collection("mine", metadata={"embed_model": "m"})
            source.add(ids=["a", "b"], documents=["one", "two"],
                       metadatas=[{"filepath": "x"}, {"filepath": "y"}],
                       embeddings=[[1.0, 0.0], [0.0, 1.0]])
            own.create_collection("unnamed")
            stale = shipped.create_collection("mine")
            stale.add(ids=["old"], documents=["old"], embeddings=[[1.0, 1.0]])

            copied = bake_state.bake_index(own, shipped, {"mine", "absent"})
            baked = shipped.get_collection("mine")
            rows = baked.get(ids=["b"], include=["documents", "metadatas", "embeddings"])

            self.assertEqual(copied, {"mine": 2})
            self.assertEqual(baked.count(), 2)
            self.assertEqual(baked.metadata["embed_model"], "m")
            self.assertEqual(rows["documents"], ["two"])
            self.assertEqual(rows["metadatas"], [{"filepath": "y"}])
            self.assertEqual([round(v) for v in rows["embeddings"][0]], [0, 1])
            self.assertEqual([c.name for c in shipped.list_collections()], ["mine"])


class Knowledge(unittest.TestCase):
    def test_active_records_of_named_workspaces_are_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = sqlite3.connect(f"{tmp}/user.sqlite3")
            source.executescript(wk._SCHEMA)

            for ident, workspace, lifecycle in (("r1", "tht", wk.Lifecycle.ACTIVE),
                                                ("r2", "adhoc:tmp", wk.Lifecycle.ACTIVE),
                                                ("r3", "tht", "proposed")):
                body = json.dumps({"version": 1, "updated_at": "t"})
                source.execute("INSERT INTO records VALUES (?, ?, ?, ?)",
                               (ident, workspace, lifecycle, body))

            source.commit()
            source.close()

            count = bake_state.bake_knowledge(f"{tmp}/user.sqlite3", f"{tmp}/common/knowledge.sqlite3",
                                              wk._SCHEMA, wk.Lifecycle.ACTIVE)
            common = sqlite3.connect(f"{tmp}/common/knowledge.sqlite3")

            self.assertEqual(count, 1)
            self.assertEqual(common.execute("SELECT record_id FROM records").fetchall(), [("r1",)])
            self.assertEqual(common.execute("SELECT record_id FROM versions").fetchall(), [("r1",)])
            common.close()

    def test_no_store_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(bake_state.bake_knowledge(f"{tmp}/absent", f"{tmp}/c/k.sqlite3",
                                                       wk._SCHEMA, wk.Lifecycle.ACTIVE), 0)


class SkillsAndLearnedRules(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.state = os.path.join(self.dir, "state")
        self.rules = os.path.join(self.dir, "rules.d")
        os.makedirs(os.path.join(self.state, "skills"))

    def write(self, path, text):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_the_users_skills_are_copied_over_the_images(self):
        target = os.path.join(self.dir, "skills")
        os.makedirs(target)
        self.write(os.path.join(target, "flash.md"), "shipped")
        self.write(os.path.join(self.state, "skills", "flash.md"), "revised")
        self.write(os.path.join(self.state, "skills", "probe.md"), "new")
        self.write(os.path.join(self.state, "skills", "notes.txt"), "not a skill")

        self.assertEqual(bake_state.bake_skills(os.path.join(self.state, "skills"), target),
                         ["flash", "probe"])
        self.assertEqual(sorted(os.listdir(target)), ["flash.md", "probe.md"])

        with open(os.path.join(target, "flash.md")) as handle:
            self.assertEqual(handle.read(), "revised")

    def test_no_skills_of_the_users_adds_nothing(self):
        self.assertEqual(bake_state.bake_skills(os.path.join(self.dir, "absent"), self.dir), [])

    def test_learned_rules_become_a_rule_that_reaches_what_a_learned_rule_reaches(self):
        from context import context_sources

        learned = os.path.join(self.state, "rules-learned.md")
        self.write(learned, "- Never push to main.  (2026-01-02)\n")

        self.assertEqual(bake_state.bake_learned_rules(learned, self.rules), 1)

        baked, = context_sources.rule_candidates(self.rules)
        own, = context_sources.learned_candidate(learned)

        self.assertIn("Never push to main.", baked.text)
        self.assertEqual((baked.scope, baked.tasks), (own.scope, own.tasks))

    def test_a_second_build_adds_only_what_is_new(self):
        learned = os.path.join(self.state, "rules-learned.md")
        self.write(learned, "- One.\n")
        bake_state.bake_learned_rules(learned, self.rules)
        self.write(learned, "- One.\n- Two.\n")

        self.assertEqual(bake_state.bake_learned_rules(learned, self.rules), 1)
        self.assertEqual(bake_state.bake_learned_rules(learned, self.rules), 0)

        with open(os.path.join(self.rules, bake_state.LEARNED_RULE)) as handle:
            self.assertEqual(handle.read(), bake_state.LEARNED_HEADER + "- One.\n- Two.\n")

    def test_nothing_learned_writes_no_rule(self):
        self.assertEqual(bake_state.bake_learned_rules(os.path.join(self.dir, "absent"),
                                                       self.rules), 0)
        self.assertFalse(os.path.exists(self.rules))


class WhatIsAlreadyOpenIsLeftAlone(unittest.TestCase):
    def test_only_a_path_lacking_the_mode_is_changed(self):
        directory = tempfile.mkdtemp()
        closed, opened = os.path.join(directory, "closed"), os.path.join(directory, "open")

        for path, mode in ((closed, 0o600), (opened, 0o777)):
            open(path, "w").close()
            os.chmod(path, mode)

        with unittest.mock.patch("os.chmod", wraps=os.chmod) as chmod:
            bake_state._open(closed, 0o666)
            bake_state._open(opened, 0o666)

        chmod.assert_called_once_with(closed, 0o666)
        self.assertEqual(os.stat(opened).st_mode & 0o777, 0o777)


if __name__ == "__main__":
    unittest.main()
