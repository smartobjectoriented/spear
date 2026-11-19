"""A user's state, written into an image: registry, index, knowledge."""

import json
import os
import sqlite3
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
