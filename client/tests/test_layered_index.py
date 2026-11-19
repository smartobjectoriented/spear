"""The user's index is read before the one an image ships, and is the one
written."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from cli import corpus_search


def vectors(count):
    return [[float(i), 1.0] for i in range(count)]


class LayeredIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.own = chromadb.PersistentClient(path=f"{self.tmp.name}/own")
        self.shipped = chromadb.PersistentClient(path=f"{self.tmp.name}/shipped")
        self.index = corpus_search.LayeredIndex(self.own, self.shipped)

        baked = self.shipped.create_collection("baked")
        baked.add(ids=["a", "b"], documents=["one", "two"], embeddings=vectors(2))

    def test_a_shipped_collection_is_found(self):
        self.assertEqual(self.index.get_collection(name="baked").count(), 2)

    def test_the_user_s_collection_of_the_same_name_wins(self):
        mine = self.own.create_collection("baked")
        mine.add(ids=["a"], documents=["mine"], embeddings=vectors(1))

        self.assertEqual(self.index.get_collection(name="baked").count(), 1)
        self.assertEqual([c.name for c in self.index.list_collections()], ["baked"])

    def test_what_is_created_goes_to_the_user_s(self):
        self.index.get_or_create_collection("new")

        self.assertEqual([c.name for c in self.own.list_collections()], ["new"])
        self.assertEqual([c.name for c in self.shipped.list_collections()], ["baked"])
        self.assertEqual(sorted(c.name for c in self.index.list_collections()),
                         ["baked", "new"])

    def test_an_unknown_collection_still_fails(self):
        with self.assertRaises(Exception):
            self.index.get_collection(name="absent")


if __name__ == "__main__":
    unittest.main()
