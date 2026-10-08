"""Selecting the discovered corpora by what the question names.

A workspace launch federates everything registered under it -- eight trees at
~/soo/so3 -- and querying all of them every turn costs an embedding call each
for corpora the question never touches.
"""
import os
import tempfile
import unittest

from cli import corpus_search, operator_input, session_workspace


class Collection:
    def __init__(self, name):
        self.name = name


PROJECTS = {
    "so3": {"path": "/home/operator/soo/so3/so3"},
    "so3-doc": {"path": "/home/operator/soo/so3/doc"},
    "avz": {"path": "/home/operator/soo/so3/avz"},
    "u-boot": {"path": "/home/operator/soo/so3/u-boot"},
    "micropython-so3": {"path": "/home/operator/soo/so3/so3/usr/src/micropython"},
    "claude": {"path": "/opt/llm/spear/claude", "shared": True},
}


class SelectCorporaTests(unittest.TestCase):
    def setUp(self):
        self.previous = (dict(corpus_search.AUTO_CORPUS_BY_COLLECTION),
                         session_workspace.PROJECT)
        corpus_search.AUTO_CORPUS_BY_COLLECTION.clear()
        corpus_search.AUTO_CORPUS_BY_COLLECTION.update({
            "c_so3": "so3", "c_doc": "so3-doc", "c_avz": "avz",
            "c_uboot": "u-boot", "c_mpy": "micropython-so3",
        })
        session_workspace.PROJECT = "workspace:so3"
        self.corpora = [(Collection(name), "") for name in
                        ("c_so3", "c_doc", "c_avz", "c_uboot", "c_mpy",
                         "c_shared")]
        self.addCleanup(self.restore)

    def restore(self):
        corpus_search.AUTO_CORPUS_BY_COLLECTION.clear()
        corpus_search.AUTO_CORPUS_BY_COLLECTION.update(self.previous[0])
        session_workspace.PROJECT = self.previous[1]

    def kept(self, query):
        return [col.name for col, _ in
                corpus_search.select_corpora(self.corpora, query, PROJECTS)]

    def test_a_named_corpus_is_the_one_queried(self):
        self.assertEqual(["c_avz", "c_shared"], self.kept("how does avz boot?"))

    def test_the_directory_name_counts_as_a_mention(self):
        """"a chapter in doc" names doc/, which is corpus so3-doc."""
        self.assertEqual(["c_doc", "c_shared"],
                         self.kept("write a chapter in doc for more.c"))

    def test_several_named_corpora_are_all_kept(self):
        self.assertEqual(["c_so3", "c_avz", "c_shared"],
                         self.kept("how does so3 talk to avz?"))

    def test_naming_nothing_falls_back_to_the_workspace_s_own_part(self):
        self.assertEqual(["c_so3", "c_shared"],
                         self.kept("where is the scheduler implemented?"))

    def test_what_was_not_discovered_is_never_dropped(self):
        """A shared corpus is a deliberate choice, not a discovery."""
        for query in ("avz", "doc", "anything at all"):
            with self.subTest(query=query):
                self.assertIn("c_shared", self.kept(query))

    def test_a_session_with_no_discovery_is_untouched(self):
        corpus_search.AUTO_CORPUS_BY_COLLECTION.clear()
        self.assertEqual(6, len(self.kept("avz")))

    def test_a_substring_is_not_a_mention(self):
        """`so3` must not be matched inside `micropython-so3`.

        `-` is a regex word boundary, so a plain \\b would attach the kernel to
        a question that named only the library.
        """
        kept = self.kept("what does micropython-so3 hold?")
        self.assertIn("c_mpy", kept)
        self.assertNotIn("c_so3", kept)

    def test_a_path_fragment_still_names_the_corpus(self):
        self.assertIn("c_so3", self.kept("read so3/usr/src/ping.c"))


if __name__ == "__main__":
    unittest.main()


class MentionHintTests(unittest.TestCase):
    """The hint tells you to cd somewhere. Not to a corpus already attached.

    An umbrella federates everything under the cwd, so `so3`'s chunks arrive as
    `so3/usr/src/...` -- openable from where bash runs. Telling the user to
    `cd so3` there is advice to leave a session that can already reach the file.
    """
    def setUp(self):
        self.previous = (session_workspace.PROJECT_SPEC, session_workspace.PROJECT_ROOT,
                         set(operator_input._HINTED_CORPORA))
        operator_input._HINTED_CORPORA.clear()

        # The hint is only offered for a corpus that EXISTS -- there is no
        # point telling someone to cd somewhere that is not there. So the
        # fixture has to be a real directory. It used to be a path that
        # happened to exist on one machine, which made the test pass there
        # and nowhere else.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.avz = os.path.join(self.tmp.name, "soo", "so3", "avz")
        os.makedirs(self.avz)

        session_workspace.PROJECT_ROOT = os.path.join(self.tmp.name, "soo", "so3")
        self.addCleanup(self.restore)

    def restore(self):
        session_workspace.PROJECT_SPEC, session_workspace.PROJECT_ROOT, hinted = self.previous
        operator_input._HINTED_CORPORA.clear()
        operator_input._HINTED_CORPORA.update(hinted)

    def test_an_attached_corpus_is_not_somewhere_to_go(self):
        session_workspace.PROJECT_SPEC = {"name": "workspace:so3", "corpora": ["avz"]}
        projects = {"avz": {"path": self.avz}}
        self.assertIsNone(operator_input.corpus_mention_hint(
            "how does avz boot?", projects, "workspace:so3"))

    def test_an_unattached_corpus_still_gets_the_hint(self):
        session_workspace.PROJECT_SPEC = {"name": "workspace:so3", "corpora": []}
        projects = {"avz": {"path": self.avz}}
        hint = operator_input.corpus_mention_hint(
            "how does avz boot?", projects, "workspace:so3")
        self.assertIsNotNone(hint)
        self.assertIn("avz", hint)
