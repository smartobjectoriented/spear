"""Tests for retrieve_context's hybrid lexical channel (rag_chat)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import embedding
import rag_chat


class IdentTermsTest(unittest.TestCase):
    """What must trigger an EXACT search, and what must not."""

    def test_identifiers_are_extracted(self):
        for q, want in [
            ("ou est defini __sys_empty ?", "__sys_empty"),
            ("a quoi sert IB_BUILDROOT_DL_DIR", "IB_BUILDROOT_DL_DIR"),
            ("montre moi sys_do_futex", "sys_do_futex"),
            ("le contenu de linux_6.6-verdin.bb", "linux_6.6-verdin.bb"),
            ("ou est doFsInitStorage", "doFsInitStorage"),
        ]:
            self.assertIn(want, rag_chat._ident_terms(q), q)

    def test_plain_french_yields_no_term(self):
        """A prose question must cost NO lexical query at all — including
        when it starts with a capital letter."""
        for q in ["Comment ajouter un paquet au rootfs buildroot",
                  "Dans quel repertoire se trouve la recette du noyau",
                  "quelle classe bitbake construit le rootfs"]:
            self.assertEqual([], rag_chat._ident_terms(q), q)

    def test_short_tokens_ignored(self):
        self.assertEqual([], rag_chat._ident_terms("a.b et X_Y"))

    def test_term_cap(self):
        q = " ".join(f"CONFIG_SYMBOL_{i}" for i in range(10))
        self.assertLessEqual(len(rag_chat._ident_terms(q)),
                             rag_chat.LEX_MAX_TERMS)

    def test_longest_first(self):
        terms = rag_chat._ident_terms("IB_DIR et IB_BUILDROOT_DL_DIR")
        self.assertEqual("IB_BUILDROOT_DL_DIR", terms[0])


class DefinitionScoreTest(unittest.TestCase):
    """The ranking we build for lexical hits, which get() does not sort."""

    def test_definition_beats_mere_usage(self):
        define = "# File: k.c\n\nstatic long __sys_empty(args)\n{\n\treturn 0;\n}"
        use = "# File: t.c\n\n\tif (fn == __sys_empty)\n\t\tgoto out;"
        self.assertGreater(
            rag_chat._definition_score(define, {"filepath": "k.c"}, "__sys_empty"),
            rag_chat._definition_score(use, {"filepath": "t.c"}, "__sys_empty"))

    def test_filename_match_dominates(self):
        """$contains is tokenised: searching a file name also brings back
        neighbours sharing a token. The path must be decisive."""
        target = "# File: linux_6.6-verdin.bb\n\nrequire linux.inc\n"
        noise = ("# File: e1c-verdin.cfg\n\n" +
                 "\n".join(["linux_6.6-verdin.bb"] * 8))
        self.assertGreater(
            rag_chat._definition_score(
                target, {"filepath": "recipes/linux_6.6-verdin.bb"},
                "linux_6.6-verdin.bb"),
            rag_chat._definition_score(
                noise, {"filepath": "files/e1c-verdin.cfg"},
                "linux_6.6-verdin.bb"))

    def test_header_is_not_counted_as_body(self):
        """The '# File: ...' header is injected by the indexer, not written by
        the author: it must not inflate the body score."""
        doc = "# File: a/b/thing.c\n\nint unrelated;\n"
        self.assertEqual(
            0, rag_chat._definition_score(doc, {"filepath": "zzz.c"}, "thing.c"))


class RRFTest(unittest.TestCase):

    def test_consensus_wins(self):
        """A document ranked well by BOTH channels beats one that is first in
        a single channel."""
        order = rag_chat._rrf([["a", "b", "c"], ["d", "b", "e"]])
        self.assertEqual("b", order[0])

    def test_single_list_is_order_preserving(self):
        self.assertEqual(["x", "y", "z"], rag_chat._rrf([["x", "y", "z"]]))

    def test_empty(self):
        self.assertEqual([], rag_chat._rrf([]))


class RetrieveContextTest(unittest.TestCase):
    """Integration against the real index, when present."""

    @classmethod
    def setUpClass(cls):
        import chromadb
        try:
            cls.col = chromadb.PersistentClient(
                path=rag_chat.DB_PATH).get_collection("edgem1_verdin")
        except Exception:
            raise unittest.SkipTest("index edgem1_verdin absent")

    def test_identifier_reaches_the_prompt(self):
        _, seen = rag_chat.retrieve_context(self.col, "ou est defini __sys_empty ?")
        self.assertIn("avz/kernel/syscalls.c", seen)

    def test_respects_char_budget(self):
        ctx, _ = rag_chat.retrieve_context(self.col, "rootfs buildroot verdin")
        # the "\n\n---\n\n" separators add to the chunk budget
        self.assertLess(len(ctx), rag_chat.MAX_CONTEXT_CHARS + 500)

    def test_lexical_channel_never_breaks_retrieval(self):
        """If the full-text filter fails, the dense search must survive."""
        class Broken:
            def __init__(self, col):
                self.col = col
                # forward metadata: that is how retrieve_context learns which
                # model indexed the collection. Without it the double falls
                # back to MiniLM (384 dims) against a 1024-dim index.
                self.metadata = col.metadata

            def query(self, **kw):
                return self.col.query(**kw)

            def get(self, **kw):
                raise RuntimeError("FTS indisponible")

        broken = Broken(self.col)
        ctx, seen = rag_chat.retrieve_context(broken, "ou est defini __sys_empty ?")
        self.assertTrue(seen)


class EmbeddingModelResolutionTest(unittest.TestCase):
    """The model follows the COLLECTION, not the config. That is the central
    protection: if index and query diverge, Chroma raises nothing and returns
    random neighbours."""

    def setUp(self):
        self._env = os.environ.get("SPEAR_EMBED_MODEL")

    def tearDown(self):
        os.environ.pop("SPEAR_EMBED_MODEL", None)
        if self._env is not None:
            os.environ["SPEAR_EMBED_MODEL"] = self._env

    def test_legacy_collection_without_stamp_stays_on_default(self):
        """Collections indexed before embedding.py lack the key: they are
        MiniLM and must stay so, even if the config changed."""
        os.environ["SPEAR_EMBED_MODEL"] = "BAAI/bge-m3"

        class Legacy:
            metadata = {"hnsw:space": "cosine"}
        self.assertEqual(embedding.DEFAULT,
                         embedding.collection_model(Legacy()))

    def test_stamped_collection_wins_over_config(self):
        os.environ["SPEAR_EMBED_MODEL"] = embedding.DEFAULT

        class Stamped:
            metadata = {"embed_model": "BAAI/bge-m3"}
        self.assertEqual("BAAI/bge-m3", embedding.collection_model(Stamped()))

    def test_missing_metadata_is_not_fatal(self):
        class NoMeta:
            pass
        self.assertEqual(embedding.DEFAULT,
                         embedding.collection_model(NoMeta()))

    def test_unknown_model_falls_back(self):
        os.environ["SPEAR_EMBED_MODEL"] = "acme/does-not-exist"
        self.assertEqual(embedding.DEFAULT, embedding.active_model())

    def test_default_model_loads_nothing(self):
        """chroma-default must load NO model: it is the safe fallback, usable
        without a GPU and without downloading anything."""
        self.assertIsNone(embedding.embed_query("x", embedding.DEFAULT))
        self.assertIsNone(embedding.embed_documents(["x"], embedding.DEFAULT))


class QueryPathTest(unittest.TestCase):
    """retrieve_context must switch to query_embeddings as soon as the
    collection declares a model — otherwise Chroma would re-embed the question
    with MiniLM against a bge-m3 index."""

    class Recorder:
        metadata = {"embed_model": "BAAI/bge-m3"}

        def __init__(self):
            self.kwargs = None

        def query(self, **kw):
            self.kwargs = kw
            return {"ids": [[]], "documents": [[]], "metadatas": [[]],
                    "distances": [[]]}

        def get(self, **kw):
            return {"ids": [], "documents": [], "metadatas": []}

    def test_stamped_collection_uses_query_embeddings(self):
        rec = self.Recorder()
        orig = embedding.embed_query
        embedding.embed_query = lambda text, model=None: [0.1, 0.2, 0.3]
        try:
            rag_chat.retrieve_context(rec, "ou est defini __sys_empty ?")
        finally:
            embedding.embed_query = orig
        self.assertIn("query_embeddings", rec.kwargs)
        self.assertNotIn("query_texts", rec.kwargs)

    def test_legacy_collection_uses_query_texts(self):
        rec = self.Recorder()
        rec.metadata = {}
        rag_chat.retrieve_context(rec, "comment ajouter un paquet")
        self.assertIn("query_texts", rec.kwargs)
        self.assertNotIn("query_embeddings", rec.kwargs)


if __name__ == "__main__":
    unittest.main()


class FederatedRetrievalTest(unittest.TestCase):
    """Several corpora searched per turn, fused by rank.

    Merging them into one index is not equivalent: in the tree this was built
    for, u-boot holds 11298 indexable files against so3's 1482, so kernel code
    would compete 7-to-1 for the same twelve slots. Kept apart, each corpus
    contributes its own best hits.
    """

    class FakeCollection:
        """Minimal Chroma stand-in: fixed ranking, no embedder."""

        def __init__(self, name, files):
            self.name = name
            self.metadata = {}
            self._files = files

        def count(self):
            return len(self._files)

        def query(self, **kw):
            n = kw.get("n_results", len(self._files))
            f = self._files[:n]
            return {"ids": [[f"{self.name}:{i}" for i, _ in enumerate(f)]],
                    "documents": [[f"# File: {p}\n\nbody of {p}" for p in f]],
                    "metadatas": [[{"filepath": p} for p in f]],
                    "distances": [[0.1 * i for i, _ in enumerate(f)]]}

        def get(self, **kw):
            return {"ids": [], "documents": [], "metadatas": []}

    def test_every_corpus_contributes(self):
        a = self.FakeCollection("a", ["kernel/sched.c", "kernel/thread.c"])
        b = self.FakeCollection("b", ["cmd/boot.c", "cmd/mem.c"])
        _, seen = rag_chat.retrieve_context([(a, "so3"), (b, "u-boot")], "sched")
        self.assertTrue(any(p.startswith("so3/") for p in seen), seen)
        self.assertTrue(any(p.startswith("u-boot/") for p in seen), seen)

    def test_paths_are_rerooted_on_the_federation(self):
        """Chunks are indexed relative to THEIR corpus; tools run at the
        federation root. Without re-rooting the model reads `kernel/sched.c`
        while bash needs `so3/kernel/sched.c`."""
        a = self.FakeCollection("a", ["kernel/sched.c"])
        ctx, seen = rag_chat.retrieve_context([(a, "so3")], "sched")
        self.assertEqual({"so3/kernel/sched.c"}, seen)
        self.assertIn("# File: so3/kernel/sched.c", ctx)

    def test_no_prefix_leaves_paths_alone(self):
        a = self.FakeCollection("a", ["kernel/sched.c"])
        ctx, seen = rag_chat.retrieve_context([(a, "")], "sched")
        self.assertEqual({"kernel/sched.c"}, seen)
        self.assertIn("# File: kernel/sched.c", ctx)

    def test_same_relative_path_in_two_corpora_does_not_collide(self):
        """Ids are md5(relpath) per corpus, so two trees sharing a path would
        overwrite each other in the pool if it were keyed by id alone."""
        a = self.FakeCollection("a", ["Makefile"])
        b = self.FakeCollection("b", ["Makefile"])
        _, seen = rag_chat.retrieve_context([(a, "so3"), (b, "u-boot")], "make")
        self.assertEqual({"so3/Makefile", "u-boot/Makefile"}, seen)

    def test_a_broken_corpus_does_not_sink_the_turn(self):
        class Broken:
            name = "broken"
            metadata = {}
            def count(self): return 0
            def query(self, **kw): raise RuntimeError("index corrupt")
            def get(self, **kw): raise RuntimeError("index corrupt")
        good = self.FakeCollection("a", ["kernel/sched.c"])
        _, seen = rag_chat.retrieve_context([(Broken(), "x"), (good, "so3")], "sched")
        self.assertEqual({"so3/kernel/sched.c"}, seen)

    def test_a_bare_collection_still_works(self):
        """Backward compatibility: callers that pass one collection."""
        a = self.FakeCollection("a", ["kernel/sched.c"])
        _, seen = rag_chat.retrieve_context(a, "sched")
        self.assertEqual({"kernel/sched.c"}, seen)
