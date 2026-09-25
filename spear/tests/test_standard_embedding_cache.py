"""The embedding cache is one matrix per configuration, not a file per vector.

A file per vector, each written with an fsync, was 262 470 files for the Arm
manual and most of a 25-minute rebuild, with every vector held as a Python
list until the end.
"""

import json
import tempfile
import unittest
from pathlib import Path

from standard_ingest import ingest_pdf
from standard_schema import canonical_json, sha256_json
from standard_store import StandardStore
from standard_vector_index import rebuild_vector_index, vector_entries
from tests.standard_fixture import synthetic_pdf_bytes
from tests.test_standard_hybrid_retrieval import (
    HYBRID_PAGES, Float32CacheEmbedder, UnavailableEmbedder,
)

SID, REV = "SYNTH-STD", "R1"


class CountingEmbedder(Float32CacheEmbedder):
    def __init__(self):
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return super().embed_documents(texts)


class EmbeddingCacheTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        pdf = root / "hybrid.pdf"
        pdf.write_bytes(synthetic_pdf_bytes(HYBRID_PAGES))
        self.store = StandardStore(root / "standards")
        ingest_pdf(self.store, pdf, standard_id=SID, revision=REV,
                   source_origin="TEST_FIXTURE")
        self.first = rebuild_vector_index(self.store, SID, REV,
                                          Float32CacheEmbedder(), created_at="fixed")
        self.cache = self.store.revision_dir(SID, REV) / "indexes" / "embedding-cache"

    def tearDown(self):
        self.temp.cleanup()

    def test_the_cache_is_one_matrix_and_its_key_list(self):
        names = sorted(path.name for path in self.cache.iterdir())
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].startswith("cache-") and names[0].endswith(".json"))
        self.assertTrue(names[1].endswith(".npy"))

    def test_a_rebuild_is_served_from_it_and_changes_nothing(self):
        again = rebuild_vector_index(self.store, SID, REV, UnavailableEmbedder(),
                                     created_at="later")
        self.assertEqual(again.vector_index_fingerprint,
                         self.first.vector_index_fingerprint)

    def test_a_corrupt_cache_is_embedded_again_not_trusted(self):
        matrix = next(self.cache.glob("*.npy"))
        data = bytearray(matrix.read_bytes()); data[-1] ^= 0xFF
        matrix.write_bytes(bytes(data))
        embedder = CountingEmbedder()

        again = rebuild_vector_index(self.store, SID, REV, embedder, created_at="later")

        self.assertGreater(embedder.embedded, 0)
        self.assertEqual(again.vector_index_fingerprint,
                         self.first.vector_index_fingerprint)

    def test_the_old_file_per_vector_cache_is_migrated_then_removed(self):
        """What the previous format cached is not embedded a second time."""
        _, index = self.store.load_vector_index(SID, REV)
        config = json.loads(next(self.cache.glob("*.json")).read_text())
        fingerprint = config["config_fingerprint"]

        for path in list(self.cache.iterdir()):
            path.unlink()

        for source_id, vector in vector_entries(index).items():
            key = sha256_json({"config": fingerprint, "retrieval_text_sha256":
                               index["retrieval_text_sha256"][source_id]})
            (self.cache / f"{key}.json").write_bytes(canonical_json(
                {"schema_version": 1, "vector": vector}))

        again = rebuild_vector_index(self.store, SID, REV, UnavailableEmbedder(),
                                     created_at="later")

        self.assertEqual(again.vector_index_fingerprint,
                         self.first.vector_index_fingerprint)
        self.assertEqual(len(list(self.cache.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
