import hashlib
import json
import inspect
import os
import unittest.mock
import math
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from standard_crossrefs import rebuild_cross_reference_index
from standard_ingest import ingest_pdf
from standard_retrieval import StandardRetrieval, rebuild_lexical_index
from standard_retrieval_eval import (
    StandardRetrievalEvaluationItem, evaluate_retrieval,
)
from standard_store import StandardStore, StandardStoreError
from standard_vector_index import rebuild_vector_index
import standard_vector_index
from standard_schema import sha256_json
from tests.standard_fixture import synthetic_pdf_bytes


HYBRID_PAGES = (
    ((760, "Synthetic Retrieval Standard"),
     (730, "Revision R1 - TEST_FIXTURE / SYNTHETIC"),
     (690, "2.1 Transport Identifier"),
     (660, "transport identifier (TID) means the token selecting a transport endpoint."),
     (620, "3.1 Reserved Identifier"),
     (590, "Implementations shall reject reserved identifier value 0xF.")),
    ((760, "3.2 Informative Example"),
     (730, "Informative example: an unused token is ignored in a demonstration."),
     (690, "4.1 Packet Class Identifier"),
     (660, "A packet class identifier selects the packet family, not validation."),
     (620, "5.1 Validation Behavior"),
     (590, "Validation behavior is specified in Section 3.1."),
     (550, "6.1 Missing Reference"),
     (520, "Refer to Section 9.9 for optional future behavior.")),
)


class FixtureSemanticEmbedder:
    model_id = "TEST_FIXTURE/semantic-features"
    model_revision = "v1"

    @staticmethod
    def _embed(text):
        text = text.casefold()
        groups = (
            ("identifier", "token", "code"),
            ("reserved", "forbidden", "unavailable"),
            ("reject", "invalid", "validation", "should happen"),
            ("packet", "class", "family"),
            ("definition", "means"),
            ("example", "demonstration"),
            ("transport", "endpoint"),
            ("specified", "refer", "section"),
        )
        values = [sum(term in text for term in group) for group in groups] + [1]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts): return [self._embed(text) for text in texts]
    def embed_query(self, text): return self._embed(text)


def _float32(value):
    """Round a float through IEEE-754 binary32, as a real embedding backend does."""
    return struct.unpack("f", struct.pack("f", value))[0]


class Float32CacheEmbedder:
    """Mirrors SentenceTransformer: unit-normalized in float32, widened to float64.

    The widened vector's float64 norm is therefore only approximately 1.0, which is
    what makes re-normalizing a cached vector observable.
    """

    model_id = "TEST_FIXTURE/float32-features"
    model_revision = "v1"
    DIMENSION = 16

    def _embed(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [_float32((digest[index] + 1) / 255.0) for index in range(self.DIMENSION)]
        norm = _float32(math.sqrt(sum(_float32(value * value) for value in raw)))
        return [_float32(value / norm) for value in raw]

    def embed_documents(self, texts): return [self._embed(text) for text in texts]
    def embed_query(self, text): return self._embed(text)


class UnavailableEmbedder(Float32CacheEmbedder):
    """Same pinned identity, but fails loudly if anything is not served from cache."""

    def embed_documents(self, texts):
        raise AssertionError(f"embedding backend invoked for {len(texts)} cached texts")


class StandardHybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        pdf = root / "hybrid.pdf"; pdf.write_bytes(synthetic_pdf_bytes(HYBRID_PAGES))
        self.store = StandardStore(root / "standards")
        self.manifest = ingest_pdf(
            self.store, pdf, standard_id="SYNTH-STD", revision="R1",
            source_origin="TEST_FIXTURE")
        self.original_ids = tuple(unit.source_id for unit in self.store.load_units(
            "SYNTH-STD", "R1"))
        rebuild_lexical_index(self.store, "SYNTH-STD", "R1", created_at="fixed")
        self.crossrefs = rebuild_cross_reference_index(
            self.store, "SYNTH-STD", "R1", created_at="fixed")
        self.embedder = FixtureSemanticEmbedder()
        self.vector = rebuild_vector_index(
            self.store, "SYNTH-STD", "R1", self.embedder, created_at="fixed")
        self.retrieval = StandardRetrieval(self.store, self.embedder)

    def tearDown(self): self.temp.cleanup()

    def source_for(self, phrase):
        return next(unit for unit in self.store.load_units("SYNTH-STD", "R1")
                    if phrase in unit.text)

    def test_indexes_preserve_canonical_identity_and_are_deterministic(self):
        self.assertEqual(self.original_ids, tuple(unit.source_id for unit in
                         self.store.load_units("SYNTH-STD", "R1")))
        rebuilt = rebuild_vector_index(
            self.store, "SYNTH-STD", "R1", self.embedder, created_at="later")
        self.assertEqual(rebuilt.vector_index_fingerprint,
                         self.vector.vector_index_fingerprint)
        self.assertEqual(rebuilt.source_corpus_sha256,
                         self.manifest.corpus_manifest_sha256)
        manifest, index = self.store.load_vector_index("SYNTH-STD", "R1")
        self.assertEqual(set(index["entries"]), set(self.original_ids))
        self.assertEqual(manifest.embedding_model_id, self.embedder.model_id)
        self.assertEqual(manifest.embedding_model_revision, "v1")

    def test_cached_vectors_are_reused_verbatim_and_rebuild_is_stable(self):
        """A cache hit must reproduce the index exactly, not re-derive the vector.

        Re-normalizing an already-normalized float32-derived vector perturbs its
        last representable digits, which changes the vector index fingerprint --
        and therefore the StandardBinding -- on a rebuild that changed nothing.
        """
        embedder = Float32CacheEmbedder()
        first = rebuild_vector_index(
            self.store, "SYNTH-STD", "R1", embedder, created_at="fixed")
        cache = (self.store.revision_dir("SYNTH-STD", "R1")
                 / "indexes" / "embedding-cache")
        before = {path.name: path.read_bytes() for path in sorted(cache.glob("*.json"))}
        self.assertTrue(before)
        _, built = self.store.load_vector_index("SYNTH-STD", "R1")

        cached_only = UnavailableEmbedder()
        second = rebuild_vector_index(
            self.store, "SYNTH-STD", "R1", cached_only, created_at="later")
        third = rebuild_vector_index(
            self.store, "SYNTH-STD", "R1", cached_only, created_at="later-still")

        self.assertEqual(second.vector_index_fingerprint,
                         first.vector_index_fingerprint)
        self.assertEqual(third.vector_index_fingerprint,
                         first.vector_index_fingerprint)
        _, reloaded = self.store.load_vector_index("SYNTH-STD", "R1")
        self.assertEqual(reloaded["entries"], built["entries"])
        self.assertEqual({path.name: path.read_bytes()
                          for path in sorted(cache.glob("*.json"))}, before)

    def test_embedding_revision_changes_vector_not_source_identity(self):
        changed = FixtureSemanticEmbedder(); changed.model_revision = "v2"
        rebuilt = rebuild_vector_index(self.store, "SYNTH-STD", "R1", changed,
                                       created_at="later")
        self.assertNotEqual(rebuilt.vector_index_fingerprint,
                            self.vector.vector_index_fingerprint)
        self.assertEqual(self.original_ids, tuple(unit.source_id for unit in
                         self.store.load_units("SYNTH-STD", "R1")))

    def test_modes_rrf_dedup_and_exact_section_precedence(self):
        exact = self.retrieval.search("SYNTH-STD", "R1", "section 3.1", mode="hybrid")
        self.assertEqual(exact[0].section, "3.1")
        self.assertTrue(exact[0].exact_section_match)
        lexical = self.retrieval.search("SYNTH-STD", "R1", "reserved identifier",
                                        mode="lexical")
        vector = self.retrieval.search("SYNTH-STD", "R1", "forbidden token code",
                                       mode="vector")
        hybrid1 = self.retrieval.search("SYNTH-STD", "R1", "reserved identifier",
                                        mode="hybrid")
        hybrid2 = self.retrieval.search("SYNTH-STD", "R1", "reserved identifier",
                                        mode="hybrid")
        self.assertTrue(lexical and vector and hybrid1)
        self.assertEqual([item.source_id for item in hybrid1],
                         [item.source_id for item in hybrid2])
        self.assertEqual(len({item.source_id for item in hybrid1}), len(hybrid1))
        self.assertTrue(any(item.lexical_rank and item.vector_rank for item in hybrid1))
        direct = self.retrieval.search(
            "SYNTH-STD", "R1", hybrid1[0].source_id, mode="hybrid")
        self.assertEqual(direct[0].source_id, hybrid1[0].source_id)
        self.assertEqual(direct[0].match_reason, "exact_source")

    def test_citation_is_identical_across_retrieval_modes(self):
        source = self.source_for("shall reject")
        citations = []
        for mode in ("lexical", "vector", "hybrid"):
            results = self.retrieval.search("SYNTH-STD", "R1", "reserved identifier",
                                            mode=mode)
            self.assertIn(source.source_id, {item.source_id for item in results})
            citations.append(self.retrieval.cite(
                "SYNTH-STD", "R1", source.source_id).render())
        self.assertEqual(len(set(citations)), 1)

    def test_parent_neighbors_and_one_hop_cross_reference(self):
        source = self.source_for("Validation behavior")
        fetched = self.retrieval.fetch("SYNTH-STD", "R1", source.source_id,
                                       neighbor_limit=1)
        self.assertIsNotNone(fetched["parent"])
        self.assertLessEqual(len(fetched["neighbors"]), 2)
        self.assertEqual(fetched["resolved_cross_references"][0]["section"], "3.1")
        results = self.retrieval.search("SYNTH-STD", "R1", "validation behavior",
                                        mode="hybrid", limit=3)
        self.assertEqual(results[0].section, "5.1")
        self.assertLessEqual(len([x for x in results if x.referenced_from]), 2)
        self.assertTrue(any(x.referenced_from == source.source_id for x in results))

    def test_missing_cross_reference_is_unresolved_not_guessed(self):
        source = self.source_for("9.9")
        fetched = self.retrieval.fetch("SYNTH-STD", "R1", source.source_id)
        self.assertEqual(fetched["unresolved_cross_references"][0]["status"],
                         "unresolved")
        self.assertEqual(fetched["unresolved_cross_references"][0][
                         "target_source_ids"], [])

    def test_corrupt_cross_reference_index_is_ignored_not_canonicalized(self):
        directory = self.store.revision_dir("SYNTH-STD", "R1") / "indexes" / "crossrefs"
        raw = json.loads((directory / "index.json").read_text())
        raw["counts"]["resolved"] += 1
        (directory / "index.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(StandardStoreError, "fingerprint mismatch"):
            self.store.load_cross_reference_index("SYNTH-STD", "R1")
        source = self.source_for("Validation behavior")
        fetched = self.retrieval.fetch("SYNTH-STD", "R1", source.source_id)
        self.assertEqual(fetched["unit"]["source_id"], source.source_id)
        self.assertEqual(self.store.verify_corpus("SYNTH-STD", "R1").corpus_manifest_sha256,
                         self.manifest.corpus_manifest_sha256)

    def test_ambiguous_cross_reference_is_not_guessed(self):
        """A corpus whose anchors are only textual can still be ambiguous.

        Schema-1 units carry no heading classification, so two lines opening with
        the same clause number are both anchors and the reference stays unresolved.
        """
        from standard_ingest import build_manifest, extract_pdf_pages
        from tests.standard_extraction_fixture import legacy_canonical_units

        root = Path(self.temp.name); pdf = root / "ambiguous.pdf"
        pdf.write_bytes(synthetic_pdf_bytes((((760, "7.1 First Anchor"),
                                               (700, "7.1 Second Anchor"),
                                               (640, "8.1 Reference"),
                                               (600, "See Section 7.1.")),)))
        store = StandardStore(root / "ambiguous-store")
        pages, warnings = extract_pdf_pages(pdf)
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        units = legacy_canonical_units(pages, standard_id="AMB", revision="R1",
                                       pdf_sha256=sha)
        manifest = build_manifest(
            units, standard_id="AMB", revision="R1", pdf_sha256=sha,
            filename=pdf.name, page_count=len(pages), warnings=warnings,
            source_origin="TEST_FIXTURE", retain_pdf=False,
            ingestion_timestamp="2026-01-01T00:00:00+00:00")
        object.__setattr__(manifest, "corpus_schema_version", 1)
        store.save_ingestion(manifest, units)
        rebuild_lexical_index(store, "AMB", "R1")
        index = rebuild_cross_reference_index(store, "AMB", "R1")
        self.assertEqual(index.ambiguous_count, 1)
        source = next(unit for unit in store.load_units("AMB", "R1")
                      if "See Section" in unit.text)
        relations = StandardRetrieval(store).fetch(
            "AMB", "R1", source.source_id)["unresolved_cross_references"]
        self.assertIn("ambiguous", {relation["status"] for relation in relations})
        self.assertTrue(all(relation["target_source_ids"] == []
                            for relation in relations))

    def test_vector_corruption_dimension_and_stale_ids_fail_closed(self):
        directory = self.store.revision_dir("SYNTH-STD", "R1") / "indexes" / "vector"
        original = (directory / "index.json").read_bytes()
        raw = json.loads(original); first = next(iter(raw["entries"]))
        raw["entries"][first] = raw["entries"][first][:-1]
        (directory / "index.json").write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(StandardStoreError, "fingerprint mismatch"):
            self.store.load_vector_index("SYNTH-STD", "R1")
        (directory / "index.json").write_bytes(original)
        self.assertEqual(self.store.verify_corpus("SYNTH-STD", "R1").corpus_manifest_sha256,
                         self.manifest.corpus_manifest_sha256)

    def test_vector_dimension_and_stale_source_validation_after_valid_hash(self):
        directory = self.store.revision_dir("SYNTH-STD", "R1") / "indexes" / "vector"
        index = json.loads((directory / "index.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        first = next(iter(index["entries"])); index["entries"][first].pop()
        manifest["vector_index_fingerprint"] = sha256_json({
            "vector_index_version": manifest["vector_index_version"],
            "source_corpus_sha256": manifest["source_corpus_sha256"], "index": index})
        (directory / "index.json").write_text(json.dumps(index))
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(StandardStoreError, "dimension mismatch"):
            self.store.load_vector_index("SYNTH-STD", "R1")
        rebuild_vector_index(self.store, "SYNTH-STD", "R1", self.embedder,
                             created_at="fixed")
        index = json.loads((directory / "index.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        index["entries"]["std-" + "a" * 32] = next(iter(index["entries"].values()))
        manifest["indexed_source_count"] += 1
        manifest["vector_index_fingerprint"] = sha256_json({
            "vector_index_version": manifest["vector_index_version"],
            "source_corpus_sha256": manifest["source_corpus_sha256"], "index": index})
        (directory / "index.json").write_text(json.dumps(index))
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(StandardStoreError, "stale source IDs"):
            self.store.load_vector_index("SYNTH-STD", "R1")

    def test_the_model_is_loaded_offline_and_no_http_client_is_used(self):
        source = inspect.getsource(standard_vector_index)
        self.assertIn("local_files_only=True", source)
        self.assertNotIn("requests.", source)

    def test_a_licensed_corpus_is_never_embedded_off_this_machine(self):
        """The offload used to be absent from the module outright. It is here
        now, because minutes of saturated CPU for a public document is a real
        cost — but indexing sends the STANDARD'S TEXT to a host shared under a
        common login, so a licensed corpus does not go, however it is
        configured."""
        with unittest.mock.patch.dict(os.environ, {
            "SPEAR_STANDARD_EMBED_MODEL": "BAAI/bge-m3",
            "SPEAR_STANDARD_EMBED_REVISION": "abc123",
            "SPEAR_STANDARD_EMBED_REMOTE": "gpu@example.invalid",
        }), unittest.mock.patch.object(
            standard_vector_index, "SentenceTransformerStandardEmbedder"
        ) as local, unittest.mock.patch.object(
            standard_vector_index, "remote_model_revision", return_value="abc123"
        ):
            licensed = standard_vector_index.configured_embedder("LICENSED_STANDARD")
            self.assertIs(licensed, local.return_value)
            self.assertIsInstance(
                standard_vector_index.configured_embedder("PUBLIC"),
                standard_vector_index.OffloadedStandardEmbedder)
            # configured_local_embedder() is the "never offload" entry point.
            self.assertIs(standard_vector_index.configured_local_embedder(),
                          local.return_value)

    def test_a_pinned_revision_the_gpu_host_does_not_have_is_refused(self):
        """The worker takes a model name and no revision, so it loads whatever
        that host resolves `main` to. Unchecked, the index would be stamped
        with a revision its vectors were not produced by."""
        with unittest.mock.patch.dict(os.environ, {
            "SPEAR_STANDARD_EMBED_MODEL": "BAAI/bge-m3",
            "SPEAR_STANDARD_EMBED_REVISION": "abc123",
            "SPEAR_STANDARD_EMBED_REMOTE": "gpu@example.invalid",
        }), unittest.mock.patch.object(
            standard_vector_index, "SentenceTransformerStandardEmbedder"
        ), unittest.mock.patch.object(
            standard_vector_index, "remote_model_revision", return_value="deadbeef"
        ):
            with self.assertRaises(RuntimeError) as raised:
                standard_vector_index.configured_embedder("PUBLIC")

        self.assertIn("not the pinned abc123", str(raised.exception))

    def test_reingesting_the_same_pdf_can_still_change_the_classification(self):
        """An identical corpus returns the stored manifest untouched, which was
        right for the units and wrong for this field: --origin PUBLIC over an
        already-ingested standard gave no error, no change, and an index built
        on the wrong machine while the command said otherwise."""
        self.assertEqual("TEST_FIXTURE", self.store.load_manifest(
            "SYNTH-STD", "R1").source_origin)
        previous = self.store.reclassify_origin("SYNTH-STD", "R1", "PUBLIC")
        self.assertEqual("TEST_FIXTURE", previous)
        self.assertEqual("PUBLIC", self.store.load_manifest(
            "SYNTH-STD", "R1").source_origin)
        # The corpus itself did not move, so nothing has to be rebuilt.
        self.assertEqual(self.manifest.corpus_manifest_sha256,
                         self.store.load_manifest(
                             "SYNTH-STD", "R1").corpus_manifest_sha256)
        self.store.verify_corpus("SYNTH-STD", "R1")
        # Reclassifying to what it already is reports no change.
        self.assertEqual("PUBLIC", self.store.reclassify_origin(
            "SYNTH-STD", "R1", "PUBLIC"))

    def test_where_the_model_ran_is_part_of_the_index_identity(self):
        """Measured on NISTIR 6556 with bge-m3 at one pinned revision: the GPU
        host's vectors differ from this machine's by up to 2.8e-4 per
        component. Reduced precision, not float32 rounding — so the same model
        and revision do NOT produce the same index everywhere, and a
        fingerprint blind to that would call two different indexes one."""
        class Offloaded(FixtureSemanticEmbedder):
            compute_label = "offload:gpu@example.invalid"

        local = rebuild_vector_index(self.store, "SYNTH-STD", "R1",
                                     FixtureSemanticEmbedder(), created_at="a")
        remote = rebuild_vector_index(self.store, "SYNTH-STD", "R1",
                                      Offloaded(), created_at="a")
        self.assertNotEqual(local.embedding_config_fingerprint,
                            remote.embedding_config_fingerprint)
        self.assertNotEqual(local.vector_index_fingerprint,
                            remote.vector_index_fingerprint)

    def test_an_operator_override_lets_a_licensed_corpus_offload(self):
        """--allow-offload is per command and never stored. Relabelling the
        corpus PUBLIC would buy the same offload AND tell training governance
        the text is exportable, which is a much larger claim."""
        with unittest.mock.patch.dict(os.environ, {
            "SPEAR_STANDARD_EMBED_MODEL": "BAAI/bge-m3",
            "SPEAR_STANDARD_EMBED_REVISION": "abc123",
            "SPEAR_STANDARD_EMBED_REMOTE": "gpu@example.invalid",
        }), unittest.mock.patch.object(
            standard_vector_index, "SentenceTransformerStandardEmbedder"
        ), unittest.mock.patch.object(
            standard_vector_index, "remote_model_revision", return_value="abc123"
        ):
            self.assertIsInstance(
                standard_vector_index.configured_embedder(
                    "LICENSED_STANDARD", allow_offload=True),
                standard_vector_index.OffloadedStandardEmbedder)
            # …and the default is still a refusal.
            self.assertNotIsInstance(
                standard_vector_index.configured_embedder("LICENSED_STANDARD"),
                standard_vector_index.OffloadedStandardEmbedder)

    def test_a_disclosure_survives_the_next_rebuild(self):
        """"Was this licensed text ever sent anywhere?" must not be answerable
        only until the index is rebuilt, so the record appends."""
        self.assertEqual([], standard_vector_index.offload_disclosures(
            self.store, "SYNTH-STD", "R1"))
        for host in ("gpu-a@example.invalid", "gpu-b@example.invalid"):
            standard_vector_index.record_offload_disclosure(
                self.store, "SYNTH-STD", "R1", target=host, unit_count=12,
                model_id="BAAI/bge-m3", at="2026-09-07T10:00:00+00:00")

        recorded = standard_vector_index.offload_disclosures(
            self.store, "SYNTH-STD", "R1")
        self.assertEqual(["gpu-a@example.invalid", "gpu-b@example.invalid"],
                         [item["target"] for item in recorded])

        # It is not part of the index identity: the same model and revision
        # produce the same vectors wherever they run.
        rebuilt = rebuild_vector_index(self.store, "SYNTH-STD", "R1",
                                       self.embedder, created_at="later")
        self.assertEqual(rebuilt.vector_index_fingerprint,
                         self.vector.vector_index_fingerprint)
        self.assertEqual(2, len(standard_vector_index.offload_disclosures(
            self.store, "SYNTH-STD", "R1")))

    def test_a_question_is_never_sent_to_the_gpu_host(self):
        """Indexing ships the standard's text; answering ships the user's
        words. Only the first is a disclosure, and only the first is offloaded."""
        with unittest.mock.patch.object(
            standard_vector_index, "SentenceTransformerStandardEmbedder"
        ) as local, unittest.mock.patch.object(
            standard_vector_index, "remote_model_revision", return_value="r1"
        ):
            embedder = standard_vector_index.OffloadedStandardEmbedder(
                "BAAI/bge-m3", "r1", "gpu@example.invalid")
            embedder.embed_query("what does G17 select?")

        local.return_value.embed_query.assert_called_once_with(
            "what does G17 select?")

    def test_corrupt_or_missing_vector_has_explicit_lexical_fallback(self):
        directory = self.store.revision_dir("SYNTH-STD", "R1") / "indexes" / "vector"
        (directory / "manifest.json").unlink()
        response = self.retrieval.search_response(
            "SYNTH-STD", "R1", "reserved identifier", mode="hybrid")
        self.assertEqual(response.retrieval_mode_used, "lexical_fallback")
        self.assertTrue(response.results)

    def test_benchmark_is_deterministic_and_hybrid_preserves_structure(self):
        requirement = self.source_for("shall reject")
        validation = self.source_for("Validation behavior")
        items = (
            StandardRetrievalEvaluationItem("section 3.1", "SYNTH-STD", "R1",
                                            "structural", expected_section="3.1"),
            StandardRetrievalEvaluationItem("reserved identifier", "SYNTH-STD", "R1",
                                            "exact-term", (requirement.source_id,)),
            StandardRetrievalEvaluationItem(
                "what should happen for a forbidden identifier code?",
                "SYNTH-STD", "R1", "conceptual", (requirement.source_id,)),
            StandardRetrievalEvaluationItem("validation behavior", "SYNTH-STD", "R1",
                                            "cross-reference", (validation.source_id,)),
            StandardRetrievalEvaluationItem("TID", "SYNTH-STD", "R1",
                                            "acronym", expected_section="2.1"),
            StandardRetrievalEvaluationItem("transport token definition", "SYNTH-STD", "R1",
                                            "definition", expected_section="2.1"),
            StandardRetrievalEvaluationItem("shall reject 0xF", "SYNTH-STD", "R1",
                                            "requirement", (requirement.source_id,)),
            StandardRetrievalEvaluationItem("nonexistent lunar checksum", "SYNTH-STD", "R1",
                                            "no-answer"),
        )
        report = evaluate_retrieval(self.retrieval, items)
        self.assertGreaterEqual(report.modes["hybrid"].recall_at_3,
                                report.modes["lexical"].recall_at_3)
        self.assertGreater(
            report.by_category["conceptual"]["hybrid"].recall_at_3,
            report.by_category["conceptual"]["lexical"].recall_at_3)
        self.assertEqual(report.by_category["structural"]["hybrid"].recall_at_1, 1)
        self.assertEqual(json.loads(report.to_json())["embedding_model"],
                         self.embedder.model_id)
        self.assertIn("hybrid:", report.to_text())
