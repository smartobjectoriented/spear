"""Private filesystem store for authoritative standard corpora and derived indexes."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from standard_progress import report
from standard_schema import (
    HumanValidationStatus, StandardBinding,
    StandardCrossReferenceIndexManifest, StandardDocumentUnit,
    StandardIndexManifest, StandardIngestionManifest,
    StandardVectorIndexManifest, canonical_json, sha256_json,
    DEFAULT_STANDARD_RETRIEVAL_CONFIGURATION,
)


_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SOURCE_ID = re.compile(r"^std-[0-9a-f]{32}$")
_CANDIDATE_ID = re.compile(r"^cand-[0-9a-f]{16}$")
_GENERATION_ID = re.compile(r"^gen-[0-9a-f]{16}$")

INDEX_STATE_READY = "READY"
INDEX_STATE_REBUILD_REQUIRED = "REBUILD_REQUIRED"

RETRIEVAL_MODES = ("lexical", "vector", "hybrid")
RETRIEVAL_SETTINGS_FILE = "retrieval.json"


def candidate_id_for(corpus_manifest_sha256: str) -> str:
    """A candidate is named by what it contains, so re-extraction is idempotent."""
    return "cand-" + corpus_manifest_sha256[:16]


def generation_id_for(corpus_manifest_sha256: str) -> str:
    return "gen-" + corpus_manifest_sha256[:16]


class StandardStoreError(RuntimeError):
    pass


class StandardCollisionError(StandardStoreError):
    pass


def corpus_fingerprint(units: Iterable[StandardDocumentUnit]) -> str:
    ordered = sorted((unit.to_dict() for unit in units), key=lambda item: item["source_id"])
    return sha256_json({"corpus_schema_version": 1, "units": ordered})


class StandardStore:
    """A confined, atomic store. Canonical corpus never depends on its index."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        candidate = Path(root).expanduser()
        candidate.mkdir(parents=True, exist_ok=True, mode=0o700)

        if candidate.is_symlink():
            raise StandardStoreError("standard root cannot be a symlink")

        self.root = candidate.resolve()
        os.chmod(self.root, 0o700)

    @staticmethod
    def _component(value: str, name: str) -> str:
        if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
            raise StandardStoreError(f"invalid {name}")

        if value in {".", ".."}:
            raise StandardStoreError(f"invalid {name}")

        return value

    def revision_dir(self, standard_id: str, revision: str, *, create: bool = False) -> Path:
        sid = self._component(standard_id, "standard_id")
        rev = self._component(revision, "revision")
        standard_dir = self.root / sid
        path = standard_dir / rev

        for probe in (standard_dir, path):
            if probe.exists() and probe.is_symlink():
                raise StandardStoreError("standard path uses a symlink")

        if create:
            standard_dir.mkdir(mode=0o700, exist_ok=True)
            path.mkdir(mode=0o700, exist_ok=True)
            os.chmod(standard_dir, 0o700)
            os.chmod(path, 0o700)

        resolved_parent = path.parent.resolve()

        if resolved_parent != standard_dir.resolve() or standard_dir.parent.resolve() != self.root:
            raise StandardStoreError("standard path escapes store")

        return path

    def save_ingestion(
        self, manifest: StandardIngestionManifest,
        units: Iterable[StandardDocumentUnit], *, source_pdf: bytes | None = None,
        progress=None,
    ) -> StandardIngestionManifest:
        units = tuple(sorted(units, key=lambda item: item.source_id))

        if corpus_fingerprint(units) != manifest.corpus_manifest_sha256:
            raise StandardStoreError("manifest corpus hash does not match canonical units")

        directory = self.revision_dir(manifest.standard_id, manifest.revision, create=True)
        existing_path = directory / "manifest.json"

        if existing_path.exists():
            existing = self.load_manifest(manifest.standard_id, manifest.revision)

            if existing.source_pdf_sha256 != manifest.source_pdf_sha256:
                raise StandardCollisionError(
                    "revision label already exists with different source PDF")

            if existing.corpus_manifest_sha256 != manifest.corpus_manifest_sha256:
                raise StandardCollisionError(
                    "existing canonical corpus differs; refusing silent overwrite")

            self.verify_corpus(manifest.standard_id, manifest.revision)

            return existing

        corpus = directory / "corpus"
        indexes = directory / "indexes" / "lexical"
        source = directory / "source"

        for path in (corpus, indexes, source):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)

        for position, unit in enumerate(units, 1):
            if (unit.standard_id, unit.revision) != (manifest.standard_id, manifest.revision):
                raise StandardStoreError("unit belongs to a different standard binding")

            report(progress, "writing corpus", position, len(units))
            self._atomic_write(corpus / f"{unit.source_id}.json", canonical_json(unit.to_dict()))

        source_metadata = {
            "schema_version": 1,
            "logical_source_filename": manifest.logical_source_filename,
            "source_pdf_sha256": manifest.source_pdf_sha256,
            "page_count": manifest.page_count,
            "raw_pdf_retained": source_pdf is not None,
        }
        self._atomic_write(source / "metadata.json", canonical_json(source_metadata))

        if source_pdf is not None:
            self._atomic_write(source / "original.pdf", source_pdf)

        # The manifest is the commit marker and is always written last.

        self._atomic_write(existing_path, canonical_json(manifest.to_dict()))

        return manifest

    def reclassify_origin(self, standard_id: str, revision: str,
                          source_origin: str) -> str:
        """Change a stored corpus's classification. Returns the previous one.

        Re-ingesting an identical PDF returns the existing manifest untouched,
        which is right for the corpus and was wrong for this one field: an
        operator passing --origin PUBLIC over an already-ingested standard got
        no error, no change, and a corpus still marked licensed — so the index
        was built on this machine while the command said otherwise.

        Only the classification moves. It is not part of the corpus
        fingerprint or of any binding, so nothing else has to be rebuilt.
        """
        existing = self.load_manifest(standard_id, revision)

        if existing.source_origin == source_origin:
            return source_origin

        path = self.revision_dir(standard_id, revision) / "manifest.json"
        raw = dict(self._read_json(path))
        raw["source_origin"] = source_origin
        self._atomic_write(path, canonical_json(raw))

        return existing.source_origin

    def load_manifest(self, standard_id: str, revision: str) -> StandardIngestionManifest:
        path = self.revision_dir(standard_id, revision) / "manifest.json"
        return StandardIngestionManifest.from_dict(self._read_json(path))

    def load_units(self, standard_id: str, revision: str) -> tuple[StandardDocumentUnit, ...]:
        directory = self.revision_dir(standard_id, revision) / "corpus"

        if not directory.is_dir() or directory.is_symlink():
            raise StandardStoreError("canonical corpus is unavailable")

        units = []

        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file() or not _SOURCE_ID.fullmatch(path.stem):
                raise StandardStoreError("malformed canonical corpus entry")

            units.append(StandardDocumentUnit.from_dict(self._read_json(path)))

        return tuple(units)

    def resolve_source(
        self, standard_id: str, revision: str, source_id: str,
    ) -> StandardDocumentUnit:
        if not isinstance(source_id, str) or not _SOURCE_ID.fullmatch(source_id):
            raise StandardStoreError("invalid source_id")

        path = self.revision_dir(standard_id, revision) / "corpus" / f"{source_id}.json"

        if path.is_symlink():
            raise StandardStoreError("source path uses a symlink")

        try:
            unit = StandardDocumentUnit.from_dict(self._read_json(path))
        except FileNotFoundError as exc:
            raise KeyError(source_id) from exc

        if unit.source_id != source_id or (unit.standard_id, unit.revision) != (
            standard_id, revision
        ):
            raise StandardStoreError("source identity mismatch")

        return unit

    def verify_corpus(self, standard_id: str, revision: str) -> StandardIngestionManifest:
        manifest = self.load_manifest(standard_id, revision)
        units = self.load_units(standard_id, revision)

        if len(units) != manifest.canonical_unit_count:
            raise StandardStoreError("canonical unit count mismatch")

        if corpus_fingerprint(units) != manifest.corpus_manifest_sha256:
            raise StandardStoreError("canonical corpus integrity check failed")

        if any(unit.source_pdf_sha256 != manifest.source_pdf_sha256 for unit in units):
            raise StandardStoreError("canonical unit PDF identity mismatch")

        retained = self.revision_dir(standard_id, revision) / "source" / "original.pdf"

        if retained.exists():
            import hashlib

            if hashlib.sha256(retained.read_bytes()).hexdigest() != manifest.source_pdf_sha256:
                raise StandardStoreError("retained source PDF checksum mismatch")

        return manifest

    def save_index(
        self, manifest: StandardIndexManifest, index: Mapping[str, object],
    ) -> None:
        source = self.verify_corpus(manifest.standard_id, manifest.revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("index source identity mismatch")

        directory = self.revision_dir(
            manifest.standard_id, manifest.revision,
        ) / "indexes" / "lexical"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)

        if directory.is_symlink():
            raise StandardStoreError("index path uses a symlink")

        self._atomic_write(directory / "index.json", canonical_json(dict(index)))
        self._atomic_write(directory / "manifest.json", canonical_json(manifest.to_dict()))

    def load_index(
        self, standard_id: str, revision: str,
    ) -> tuple[StandardIndexManifest, Mapping[str, object]]:
        directory = self.revision_dir(standard_id, revision) / "indexes" / "lexical"

        if directory.is_symlink():
            raise StandardStoreError("index path uses a symlink")

        manifest = StandardIndexManifest.from_dict(self._read_json(directory / "manifest.json"))
        index = self._read_json(directory / "index.json")
        source = self.verify_corpus(standard_id, revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("lexical index is stale for canonical corpus")

        expected_fingerprint = sha256_json({
            "indexer_version": manifest.indexer_version,
            "source_corpus_sha256": manifest.source_corpus_sha256,
            "index": index,
        })

        if expected_fingerprint != manifest.index_fingerprint:
            raise StandardStoreError("lexical index fingerprint mismatch")

        if sha256_json(index.get("tokenizer_config")) != (
            manifest.tokenizer_config_fingerprint
        ):
            raise StandardStoreError("lexical tokenizer fingerprint mismatch")

        return manifest, index

    def save_vector_index(
        self, manifest: StandardVectorIndexManifest, index: Mapping[str, object],
    ) -> None:
        source = self.verify_corpus(manifest.standard_id, manifest.revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("vector index source identity mismatch")

        directory = self.revision_dir(
            manifest.standard_id, manifest.revision) / "indexes" / "vector"

        if directory.exists() and directory.is_symlink():
            raise StandardStoreError("vector index path uses a symlink")

        self._atomic_write(directory / "index.json", canonical_json(dict(index)))
        self._atomic_write(directory / "manifest.json", canonical_json(manifest.to_dict()))

    def load_vector_index(
        self, standard_id: str, revision: str,
    ) -> tuple[StandardVectorIndexManifest, Mapping[str, object]]:
        directory = self.revision_dir(standard_id, revision) / "indexes" / "vector"

        if directory.is_symlink():
            raise StandardStoreError("vector index path uses a symlink")

        manifest = StandardVectorIndexManifest.from_dict(
            self._read_json(directory / "manifest.json"))
        index = self._read_json(directory / "index.json")
        source = self.verify_corpus(standard_id, revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("vector index is stale for canonical corpus")

        expected = sha256_json({
            "vector_index_version": manifest.vector_index_version,
            "source_corpus_sha256": manifest.source_corpus_sha256,
            "index": index,
        })

        if expected != manifest.vector_index_fingerprint:
            raise StandardStoreError("vector index fingerprint mismatch")

        config = index.get("embedding_config")

        if not isinstance(config, Mapping) or sha256_json(config) != (
            manifest.embedding_config_fingerprint
        ):
            raise StandardStoreError("embedding configuration fingerprint mismatch")

        if (config.get("model_id") != manifest.embedding_model_id
                or config.get("model_revision") != manifest.embedding_model_revision
                or config.get("normalization") != manifest.normalization_policy
                or index.get("dimension") != manifest.embedding_dimension):
            raise StandardStoreError("vector manifest metadata mismatch")

        entries = index.get("entries")

        if not isinstance(entries, Mapping):
            raise StandardStoreError("vector entries are malformed")

        canonical_ids = {unit.source_id for unit in self.load_units(standard_id, revision)
                         if unit.retrievable}

        if set(entries) != canonical_ids or len(entries) != manifest.indexed_source_count:
            raise StandardStoreError("vector index has missing or stale source IDs")

        text_hashes = index.get("retrieval_text_sha256")

        if not isinstance(text_hashes, Mapping) or set(text_hashes) != canonical_ids:
            raise StandardStoreError("vector retrieval-text hashes are incomplete")

        for vector in entries.values():
            if (not isinstance(vector, list)
                    or len(vector) != manifest.embedding_dimension
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in vector)):
                raise StandardStoreError("vector dimension mismatch")

            norm = math.sqrt(sum(float(value) ** 2 for value in vector))

            if abs(norm - 1.0) > 1e-5:
                raise StandardStoreError("vector normalization mismatch")

        return manifest, index

    def save_cross_reference_index(
        self, manifest: StandardCrossReferenceIndexManifest,
        index: Mapping[str, object],
    ) -> None:
        source = self.verify_corpus(manifest.standard_id, manifest.revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("cross-reference index source identity mismatch")

        directory = self.revision_dir(
            manifest.standard_id, manifest.revision) / "indexes" / "crossrefs"

        if directory.exists() and directory.is_symlink():
            raise StandardStoreError("cross-reference index path uses a symlink")

        self._atomic_write(directory / "index.json", canonical_json(dict(index)))
        self._atomic_write(directory / "manifest.json", canonical_json(manifest.to_dict()))

    def load_cross_reference_index(
        self, standard_id: str, revision: str,
    ) -> tuple[StandardCrossReferenceIndexManifest, Mapping[str, object]]:
        directory = self.revision_dir(standard_id, revision) / "indexes" / "crossrefs"

        if directory.is_symlink():
            raise StandardStoreError("cross-reference index path uses a symlink")

        manifest = StandardCrossReferenceIndexManifest.from_dict(
            self._read_json(directory / "manifest.json"))
        index = self._read_json(directory / "index.json")
        source = self.verify_corpus(standard_id, revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("cross-reference index is stale for canonical corpus")

        expected = sha256_json({
            "resolver_version": manifest.resolver_version,
            "source_corpus_sha256": manifest.source_corpus_sha256,
            "index": index,
        })

        if expected != manifest.cross_reference_index_fingerprint:
            raise StandardStoreError("cross-reference index fingerprint mismatch")

        entries = index.get("entries")

        if not isinstance(entries, Mapping):
            raise StandardStoreError("cross-reference entries are malformed")

        counts = index.get("counts")

        if (not isinstance(counts, Mapping)
                or counts.get("resolved") != manifest.resolved_count
                or counts.get("ambiguous") != manifest.ambiguous_count
                or counts.get("unresolved") != manifest.unresolved_count):
            raise StandardStoreError("cross-reference manifest metadata mismatch")

        canonical_ids = {unit.source_id for unit in self.load_units(standard_id, revision)}

        if not set(entries).issubset(canonical_ids):
            raise StandardStoreError("cross-reference index has a stale source ID")

        for relations in entries.values():
            if not isinstance(relations, list):
                raise StandardStoreError("cross-reference relation is malformed")

            for relation in relations:
                targets = relation.get("target_source_ids", []) if isinstance(
                    relation, Mapping) else []

                if not isinstance(targets, list) or not set(targets).issubset(canonical_ids):
                    raise StandardStoreError("cross-reference target is stale")

        return manifest, index

    def binding(self, standard_id: str, revision: str, *, bound_at: str | None = None) -> StandardBinding:
        """The identity a session binds to: the corpus and every index over it."""

        manifest = self.verify_corpus(standard_id, revision)
        index_manifest, _ = self.load_index(standard_id, revision)
        vector_fingerprint = crossref_fingerprint = None

        try:
            vector_fingerprint = self.load_vector_index(
                standard_id, revision)[0].vector_index_fingerprint
        except (FileNotFoundError, StandardStoreError):
            pass

        try:
            crossref_fingerprint = self.load_cross_reference_index(
                standard_id, revision)[0].cross_reference_index_fingerprint
        except (FileNotFoundError, StandardStoreError):
            pass

        # One fingerprint over every retrieval input. A session compares this
        # to detect that retrieval would now answer differently -- which the
        # lexical fingerprint alone would miss when only vectors were rebuilt.

        retrieval_fingerprint = sha256_json({
            "lexical": index_manifest.index_fingerprint,
            "vector": vector_fingerprint,
            "crossrefs": crossref_fingerprint,
            "configuration": DEFAULT_STANDARD_RETRIEVAL_CONFIGURATION,
        })

        return StandardBinding(
            standard_id, revision, manifest.source_pdf_sha256,
            manifest.corpus_manifest_sha256, index_manifest.index_fingerprint,
            manifest.extractor_version, manifest.corpus_schema_version, bound_at,
            manifest.source_origin, vector_fingerprint, crossref_fingerprint,
            retrieval_fingerprint,
        )

    # ------------------------------------------------------------------
    # candidate corpora, generations and promotion
    # ------------------------------------------------------------------

    def _confined(self, standard_id: str, revision: str, *parts: str,
                  create: bool = False) -> Path:
        """Resolve a path under one revision, refusing symlinks and escapes."""

        base = self.revision_dir(standard_id, revision, create=create)
        path = base

        for part in parts:
            path = path / self._component(part, "path component")

        for probe in (base, *path.parents):
            if probe == base.parent:
                break

            if probe.exists() and probe.is_symlink():
                raise StandardStoreError("standard path uses a symlink")

        if path.exists() and path.is_symlink():
            raise StandardStoreError("standard path uses a symlink")

        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)

        resolved = path.resolve() if path.exists() else (
            base.resolve() / Path(*parts))

        if not str(resolved).startswith(str(base.resolve())):
            raise StandardStoreError("standard path escapes its revision")

        return path

    def save_candidate(
        self, manifest: StandardIngestionManifest,
        units: Iterable[StandardDocumentUnit], *,
        diagnostics: Mapping[str, object], layout: bytes | None = None,
    ) -> str:
        """Store a re-extraction beside the active corpus without activating it."""

        # Sorted before hashing, so the fingerprint depends on the units and
        # not on the order they happened to be extracted in.

        units = tuple(sorted(units, key=lambda item: item.source_id))

        if corpus_fingerprint(units) != manifest.corpus_manifest_sha256:
            raise StandardStoreError("candidate corpus hash does not match its units")

        active = self.load_manifest(manifest.standard_id, manifest.revision)

        if active.source_pdf_sha256 != manifest.source_pdf_sha256:
            raise StandardCollisionError(
                "candidate was extracted from a different source PDF")

        candidate_id = candidate_id_for(manifest.corpus_manifest_sha256)
        directory = self._confined(manifest.standard_id, manifest.revision,
                                   "candidates", candidate_id, create=True)
        corpus = directory / "corpus"
        corpus.mkdir(parents=True, exist_ok=True, mode=0o700)

        for stale in corpus.iterdir():
            stale.unlink()

        for unit in units:
            if (unit.standard_id, unit.revision) != (manifest.standard_id,
                                                     manifest.revision):
                raise StandardStoreError("unit belongs to a different standard binding")

            self._atomic_write(corpus / f"{unit.source_id}.json",
                               canonical_json(unit.to_dict()))

        self._atomic_write(directory / "diagnostics.json",
                           canonical_json(dict(diagnostics)))

        if layout is not None:
            self._atomic_write(directory / "layout.json", layout)

        # The manifest is the commit marker and is always written last.

        self._atomic_write(directory / "manifest.json",
                           canonical_json(manifest.to_dict()))

        return candidate_id

    def list_candidates(self, standard_id: str, revision: str) -> tuple[str, ...]:
        directory = self.revision_dir(standard_id, revision) / "candidates"

        if not directory.is_dir() or directory.is_symlink():
            return ()

        return tuple(sorted(
            entry.name for entry in directory.iterdir()
            if entry.is_dir() and not entry.is_symlink()
            and _CANDIDATE_ID.fullmatch(entry.name)
            and (entry / "manifest.json").is_file()))

    def _candidate_dir(self, standard_id: str, revision: str, candidate_id: str) -> Path:
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise StandardStoreError("invalid candidate id")

        return self._confined(standard_id, revision, "candidates", candidate_id)

    def load_candidate(
        self, standard_id: str, revision: str, candidate_id: str,
    ) -> tuple[StandardIngestionManifest, tuple[StandardDocumentUnit, ...]]:
        directory = self._candidate_dir(standard_id, revision, candidate_id)

        try:
            manifest = StandardIngestionManifest.from_dict(
                self._read_json(directory / "manifest.json"))
        except FileNotFoundError as exc:
            raise StandardStoreError(f"unknown candidate {candidate_id}") from exc

        corpus = directory / "corpus"

        if not corpus.is_dir() or corpus.is_symlink():
            raise StandardStoreError("candidate corpus is unavailable")

        units = []

        for path in sorted(corpus.iterdir()):
            if path.is_symlink() or not path.is_file() or not _SOURCE_ID.fullmatch(path.stem):
                raise StandardStoreError("malformed candidate corpus entry")

            units.append(StandardDocumentUnit.from_dict(self._read_json(path)))

        units = tuple(units)

        if (len(units) != manifest.canonical_unit_count
                or corpus_fingerprint(units) != manifest.corpus_manifest_sha256):
            raise StandardStoreError("candidate corpus integrity check failed")

        if candidate_id_for(manifest.corpus_manifest_sha256) != candidate_id:
            raise StandardStoreError("candidate id does not match its corpus")

        if any(unit.source_pdf_sha256 != manifest.source_pdf_sha256 for unit in units):
            raise StandardStoreError("candidate unit PDF identity mismatch")

        return manifest, units

    def load_candidate_diagnostics(
        self, standard_id: str, revision: str, candidate_id: str,
    ) -> Mapping[str, object]:
        directory = self._candidate_dir(standard_id, revision, candidate_id)
        return self._read_json(directory / "diagnostics.json")

    def list_generations(self, standard_id: str, revision: str) -> tuple[str, ...]:
        directory = self.revision_dir(standard_id, revision) / "generations"

        if not directory.is_dir() or directory.is_symlink():
            return ()

        return tuple(sorted(
            entry.name for entry in directory.iterdir()
            if entry.is_dir() and not entry.is_symlink()
            and _GENERATION_ID.fullmatch(entry.name)))

    def index_state(self, standard_id: str, revision: str) -> str:
        """Derive index freshness from what the indexes claim to be built on.

        A stored flag would go stale the moment an index was rebuilt; comparing
        each index manifest to the active corpus cannot.
        """

        try:
            corpus = self.load_manifest(
                standard_id, revision).corpus_manifest_sha256
        except (FileNotFoundError, StandardStoreError):
            return INDEX_STATE_REBUILD_REQUIRED

        indexes = self.revision_dir(standard_id, revision) / "indexes"

        if not (indexes / "lexical" / "manifest.json").is_file():
            return INDEX_STATE_REBUILD_REQUIRED

        for name in ("lexical", "vector", "crossrefs"):
            path = indexes / name / "manifest.json"

            if not path.is_file() or path.is_symlink():
                continue

            if self._read_json(path).get("source_corpus_sha256") != corpus:
                return INDEX_STATE_REBUILD_REQUIRED

        return INDEX_STATE_READY

    def _invalidate_indexes(self, standard_id: str, revision: str,
                            reason: str, corpus_sha: str) -> None:
        """Derived indexes never survive a canonical corpus replacement.

        The embedding cache is keyed by model and retrieval text, so it stays:
        it lets an unchanged unit be re-indexed without being re-embedded.
        """

        indexes = self.revision_dir(standard_id, revision) / "indexes"

        for name in ("lexical", "vector", "crossrefs"):
            directory = indexes / name

            if directory.is_dir() and not directory.is_symlink():
                for entry in directory.iterdir():
                    if entry.is_file() and not entry.is_symlink():
                        entry.unlink()

        indexes.mkdir(parents=True, exist_ok=True, mode=0o700)

        # An audit record of why the indexes went away; freshness itself is
        # derived from the index manifests by index_state().

        self._atomic_write(indexes / "status.json", canonical_json({
            "schema_version": 1, "last_invalidation_reason": reason,
            "invalidated_for_corpus_sha256": corpus_sha,
        }))

    def promote_candidate(
        self, standard_id: str, revision: str, candidate_id: str,
    ) -> StandardIngestionManifest:
        """Replace the active corpus, keeping the previous one recoverable."""

        # The three checks below establish that this candidate is a re-extraction
        # OF THIS corpus and actually differs from it; anything else would either
        # cross standards or replace a corpus with itself.

        candidate, units = self.load_candidate(standard_id, revision, candidate_id)
        active = self.verify_corpus(standard_id, revision)

        if (candidate.standard_id, candidate.revision) != (standard_id, revision):
            raise StandardStoreError("candidate belongs to a different standard binding")

        if candidate.source_pdf_sha256 != active.source_pdf_sha256:
            raise StandardCollisionError(
                "candidate was extracted from a different source PDF")

        if candidate.corpus_manifest_sha256 == active.corpus_manifest_sha256:
            raise StandardStoreError("candidate corpus is identical to the active corpus")

        base = self.revision_dir(standard_id, revision)
        generation = self._confined(
            standard_id, revision, "generations",
            generation_id_for(active.corpus_manifest_sha256), create=True)

        if (generation / "corpus").exists():
            raise StandardStoreError("generation directory is already populated")

        self._atomic_write(generation / "manifest.json",
                           canonical_json(active.to_dict()))
        candidate_dir = self._candidate_dir(standard_id, revision, candidate_id)

        # Archive first: if anything below fails the corpus is missing and the
        # store fails closed, with the previous generation still recoverable.

        os.rename(base / "corpus", generation / "corpus")
        os.rename(candidate_dir / "corpus", base / "corpus")

        for name in ("diagnostics.json", "layout.json"):
            source = candidate_dir / name

            if source.is_file() and not source.is_symlink():
                os.replace(source, base / name)

        # A review belongs to the corpus it was written against; keep it with
        # that generation rather than leaving it to be overwritten.

        review = base / "evaluation" / "std1e-operator-review.json"

        if review.is_file() and not review.is_symlink():
            os.replace(review, generation / "operator-review.json")

        self._atomic_write(base / "manifest.json", canonical_json(candidate.to_dict()))
        self._invalidate_indexes(standard_id, revision, "canonical corpus promoted",
                                 candidate.corpus_manifest_sha256)

        for entry in sorted(candidate_dir.iterdir(), reverse=True):
            if entry.is_file() and not entry.is_symlink():
                entry.unlink()

        candidate_dir.rmdir()

        return self.verify_corpus(standard_id, revision)

    def record_human_validation(
        self, standard_id: str, revision: str, status: "HumanValidationStatus", *,
        reviewer: str, reviewed_at: str, evidence: Mapping[str, object],
    ) -> StandardIngestionManifest:
        """Record an operator's verdict on the corpus, leaving its identity alone."""

        # Only the validation status changes: the corpus hash and extractor
        # version identify the bytes, and approving them must not restate them.

        manifest = self.verify_corpus(standard_id, revision)
        updated = replace(manifest, human_validation_status=status)
        directory = self.revision_dir(standard_id, revision)
        self._atomic_write(directory / "manifest.json",
                           canonical_json(updated.to_dict()))
        self._atomic_write(directory / "evaluation" / "approval.json", canonical_json({
            "schema_version": 1,
            "standard_id": standard_id, "revision": revision,
            "human_validation_status": status.value,
            "corpus_manifest_sha256": manifest.corpus_manifest_sha256,
            "source_pdf_sha256": manifest.source_pdf_sha256,
            "extractor_version": manifest.extractor_version,
            "reviewer": reviewer, "reviewed_at": reviewed_at,
            "evidence": dict(evidence),
        }))

        return updated

    def list_standards(self) -> tuple[tuple[str, str], ...]:
        """Every ingested (standard, revision), skipping anything malformed."""

        result = []

        for standard in sorted(self.root.iterdir()):
            if standard.name.startswith(".") or not standard.is_dir() or standard.is_symlink():
                continue

            if not _COMPONENT.fullmatch(standard.name):
                continue

            for revision in sorted(standard.iterdir()):
                if (revision.is_dir() and not revision.is_symlink()
                        and (revision / "manifest.json").is_file()):
                    result.append((standard.name, revision.name))

        return tuple(result)

    def save_retrieval_settings(self, standard_id: str, revision: str, *,
                                mode: str, evidence_completion: int) -> None:
        """How this document is searched, stored with the document.

        Retrieval was tuned per deployment, through the environment, while the
        thing it was measured on is one document: a second standard bound on
        the same machine inherited the first one's settings. Kept out of every
        fingerprint on purpose -- the indexes are the same whichever way they
        are searched.
        """
        if mode not in RETRIEVAL_MODES:
            raise StandardStoreError(
                f"retrieval mode must be one of {', '.join(RETRIEVAL_MODES)}")

        if (not isinstance(evidence_completion, int)
                or isinstance(evidence_completion, bool)
                or evidence_completion < 0):
            raise StandardStoreError("evidence completion must be a whole number >= 0")

        self.load_manifest(standard_id, revision)
        path = self.revision_dir(standard_id, revision) / RETRIEVAL_SETTINGS_FILE
        self._atomic_write(path, canonical_json(
            {"mode": mode, "evidence_completion": evidence_completion}))

    def load_retrieval_settings(self, standard_id: str,
                                revision: str) -> Mapping[str, object]:
        """What save_retrieval_settings recorded, or {} when nothing was.

        A malformed file is an error, not an empty answer: silently searching
        a validated document some other way is what this file exists to stop.
        """
        path = self.revision_dir(standard_id, revision) / RETRIEVAL_SETTINGS_FILE

        if not path.exists() and not path.is_symlink():
            return {}

        raw = self._read_json(path)
        mode = raw.get("mode")
        completion = raw.get("evidence_completion")

        if (mode not in RETRIEVAL_MODES or not isinstance(completion, int)
                or isinstance(completion, bool) or completion < 0):
            raise StandardStoreError(f"malformed {RETRIEVAL_SETTINGS_FILE}")

        return {"mode": mode, "evidence_completion": completion}

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, object]:
        if path.is_symlink():
            raise StandardStoreError("standard store entry uses a symlink")

        raw = json.loads(path.read_text("utf-8"))

        if not isinstance(raw, Mapping):
            raise StandardStoreError("standard store JSON must be an object")

        return raw

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        if path.exists() and path.is_symlink():
            raise StandardStoreError("refusing to replace a symlink")

        fd, temporary = tempfile.mkstemp(prefix=".standard-", dir=path.parent)

        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)

            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
