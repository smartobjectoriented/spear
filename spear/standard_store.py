"""Private filesystem store for authoritative standard corpora and derived indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
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

#: The vector index's matrix, beside its index.json: float32, one row per
#: source id in index["ids"] order. Read by mmap, never parsed.
VECTORS_FILE = "vectors.npy"
VECTOR_FORMAT = "float32-npy-v1"

#: Per revision: what was last verified in full, and the on-disk stamp of
#: the files it was verified from. See StandardStore._verified.
VERIFICATION_FILE = "verified.json"

#: How long a tree stamp is reused within one process. A search reads the
#: corpus stamp several times in a row; restat'ing 300 000 files for each is
#: seconds per query for nothing.
_TREE_STAMP_TTL = 2.0
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

        # What this process has already read, keyed by what it was read from:
        # (standard, revision, artefact) -> (stamp, value).
        self._memo: dict[tuple[str, str, str], tuple[object, object]] = {}
        self._tree_stamps: dict[Path, tuple[float, list | None, str]] = {}

    # ------------------------------------------------------------------
    # verification stamps
    #
    # Checking a document in full -- rehashing every unit, recomputing each
    # index's fingerprint over its whole content -- was done on EVERY binding
    # check and every search. For a document of a few hundred pages that was
    # a fraction of a second. For the Arm A-profile manual, 298 709 units and
    # a 5.8 GB vector index, it was five minutes, at startup and per query.
    #
    # A full check now records the stamp of the files it read: size, mtime,
    # ctime and inode of each. Until one of them changes on disk the check is
    # not repeated, the way git trusts its index. Any write -- in place or by
    # atomic rename -- moves ctime or the inode, so a changed file is always
    # checked again. `/standard verify` forces a full check regardless.
    # ------------------------------------------------------------------

    @staticmethod
    def _file_stamp(path: Path) -> list[int] | None:
        try:
            st = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return None

        return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]

    def _tree_stamp(self, directory: Path) -> str:
        """One digest over the stamp of every entry of `directory`."""
        now = time.monotonic()
        # The directory's own stat moves whenever an entry is added, removed
        # or renamed over -- which is how the store writes -- and when the
        # directory itself is moved away. Only an in-place edit of an entry
        # leaves it alone, and that is what the short reuse window accepts.
        own = self._file_stamp(directory)
        cached = self._tree_stamps.get(directory)

        if (cached is not None and now - cached[0] < _TREE_STAMP_TTL
                and cached[1] == own):
            return cached[2]

        entries = []

        try:
            with os.scandir(directory) as found:
                for entry in found:
                    st = entry.stat(follow_symlinks=False)
                    entries.append(f"{entry.name} {st.st_size} {st.st_mtime_ns} "
                                   f"{st.st_ctime_ns} {st.st_ino}")
        except FileNotFoundError:
            entries = ["<absent>"]

        entries.sort()
        digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
        self._tree_stamps[directory] = (now, own, digest)

        return digest

    def _corpus_stamp(self, standard_id: str, revision: str) -> list:
        directory = self.revision_dir(standard_id, revision)

        return [self._file_stamp(directory / "manifest.json"),
                self._tree_stamp(directory / "corpus"),
                self._file_stamp(directory / "source" / "original.pdf")]

    def _index_stamp(self, standard_id: str, revision: str, kind: str,
                     corpus: list | None = None) -> list:
        directory = self.revision_dir(standard_id, revision) / "indexes" / kind
        stamp = [corpus or self._corpus_stamp(standard_id, revision),
                 self._file_stamp(directory / "manifest.json"),
                 self._file_stamp(directory / "index.json")]

        if kind == "vector":
            stamp.append(self._file_stamp(directory / VECTORS_FILE))

        return stamp

    def _verified(self, standard_id: str, revision: str, key: str, stamp) -> bool:
        path = self.revision_dir(standard_id, revision) / VERIFICATION_FILE

        try:
            return json.loads(path.read_text("utf-8")).get(key) == stamp
        except (OSError, ValueError, AttributeError):
            return False

    def _record_verified(self, standard_id: str, revision: str, key: str,
                         stamp) -> None:
        """Remember a full check -- only if nothing moved while it ran."""
        self._tree_stamps.clear()
        now = (self._corpus_stamp(standard_id, revision) if key == "corpus"
               else self._index_stamp(standard_id, revision, key))

        if now != stamp:
            return

        path = self.revision_dir(standard_id, revision) / VERIFICATION_FILE

        try:
            record = json.loads(path.read_text("utf-8"))
            record = record if isinstance(record, dict) else {}
        except (OSError, ValueError):
            record = {}

        record[key] = stamp

        try:
            self._atomic_write(path, canonical_json(record))
        except OSError:
            pass                    # a read-only store simply checks again

    def _record_written(self, standard_id: str, revision: str, key: str) -> None:
        """What the store has just written itself is as checked as it gets:
        its fingerprint was computed from the very content written."""
        self._tree_stamps.clear()
        stamp = (self._corpus_stamp(standard_id, revision) if key == "corpus"
                 else self._index_stamp(standard_id, revision, key))
        self._record_verified(standard_id, revision, key, stamp)

    def _forget(self, standard_id: str, revision: str) -> None:
        self._tree_stamps.clear()

        for key in [key for key in self._memo if key[:2] == (standard_id, revision)]:
            del self._memo[key]

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
        self._record_written(manifest.standard_id, manifest.revision, "corpus")

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
        """The canonical units, read once per process while the corpus is unchanged."""
        stamp = self._corpus_stamp(standard_id, revision)
        memo = self._memo.get((standard_id, revision, "units"))

        if memo is not None and memo[0] == stamp:
            return memo[1]

        units = self._read_units(standard_id, revision)
        self._memo[(standard_id, revision, "units")] = (stamp, units)

        return units

    def _read_units(self, standard_id: str, revision: str) -> tuple[StandardDocumentUnit, ...]:
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

    def verify_corpus(self, standard_id: str, revision: str, *,
                      force: bool = False) -> StandardIngestionManifest:
        stamp = self._corpus_stamp(standard_id, revision)

        if not force:
            memo = self._memo.get((standard_id, revision, "corpus"))

            if memo is not None and memo[0] == stamp:
                return memo[1]

            if self._verified(standard_id, revision, "corpus", stamp):
                manifest = self.load_manifest(standard_id, revision)
                self._memo[(standard_id, revision, "corpus")] = (stamp, manifest)

                return manifest

        manifest = self.load_manifest(standard_id, revision)
        units = self._read_units(standard_id, revision)

        if len(units) != manifest.canonical_unit_count:
            raise StandardStoreError("canonical unit count mismatch")

        if corpus_fingerprint(units) != manifest.corpus_manifest_sha256:
            raise StandardStoreError("canonical corpus integrity check failed")

        if any(unit.source_pdf_sha256 != manifest.source_pdf_sha256 for unit in units):
            raise StandardStoreError("canonical unit PDF identity mismatch")

        retained = self.revision_dir(standard_id, revision) / "source" / "original.pdf"

        if retained.exists():
            if hashlib.sha256(retained.read_bytes()).hexdigest() != manifest.source_pdf_sha256:
                raise StandardStoreError("retained source PDF checksum mismatch")

        self._record_verified(standard_id, revision, "corpus", stamp)
        self._memo[(standard_id, revision, "corpus")] = (stamp, manifest)
        self._memo[(standard_id, revision, "units")] = (stamp, units)

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
        self._record_written(manifest.standard_id, manifest.revision, "lexical")

    def load_index(
        self, standard_id: str, revision: str, *, force: bool = False,
    ) -> tuple[StandardIndexManifest, Mapping[str, object]]:
        directory = self.revision_dir(standard_id, revision) / "indexes" / "lexical"

        if directory.is_symlink():
            raise StandardStoreError("index path uses a symlink")

        stamp = self._index_stamp(standard_id, revision, "lexical")
        memo = self._memo.get((standard_id, revision, "lexical"))

        if not force and memo is not None and memo[0] == stamp:
            return memo[1]

        manifest = StandardIndexManifest.from_dict(self._read_json(directory / "manifest.json"))
        index = self._read_json(directory / "index.json")
        source = self.verify_corpus(standard_id, revision, force=force)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("lexical index is stale for canonical corpus")

        if not force and self._verified(standard_id, revision, "lexical", stamp):
            self._memo[(standard_id, revision, "lexical")] = (stamp, (manifest, index))

            return manifest, index

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

        self._record_verified(standard_id, revision, "lexical", stamp)
        self._memo[(standard_id, revision, "lexical")] = (stamp, (manifest, index))

        return manifest, index

    def save_vector_index(
        self, manifest: StandardVectorIndexManifest, index: Mapping[str, object],
        vectors: bytes,
    ) -> None:
        """`vectors` is the .npy file, whose sha256 index["vectors"] names."""
        source = self.verify_corpus(manifest.standard_id, manifest.revision)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("vector index source identity mismatch")

        directory = self.revision_dir(
            manifest.standard_id, manifest.revision) / "indexes" / "vector"

        if directory.exists() and directory.is_symlink():
            raise StandardStoreError("vector index path uses a symlink")

        if hashlib.sha256(vectors).hexdigest() != (index.get("vectors") or {}).get("sha256"):
            raise StandardStoreError("vector data does not match its index")

        # Matrix first, manifest last: the manifest is what makes it an index.
        self._atomic_write(directory / VECTORS_FILE, vectors)
        self._atomic_write(directory / "index.json", canonical_json(dict(index)))
        self._atomic_write(directory / "manifest.json", canonical_json(manifest.to_dict()))
        self._record_written(manifest.standard_id, manifest.revision, "vector")

    def load_vector_index(
        self, standard_id: str, revision: str, *, force: bool = False,
    ) -> tuple[StandardVectorIndexManifest, Mapping[str, object]]:
        """The manifest and index; index["matrix"] is the vectors, memory-mapped.

        Row i of the matrix is the vector of index["ids"][i]. The matrix is
        read by mmap and shared by every search of the process: parsed as
        JSON it was 5.8 GB and four minutes for a 300 000-unit document.
        """
        import numpy

        directory = self.revision_dir(standard_id, revision) / "indexes" / "vector"

        if directory.is_symlink():
            raise StandardStoreError("vector index path uses a symlink")

        stamp = self._index_stamp(standard_id, revision, "vector")
        memo = self._memo.get((standard_id, revision, "vector"))

        if not force and memo is not None and memo[0] == stamp:
            return memo[1]

        manifest = StandardVectorIndexManifest.from_dict(
            self._read_json(directory / "manifest.json"))

        if manifest.vector_index_format != VECTOR_FORMAT:
            raise StandardStoreError(
                f"vector index is in the retired {manifest.vector_index_format} "
                f"format; rebuild it: /standard rebuild {standard_id} {revision}")

        index = dict(self._read_json(directory / "index.json"))
        source = self.verify_corpus(standard_id, revision, force=force)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("vector index is stale for canonical corpus")

        stored = index.get("vectors")

        if not isinstance(stored, Mapping) or stored.get("file") != VECTORS_FILE:
            raise StandardStoreError("vector entries are malformed")

        path = directory / VECTORS_FILE

        if path.is_symlink():
            raise StandardStoreError("vector index path uses a symlink")

        try:
            matrix = numpy.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as exc:
            raise StandardStoreError(f"vector data is unreadable: {exc}") from exc

        if force or not self._verified(standard_id, revision, "vector", stamp):
            self._check_vector_index(standard_id, revision, manifest, index,
                                     matrix, path)
            self._record_verified(standard_id, revision, "vector", stamp)

        index["matrix"] = matrix
        self._memo[(standard_id, revision, "vector")] = (stamp, (manifest, index))

        return manifest, index

    def _check_vector_index(self, standard_id, revision, manifest, index,
                            matrix, path) -> None:
        import numpy

        expected = sha256_json({
            "vector_index_version": manifest.vector_index_version,
            "source_corpus_sha256": manifest.source_corpus_sha256,
            "index": index,
        })

        if expected != manifest.vector_index_fingerprint:
            raise StandardStoreError("vector index fingerprint mismatch")

        digest = hashlib.sha256()

        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)

        # The fingerprint covers index.json, which names the matrix's hash;
        # this is what ties the fingerprint to the vectors themselves.
        if digest.hexdigest() != index["vectors"].get("sha256"):
            raise StandardStoreError("vector data checksum mismatch")

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

        ids = index.get("ids")

        if not isinstance(ids, list) or ids != sorted(set(ids)):
            raise StandardStoreError("vector entries are malformed")

        canonical_ids = {unit.source_id for unit in self.load_units(standard_id, revision)
                         if unit.retrievable}

        if set(ids) != canonical_ids or len(ids) != manifest.indexed_source_count:
            raise StandardStoreError("vector index has missing or stale source IDs")

        text_hashes = index.get("retrieval_text_sha256")

        if not isinstance(text_hashes, Mapping) or set(text_hashes) != canonical_ids:
            raise StandardStoreError("vector retrieval-text hashes are incomplete")

        if (matrix.dtype != numpy.dtype("<f4") or matrix.ndim != 2
                or matrix.shape != (len(ids), manifest.embedding_dimension)
                or list(index["vectors"].get("shape", ())) != list(matrix.shape)):
            raise StandardStoreError("vector dimension mismatch")

        # In blocks: a norm over the whole matrix would allocate its size again.
        for start in range(0, matrix.shape[0], 65536):
            block = numpy.asarray(matrix[start:start + 65536], dtype=numpy.float64)

            if not numpy.isfinite(block).all():
                raise StandardStoreError("vector dimension mismatch")

            if (numpy.abs(numpy.linalg.norm(block, axis=1) - 1.0) > 1e-5).any():
                raise StandardStoreError("vector normalization mismatch")

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
        self._record_written(manifest.standard_id, manifest.revision, "crossrefs")

    def load_cross_reference_index(
        self, standard_id: str, revision: str, *, force: bool = False,
    ) -> tuple[StandardCrossReferenceIndexManifest, Mapping[str, object]]:
        directory = self.revision_dir(standard_id, revision) / "indexes" / "crossrefs"

        if directory.is_symlink():
            raise StandardStoreError("cross-reference index path uses a symlink")

        stamp = self._index_stamp(standard_id, revision, "crossrefs")
        memo = self._memo.get((standard_id, revision, "crossrefs"))

        if not force and memo is not None and memo[0] == stamp:
            return memo[1]

        manifest = StandardCrossReferenceIndexManifest.from_dict(
            self._read_json(directory / "manifest.json"))
        index = self._read_json(directory / "index.json")
        source = self.verify_corpus(standard_id, revision, force=force)

        if (manifest.source_pdf_sha256 != source.source_pdf_sha256
                or manifest.source_corpus_sha256 != source.corpus_manifest_sha256):
            raise StandardStoreError("cross-reference index is stale for canonical corpus")

        if not force and self._verified(standard_id, revision, "crossrefs", stamp):
            self._memo[(standard_id, revision, "crossrefs")] = (stamp, (manifest, index))

            return manifest, index

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

        self._record_verified(standard_id, revision, "crossrefs", stamp)
        self._memo[(standard_id, revision, "crossrefs")] = (stamp, (manifest, index))

        return manifest, index

    def binding(self, standard_id: str, revision: str, *, bound_at: str | None = None) -> StandardBinding:
        """The identity a session binds to: the corpus and every index over it."""

        manifest = self.verify_corpus(standard_id, revision)
        index_manifest = self._checked_index_manifest(
            standard_id, revision, "lexical", manifest)
        vector_fingerprint = crossref_fingerprint = None

        try:
            vector_fingerprint = self._checked_index_manifest(
                standard_id, revision, "vector", manifest).vector_index_fingerprint
        except (FileNotFoundError, StandardStoreError):
            pass

        try:
            crossref_fingerprint = self._checked_index_manifest(
                standard_id, revision, "crossrefs",
                manifest).cross_reference_index_fingerprint
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

    def _checked_index_manifest(self, standard_id: str, revision: str,
                                kind: str, corpus: StandardIngestionManifest):
        """An index's manifest, trusted only as far as it has been checked.

        A binding needs each index's fingerprint, not its content. Once the
        index has been checked in full and nothing under it has moved, its
        manifest says the same thing the whole file would; otherwise the full
        check runs, and records itself for next time.
        """
        stamp = self._index_stamp(standard_id, revision, kind)

        if not self._verified(standard_id, revision, kind, stamp):
            loader = {"lexical": self.load_index, "vector": self.load_vector_index,
                      "crossrefs": self.load_cross_reference_index}[kind]

            return loader(standard_id, revision)[0]

        cls = {"lexical": StandardIndexManifest, "vector": StandardVectorIndexManifest,
               "crossrefs": StandardCrossReferenceIndexManifest}[kind]
        directory = self.revision_dir(standard_id, revision) / "indexes" / kind

        if directory.is_symlink():
            raise StandardStoreError(f"{kind} index path uses a symlink")

        manifest = cls.from_dict(self._read_json(directory / "manifest.json"))

        if (manifest.source_pdf_sha256 != corpus.source_pdf_sha256
                or manifest.source_corpus_sha256 != corpus.corpus_manifest_sha256):
            raise StandardStoreError(f"{kind} index is stale for canonical corpus")

        return manifest

    def verify_everything(self, standard_id: str, revision: str) -> dict[str, str]:
        """Check the corpus and every index in full, whatever was recorded."""
        results = {}

        for name, check in (
                ("corpus", lambda: self.verify_corpus(standard_id, revision, force=True)),
                ("lexical", lambda: self.load_index(standard_id, revision, force=True)),
                ("vector", lambda: self.load_vector_index(standard_id, revision, force=True)),
                ("crossrefs", lambda: self.load_cross_reference_index(
                    standard_id, revision, force=True))):
            try:
                check()
                results[name] = "ok"
            except FileNotFoundError:
                results[name] = "absent"
            except StandardStoreError as exc:
                results[name] = f"FAILED: {exc}"

        return results

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

        # What the document IS does not change with how it was extracted: the
        # promoted corpus keeps the active one's classification. Not part of
        # the corpus fingerprint, so nothing else moves with it.
        candidate = replace(candidate, source_origin=active.source_origin)
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
