"""Confined, rebuildable vector index for canonical standard source units."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from standard_schema import (
    StandardVectorIndexManifest, canonical_json, sha256_json,
)
from standard_progress import report
from standard_store import (
    VECTOR_FORMAT, VECTORS_FILE, StandardStore, StandardStoreError,
)


#: 2: the vectors moved out of index.json into a float32 .npy matrix.
VECTOR_INDEX_VERSION = 2
EMBEDDING_CONFIG_VERSION = "standard-retrieval-text-v1"


class LocalStandardEmbedder(Protocol):
    model_id: str
    model_revision: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def structural_context(unit) -> tuple[str, ...]:
    """What the DOCUMENT prints around this unit, where the store recorded it.

    For a table row: the table's caption and its column headers. A row reads
    "20 ReqV Request Validation Acknowledge packet Set to 1: ..." and nothing
    in those words says it is a bit of the Control packet CAM field -- the
    caption above it says that, and the column header "Bit#" says what the 20
    is. Measured on one bound standard, four bit-level questions found their
    row nowhere in the vector top twenty without it.

    Structural only, and never a neighbouring provision: a caption and a
    column header carry no normative force of their own, so nothing here can
    lend one provision the modality of the one beside it. The evidence
    returned to a caller is unchanged; this is what the row is FOUND by.

    A unit whose store never recorded a grid contributes nothing, so a corpus
    ingested under an earlier schema indexes exactly as it did before.
    """
    structure = getattr(unit, "table_structure", None)

    if structure is None:
        return ()

    return tuple(part for part in (structure.caption, *structure.columns) if part)


def retrieval_text(unit) -> str:
    """Visible semantic metadata only; identity and storage metadata are excluded."""

    return "\n".join(part for part in (
        f"Section {unit.section}" if unit.section else "",
        " > ".join(unit.heading_path),
        " ".join(structural_context(unit)),
        unit.text,
    ) if part)


@dataclass
class SentenceTransformerStandardEmbedder:
    """Strictly local adapter around SPEAR's configured embedding models."""

    model_id: str
    model_revision: str

    def __post_init__(self) -> None:
        import embedding

        # A standard's vector index is part of its identity, so the model must
        # be named outright: the default could change under it.

        if self.model_id == embedding.DEFAULT:
            raise ValueError("standards require an explicit local SentenceTransformer model")

        model_path = Path(self.model_id).expanduser()

        if self.model_id not in embedding.MODELS and not model_path.exists():
            raise ValueError("unsupported standard embedding model ID")

        # Refused up front rather than discovered as a download attempt during
        # a rebuild: indexing a licensed standard stays entirely local.

        if not model_path.exists() and not embedding._is_cached(self.model_id):
            raise RuntimeError("standard embedding model is not present in the local cache")

        self._document_prefix, self._query_prefix, self._trust = embedding.MODELS.get(
            self.model_id, ("", "", False))
        self._model = None

    def _load(self):
        """Load the model on first use, offline and pinned to its revision."""

        if self._model is None:
            # Set before the import: the hub client reads these at import time,
            # and the point is that no request ever leaves this machine.

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, revision=self.model_revision,
                device=os.environ.get("SPEAR_STANDARD_EMBED_DEVICE", "cpu"),
                trust_remote_code=self._trust, local_files_only=True,
            )

        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._load().encode(
            [self._document_prefix + text for text in texts],
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._load().encode(
            [self._query_prefix + text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0].tolist()


LICENSED_ORIGIN = "LICENSED_STANDARD"


@dataclass
class OffloadedStandardEmbedder:
    """Documents embedded on the GPU host; queries never leave this machine.

    The offload exists because bge-m3 over a corpus is minutes of saturated CPU
    here and seconds on the RTX 6000 there. It is refused for a licensed
    standard and allowed for a public one, because the two are not the same
    act: indexing sends the STANDARD'S TEXT to another machine — one shared
    under a common login — while answering sends only the user's question,
    which is why embed_query stays local whatever the origin.

    The pinned revision is enforced on both ends. The remote worker takes a
    model name and no revision, so it loads whatever that host's cache resolves
    `main` to; unchecked, the fingerprint this index is stamped with would name
    a revision the vectors were not produced by.
    """

    model_id: str
    model_revision: str
    target: str

    @property
    def compute_label(self) -> str:
        return f"offload:{self.target}"

    def __post_init__(self) -> None:
        self._local = SentenceTransformerStandardEmbedder(
            self.model_id, self.model_revision)
        remote = remote_model_revision(self.model_id, self.target)

        if remote != self.model_revision:
            raise RuntimeError(
                f"{self.target} resolves {self.model_id} to {remote or 'nothing'}, "
                f"not the pinned {self.model_revision}: the vectors would not "
                f"match the fingerprint they are stamped with")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import embedding

        vectors = embedding._embed_remote(
            list(texts), self.model_id, self.target, 16, True)

        if vectors is None:
            raise RuntimeError(
                f"embedding on {self.target} failed; refusing to fall back to "
                f"local CPU for a corpus this size — fix the host or unset "
                f"SPEAR_STANDARD_EMBED_REMOTE")

        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._local.embed_query(text)


def remote_model_revision(model_id: str, target: str) -> str | None:
    """What `main` resolves to for this model in the GPU host's cache."""
    import subprocess

    import embedding

    slug = "models--" + model_id.replace("/", "--")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    command += embedding.remote_ssh_opts()
    command += [target, f"cat ~/.cache/huggingface/hub/{slug}/refs/main"]

    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot reach {target}: {exc}") from exc

    # 255 is ssh's own failure: no connection, no authentication. Read as an
    # empty cache it became "the host resolves the model to nothing", and the
    # operator went looking for a cache mismatch on a host that was simply
    # off the network. The remote `cat` failing is the only real "nothing".

    if done.returncode == 255:
        reason = (done.stderr.strip().splitlines() or ["ssh failed"])[-1]
        raise RuntimeError(f"cannot reach {target}: {reason}")

    return done.stdout.strip() or None


DISCLOSURE_FILE = "offload-disclosure.json"


def offload_disclosures(store, standard_id: str, revision: str) -> list:
    """Every time this LICENSED corpus's text was sent to another machine."""
    path = store.revision_dir(standard_id, revision) / DISCLOSURE_FILE

    if not path.exists() or path.is_symlink():
        return []

    try:
        return list(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        return []


def record_offload_disclosure(store, standard_id: str, revision: str, *,
                              target: str, unit_count: int, model_id: str,
                              at: str | None = None) -> None:
    """Append the fact that a licensed corpus left this machine.

    Not part of any fingerprint, deliberately: the same model and revision
    produce the same vectors wherever they run, so where they ran must not
    change the index's identity. It is not an index property at all — it is a
    disclosure, and the question it answers ("was this licensed text ever sent
    anywhere?") has to survive the next rebuild, which is why it appends.
    """
    entries = offload_disclosures(store, standard_id, revision)
    entries.append({
        "target": target, "unit_count": unit_count, "model_id": model_id,
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    path = store.revision_dir(standard_id, revision) / DISCLOSURE_FILE
    path.write_text(json.dumps(entries, indent=2, sort_keys=True), "utf-8")


def configured_embedder(source_origin: str = LICENSED_ORIGIN, *,
                        allow_offload: bool = False):
    """The embedder for a standard of this origin, or None if none is set up.

    No model configured is not an error: vector search is optional, and
    retrieval falls back to lexical.

    `allow_offload` is the operator saying, for this one command, that this
    licensed corpus may be embedded on the GPU host anyway. It is a flag and
    not a stored property on purpose: relabelling the corpus PUBLIC would buy
    the same offload and would also tell training governance the text is
    exportable, which is a different and much larger claim.
    """
    model_id = os.environ.get("SPEAR_STANDARD_EMBED_MODEL")

    if not model_id:
        return None

    # An unpinned revision is an error: the index fingerprint would otherwise
    # depend on whatever the cache happened to hold.

    revision = os.environ.get("SPEAR_STANDARD_EMBED_REVISION")

    if not revision:
        raise ValueError("SPEAR_STANDARD_EMBED_REVISION must pin the local model revision")

    target = (os.environ.get("SPEAR_STANDARD_EMBED_REMOTE") or "").strip()

    if target and (source_origin != LICENSED_ORIGIN or allow_offload):
        return OffloadedStandardEmbedder(model_id, revision, target)

    return SentenceTransformerStandardEmbedder(model_id, revision)


def configured_local_embedder() -> SentenceTransformerStandardEmbedder | None:
    """Strictly local, whatever the origin. Kept for callers that must not
    offload under any circumstance."""

    return configured_embedder(LICENSED_ORIGIN)


UNIT_NORM_TOLERANCE = 1e-6

#: Texts per embed_documents call during a rebuild. The remote path already
#: ships 20 000 texts per worker process (embedding._embed_remote), each one
#: paying a model load, so a smaller chunk would only add loads. A multiple of
#: the worker's batch (16) and of SentenceTransformer's default (32).
EMBED_CHUNK = 20000


def _validated(vector: Sequence[float], *, dimension: int | None = None) -> list[float]:
    values = [float(value) for value in vector]

    if not values or (dimension is not None and len(values) != dimension):
        raise StandardStoreError("vector dimension mismatch")

    if not all(math.isfinite(value) for value in values):
        raise StandardStoreError("vector contains a non-finite value")

    return values


def _normalized(vector: Sequence[float], *, dimension: int | None = None) -> list[float]:
    values = _validated(vector, dimension=dimension)
    norm = math.sqrt(sum(value * value for value in values))

    if norm <= 0:
        raise StandardStoreError("zero-length vector is invalid")

    return [value / norm for value in values]


def _already_normalized(vector: Sequence[float], *,
                        dimension: int | None = None) -> list[float]:
    """Accept a stored vector verbatim; never re-derive one that is already unit length.

    Dividing an already-normalized float32-derived vector by its float64 norm shifts
    its last representable digits, which would change the vector index fingerprint --
    and so the StandardBinding -- on a rebuild that changed nothing.
    """

    values = _validated(vector, dimension=dimension)
    norm = math.sqrt(sum(value * value for value in values))

    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise StandardStoreError("cached vector is not unit-normalized")

    return values


def rebuild_vector_index(
    store: StandardStore, standard_id: str, revision: str,
    embedder: LocalStandardEmbedder, *, created_at: str | None = None,
    progress=None,
) -> StandardVectorIndexManifest:
    report(progress, "verifying corpus")
    source = store.verify_corpus(standard_id, revision)
    units = [unit for unit in store.load_units(standard_id, revision)
             if unit.retrievable]

    if not units:
        raise StandardStoreError("canonical corpus has no retrievable units")

    texts = [retrieval_text(unit) for unit in units]

    # WHERE the model ran belongs in the config, because it changes the answer.
    # Measured on NISTIR 6556 with bge-m3 at one pinned revision: vectors from
    # the GPU host differ from vectors computed here by up to 2.8e-4 per
    # component — reduced precision, not float32 rounding. Left out, the
    # fingerprint would claim two materially different indexes are the same
    # one, and a rebuild that moved between machines would look like a no-op.
    config = {
        "version": EMBEDDING_CONFIG_VERSION,
        "model_id": embedder.model_id,
        "model_revision": embedder.model_revision,
        "compute": getattr(embedder, "compute_label", "local"),
        "normalization": "l2-unit",
    }
    config_fingerprint = sha256_json(config)
    cache_dir = store.revision_dir(standard_id, revision) / "indexes" / "embedding-cache"

    if cache_dir.exists() and cache_dir.is_symlink():
        raise StandardStoreError("embedding cache path uses a symlink")

    import numpy

    # Rows follow the index's order -- source ids, sorted -- so the matrix
    # built here is the index's matrix and the cache's, byte for byte.
    order = sorted(range(len(units)), key=lambda position: units[position].source_id)
    units = [units[position] for position in order]
    texts = [texts[position] for position in order]
    text_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    cached_rows, cached = _load_cache(store, cache_dir, config_fingerprint)
    legacy = _legacy_cache_names(cache_dir)
    matrix = None
    missing: list[int] = []
    migrated: list[str] = []

    def rows_for(dimension):
        nonlocal matrix

        if matrix is None:
            matrix = numpy.empty((len(units), dimension), dtype="<f4")
        elif matrix.shape[1] != dimension:
            raise StandardStoreError("vector dimension mismatch")

        return matrix

    for position, text_hash in enumerate(text_hashes):
        report(progress, "embedding cache", position + 1, len(texts))
        row = cached_rows.get(text_hash)

        if row is not None:
            vector = numpy.asarray(cached[row], dtype=numpy.float64)

            # A corrupt or non-unit cache entry is re-embedded, never trusted.
            if numpy.isfinite(vector).all() and abs(
                    math.sqrt(float(vector @ vector)) - 1.0) <= UNIT_NORM_TOLERANCE:
                rows_for(len(vector))[position] = cached[row]
                continue

        key = sha256_json({"config": config_fingerprint,
                           "retrieval_text_sha256": text_hash})

        if f"{key}.json" in legacy:
            try:
                vector = _already_normalized(
                    store._read_json(cache_dir / f"{key}.json")["vector"])
            except Exception:
                pass
            else:
                rows_for(len(vector))[position] = vector
                migrated.append(f"{key}.json")
                continue

        missing.append(position)

    if missing:
        # In chunks, so a corpus of hundreds of thousands of units shows its
        # progress, does not travel as one payload, and is never held as
        # Python lists: each chunk is normalised straight into the matrix.
        # A multiple of every backend's batch size, so each batch holds the
        # same texts it would have held in one call and the vectors do not
        # move.

        label = f"embedding ({getattr(embedder, 'compute_label', 'local')})"

        for start in range(0, len(missing), EMBED_CHUNK):
            # Unthrottled: one call per chunk is already few, and the
            # throttle's step would not fall on chunk boundaries.
            if progress is not None:
                progress(label, start, len(missing))

            positions = missing[start:start + EMBED_CHUNK]
            generated = embedder.embed_documents([texts[index] for index in positions])

            if len(generated) != len(positions):
                raise StandardStoreError("embedding backend returned a short result")

            dimension = len(generated[0]) if generated else 0
            rows_for(dimension)[positions] = _normalized_rows(generated, dimension)

        if progress is not None:
            progress(label, len(missing), len(missing))

    if matrix is None or matrix.shape[1] < 1:
        raise StandardStoreError("vector dimension mismatch")

    dimension = matrix.shape[1]
    ids = [unit.source_id for unit in units]
    matrix = encode_vectors(matrix, dimension)
    _save_cache(store, cache_dir, config_fingerprint, config, text_hashes, matrix)

    # The entries just carried into the single-file cache are dead weight as
    # files: one each, hundreds of thousands for a large document.
    for name in migrated:
        try:
            (cache_dir / name).unlink()
        except OSError:
            pass

    entries = dict.fromkeys(ids)
    text_hashes = dict(zip(ids, text_hashes))
    index = {
        "schema_version": 2, "standard_id": standard_id, "revision": revision,
        "embedding_config": config, "dimension": dimension, "ids": ids,
        "retrieval_text_sha256": dict(sorted(text_hashes.items())),
        "vectors": {"file": VECTORS_FILE, "dtype": "float32",
                    "shape": [len(ids), dimension],
                    "sha256": hashlib.sha256(matrix).hexdigest()},
    }
    fingerprint = sha256_json({
        "vector_index_version": VECTOR_INDEX_VERSION,
        "source_corpus_sha256": source.corpus_manifest_sha256,
        "index": index,
    })
    manifest = StandardVectorIndexManifest(
        standard_id, revision, source.source_pdf_sha256,
        source.corpus_manifest_sha256, embedder.model_id,
        embedder.model_revision, dimension, "l2-unit", VECTOR_FORMAT,
        VECTOR_INDEX_VERSION, len(entries), config_fingerprint, fingerprint,
        created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    store.save_vector_index(manifest, index, matrix)

    return manifest


#: The embedding cache: one float32 matrix per embedding configuration, with
#: the retrieval-text hash of each row beside it. It used to be one JSON file
#: per vector, each written with an fsync -- 262 470 of them for the Arm
#: manual, most of a 25-minute rebuild, and every vector held as a Python
#: list until the end, 15 GB of it.
CACHE_SCHEMA_VERSION = 2
_LEGACY_CACHE_ENTRY = re.compile(r"[0-9a-f]{64}\.json")


def _cache_paths(cache_dir: Path, config_fingerprint: str) -> tuple[Path, Path]:
    return (cache_dir / f"cache-{config_fingerprint}.npy",
            cache_dir / f"cache-{config_fingerprint}.json")


def _load_cache(store, cache_dir: Path, config_fingerprint: str):
    """({text_sha256: row}, matrix) for this configuration, or ({}, None).

    Anything that does not check out -- a missing file, a checksum that does
    not match, a shape that disagrees -- is an empty cache: the vectors are
    embedded again rather than trusted.
    """
    import numpy

    matrix_path, meta_path = _cache_paths(cache_dir, config_fingerprint)

    if (not matrix_path.is_file() or not meta_path.is_file()
            or matrix_path.is_symlink() or meta_path.is_symlink()):
        return {}, None

    try:
        meta = store._read_json(meta_path)
        hashes = meta["text_sha256"]
        digest = hashlib.sha256()

        with open(matrix_path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)

        if (meta.get("schema_version") != CACHE_SCHEMA_VERSION
                or meta.get("config_fingerprint") != config_fingerprint
                or digest.hexdigest() != meta.get("sha256")):
            return {}, None

        matrix = numpy.load(matrix_path, mmap_mode="r", allow_pickle=False)

        if matrix.ndim != 2 or matrix.shape[0] != len(hashes):
            return {}, None
    except Exception:
        return {}, None

    return {text_hash: row for row, text_hash in enumerate(hashes)}, matrix


def _save_cache(store, cache_dir: Path, config_fingerprint: str, config,
                text_hashes: list[str], matrix: bytes) -> None:
    """Replace this configuration's cache with the vectors of this rebuild."""
    if cache_dir.exists() and cache_dir.is_symlink():
        raise StandardStoreError("embedding cache path uses a symlink")

    matrix_path, meta_path = _cache_paths(cache_dir, config_fingerprint)
    store._atomic_write(matrix_path, matrix)
    store._atomic_write(meta_path, canonical_json({
        "schema_version": CACHE_SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint, "config": config,
        "sha256": hashlib.sha256(matrix).hexdigest(),
        "text_sha256": text_hashes,
    }))


def _legacy_cache_names(cache_dir: Path) -> frozenset[str]:
    """The one-file-per-vector entries a previous format left, read once to
    migrate them."""
    try:
        with os.scandir(cache_dir) as found:
            return frozenset(entry.name for entry in found
                             if _LEGACY_CACHE_ENTRY.fullmatch(entry.name))
    except FileNotFoundError:
        return frozenset()


def _normalized_rows(vectors: Sequence[Sequence[float]], dimension: int):
    """Unit-length float32 rows, refused whole if any row is not a vector."""
    import numpy

    try:
        rows = numpy.asarray(vectors, dtype=numpy.float64)
    except (TypeError, ValueError):
        raise StandardStoreError("vector dimension mismatch") from None

    if rows.ndim != 2 or rows.shape[1] != dimension or dimension < 1:
        raise StandardStoreError("vector dimension mismatch")

    if not numpy.isfinite(rows).all():
        raise StandardStoreError("vector contains a non-finite value")

    norms = numpy.sqrt(numpy.einsum("ij,ij->i", rows, rows))

    if (norms <= 0).any():
        raise StandardStoreError("zero-length vector is invalid")

    return (rows / norms[:, None]).astype("<f4")


def encode_vectors(rows: Sequence[Sequence[float]], dimension: int) -> bytes:
    """The .npy bytes of `rows` as a float32 matrix, deterministically."""
    import io

    import numpy

    matrix = numpy.asarray(rows, dtype="<f4").reshape(len(rows), dimension)
    buffer = io.BytesIO()
    numpy.save(buffer, matrix, allow_pickle=False)

    return buffer.getvalue()


def vector_entries(index: Mapping[str, object]) -> dict[str, list[float]]:
    """{source_id: vector} from a loaded index. For inspection and tests: a
    search reads index["matrix"] directly."""
    return {source_id: [float(value) for value in row]
            for source_id, row in zip(index["ids"], index["matrix"])}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise StandardStoreError("vector dimension mismatch")

    return sum(float(a) * float(b) for a, b in zip(left, right))
